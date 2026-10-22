// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <numbers>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <ddc/pdi.hpp>

#include <benchmark/benchmark.h>

#include <hdf5.h>
#include <paraconf.h>

#include "electromagnetic_solver.hpp"

namespace {
using taylor_green_vortex::CompressiblePhysics;
using taylor_green_vortex::CompressibleSolver;
using taylor_green_vortex::Configuration;
using taylor_green_vortex::Coupling;
using taylor_green_vortex::ElectromagneticSolver;
using taylor_green_vortex::Field;
using taylor_green_vortex::IncompressibleSolver;
using taylor_green_vortex::Physics;
using taylor_green_vortex::ProfilingRegion;
using taylor_green_vortex::Reference;
using taylor_green_vortex::StructuredGrid;
using Clock = std::chrono::steady_clock;
int nodes = 64;
Configuration configuration;
Physics physics = Physics::Incompressible;
std::filesystem::path output_directory = "taylor_green_vortex_results";
bool verify = false;
bool reference_run = false;
bool strategy_suite = false, reference_suite = false;
bool benchmark_output_requested = false;
int strategy_index = 0;
double reference_final_time = 10;
bool write_snapshots = true;
bool benchmark_failed = false;

void check_hdf(herr_t status)
{
    if (status < 0)
        throw std::runtime_error("HDF5 write failed");
}
void dataset(
        hid_t parent,
        char const* name,
        std::vector<hsize_t> const& shape,
        double const* values)
{
    hid_t const space = H5Screate_simple(static_cast<int>(shape.size()), shape.data(), nullptr);
    if (space < 0)
        throw std::runtime_error("HDF5 dataspace creation failed");
    hid_t const data = H5Dcreate2(
            parent,
            name,
            H5T_IEEE_F64LE,
            space,
            H5P_DEFAULT,
            H5P_DEFAULT,
            H5P_DEFAULT);
    if (data < 0) {
        H5Sclose(space);
        throw std::runtime_error("HDF5 dataset creation failed");
    }
    herr_t const status = H5Dwrite(data, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, values);
    H5Dclose(data);
    H5Sclose(space);
    check_hdf(status);
}

void string_attribute(hid_t parent, char const* name, std::string const& value)
{
    hid_t const type = H5Tcopy(H5T_C_S1);
    check_hdf(H5Tset_size(type, value.size() + 1));
    hid_t const space = H5Screate(H5S_SCALAR);
    hid_t const attribute = H5Acreate2(parent, name, type, space, H5P_DEFAULT, H5P_DEFAULT);
    check_hdf(H5Awrite(attribute, type, value.c_str()));
    H5Aclose(attribute);
    H5Sclose(space);
    H5Tclose(type);
}

struct Monitor
{
    double time, energy, derivative, enstrophy, dissipation, pressure_dilatation,
            viscous_dissipation;
    double mass, total_energy, divergence, rho_min, pressure_min, step_seconds;
    double potential_gauge, vorticity_divergence, curl_residual, reconstruction_residual;
    std::array<double, 9> electromagnetic {};
};

class Simulation
{
public:
    StructuredGrid dec;
    Physics const kind;
    int const components;
    Field state, initial, stage, rhs, velocity, gradient, vorticity, density, pressure, temperature,
            heat_flux, velocity_divergence, velocity_rate, vector_potential;
    std::unique_ptr<IncompressibleSolver> incompressible;
    std::unique_ptr<CompressibleSolver> compressible;
    std::unique_ptr<ElectromagneticSolver> electromagnetic;
    explicit Simulation(int n, Physics selected)
        : dec(n, configuration)
        , kind(selected)
        , components(selected == Physics::Compressible ? 5 : 3)
        , state("state", components, dec.cells)
        , initial("initial", components, dec.cells)
        , stage("stage", components, dec.cells)
        , rhs("rhs", components, dec.cells)
        , velocity("velocity", 3, dec.cells)
        , gradient("gradient", 9, dec.cells)
        , vorticity("vorticity", 3, dec.cells)
        , density("density", 1, dec.cells)
        , pressure("pressure", 1, dec.cells)
        , temperature("temperature", 1, dec.cells)
        , heat_flux("heat_flux", 3, dec.cells)
        , velocity_divergence("velocity_divergence", 1, dec.cells)
        , velocity_rate("reconstructed_velocity_rate", 3, dec.cells)
        , vector_potential("export_vector_potential", 3, dec.cells)
        , incompressible(
                  selected == Physics::Incompressible ? std::make_unique<IncompressibleSolver>(dec)
                                                      : nullptr)
        , compressible(
                  selected == Physics::Compressible ? std::make_unique<CompressibleSolver>(dec)
                                                    : nullptr)
        , electromagnetic(
                  configuration.coupling == Coupling::HD
                          ? nullptr
                          : std::make_unique<ElectromagneticSolver>(dec, configuration))
    {
        if (selected != Physics::Incompressible && selected != Physics::Compressible)
            throw std::runtime_error("Selected physics is not implemented");
        Kokkos::deep_copy(vector_potential, 0.0);
        reset();
    }
    void reset()
    {
        if (electromagnetic)
            electromagnetic->reset();
        if (incompressible) {
            incompressible->mean_velocity = {};
            incompressible->mean_rate = {};
        }
        Field const q = state;
        bool const gas = kind == Physics::Compressible;
        int const n = dec.n;
        double const h = dec.h;
        Kokkos::parallel_for(
                "taylor_green_initial_condition",
                dec.cells,
                KOKKOS_LAMBDA(int i) {
                    double const x = -std::numbers::pi + (i / (n * n)) * h;
                    double const y = -std::numbers::pi + ((i / n) % n) * h;
                    double const z = -std::numbers::pi + (i % n) * h;
                    double const u = sin(x) * cos(y) * cos(z), v = -cos(x) * sin(y) * cos(z);
                    double const p = Reference::reference_pressure
                                     + (cos(2 * x) + cos(2 * y)) * (cos(2 * z) + 2) / 16;
                    if (gas) {
                        double const rho = p / Reference::reference_pressure;
                        q(0, i) = rho;
                        q(1, i) = rho * u;
                        q(2, i) = rho * v;
                        q(3, i) = 0;
                        q(4, i) = p / (Reference::gamma - 1) + 0.5 * rho * (u * u + v * v);
                    } else {
                        q(0, i) = u;
                        q(1, i) = v;
                        q(2, i) = 0;
                    }
                });
        if (!gas)
            incompressible->initialize(state);
    }
    void evaluate(Field const& q, Field const& result, bool reconstruct_pressure = false)
    {
        ProfilingRegion const region("rhs");
        Field acceleration;
        if (electromagnetic) {
            if (incompressible) {
                incompressible->reconstruct_velocity(q, incompressible->velocity);
                Kokkos::deep_copy(velocity, incompressible->velocity);
            } else {
                Field const u = velocity;
                Kokkos::parallel_for(
                        "coupled_velocity",
                        dec.cells,
                        KOKKOS_LAMBDA(int i) {
                            for (int a = 0; a < 3; ++a)
                                u(a, i) = q(a + 1, i) / q(0, i);
                        });
            }
            electromagnetic->evaluate(velocity);
            acceleration = electromagnetic->force;
            if (incompressible)
                for (int a = 0; a < 3; ++a) {
                    double sum;
                    Kokkos::parallel_reduce(
                            "mean_lorentz_force",
                            dec.cells,
                            KOKKOS_LAMBDA(int i, double& value) { value += acceleration(a, i); },
                            sum);
                    incompressible->mean_rate[a] = sum / dec.cells;
                }
        }
        if (kind == Physics::Compressible) {
            compressible->rhs(q, result);
            if (electromagnetic) {
                Field const power = electromagnetic->fluid_power;
                Kokkos::parallel_for(
                        "compressible_field_exchange",
                        dec.cells,
                        KOKKOS_LAMBDA(int i) {
                            for (int a = 0; a < 3; ++a)
                                result(a + 1, i) += acceleration(a, i);
                            result(4, i) += power(0, i);
                        });
            }
        } else
            incompressible
                    ->rhs(q,
                          result,
                          reconstruct_pressure,
                          acceleration,
                          electromagnetic != nullptr);
    }
    // Reconstruct pressure directly from the current state for export.
    void compute_pressure()
    {
        if (kind == Physics::Incompressible) {
            if (electromagnetic) {
                incompressible->reconstruct_velocity(state, velocity);
                electromagnetic->evaluate(velocity);
            }
            incompressible->compute_pressure(
                    state,
                    pressure,
                    Reference::reference_pressure,
                    electromagnetic ? electromagnetic->force : Field());
            incompressible->monitor_constraints(state);
            Kokkos::deep_copy(velocity, incompressible->velocity);
            Kokkos::deep_copy(vorticity, incompressible->vorticity);
            Kokkos::deep_copy(vector_potential, incompressible->vector_potential);
        } else {
            Field const q = state, output = pressure;
            Kokkos::parallel_for(
                    "export_compressible_pressure",
                    dec.cells,
                    KOKKOS_LAMBDA(int i) { output(0, i) = Reference::pressure(q, i); });
        }
    }
    double timestep()
    {
        ProfilingRegion const region("timestep");
        bool const gas = kind == Physics::Compressible;
        if (!gas)
            incompressible->reconstruct_velocity(state, velocity);
        if (electromagnetic) {
            if (gas) {
                Field const fluid = state, u = velocity;
                Kokkos::parallel_for(
                        "cfl_coupled_velocity",
                        dec.cells,
                        KOKKOS_LAMBDA(int i) {
                            for (int a = 0; a < 3; ++a)
                                u(a, i) = fluid(a + 1, i) / fluid(0, i);
                        });
            }
            electromagnetic->evaluate(velocity);
        }
        Field const q = gas ? state : velocity;
        Field const magnetic = electromagnetic ? electromagnetic->induction : Field();
        bool const has_magnetic = electromagnetic && electromagnetic->magnetic();
        double maximum;
        Kokkos::parallel_reduce(
                "cfl_speed",
                dec.cells,
                KOKKOS_LAMBDA(int i, double& value) {
                    double speed = 0;
                    if (gas) {
                        double alfven_squared = 0;
                        if (has_magnetic)
                            for (int a = 0; a < 3; ++a)
                                alfven_squared += magnetic(a, i) * magnetic(a, i) / q(0, i);
                        double const sound = sqrt(
                                Reference::model().sound_speed_squared(Reference::state(q, i))
                                + alfven_squared);
                        for (int a = 0; a < 3; ++a)
                            speed += fabs(q(a + 1, i) / q(0, i)) + sound;
                    } else
                        for (int a = 0; a < 3; ++a)
                            speed += fabs(q(a, i));
                    if (!gas && has_magnetic) {
                        double alfven_squared = 0;
                        for (int a = 0; a < 3; ++a)
                            alfven_squared += magnetic(a, i) * magnetic(a, i);
                        speed += 3 * sqrt(alfven_squared);
                    }
                    value = fmax(value, speed);
                },
                Kokkos::Max<double>(maximum));
        double diffusivity = 4.0 * Reference::viscosity / 3.0;
        if (gas) {
            Kokkos::parallel_reduce(
                    "thermal_viscous_diffusivity",
                    dec.cells,
                    KOKKOS_LAMBDA(int i, double& value) {
                        double const local
                                = fmax(4.0 * Reference::viscosity / 3.0,
                                       Reference::model().conductivity(Reference::state(q, i))
                                               / Reference::model()
                                                         .heat_capacity_at_constant_volume(
                                                                 Reference::state(q, i)))
                                  / q(0, i);
                        value = fmax(value, local);
                    },
                    Kokkos::Max<double>(diffusivity));
        }
        // Fixed CFL and diffusion constants; no user-controlled time step.
        double const fluid_limit = std::
                min(0.3 * dec.h / std::max(maximum, 1e-12), 0.1 * dec.h * dec.h / diffusivity);
        return electromagnetic ? std::min(fluid_limit, electromagnetic->timestep_limit())
                               : fluid_limit;
    }
    void advance(double dt)
    {
        ProfilingRegion const region("evolution");
        Kokkos::deep_copy(initial, state);
        if (electromagnetic)
            electromagnetic->begin_step();
        std::array<double, 3> const initial_mean
                = incompressible ? incompressible->mean_velocity : std::array<double, 3> {};
        Field const q = state, q0 = initial, s = stage, f = rhs;
        int const count = components;
        for (int step = 0; step < 3; ++step) {
            ProfilingRegion const stage_region("ssprk3-stage");
            evaluate(q, f);
            double const a = step == 0 ? 0.0 : (step == 1 ? 0.75 : 1.0 / 3.0);
            Kokkos::parallel_for(
                    "ssprk3_stage",
                    dec.cells,
                    KOKKOS_LAMBDA(int i) {
                        for (int c = 0; c < count; ++c)
                            s(c, i) = a * q0(c, i) + (1 - a) * (q(c, i) + dt * f(c, i));
                    });
            Kokkos::deep_copy(q, s);
            if (electromagnetic)
                electromagnetic->advance_stage(step, dt);
            if (incompressible)
                for (int axis = 0; axis < 3; ++axis)
                    incompressible->mean_velocity[axis]
                            = a * initial_mean[axis]
                              + (1 - a)
                                        * (incompressible->mean_velocity[axis]
                                           + dt * incompressible->mean_rate[axis]);
        }
    }
    Monitor monitor(double time, double step_seconds = 0)
    {
        evaluate(state, rhs, true);
        bool const gas = kind == Physics::Compressible;
        double const reconstruction_residual = gas ? 0 : incompressible->last_relative_residual;
        if (!gas) {
            Kokkos::deep_copy(velocity, incompressible->velocity);
            Kokkos::deep_copy(vector_potential, incompressible->vector_potential);
            Kokkos::deep_copy(vorticity, incompressible->vorticity);
            incompressible->divergence(velocity_divergence);
            // Differentiate the linear elliptic reconstruction to obtain du/dt
            // from dW/dt; u dot dW/dt is not the kinetic-energy derivative.
            incompressible->reconstruct_velocity(rhs, velocity_rate, true);
        }
        Field const q = state, u = velocity, rho = density, p = pressure, t = temperature,
                    g = gradient, w = vorticity, f = gas ? rhs : velocity_rate;
        Field const projected_pressure = gas ? pressure : incompressible->pressure;
        Kokkos::parallel_for(
                "monitor_primitives",
                dec.cells,
                KOKKOS_LAMBDA(int i) {
                    rho(0, i) = gas ? q(0, i) : 1.0;
                    for (int a = 0; a < 3; ++a)
                        u(a, i) = gas ? q(a + 1, i) / q(0, i) : u(a, i);
                    p(0, i) = gas ? Reference::pressure(q, i)
                                  : Reference::reference_pressure + projected_pressure(0, i);
                    t(0,
                      i) = gas ? Reference::model().primitives(Reference::state(q, i)).temperature
                               : Reference::reference_pressure / Reference::gas_constant;
                });
        for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b)
                dec.nodal_derivative_component(u, a, g, 3 * a + b, b);
        if (gas) {
            dec.reconstructed_derivative_one_form(u, w);
            dec.negative_reduced_codifferential(u, velocity_divergence);
        }
        Field const div_field = velocity_divergence;
        CompressiblePhysics const physical_model = Reference::model();
        if (gas) {
            Field const heat = heat_flux;
            for (int a = 0; a < 3; ++a)
                dec.nodal_derivative_component(t, 0, heat, a, a);
            Kokkos::parallel_for(
                    "fourier_heat_flux",
                    dec.cells,
                    KOKKOS_LAMBDA(int i) {
                        std::array<double, 3> const values = physical_model.heat_flux(
                                Reference::state(q, i),
                                {heat(0, i), heat(1, i), heat(2, i)});
                        for (int a = 0; a < 3; ++a)
                            heat(a, i) = values[a];
                    });
        } else
            Kokkos::deep_copy(heat_flux, 0.0);
        double sums[8] = {};
        // Independent reductions keep the diagnostic definition explicit.
        for (int diagnostic = 0; diagnostic < 8; ++diagnostic) {
            double value;
            Kokkos::parallel_reduce(
                    "taylor_green_monitor",
                    dec.cells,
                    KOKKOS_LAMBDA(int i, double& sum) {
                        double speed2 = 0, omega2 = 0, rate = 0;
                        for (int a = 0; a < 3; ++a) {
                            speed2 += u(a, i) * u(a, i);
                            omega2 += w(a, i) * w(a, i);
                            rate += u(a, i) * f(gas ? a + 1 : a, i);
                        }
                        if (gas)
                            rate -= 0.5 * speed2 * f(0, i);
                        double const div = div_field(0, i);
                        CompressiblePhysics::VelocityGradient grad {};
                        for (int a = 0; a < 3; ++a)
                            for (int b = 0; b < 3; ++b)
                                grad[a][b] = g(3 * a + b, i);
                        CompressiblePhysics::VelocityGradient const tau
                                = physical_model.stress(grad);
                        double stress_work = 0;
                        for (int a = 0; a < 3; ++a)
                            for (int b = 0; b < 3; ++b)
                                stress_work += tau[a][b] * grad[a][b];
                        if (diagnostic == 0)
                            sum += 0.5 * rho(0, i) * speed2;
                        else if (diagnostic == 1)
                            sum += rate;
                        else if (diagnostic == 2)
                            sum += 0.5 * rho(0, i) * omega2;
                        else if (diagnostic == 3)
                            sum += p(0, i) * div;
                        else if (diagnostic == 4)
                            sum += stress_work;
                        else if (diagnostic == 5)
                            sum += rho(0, i);
                        else if (diagnostic == 6)
                            sum += gas ? q(4, i) : 0.5 * speed2;
                        else
                            sum += div * div;
                    },
                    value);
            sums[diagnostic] = value / dec.cells;
        }
        double minrho, minpressure;
        Kokkos::parallel_reduce(
                "density_min",
                dec.cells,
                KOKKOS_LAMBDA(int i, double& value) { value = fmin(value, rho(0, i)); },
                Kokkos::Min<double>(minrho));
        Kokkos::parallel_reduce(
                "pressure_min",
                dec.cells,
                KOKKOS_LAMBDA(int i, double& value) { value = fmin(value, p(0, i)); },
                Kokkos::Min<double>(minpressure));
        Monitor
                result {time,
                        sums[0],
                        sums[1],
                        sums[2],
                        2 * Reference::viscosity * sums[2],
                        sums[3],
                        sums[4],
                        sums[5],
                        sums[6],
                        sqrt(sums[7]),
                        minrho,
                        minpressure,
                        step_seconds,
                        gas ? 0 : incompressible->gauge_rms,
                        gas ? 0 : incompressible->vorticity_divergence_rms,
                        gas ? 0 : incompressible->curl_residual_rms,
                        reconstruction_residual};
        if (electromagnetic) {
            result.electromagnetic = electromagnetic->monitor(time);
            result.total_energy += result.electromagnetic[1] + result.electromagnetic[2];
        }
        if (!std::
                    all_of(std::begin(sums),
                           std::end(sums),
                           [](double value) { return std::isfinite(value); })
            || !(minrho > 0) || !(minpressure > 0) || !std::isfinite(minrho)
            || !std::isfinite(minpressure))
            throw std::runtime_error("Nonfinite state or nonpositive density/pressure");
        return result;
    }
};

std::string header(int n, Physics kind)
{
    std::ostringstream text;
    text << "# participant: SimiLie\n# code name: SimiLie\n# mesh resolution: " << n << " " << n
         << " " << n << "\n# discretization: native Cartesian DEC, "
         << (kind == Physics::Compressible ? "centered kinetic-energy-preserving flux"
                                           : "vorticity with Coulomb-gauge vector potential")
         << "\n# order of convergence: " << 2
         << " spatial; SSPRK3 temporal\n# mesh file name: implicit periodic Cartesian grid\n# "
            "density: 1\n# velocity: 1\n# dynamic viscosity: 0.000625\n# reference length: 1\n";
    text << "# configuration: " << configuration.name() << "\n";
    if (configuration.coupling != Coupling::HD)
        text << "# extended Taylor-Green case; not the workshop HD reference\n"
             << "# epsilon: 1\n# permeability: 1\n# guide field: 0.25\n"
             << "# magnetic amplitude: 0.15\n# electric potential amplitude: 0.1\n"
             << "# magnetic resistivity: 0.001\n# EHD/FMHD electrical resistivity: 10\n";
    return text.str();
}

// One PDI configuration serves every resolution in a suite.
struct PdiSession
{
    PC_tree_t config {};
    bool initialized = false;
    ~PdiSession()
    {
        if (initialized) {
            PDI_finalize();
            PC_tree_destroy(&config);
        }
    }
};
PdiSession* pdi_session = nullptr;

class Export
{
    hid_t m_file;
    std::filesystem::path m_base;
    int m_n;
    std::vector<double> m_times;

public:
    Export(std::filesystem::path base, int n) : m_base(std::move(base)), m_n(n)
    {
        m_file = H5Fcreate(
                (m_base.string() + ".h5").c_str(),
                H5F_ACC_TRUNC,
                H5P_DEFAULT,
                H5P_DEFAULT);
        if (m_file < 0)
            throw std::runtime_error("Cannot create HDF5 output");
        string_attribute(m_file, "configuration", configuration.name());
        if (configuration.coupling != Coupling::HD)
            string_attribute(
                    m_file,
                    "electromagnetic_parameters",
                    "epsilon=1,mu=1,Bguide=0.25,Bamplitude=0.15,phi_amplitude=0.1,"
                    "eta_mhd=0.001,eta_ehd_fmhd=10");
        if (!write_snapshots)
            return;
        std::vector<double> position(3 * n * n * n);
        double const h = 2 * std::numbers::pi / n;
        for (int i = 0; i < n * n * n; ++i) {
            position[3 * i] = -std::numbers::pi + (i / (n * n)) * h;
            position[3 * i + 1] = -std::numbers::pi + ((i / n) % n) * h;
            position[3 * i + 2] = -std::numbers::pi + (i % n) * h;
        }
        dataset(m_file, "position", {hsize_t(n), hsize_t(n), hsize_t(n), 3}, position.data());
    }
    ~Export()
    {
        if (m_file >= 0)
            H5Fclose(m_file);
    }
    void final_state(Simulation const& simulation)
    {
        auto save = [&](char const* name, Field const& field) {
            Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> const host
                    = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), field);
            dataset(m_file, name, {host.extent(0), host.extent(1)}, host.data());
        };
        save("final_state", simulation.state);
        if (simulation.electromagnetic)
            save("final_electromagnetic_state", simulation.electromagnetic->state);
    }
    void snapshot(Simulation& simulation, double time)
    {
        simulation.compute_pressure();
        std::vector<std::pair<char const*, Field>>
                fields {{"velocity", simulation.velocity},
                        std::pair {"vorticity", simulation.vorticity},
                        std::pair {"density", simulation.density},
                        std::pair {"pressure", simulation.pressure},
                        std::pair {"temperature", simulation.temperature},
                        std::pair {"heat_flux", simulation.heat_flux},
                        std::pair {"vector_potential", simulation.vector_potential}};
        if (simulation.electromagnetic) {
            ElectromagneticSolver const& em = *simulation.electromagnetic;
            fields
                    .insert(fields.end(),
                            {{"electric_field", em.electric_field},
                             {"current", em.current},
                             {"lorentz_force", em.force},
                             {"joule_heating", em.joule_heating}});
            if (em.magnetic())
                fields.emplace_back("induction", em.induction);
            if (em.electric())
                fields
                        .insert(fields.end(),
                                {{"displacement", em.displacement},
                                 {"charge_density", em.charge_density},
                                 {"gauss_error", em.gauss_error}});
            if (configuration.coupling == Coupling::EHD)
                fields.emplace_back("electric_potential", em.potential);
        }
        // PDI owns each snapshot write, just as in the scalar-field example.
        // Close our monitor handle while the declarative plugin opens the file.
        check_hdf(H5Fclose(m_file));
        m_file = -1;
        if (!pdi_session->initialized) {
            std::ostringstream yaml;
            yaml << "logging: error\nmetadata:\n  N: int\n  export_id: int\n"
                 << "  file_name_size: size_t\n  file_name: {type: array, subtype: char, size: "
                    "'$file_name_size'}\ndata:\n"
                 << "  time: {type: array, subtype: double, size: 1}\n";
            for (std::pair<char const*, Field> const& item : fields) {
                yaml << "  " << item.first << ": {type: array, subtype: double, size: ["
                     << "'$N', '$N', '$N'";
                if (item.second.extent(0) > 1)
                    yaml << ", " << item.second.extent(0);
                yaml << "]}\n";
            }
            yaml << "plugins:\n  decl_hdf5:\n    file: '${file_name}'\n"
                 << "    on_event: export\n    collision_policy: write_into\n    write:\n"
                 << "      time: {dataset: 'snapshot_${export_id}/time'}\n";
            for (std::pair<char const*, Field> const& item : fields)
                yaml << "      " << item.first << ": {dataset: 'snapshot_${export_id}/"
                     << item.first << "'}\n";
            pdi_session->config = PC_parse_string(yaml.str().c_str());
            if (PDI_init(pdi_session->config) != PDI_OK) {
                PC_tree_destroy(&pdi_session->config);
                throw std::runtime_error("Cannot initialize Taylor-Green PDI output");
            }
            pdi_session->initialized = true;
        }
        int export_id = static_cast<int>(m_times.size());
        std::string const filename = m_base.string() + ".h5";
        std::size_t const filename_size = filename.size();
        if (PDI_share("N", &m_n, PDI_OUT) != PDI_OK
            || PDI_share("file_name_size", &filename_size, PDI_OUT) != PDI_OK
            || PDI_share("file_name", filename.c_str(), PDI_OUT) != PDI_OK
            || PDI_share("export_id", &export_id, PDI_OUT) != PDI_OK
            || PDI_share("time", &time, PDI_OUT) != PDI_OK)
            throw std::runtime_error("Cannot expose PDI snapshot metadata");
        std::vector<std::vector<double>> buffers;
        buffers.reserve(fields.size());
        for (std::pair<char const*, Field> const& item : fields) {
            Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> const host
                    = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), item.second);
            int const components = static_cast<int>(host.extent(0));
            buffers.emplace_back(simulation.dec.cells * components);
            std::vector<double>& values = buffers.back();
            for (int i = 0; i < simulation.dec.cells; ++i)
                for (int c = 0; c < components; ++c)
                    values[i * components + c] = host(c, i);
            if (PDI_share(item.first, values.data(), PDI_OUT) != PDI_OK)
                throw std::runtime_error("Cannot expose PDI snapshot field");
        }
        if (PDI_event("export") != PDI_OK)
            throw std::runtime_error("PDI snapshot export failed");
        for (std::pair<char const*, Field> const& item : fields)
            PDI_reclaim(item.first);
        PDI_reclaim("time");
        PDI_reclaim("export_id");
        PDI_reclaim("file_name");
        PDI_reclaim("file_name_size");
        PDI_reclaim("N");
        m_file = H5Fopen((m_base.string() + ".h5").c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
        if (m_file < 0)
            throw std::runtime_error("Cannot reopen monitor output after PDI export");
        m_times.push_back(time);
        write_xmf();
    }
    void monitors(std::vector<Monitor> const& monitors)
    {
        std::vector<double> table;
        for (Monitor const& m : monitors)
            for (double v :
                 {m.time,
                  m.energy,
                  m.derivative,
                  -m.derivative,
                  m.enstrophy,
                  m.dissipation,
                  m.pressure_dilatation,
                  m.viscous_dissipation,
                  m.mass,
                  m.total_energy,
                  m.divergence,
                  m.rho_min,
                  m.pressure_min,
                  m.step_seconds})
                table.push_back(v);
        dataset(m_file, "monitors", {monitors.size(), 14}, table.data());
        std::vector<double> constraints;
        for (Monitor const& m : monitors)
            for (double value :
                 {m.time,
                  m.potential_gauge,
                  m.vorticity_divergence,
                  m.curl_residual,
                  m.reconstruction_residual})
                constraints.push_back(value);
        dataset(m_file, "reconstruction_monitors", {monitors.size(), 5}, constraints.data());
        if (configuration.coupling != Coupling::HD) {
            std::vector<double> electromagnetic;
            for (Monitor const& m : monitors)
                electromagnetic
                        .insert(electromagnetic.end(),
                                m.electromagnetic.begin(),
                                m.electromagnetic.end());
            dataset(m_file,
                    "electromagnetic_monitors",
                    {monitors.size(), 9},
                    electromagnetic.data());
            string_attribute(
                    m_file,
                    "electromagnetic_monitor_columns",
                    "time,magnetic_energy,electric_energy,fluid_electric_work,ohmic_heating,"
                    "magnetic_divergence_rms,gauss_rms,mean_charge,energy_exchange_residual");
        }
        std::string const names
                = "time,kinetic_energy,dEk_dt,kinetic_dissipation,enstrophy,enstrophy_dissipation,"
                  "pressure_dilatation,viscous_dissipation,mass,total_energy,divergence_rms,"
                  "density_min,pressure_min,step_seconds";
        string_attribute(m_file, "monitor_columns", names);
    }
    void write_xmf()
    {
        std::ofstream out(m_base.string() + ".xmf", std::ios::trunc);
        if (!out)
            throw std::runtime_error("Cannot open XDMF output");
        out << R"XMF(<?xml version="1.0"?>
<Xdmf Version="3.0">
 <Domain>
  <Grid Name="TimeSeries" GridType="Collection" CollectionType="Temporal">
)XMF";
        std::string const file = m_base.filename().string() + ".h5";
        std::string const dimensions
                = std::to_string(m_n) + " " + std::to_string(m_n) + " " + std::to_string(m_n);
        for (std::size_t i = 0; i < m_times.size(); ++i) {
            out << "   <Grid Name=\"snapshot_" << i << "\" GridType=\"Uniform\">\n"
                << "    <Time Value=\"" << m_times[i] << "\"/>\n"
                << "    <Topology TopologyType=\"3DSMesh\" Dimensions=\"" << dimensions << "\"/>\n"
                << "    <Geometry GeometryType=\"XYZ\">\n"
                << "     <DataItem Dimensions=\"" << dimensions
                << " 3\" NumberType=\"Float\" Precision=\"8\" Format=\"HDF\">\n      " << file
                << ":/position\n"
                << R"XMF(     </DataItem>
    </Geometry>
)XMF";
            std::vector<char const*>
                    fields {"velocity",
                            "vorticity",
                            "density",
                            "pressure",
                            "temperature",
                            "heat_flux",
                            "vector_potential"};
            if (configuration.coupling != Coupling::HD) {
                fields
                        .insert(fields.end(),
                                {"electric_field", "current", "lorentz_force", "joule_heating"});
                if (configuration.coupling != Coupling::EHD)
                    fields.push_back("induction");
                if (configuration.coupling != Coupling::MHD)
                    fields.insert(fields.end(), {"displacement", "charge_density", "gauss_error"});
                if (configuration.coupling == Coupling::EHD)
                    fields.push_back("electric_potential");
            }
            for (char const* name : fields) {
                bool const vector
                        = std::string(name) == "velocity" || std::string(name) == "vorticity"
                          || std::string(name) == "heat_flux"
                          || std::string(name) == "vector_potential"
                          || std::string(name) == "induction" || std::string(name) == "displacement"
                          || std::string(name) == "electric_field" || std::string(name) == "current"
                          || std::string(name) == "lorentz_force";
                out << "    <Attribute Name=\"" << name << "\" AttributeType=\""
                    << (vector ? "Vector" : "Scalar")
                    << "\" Center=\"Node\">\n     <DataItem Dimensions=\"" << dimensions
                    << (vector ? " 3" : "")
                    << "\" NumberType=\"Float\" Precision=\"8\" Format=\"HDF\">\n      " << file
                    << ":/snapshot_" << i << "/" << name << R"XMF(
     </DataItem>
    </Attribute>
)XMF";
            }
            out << "   </Grid>\n";
        }
        out << R"XMF(  </Grid>
 </Domain>
</Xdmf>
)XMF";
        if (!out)
            throw std::runtime_error("Cannot write XDMF output");
    }
};

void validate()
{
    Simulation simulation(nodes, physics);
    Monitor const initial = simulation.monitor(0);
    double const expected = 0.125;
    if (std::abs(initial.energy - expected) > 1e-10 || initial.divergence > 1e-10)
        throw std::runtime_error("Initial kinetic energy or periodic divergence is incorrect");
    double const symbol = std::sin(simulation.dec.h) / simulation.dec.h;
    if (physics == Physics::Incompressible
        && std::abs(initial.enstrophy - 0.375 * symbol * symbol) > 1e-10)
        throw std::runtime_error("Initial DEC enstrophy is incorrect");
    Field constant("constant", 1, simulation.dec.cells), lap("lap", 1, simulation.dec.cells);
    Kokkos::deep_copy(constant, 1.0);
    simulation.dec.laplacian(constant, 0, lap, 0);
    double error;
    Kokkos::parallel_reduce(
            "constant_laplacian_check",
            simulation.dec.cells,
            KOKKOS_LAMBDA(int i, double& sum) { sum += lap(0, i) * lap(0, i); },
            error);
    if (error != 0)
        throw std::runtime_error("Periodic DEC Laplacian of constant is nonzero");

    Field wave("periodic_fourier_mode", 1, simulation.dec.cells);
    Field derivative("periodic_fourier_derivative", 1, simulation.dec.cells);
    int const n = simulation.dec.n;
    double const h = simulation.dec.h;
    for (int axis = 0; axis < 3; ++axis) {
        Kokkos::parallel_for(
                "initialize_fourier_mode",
                simulation.dec.cells,
                KOKKOS_LAMBDA(int i) {
                    int const stride = axis == 0 ? n * n : (axis == 1 ? n : 1);
                    wave(0, i) = sin(-std::numbers::pi + ((i / stride) % n) * h);
                });
        simulation.dec.nodal_derivative_component(wave, 0, derivative, 0, axis);
        simulation.dec.laplacian(wave, 0, lap, 0);
        Kokkos::parallel_reduce(
                "periodic_fourier_symbols",
                simulation.dec.cells,
                KOKKOS_LAMBDA(int i, double& sum) {
                    int const stride = axis == 0 ? n * n : (axis == 1 ? n : 1);
                    double const x = -std::numbers::pi + ((i / stride) % n) * h;
                    double const gradient_error = derivative(0, i) - sin(h) / h * cos(x);
                    double const laplacian_error
                            = lap(0, i) + 4 * sin(h / 2) * sin(h / 2) / (h * h) * sin(x);
                    sum += gradient_error * gradient_error + laplacian_error * laplacian_error;
                },
                error);
        if (error / simulation.dec.cells > 1e-22)
            throw std::runtime_error("DEC Fourier symbol or periodic boundary sign is incorrect");
    }
    if (physics == Physics::Incompressible) {
        Field saved("saved_vorticity", 3, simulation.dec.cells);
        Kokkos::deep_copy(saved, simulation.state);
        simulation.compute_pressure();
        simulation.incompressible->monitor_constraints(simulation.state);
        Field const q = simulation.state, p = simulation.pressure,
                    u = simulation.incompressible->velocity;
        double mean_pressure;
        Kokkos::parallel_reduce(
                "reconstructed_pressure_mean",
                simulation.dec.cells,
                KOKKOS_LAMBDA(int i, double& sum) { sum += p(0, i); },
                mean_pressure);
        Kokkos::parallel_reduce(
                "initial_velocity_reconstruction",
                simulation.dec.cells,
                KOKKOS_LAMBDA(int i, double& sum) {
                    double const x = -std::numbers::pi + (i / (n * n)) * h;
                    double const y = -std::numbers::pi + ((i / n) % n) * h;
                    double const z = -std::numbers::pi + (i % n) * h;
                    double const du = u(0, i) - sin(x) * cos(y) * cos(z);
                    double const dv = u(1, i) + cos(x) * sin(y) * cos(z);
                    sum += du * du + dv * dv + u(2, i) * u(2, i);
                    for (int c = 0; c < 3; ++c) {
                        double const change = q(c, i) - saved(c, i);
                        sum += change * change;
                    }
                },
                error);
        if (error / simulation.dec.cells > 1e-18
            || std::abs(mean_pressure / simulation.dec.cells - Reference::reference_pressure)
                       > 1e-10)
            throw std::runtime_error(
                    "Velocity reconstruction, pressure gauge, or unchanged vorticity check failed");
        Field zero("zero_vorticity", 3, simulation.dec.cells);
        Field uniform_pressure("uniform_pressure", 1, simulation.dec.cells);
        Kokkos::deep_copy(zero, 0.0);
        simulation.incompressible
                ->compute_pressure(zero, uniform_pressure, Reference::reference_pressure);
        Kokkos::parallel_reduce(
                "zero_vorticity_pressure",
                simulation.dec.cells,
                KOKKOS_LAMBDA(int i, double& sum) {
                    double const difference
                            = uniform_pressure(0, i) - Reference::reference_pressure;
                    sum += difference * difference;
                },
                error);
        if (error / simulation.dec.cells > 1e-20)
            throw std::runtime_error("Zero-vorticity pressure reconstruction failed");
        // A periodic gradient force has zero solenoidal response and a known
        // Bernoulli pressure. Use its analytic DEC Fourier symbol to check the
        // diagnostic solve independently of the Krylov convergence report.
        Field gradient_force("pressure_gradient_force", 3, simulation.dec.cells);
        Kokkos::parallel_for(
                "initialize_pressure_gradient_force",
                simulation.dec.cells,
                KOKKOS_LAMBDA(int i) {
                    double const x = -std::numbers::pi + (i / (n * n)) * h;
                    gradient_force(0, i) = sin(x);
                    gradient_force(1, i) = 0;
                    gradient_force(2, i) = 0;
                });
        simulation.incompressible->compute_pressure(
                zero,
                uniform_pressure,
                Reference::reference_pressure,
                gradient_force);
        Kokkos::parallel_reduce(
                "manufactured_pressure",
                simulation.dec.cells,
                KOKKOS_LAMBDA(int i, double& sum) {
                    double const x = -std::numbers::pi + (i / (n * n)) * h;
                    double const expected = Reference::reference_pressure
                                            - h * cos(h / 2) / (2 * sin(h / 2)) * cos(x);
                    double const difference = uniform_pressure(0, i) - expected;
                    sum += difference * difference;
                },
                error);
        if (error / simulation.dec.cells > 1e-18)
            throw std::runtime_error("Manufactured Bernoulli pressure reconstruction failed");
    } else {
        Field uniform("uniform_gas", 5, simulation.dec.cells);
        Field uniform_rhs("uniform_gas_rhs", 5, simulation.dec.cells);
        Kokkos::parallel_for(
                "initialize_uniform_gas",
                simulation.dec.cells,
                KOKKOS_LAMBDA(int i) {
                    uniform(0, i) = 1;
                    uniform(1, i) = 0.2;
                    uniform(2, i) = 0;
                    uniform(3, i) = 0;
                    uniform(4, i) = Reference::reference_pressure / (Reference::gamma - 1) + 0.02;
                });
        simulation.compressible->rhs(uniform, uniform_rhs);
        Kokkos::parallel_reduce(
                "uniform_gas_residual",
                simulation.dec.cells,
                KOKKOS_LAMBDA(int i, double& sum) {
                    for (int c = 0; c < 5; ++c)
                        sum += uniform_rhs(c, i) * uniform_rhs(c, i);
                },
                error);
        if (error != 0)
            throw std::runtime_error("Uniform gas is not a discrete steady state");
        // Isobaric, motionless gas isolates Fourier conduction: E_rhs=kappa*Delta T.
        Kokkos::parallel_for(
                "initialize_thermal_mode",
                simulation.dec.cells,
                KOKKOS_LAMBDA(int i) {
                    double const x = -std::numbers::pi + (i / (n * n)) * h;
                    double const temperature = Reference::reference_pressure
                                               / Reference::gas_constant * (1 + 0.001 * sin(x));
                    uniform(0, i) = Reference::reference_pressure
                                    / (Reference::gas_constant * temperature);
                    uniform(1, i) = 0;
                    uniform(2, i) = 0;
                    uniform(3, i) = 0;
                    uniform(4, i) = Reference::reference_pressure / (Reference::gamma - 1);
                });
        simulation.compressible->rhs(uniform, uniform_rhs);
        Kokkos::parallel_reduce(
                "fourier_heat_closure",
                simulation.dec.cells,
                KOKKOS_LAMBDA(int i, double& sum) {
                    double const x = -std::numbers::pi + (i / (n * n)) * h;
                    double const expected = -Reference::conductivity() * 4 * sin(h / 2) * sin(h / 2)
                                            / (h * h) * Reference::reference_pressure * 0.001
                                            * sin(x);
                    double const difference = uniform_rhs(4, i) - expected;
                    sum += difference * difference;
                    for (int c = 0; c < 4; ++c)
                        sum += uniform_rhs(c, i) * uniform_rhs(c, i);
                },
                error);
        if (error / simulation.dec.cells > 1e-22)
            throw std::runtime_error(
                    "Prandtl/Fourier closure has an incorrect diffusion sign or magnitude");
    }
    double time = 0;
    for (int i = 0; i < 5; ++i) {
        double const dt = simulation.timestep();
        simulation.advance(dt);
        time += dt;
    }
    Monitor const final = simulation.monitor(time);
    if (physics == Physics::Compressible
        && (std::abs(final.mass - initial.mass) > 1e-11
            || (configuration.coupling == Coupling::HD
                && std::abs(final.total_energy - initial.total_energy) > 1e-10)))
        throw std::runtime_error("Conservative solver failed mass/energy conservation");
    if (physics == Physics::Incompressible && final.divergence > 1e-8)
        throw std::runtime_error("Vorticity reconstruction failed divergence check");
    if (configuration.coupling == Coupling::HD && final.energy > initial.energy + 1e-8)
        throw std::runtime_error("Unforced flow gained kinetic energy");
    if (simulation.electromagnetic) {
        if (initial.electromagnetic[5] > 1e-10 || initial.electromagnetic[6] > 1e-9
            || final.electromagnetic[5] > 1e-9 || final.electromagnetic[6] > 1e-8
            || std::abs(final.electromagnetic[7] - initial.electromagnetic[7]) > 1e-11
            || std::abs(initial.electromagnetic[8]) > 1e-10
            || std::abs(final.electromagnetic[8]) > 1e-10)
            throw std::runtime_error(
                    "Magnetic divergence, Gauss constraint, or charge conservation failed");
        if (physics == Physics::Compressible
            && std::abs(final.total_energy - initial.total_energy) > 1e-5)
            throw std::runtime_error("Coupled total energy drift exceeds short-run tolerance");
    }
    std::cout << "Validation passed: N=" << nodes << ", initial Ek=" << initial.energy
              << ", final divergence=" << final.divergence << "\n";
}

double run_trajectory(benchmark::State* benchmark_state)
{
    static int invocation = 0;
    try {
        ProfilingRegion const case_region(
                "N=" + std::to_string(nodes) + "/" + configuration.strategy_flags());
        ProfilingRegion setup_region("setup");
        std::filesystem::create_directories(output_directory);
        Clock::time_point const setup_start = Clock::now();
        Simulation simulation(nodes, physics);
        Kokkos::fence("taylor_green_setup_complete");
        double setup_seconds = 0;
        std::string const prefix = std::string("tgv_") + configuration.name() + "_N"
                                   + std::to_string(nodes) + "_run";
        std::filesystem::path base;
        do {
            base = output_directory / (prefix + std::to_string(invocation++));
        } while (std::filesystem::exists(base.string() + ".h5"));
        double solver_seconds = 0, monitor_seconds = 0, export_seconds = 0;

        std::vector<Monitor> monitors;
        std::ofstream energy(base.string() + ".dat");
        if (!energy)
            throw std::runtime_error("Cannot create energy history");
        energy << header(nodes, physics) << "# time Ek dEk/dt epsilon_enstrophy\n"
               << std::setprecision(17);
        std::unique_ptr<Export> output;
        double last_snapshot_time = 0;
        auto sample = [&](double time, double step_seconds, bool snapshot) {
            Clock::time_point begin = Clock::now();
            ProfilingRegion monitor_region("monitoring");
            Monitor const m = simulation.monitor(time, step_seconds);
            monitor_region.end();
            monitor_seconds += std::chrono::duration<double>(Clock::now() - begin).count();
            monitors.push_back(m);
            begin = Clock::now();
            ProfilingRegion const export_region("export");
            energy << m.time << " " << m.energy << " " << m.derivative << " " << m.dissipation
                   << "\n";
            if (snapshot && write_snapshots) {
                output->snapshot(simulation, time);
                last_snapshot_time = time;
            }
            export_seconds += std::chrono::duration<double>(Clock::now() - begin).count();
        };
        Clock::time_point const export_start = Clock::now();
        output = std::make_unique<Export>(base, nodes);
        export_seconds += std::chrono::duration<double>(Clock::now() - export_start).count();
        sample(0, 0, true);
        setup_seconds = std::chrono::duration<double>(Clock::now() - setup_start).count();
        setup_region.end();
        Clock::time_point progress_start = Clock::now();
        double time = 0;
        std::int64_t steps = 0, solver_reconstruction_iterations = 0;
        double solver_reconstruction_seconds = 0, electromagnetic_seconds = 0;
        std::int64_t electric_iterations = 0;
        std::int64_t next_snapshot = 1;
        // A single Google Benchmark iteration advances the existing state once.
        auto advance_step = [&]() {
            ProfilingRegion const iteration_region("iteration");
            ProfilingRegion solver_region("solver");
            Clock::time_point const step_start = Clock::now();
            std::int64_t const cg_before
                    = simulation.incompressible ? simulation.incompressible->total_iterations : 0;
            double const reconstruction_before
                    = simulation.incompressible ? simulation.incompressible->reconstruction_seconds
                                                : 0;
            double const em_before = simulation.electromagnetic
                                             ? simulation.electromagnetic->evaluation_seconds
                                             : 0;
            std::int64_t const electric_before
                    = simulation.electromagnetic ? simulation.electromagnetic->poisson_iterations
                                                 : 0;
            double const target
                    = reference_run
                              ? std::min(
                                        {0.033 * next_snapshot,
                                         time < 8 ? 8.0 : reference_final_time,
                                         reference_final_time})
                              : (time < 8 ? 8.0
                                          : (time < 10 ? 10.0
                                                       : std::numeric_limits<double>::infinity()));
            double const dt = std::min(simulation.timestep(), target - time);
            if (!(dt > 0) || !std::isfinite(dt) || time + dt == time)
                throw std::runtime_error("Invalid or non-advancing time step");
            simulation.advance(dt);
            if (simulation.incompressible) {
                solver_reconstruction_iterations
                        += simulation.incompressible->total_iterations - cg_before;
                solver_reconstruction_seconds += simulation.incompressible->reconstruction_seconds
                                                 - reconstruction_before;
            }

            if (simulation.electromagnetic) {
                electromagnetic_seconds
                        += simulation.electromagnetic->evaluation_seconds - em_before;
                electric_iterations
                        += simulation.electromagnetic->poisson_iterations - electric_before;
            }
            std::int64_t const optimizer_iterations
                    = simulation.incompressible
                              ? simulation.incompressible->total_iterations - cg_before
                              : 0;
            Kokkos::fence("taylor_green_timing");
            double const seconds = std::chrono::duration<double>(Clock::now() - step_start).count();
            solver_region.end();
            solver_seconds += seconds;
            if (benchmark_state)
                benchmark_state->SetIterationTime(seconds);
            bool const reached = std::isfinite(target) && dt == target - time;
            time = reached ? target : time + dt;
            if (reference_run && reached && time == 0.033 * next_snapshot)
                ++next_snapshot;
            ++steps;
            if (benchmark_state)
                benchmark_state->PauseTiming();
            double step_execution_seconds = seconds;
            try {
                sample(time, seconds, reached);
                step_execution_seconds
                        = std::chrono::duration<double>(Clock::now() - step_start).count();
            } catch (...) {
                if (benchmark_state)
                    benchmark_state->ResumeTiming();
                throw;
            }
            if (reference_run && (steps % 100 == 0 || time == reference_final_time)) {
                double const progress_interval_seconds
                        = std::chrono::duration<double>(Clock::now() - progress_start).count();
                std::cout << "TGV N=" << nodes << " step=" << steps
                          << " time=" << std::setprecision(8) << time
                          << " optimizer_iterations=" << optimizer_iterations
                          << " solver_duration_seconds=" << seconds
                          << " step_execution_seconds=" << step_execution_seconds
                          << " elapsed_since_previous_progress_seconds="
                          << progress_interval_seconds << std::endl;
                progress_start = Clock::now();
            }
            if (benchmark_state)
                benchmark_state->ResumeTiming();
        };
        if (benchmark_state) {
            for ([[maybe_unused]] auto iteration : *benchmark_state)
                advance_step();
        } else {
            while (time < reference_final_time)
                advance_step();
        }
        ProfilingRegion const finalization_region("finalization");
        Clock::time_point const finish_export = Clock::now();
        if (write_snapshots && time != last_snapshot_time)
            output->snapshot(simulation, time);
        output->final_state(simulation);
        output->monitors(monitors);
        output.reset();
        energy.close();
        if (!energy)
            throw std::runtime_error("Cannot write energy history");
        export_seconds += std::chrono::duration<double>(Clock::now() - finish_export).count();
        double const wall = std::chrono::duration<double>(Clock::now() - setup_start).count();
        if (!(solver_seconds > 0))
            throw std::runtime_error("Invalid solver timing");
        std::ofstream performance(base.string() + ".performance.json");
        performance << std::setprecision(17) << "{\"configuration\":\"" << configuration.name()
                    << "\",\"laplacian\":\""
                    << (configuration.staged_laplacian ? "staged" : "direct") << "\",\"staging\":\""
                    << configuration.staging_name() << "\",\"operators\":\""
                    << configuration.operator_name() << "\",\"mode\":\""
                    << (benchmark_state ? "performance" : "reference")
                    << "\",\"strategy\":" << strategy_index << ",\"flags\":\""
                    << configuration.strategy_flags() << "\",\"N\":" << nodes
                    << ",\"steps\":" << steps << ",\"final_time\":" << time
                    << ",\"mean_step_seconds\":" << solver_seconds / steps
                    << ",\"setup_seconds\":" << setup_seconds
                    << ",\"solver_seconds\":" << solver_seconds
                    << ",\"monitor_seconds\":" << monitor_seconds
                    << ",\"export_seconds\":" << export_seconds << ",\"wall_seconds\":" << wall
                    << ",\"reconstruction_iterations\":" << solver_reconstruction_iterations
                    << ",\"reconstruction_seconds\":" << solver_reconstruction_seconds
                    << ",\"electromagnetic_seconds\":" << electromagnetic_seconds
                    << ",\"electric_poisson_iterations\":" << electric_iterations
                    << ",\"cell_updates_per_second\":"
                    << double(steps) * simulation.dec.cells / solver_seconds << "}\n";
        if (!performance)
            throw std::runtime_error("Cannot write performance output");
        if (benchmark_state && steps != benchmark_state->iterations())
            throw std::runtime_error("Google Benchmark iterations and solver steps differ");
        if (benchmark_state) {
            benchmark_state->counters["setup_seconds"] = setup_seconds;
            benchmark_state->counters["repeats"] = 1;
        }
        return solver_seconds;
    } catch (std::exception const& error) {
        benchmark_failed = true;
        if (benchmark_state)
            benchmark_state->SkipWithError(error.what());
        else
            throw;
        return 0;
    }
}

// Keep manual solver timing, but omit its implementation suffix from displayed names.
class ConsoleReporter : public benchmark::ConsoleReporter
{
    void ReportRuns(std::vector<Run> const& reports) override
    {
        std::vector<Run> clean = reports;
        for (Run& report : clean)
            report.run_name.time_type.clear();
        benchmark::ConsoleReporter::ReportRuns(clean);
    }
};

class JsonReporter : public benchmark::JSONReporter
{
    void ReportRuns(std::vector<Run> const& reports) override
    {
        std::vector<Run> clean = reports;
        for (Run& report : clean)
            report.run_name.time_type.clear();
        benchmark::JSONReporter::ReportRuns(clean);
    }
};

Configuration select_strategy(Configuration selected, int strategy)
{
    selected.levi_civita_continuous_hodge_star = strategy >= 1 && strategy <= 6;
    selected.weighted_levi_civita_discrete_hodge_star
            = strategy == 2 || strategy == 5 || strategy == 6;
    selected.staged_weighted_levi_civita_discrete_hodge_star = strategy == 3;
    selected.staged_discrete_hodge_star = strategy == 4;
    selected.diagonal_reduction = strategy >= 2;
    selected.diagonal_reconstruction = strategy >= 2;
    selected.staged_codifferential = strategy == 5;
    selected.staged_laplacian = strategy == 6;
    return selected;
}

void register_case(Configuration selected, int n, int strategy, std::filesystem::path directory)
{
    std::string const name = "TaylorGreen/" + selected.name() + "/N" + std::to_string(n) + "/"
                             + selected.strategy_flags();
    benchmark::internal::Benchmark* const registered = benchmark::RegisterBenchmark(
            name.c_str(),
            [selected, n, strategy, directory](benchmark::State& state) {
                configuration = selected;
                physics = selected.fluid;
                nodes = n;
                strategy_index = strategy;
                output_directory = directory;
                if (reference_run) {
                    try {
                        for ([[maybe_unused]] auto iteration : state) {
                            state.SetIterationTime(run_trajectory(nullptr));
                        }
                    } catch (std::exception const& error) {
                        benchmark_failed = true;
                        state.SkipWithError(error.what());
                    }
                } else
                    run_trajectory(&state);
            });
    registered->UseManualTime()->Unit(benchmark::kSecond);
    if (reference_run)
        registered->Iterations(1)->Repetitions(1);
}
} // namespace

int main(int argc, char** argv)
{
    try {
        int retained = 1;
        for (int i = 1; i < argc; ++i) {
            std::string const arg = argv[i];
            if (arg.starts_with("--benchmark_out=") || arg == "--benchmark_out")
                benchmark_output_requested = true;
            if (arg.starts_with("--N=")) {
                std::string const value = arg.substr(4);
                std::size_t consumed = 0;
                nodes = std::stoi(value, &consumed);
                if (consumed != value.size())
                    throw std::runtime_error("N must be an integer");
            } else if (configuration.consume(arg)) {
                // Benchmark selectors are removed before Google Benchmark parses its flags.
            } else if (arg.starts_with("--output-dir="))
                output_directory = arg.substr(13);
            else if (arg == "--validate")
                verify = true;
            else if (arg == "--strategy-suite")
                strategy_suite = true;
            else if (arg == "--reference-suite")
                reference_run = reference_suite = true;
            else if (arg.starts_with("--reference-final-time="))
                reference_final_time = std::stod(arg.substr(23));
            else if (arg == "--reference")
                reference_run = true;
            else if (arg == "--no-snapshots")
                write_snapshots = false;
            else {
                if (arg == "--help")
                    std::cout << "Taylor-Green options: --N=64 --incompressible|--compressible "
                                 "--ideal|--resistive --hd|--ehd|--mhd|--fmhd --validate "
                                 "--output-dir=PATH --reference --no-snapshots\n"
                                 "Suites: --strategy-suite (seven strategies at N=64,128); "
                                 "--reference-suite (staged weighted discrete Hodge star at "
                                 "N=64,128).\n"
                                 "Staging: --staged-laplacian --staged-codifferential "
                                 "--staged-discrete-hodge-star "
                                 "--staged-reduction --staged-reconstruction "
                                 "--staged-diagonal-reduction --staged-diagonal-reconstruction\n"
                                 "Operator choices: --weighted-levi-civita-discrete-hodge-star "
                                 "--staged-weighted-levi-civita-discrete-hodge-star "
                                 "--levi-civita-continuous-hodge-star "
                                 "--diagonal-reduction --diagonal-reconstruction\n"
                                 "Flags can be combined; outer caches supersede inner stages. "
                                 "--staged-codifferential requires "
                                 "--weighted-levi-civita-discrete-hodge-star. "
                                 "Plain diagonal flags select non-staged operators; the "
                                 "--staged-diagonal-* flags select staged diagonal caches. "
                                 "--staged aliases "
                                 "--staged-laplacian.\n"
                                 "One Google Benchmark iteration advances one time step. "
                                 "Use --benchmark_min_time=100x for 100 steps.\n"
                                 "Defaults: --incompressible --ideal --hd. Resistive HD is "
                                 "invalid. General continuous Hodge star, reduction and "
                                 "reconstruction are defaults; specialized operators are "
                                 "selected by their flags. Laplacians evaluate geometry during "
                                 "application by default; --staged-laplacian caches their full "
                                 "stencils.\n";
                argv[retained++] = argv[i];
            }
        }
        configuration.validate();
        if (reference_run && (configuration.coupling != Coupling::HD || verify))
            throw std::runtime_error(
                    "--reference requires HD and cannot be combined with --validate");
        physics = configuration.fluid;
        argc = retained;
        argv[argc] = nullptr;
        if (nodes < 8 || nodes > 512)
            throw std::runtime_error("N must be between 8 and 512");
        benchmark::Initialize(&argc, argv);
        if (benchmark::ReportUnrecognizedArguments(argc, argv))
            return 1;
        Kokkos::ScopeGuard const kokkos_scope(argc, argv);
        ddc::ScopeGuard const ddc_scope(argc, argv);
#if defined(KOKKOS_ENABLE_CUDA)
        // Debug builds cannot infer the full stack depth of staged DEC operators.
        std::size_t stack_size = 0;
        if (cudaDeviceGetLimit(&stack_size, cudaLimitStackSize) != cudaSuccess
            || (stack_size < 32768 && cudaDeviceSetLimit(cudaLimitStackSize, 32768) != cudaSuccess))
            throw std::runtime_error(
                    "Cannot reserve the CUDA stack for Taylor-Green DEC operators");
#endif
        PdiSession pdi;
        pdi_session = &pdi;
        if (verify) {
            validate();
            return 0;
        }
        if (strategy_suite && (reference_run || verify))
            throw std::runtime_error("--strategy-suite cannot be combined with validation");
        if (!(reference_final_time > 0) || reference_final_time > 10)
            throw std::runtime_error("Reference final time must be in (0,10]");
        // Library diagnostics are optional; campaign stdout is reserved for Google Benchmark.
        setenv("SIMILIE_SOLVER_QUIET", "1", 1);
        setenv("SIMILIE_SOLVER_PROGRESS_STRIDE", "0", 1);
        benchmark::AddCustomContext("execution_space", Kokkos::DefaultExecutionSpace::name());
        benchmark::AddCustomContext("configuration", configuration.name());
        benchmark::AddCustomContext("Re", "1600");
        benchmark::AddCustomContext("Pr", "0.71");
        if (strategy_suite) {
            for (int strategy = 0; strategy < 7; ++strategy)
                for (int n : {64, 128})
                    register_case(
                            select_strategy(configuration, strategy),
                            n,
                            strategy,
                            output_directory / ("strategy-" + std::to_string(strategy + 1))
                                    / ("N" + std::to_string(n)));
        } else if (reference_suite) {
            for (int n : {64, 128})
                register_case(
                        select_strategy(configuration, 3),
                        n,
                        3,
                        output_directory / ("N" + std::to_string(n)));
        } else
            register_case(configuration, nodes, -1, output_directory);
        ConsoleReporter console;
        JsonReporter json;
        benchmark::RunSpecifiedBenchmarks(&console, benchmark_output_requested ? &json : nullptr);
        benchmark::Shutdown();
        return benchmark_failed ? 1 : 0;
    } catch (std::exception const& error) {
        std::cerr << error.what() << "\n";
        return 1;
    }
}
