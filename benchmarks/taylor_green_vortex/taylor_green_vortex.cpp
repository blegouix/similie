// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <memory>
#include <numbers>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

#include <benchmark/benchmark.h>

#include <hdf5.h>

#include "solvers.hpp"

namespace {
using taylor_green_vortex::CompressibleSolver;
using taylor_green_vortex::Field;
using taylor_green_vortex::IncompressibleSolver;
using taylor_green_vortex::Reference;
using taylor_green_vortex::StructuredGrid;
enum class Physics { Incompressible, Compressible, ResistiveMHD, TwoFluidPlasmaMHD };
using Clock = std::chrono::steady_clock;
int nodes = 64;
Physics physics = Physics::Incompressible;
std::filesystem::path output_directory = "taylor_green_vortex_results";
bool verify = false;
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

struct Monitor
{
    double time, energy, derivative, enstrophy, dissipation, pressure_dilatation,
            viscous_dissipation;
    double mass, total_energy, divergence, rho_min, pressure_min, step_seconds;
    double potential_gauge, vorticity_divergence, curl_residual, reconstruction_residual;
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
    explicit Simulation(int n, Physics selected)
        : dec(n)
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
    {
        if (selected != Physics::Incompressible && selected != Physics::Compressible)
            throw std::runtime_error("Selected physics is not implemented");
        Kokkos::deep_copy(vector_potential, 0.0);
        reset();
    }
    void reset()
    {
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
    void evaluate(Field q, Field result, bool reconstruct_pressure = false)
    {
        if (kind == Physics::Compressible)
            compressible->rhs(q, result);
        else
            incompressible->rhs(q, result, reconstruct_pressure);
    }
    // Reconstruct pressure directly from the current state for export.
    void compute_pressure()
    {
        if (kind == Physics::Incompressible) {
            incompressible->compute_pressure(state, pressure, Reference::reference_pressure);
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
        bool const gas = kind == Physics::Compressible;
        if (!gas)
            incompressible->reconstruct_velocity(state, velocity);
        Field const q = gas ? state : velocity;
        double maximum;
        Kokkos::parallel_reduce(
                "cfl_speed",
                dec.cells,
                KOKKOS_LAMBDA(int i, double& value) {
                    double speed = 0;
                    if (gas) {
                        double const sound
                                = sqrt(Reference::gamma * Reference::pressure(q, i) / q(0, i));
                        for (int a = 0; a < 3; ++a)
                            speed += fabs(q(a + 1, i) / q(0, i)) + sound;
                    } else
                        for (int a = 0; a < 3; ++a)
                            speed += fabs(q(a, i));
                    value = fmax(value, speed);
                },
                Kokkos::Max<double>(maximum));
        double diffusivity = 4.0 * Reference::viscosity / 3.0;
        if (gas) {
            Kokkos::parallel_reduce(
                    "thermal_viscous_diffusivity",
                    dec.cells,
                    KOKKOS_LAMBDA(int i, double& value) {
                        double const local = fmax(4.0 * Reference::viscosity / 3.0,
                                                  Reference::conductivity() * (Reference::gamma - 1)
                                                          / Reference::gas_constant)
                                             / q(0, i);
                        value = fmax(value, local);
                    },
                    Kokkos::Max<double>(diffusivity));
        }
        // Fixed CFL and diffusion constants; no user-controlled time step.
        return std::min(0.3 * dec.h / std::max(maximum, 1e-12), 0.1 * dec.h * dec.h / diffusivity);
    }
    void advance(double dt)
    {
        Kokkos::deep_copy(initial, state);
        Field const q = state, q0 = initial, s = stage, f = rhs;
        int const count = components;
        for (int step = 0; step < 3; ++step) {
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
            incompressible->reconstruct_velocity(rhs, velocity_rate);
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
                    t(0, i) = gas ? p(0, i) / (rho(0, i) * Reference::gas_constant)
                                  : Reference::reference_pressure / Reference::gas_constant;
                });
        for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b)
                dec.derivative(u, a, g, 3 * a + b, b);
        if (gas) {
            dec.curl(u, w);
            dec.divergence(u, velocity_divergence);
        }
        Field const div_field = velocity_divergence;
        similie::physics::fluid::NavierStokesFourier const physical_model = Reference::model();
        if (gas) {
            Field const heat = heat_flux;
            for (int a = 0; a < 3; ++a)
                dec.derivative(t, 0, heat, a, a);
            Kokkos::parallel_for(
                    "fourier_heat_flux",
                    dec.cells,
                    KOKKOS_LAMBDA(int i) {
                        std::array<double, 3> const q
                                = physical_model.heat_flux({heat(0, i), heat(1, i), heat(2, i)});
                        for (int a = 0; a < 3; ++a)
                            heat(a, i) = q[a];
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
                        similie::physics::fluid::NavierStokesFourier::VelocityGradient grad {};
                        for (int a = 0; a < 3; ++a)
                            for (int b = 0; b < 3; ++b)
                                grad[a][b] = g(3 * a + b, i);
                        similie::physics::fluid::NavierStokesFourier::VelocityGradient const tau
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
    return text.str();
}

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
        H5Fclose(m_file);
    }
    void snapshot(Simulation& simulation, double time)
    {
        simulation.compute_pressure();
        std::string const group_name = "snapshot_" + std::to_string(m_times.size());
        hid_t const group
                = H5Gcreate2(m_file, group_name.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        if (group < 0)
            throw std::runtime_error("Cannot create HDF5 snapshot");
        dataset(group, "time", {1}, &time);
        for (std::pair<char const*, Field> const& item :
             {std::pair {"velocity", simulation.velocity},
              std::pair {"vorticity", simulation.vorticity},
              std::pair {"density", simulation.density},
              std::pair {"pressure", simulation.pressure},
              std::pair {"temperature", simulation.temperature},
              std::pair {"heat_flux", simulation.heat_flux},
              std::pair {"vector_potential", simulation.vector_potential}}) {
            Field::HostMirror const host
                    = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), item.second);
            int const components = static_cast<int>(host.extent(0));
            std::vector<double> values(simulation.dec.cells * components);
            for (int i = 0; i < simulation.dec.cells; ++i)
                for (int c = 0; c < components; ++c)
                    values[i * components + c] = host(c, i);
            std::vector<hsize_t> shape {hsize_t(m_n), hsize_t(m_n), hsize_t(m_n)};
            if (components > 1)
                shape.push_back(components);
            dataset(group, item.first, shape, values.data());
        }
        H5Gclose(group);
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
        std::string const names
                = "time,kinetic_energy,dEk_dt,kinetic_dissipation,enstrophy,enstrophy_dissipation,"
                  "pressure_dilatation,viscous_dissipation,mass,total_energy,divergence_rms,"
                  "density_min,pressure_min,step_seconds";
        hid_t const type = H5Tcopy(H5T_C_S1);
        check_hdf(H5Tset_size(type, names.size() + 1));
        hid_t const space = H5Screate(H5S_SCALAR);
        hid_t const attribute
                = H5Acreate2(m_file, "monitor_columns", type, space, H5P_DEFAULT, H5P_DEFAULT);
        check_hdf(H5Awrite(attribute, type, names.c_str()));
        H5Aclose(attribute);
        H5Sclose(space);
        H5Tclose(type);
    }
    void write_xmf()
    {
        std::ofstream out(m_base.string() + ".xmf");
        out << "<?xml version=\"1.0\"?>\n<Xdmf Version=\"3.0\"><Domain><Grid Name=\"TimeSeries\" "
               "GridType=\"Collection\" CollectionType=\"Temporal\">\n";
        std::string const file = m_base.filename().string() + ".h5";
        std::string const dimensions
                = std::to_string(m_n) + " " + std::to_string(m_n) + " " + std::to_string(m_n);
        for (std::size_t i = 0; i < m_times.size(); ++i) {
            out << "<Grid Name=\"snapshot_" << i << "\" GridType=\"Uniform\"><Time Value=\""
                << m_times[i] << "\"/>\n"
                << "<Topology TopologyType=\"3DSMesh\" Dimensions=\"" << dimensions << "\"/>\n"
                << "<Geometry GeometryType=\"XYZ\"><DataItem Dimensions=\"" << dimensions
                << " 3\" NumberType=\"Float\" Precision=\"8\" Format=\"HDF\">" << file
                << ":/position</DataItem></Geometry>\n";
            for (char const* name :
                 {"velocity",
                  "vorticity",
                  "density",
                  "pressure",
                  "temperature",
                  "heat_flux",
                  "vector_potential"}) {
                bool const vector = std::string(name) == "velocity"
                                    || std::string(name) == "vorticity"
                                    || std::string(name) == "heat_flux"
                                    || std::string(name) == "vector_potential";
                out << "<Attribute Name=\"" << name << "\" AttributeType=\""
                    << (vector ? "Vector" : "Scalar")
                    << "\" Center=\"Node\"><DataItem Dimensions=\"" << dimensions
                    << (vector ? " 3" : "")
                    << "\" NumberType=\"Float\" Precision=\"8\" Format=\"HDF\">" << file
                    << ":/snapshot_" << i << "/" << name << "</DataItem></Attribute>\n";
            }
            out << "</Grid>\n";
        }
        out << "</Grid></Domain></Xdmf>\n";
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
        simulation.dec.derivative(wave, 0, derivative, 0, axis);
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
            || std::abs(final.total_energy - initial.total_energy) > 1e-10))
        throw std::runtime_error("Conservative solver failed mass/energy conservation");
    if (physics == Physics::Incompressible && final.divergence > 1e-8)
        throw std::runtime_error("Vorticity reconstruction failed divergence check");
    if (final.energy > initial.energy + 1e-8)
        throw std::runtime_error("Unforced flow gained kinetic energy");
    std::cout << "Validation passed: N=" << nodes << ", initial Ek=" << initial.energy
              << ", final divergence=" << final.divergence << "\n";
}

void run(benchmark::State& benchmark_state)
{
    static int invocation = 0;
    try {
        std::filesystem::create_directories(output_directory);
        double total_solver = 0, total_monitors = 0, total_export = 0, total_wall = 0,
               total_reconstruction = 0, total_reconstruction_iterations = 0;
        std::int64_t total_steps = 0;
        for ([[maybe_unused]] auto iteration : benchmark_state) {
            Simulation simulation(nodes, physics);
            Kokkos::fence("taylor_green_setup_complete");
            std::string const prefix
                    = std::string("tgv_")
                      + (physics == Physics::Compressible ? "compressible" : "incompressible")
                      + "_N" + std::to_string(nodes) + "_run";
            std::filesystem::path base;
            do {
                base = output_directory / (prefix + std::to_string(invocation++));
            } while (std::filesystem::exists(base.string() + ".h5"));
            Clock::time_point const start = Clock::now();
            double solver_seconds = 0, monitor_seconds = 0, export_seconds = 0;

            std::vector<Monitor> monitors;
            std::ofstream energy(base.string() + ".dat");
            if (!energy)
                throw std::runtime_error("Cannot create energy history");
            energy << header(nodes, physics) << "# time Ek dEk/dt epsilon_enstrophy\n"
                   << std::setprecision(17);
            std::unique_ptr<Export> output;
            auto sample = [&](double time, double step_seconds, bool snapshot) {
                Clock::time_point begin = Clock::now();
                Monitor const m = simulation.monitor(time, step_seconds);
                monitor_seconds += std::chrono::duration<double>(Clock::now() - begin).count();
                monitors.push_back(m);
                begin = Clock::now();
                energy << m.time << " " << m.energy << " " << m.derivative << " " << m.dissipation
                       << "\n";
                if (snapshot)
                    output->snapshot(simulation, time);
                export_seconds += std::chrono::duration<double>(Clock::now() - begin).count();
            };
            Clock::time_point const export_start = Clock::now();
            output = std::make_unique<Export>(base, nodes);
            export_seconds += std::chrono::duration<double>(Clock::now() - export_start).count();
            sample(0, 0, true);
            double time = 0;
            bool at_eight = false;
            std::int64_t steps = 0, solver_reconstruction_iterations = 0;
            double solver_reconstruction_seconds = 0;
            while (time < 10) {
                Clock::time_point const step_start = Clock::now();
                std::int64_t const cg_before = simulation.incompressible
                                                       ? simulation.incompressible->total_iterations
                                                       : 0;
                double const reconstruction_before
                        = simulation.incompressible
                                  ? simulation.incompressible->reconstruction_seconds
                                  : 0;
                double const target = at_eight ? 10.0 : 8.0;
                double const dt = std::min(simulation.timestep(), target - time);
                if (!(dt > 0) || !std::isfinite(dt) || time + dt == time)
                    throw std::runtime_error("Invalid or non-advancing time step");
                simulation.advance(dt);
                if (simulation.incompressible) {
                    solver_reconstruction_iterations
                            += simulation.incompressible->total_iterations - cg_before;
                    solver_reconstruction_seconds
                            += simulation.incompressible->reconstruction_seconds
                               - reconstruction_before;
                }

                Kokkos::fence("taylor_green_timing");
                double const seconds
                        = std::chrono::duration<double>(Clock::now() - step_start).count();
                solver_seconds += seconds;
                bool const reached = dt == target - time;
                time = reached ? target : time + dt;
                sample(time, seconds, reached);
                if (time == 8)
                    at_eight = true;
                ++steps;
            }
            Clock::time_point const finish_export = Clock::now();
            output->monitors(monitors);
            output.reset();
            energy.close();
            if (!energy)
                throw std::runtime_error("Cannot write energy history");
            export_seconds += std::chrono::duration<double>(Clock::now() - finish_export).count();
            double const wall = std::chrono::duration<double>(Clock::now() - start).count();
            if (!(solver_seconds > 0))
                throw std::runtime_error("Invalid solver timing");
            benchmark_state.SetIterationTime(solver_seconds);
            total_solver += solver_seconds;
            total_monitors += monitor_seconds;
            total_export += export_seconds;
            total_wall += wall;
            total_steps += steps;
            total_reconstruction += solver_reconstruction_seconds;
            total_reconstruction_iterations += solver_reconstruction_iterations;
            std::ofstream performance(base.string() + ".performance.json");
            performance << std::setprecision(17) << "{\"N\":" << nodes << ",\"steps\":" << steps
                        << ",\"solver_seconds\":" << solver_seconds
                        << ",\"monitor_seconds\":" << monitor_seconds
                        << ",\"export_seconds\":" << export_seconds << ",\"wall_seconds\":" << wall
                        << ",\"reconstruction_iterations\":" << solver_reconstruction_iterations
                        << ",\"reconstruction_seconds\":" << solver_reconstruction_seconds
                        << ",\"cell_updates_per_second\":"
                        << double(steps) * simulation.dec.cells / solver_seconds << "}\n";
            if (!performance)
                throw std::runtime_error("Cannot write performance output");
        }
        double const iterations = benchmark_state.iterations();
        benchmark_state.counters["steps"] = total_steps / iterations;
        benchmark_state.counters["solver_seconds"] = total_solver / iterations;
        benchmark_state.counters["monitor_seconds"] = total_monitors / iterations;
        benchmark_state.counters["export_seconds"] = total_export / iterations;
        benchmark_state.counters["wall_seconds"] = total_wall / iterations;
        benchmark_state.counters["reconstruction_seconds"] = total_reconstruction / iterations;
        benchmark_state.counters["reconstruction_iterations"]
                = total_reconstruction_iterations / iterations;
        benchmark_state.counters["cell_updates/s"]
                = double(total_steps) * nodes * nodes * nodes / total_solver;
        benchmark_state.SetItemsProcessed(total_steps * std::int64_t(nodes) * nodes * nodes);
    } catch (std::exception const& error) {
        benchmark_failed = true;
        benchmark_state.SkipWithError(error.what());
    }
}
} // namespace

int main(int argc, char** argv)
{
    try {
        int retained = 1;
        for (int i = 1; i < argc; ++i) {
            std::string const arg = argv[i];
            if (arg.starts_with("--N=")) {
                std::string const value = arg.substr(4);
                std::size_t consumed = 0;
                nodes = std::stoi(value, &consumed);
                if (consumed != value.size())
                    throw std::runtime_error("N must be an integer");
            } else if (arg.starts_with("--physics=")) {
                std::string const name = arg.substr(10);
                if (name == "incompressible")
                    physics = Physics::Incompressible;
                else if (name == "compressible")
                    physics = Physics::Compressible;
                else if (name == "resistive_mhd" || name == "two_fluid_plasma_mhd")
                    throw std::runtime_error(
                            name + " is reserved for a future solver and is not implemented");
                else
                    throw std::runtime_error("Unknown physics: " + name);
            } else if (arg.starts_with("--output-dir="))
                output_directory = arg.substr(13);
            else if (arg == "--validate")
                verify = true;
            else
                argv[retained++] = argv[i];
        }
        argc = retained;
        argv[argc] = nullptr;
        if (nodes < 8 || nodes > 512)
            throw std::runtime_error("N must be between 8 and 512");
        benchmark::Initialize(&argc, argv);
        if (benchmark::ReportUnrecognizedArguments(argc, argv))
            return 1;
        Kokkos::ScopeGuard const kokkos_scope(argc, argv);
        ddc::ScopeGuard const ddc_scope(argc, argv);
        if (verify) {
            validate();
            return 0;
        }
        std::string const name
                = std::string("TaylorGreen/")
                  + (physics == Physics::Compressible ? "compressible" : "incompressible") + "/N"
                  + std::to_string(nodes);
        benchmark::AddCustomContext("execution_space", Kokkos::DefaultExecutionSpace::name());
        benchmark::AddCustomContext("N", std::to_string(nodes));
        benchmark::AddCustomContext("Re", "1600");
        benchmark::AddCustomContext("Pr", "0.71");
        benchmark::AddCustomContext(
                "Mach",
                physics == Physics::Compressible ? "0.1" : "incompressible");
        benchmark::AddCustomContext(
                "timing",
                "manual_time=solver; monitors and export reported separately");
        benchmark::RegisterBenchmark(name.c_str(), run)
                ->UseManualTime()
                ->Iterations(1)
                ->Unit(benchmark::kSecond);
        benchmark::RunSpecifiedBenchmarks();
        benchmark::Shutdown();
        return benchmark_failed ? 1 : 0;
    } catch (std::exception const& error) {
        std::cerr << error.what() << "\n";
        return 1;
    }
}
