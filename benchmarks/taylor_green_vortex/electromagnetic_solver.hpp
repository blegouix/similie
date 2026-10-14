// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numbers>
#include <stdexcept>

#include <similie/physics/electrodynamics/resistive_electrodynamics.hpp>
#include <similie/physics/electromagnetism/resistive_electromagnetism.hpp>
#include <similie/physics/magnetodynamics/resistive_magnetodynamics.hpp>

#include "configuration.hpp"
#include "solvers.hpp"

namespace taylor_green_vortex {
/** Fixed nondimensional extension parameters, independent of resolution.
 * \important This documentation is fully AI-generated.
 */
struct ElectromagneticReference
{
    static constexpr double permittivity = 1, permeability = 1;
    static constexpr double guide_field = 0.25, magnetic_amplitude = 0.15;
    static constexpr double potential_amplitude = 0.1;
    static constexpr double magnetic_resistivity = 0.001;
    static constexpr double electric_resistivity = 10;
};

/** Matrix-free positive scalar Hodge Laplacian, with periodic constant gauge.
 * \important This operator and documentation is fully AI-generated.
 */
class ElectricPoissonOperator
{
    StructuredGrid& m_grid;
    Field m_input, m_output;

public:
    static constexpr bool IS_LINEAR = true, IS_SYMMETRIC = true;
    explicit ElectricPoissonOperator(StructuredGrid& grid)
        : m_grid(grid)
        , m_input("electric_poisson_input", 1, grid.cells)
        , m_output("electric_poisson_output", 1, grid.cells)
    {
    }
    std::size_t size() const
    {
        return m_grid.cells;
    }
    template <class Input, class Output>
    void apply(Kokkos::DefaultExecutionSpace exec, Input source, Output target) const
    {
        Field const input = m_input, output = m_output;
        Kokkos::parallel_for(
                "unpack_electric_poisson",
                Kokkos::RangePolicy<>(exec, 0, m_grid.cells),
                KOKKOS_LAMBDA(int i) { input(0, i) = source(i, 0); });
        m_grid.laplacian(input, 0, output, 0);
        Kokkos::parallel_for(
                "pack_electric_poisson",
                Kokkos::RangePolicy<>(exec, 0, m_grid.cells),
                KOKKOS_LAMBDA(int i) { target(i, 0) = -output(0, i); });
    }
};

/** EHD: evolve charge and reconstruct electrostatic D by a periodic Gauss solve.
 * MHD: evolve closed primal magnetic two-cochains. FMHD: also evolve electric
 * one-cochains and charge, using the same current in Ampere and continuity.
 * Native d and delta give d B_t=0 and (div D-charge)_t=0 in FMHD.
 * Fluid energy is evolved separately with J.E, paired on the same edges as
 * Ampere/Ohm. It is not the combined-energy state of the library MHD models.
 * \important This operator and documentation is fully AI-generated.
 */
class ElectromagneticSolver
{
    StructuredGrid& m_grid;
    VorticityDEC m_dec;
    ElectricPoissonOperator m_poisson;
    Configuration const m_configuration;
    Field m_initial, m_stage, m_rate, m_b, m_d, m_j, m_e, m_curl, m_ampere, m_work, m_motional,
            m_edges, m_current_edges, m_electric_edges, m_motional_edges, m_displacement_edges,
            m_magnetic_faces, m_poisson_rhs, m_poisson_solution;

public:
    // NVCC requires methods enclosing extended device lambdas to be public.
    void potential_displacement()
    {
        for (int a = 0; a < 3; ++a)
            m_grid.forward_derivative(potential, 0, m_edges, a, a);
        Field const edges = m_edges;
        Kokkos::parallel_for(
                "electric_potential_gradient",
                m_grid.cells,
                KOKKOS_LAMBDA(int i) {
                    for (int a = 0; a < 3; ++a)
                        edges(a, i) *= -ElectromagneticReference::permittivity;
                });
        m_dec.reduce_edges(edges, m_d);
    }
    void reconstruct_electrostatic()
    {
        Field const charge = charge_density, source = m_poisson_rhs;
        double mean, square;
        Kokkos::parallel_reduce(
                "net_charge",
                m_grid.cells,
                KOKKOS_LAMBDA(int i, double& sum) { sum += charge(0, i); },
                mean);
        Kokkos::parallel_reduce(
                "charge_norm",
                m_grid.cells,
                KOKKOS_LAMBDA(int i, double& sum) { sum += charge(0, i) * charge(0, i); },
                square);
        mean /= m_grid.cells;
        if (std::abs(mean) > 1e-10 * std::max(std::sqrt(square / m_grid.cells), 1e-12))
            throw std::runtime_error("Periodic EHD requires zero net charge");
        Kokkos::parallel_for(
                "electric_poisson_source",
                m_grid.cells,
                KOKKOS_LAMBDA(int i) {
                    source(i, 0) = (charge(0, i) - mean) / ElectromagneticReference::permittivity;
                });
        similie::solvers::StrongFormulationSolverSettings settings;
        settings.use_matrix_free = true;
        settings.preconditioner = similie::solvers::PreconditionerType::Identity;
        settings.relative_tolerance = 1e-11;
        settings.max_iterations = 8 * m_grid.n;
        similie::solvers::StrongFormulationSolverDiagnostics const result
                = similie::solvers::minimize_strong_formulation_residual(
                        Kokkos::DefaultExecutionSpace(),
                        m_poisson,
                        source,
                        m_poisson_solution,
                        settings);
        if (!std::isfinite(result.final_relative_residual)
            || result.final_relative_residual > 1e-10)
            throw std::runtime_error("Electrostatic reconstruction did not converge");
        poisson_iterations += result.iterations;
        Field const phi = potential, solution = m_poisson_solution;
        Kokkos::parallel_for(
                "unpack_electric_potential",
                m_grid.cells,
                KOKKOS_LAMBDA(int i) { phi(0, i) = solution(i, 0); });
        double sum;
        Kokkos::parallel_reduce(
                "electric_potential_mean",
                m_grid.cells,
                KOKKOS_LAMBDA(int i, double& value) { value += phi(0, i); },
                sum);
        double const shift = sum / m_grid.cells;
        Kokkos::parallel_for(
                "electric_potential_gauge",
                m_grid.cells,
                KOKKOS_LAMBDA(int i) { phi(0, i) -= shift; });
        potential_displacement();
    }

    // State slots: charge, D(one-cochain)[3], B(two-cochain)[3]. Inactive slots stay zero.
    Field state, induction, displacement, electric_field, current, charge_density, potential, force,
            fluid_power, joule_heating, gauss_error;
    std::int64_t poisson_iterations = 0;
    double evaluation_seconds = 0;
    explicit ElectromagneticSolver(StructuredGrid& grid, Configuration configuration)
        : m_grid(grid)
        , m_dec(grid)
        , m_poisson(grid)
        , m_configuration(configuration)
        , m_initial("initial_em", 7, grid.cells)
        , m_stage("stage_em", 7, grid.cells)
        , m_rate("rate_em", 7, grid.cells)
        , m_b("induction_cochain", 3, grid.cells)
        , m_d("displacement_cochain", 3, grid.cells)
        , m_j("current_cochain", 3, grid.cells)
        , m_e("electric_cochain", 3, grid.cells)
        , m_curl("faraday_curl", 3, grid.cells)
        , m_ampere("ampere_curl", 3, grid.cells)
        , m_work("em_divergence", 1, grid.cells)
        , m_motional("motional_electric", 3, grid.cells)
        , m_edges("em_edge_values", 3, grid.cells)
        , m_current_edges("current_edges", 3, grid.cells)
        , m_electric_edges("electric_edges", 3, grid.cells)
        , m_motional_edges("motional_edges", 3, grid.cells)
        , m_displacement_edges("displacement_edges", 3, grid.cells)
        , m_magnetic_faces("magnetic_faces", 3, grid.cells)
        , m_poisson_rhs("electric_poisson_rhs", grid.cells, 1)
        , m_poisson_solution("electric_poisson_solution", grid.cells, 1)
        , state("em_state", 7, grid.cells)
        , induction("induction", 3, grid.cells)
        , displacement("displacement", 3, grid.cells)
        , electric_field("electric_field", 3, grid.cells)
        , current("current", 3, grid.cells)
        , charge_density("charge_density", 1, grid.cells)
        , potential("electric_potential", 1, grid.cells)
        , force("lorentz_force", 3, grid.cells)
        , fluid_power("electric_work", 1, grid.cells)
        , joule_heating("joule_heating", 1, grid.cells)
        , gauss_error("gauss_error", 1, grid.cells)
    {
        reset();
    }

    bool magnetic() const
    {
        return m_configuration.coupling != Coupling::EHD;
    }
    bool electric() const
    {
        return m_configuration.coupling != Coupling::MHD;
    }
    void reset()
    {
        Kokkos::deep_copy(state, 0.0);
        Kokkos::deep_copy(m_b, 0.0);
        Kokkos::deep_copy(m_d, 0.0);
        Kokkos::deep_copy(potential, 0.0);
        int const n = m_grid.n;
        double const h = m_grid.h;
        Field const faces = m_edges, phi = potential;
        Kokkos::parallel_for(
                "electromagnetic_initial_condition",
                m_grid.cells,
                KOKKOS_LAMBDA(int i) {
                    double const x = -std::numbers::pi + (i / (n * n)) * h;
                    double const y = -std::numbers::pi + ((i / n) % n) * h;
                    double const z = -std::numbers::pi + (i % n) * h;
                    double const amplitude = ElectromagneticReference::magnetic_amplitude;
                    faces(0, i) = amplitude * sin(x) * cos(y + h / 2) * sin(z + h / 2);
                    faces(1, i) = -amplitude * cos(x + h / 2) * sin(y) * sin(z + h / 2);
                    faces(2, i) = ElectromagneticReference::guide_field;
                    phi(0, i) = ElectromagneticReference::potential_amplitude * cos(x) * cos(y)
                                * cos(z);
                });
        if (magnetic())
            m_dec.reduce_faces(faces, m_b);
        if (electric()) {
            potential_displacement();
            m_dec.divergence(m_d, charge_density);
        } else
            Kokkos::deep_copy(charge_density, 0.0);
        Field const q = state, b = m_b, d = m_d, charge = charge_density;
        Kokkos::parallel_for(
                "pack_initial_em_state",
                m_grid.cells,
                KOKKOS_LAMBDA(int i) {
                    q(0, i) = charge(0, i);
                    for (int a = 0; a < 3; ++a) {
                        q(a + 1, i) = d(a, i);
                        q(a + 4, i) = b(a, i);
                    }
                });
    }
    // Reconstruct geometry-dependent fields, then evaluate exchange and all EM rates.
    void evaluate(Field velocity)
    {
        std::chrono::steady_clock::time_point const started = std::chrono::steady_clock::now();
        Coupling const coupling = m_configuration.coupling;
        bool const resistive = m_configuration.transport == Transport::Resistive;
        Field const q = state, b = m_b, d = m_d, charge = charge_density;
        Kokkos::parallel_for(
                "unpack_em_state",
                m_grid.cells,
                KOKKOS_LAMBDA(int i) {
                    charge(0, i) = q(0, i);
                    for (int a = 0; a < 3; ++a) {
                        d(a, i) = q(a + 1, i);
                        b(a, i) = q(a + 4, i);
                    }
                });
        if (coupling == Coupling::EHD)
            reconstruct_electrostatic();
        m_dec.axial_nodes(b, induction);
        m_dec.velocity(d, displacement);
        Field const ampere = m_ampere;
        if (magnetic()) {
            m_dec.codifferential(b, ampere);
            Kokkos::parallel_for(
                    "magnetic_field_curl",
                    m_grid.cells,
                    KOKKOS_LAMBDA(int i) {
                        for (int a = 0; a < 3; ++a)
                            ampere(a, i) /= ElectromagneticReference::permeability;
                    });
        } else
            Kokkos::deep_copy(ampere, 0.0);
        Field const j = m_j, e = m_e, motion = m_motional, work = m_edges;
        Field const bn = induction, dn = displacement;
        // Reuse sector constitutive laws; resistive terms below retain native edge placement.
        Kokkos::parallel_for(
                "electromagnetic_transport",
                m_grid.cells,
                KOKKOS_LAMBDA(int i) {
                    std::array<double, 3> const u {velocity(0, i), velocity(1, i), velocity(2, i)};
                    std::array<double, 3> const magnetic {bn(0, i), bn(1, i), bn(2, i)};
                    similie::physics::magnetodynamics::Magnetodynamics const magnetic_model(
                            ElectromagneticReference::permeability);
                    similie::physics::electromagnetism::Electromagnetism const maxwell_model(
                            ElectromagneticReference::permittivity,
                            ElectromagneticReference::permeability);
                    similie::physics::electrodynamics::Electrodynamics const electric_model(
                            ElectromagneticReference::permittivity);
                    std::array<double, 3> const motional
                            = coupling == Coupling::MHD
                                      ? magnetic_model.ohm_electric_field(u, magnetic, {})
                                      : maxwell_model.rest_frame_electric_field(u, {}, magnetic);
                    std::array<double, 3> const advected
                            = electric_model.current(charge(0, i), u, {});
                    for (int a = 0; a < 3; ++a) {
                        motion(a, i) = motional[a];
                        work(a, i) = advected[a];
                    }
                });
        m_dec.reduce_nodes(motion, m_motional);
        // Keep the motional cochain for the rest-frame Ohm law and heating.
        Kokkos::deep_copy(e, m_motional);
        if (coupling == Coupling::MHD) {
            Kokkos::deep_copy(j, ampere);
            Kokkos::parallel_for(
                    "mhd_ohm_law",
                    m_grid.cells,
                    KOKKOS_LAMBDA(int i) {
                        for (int a = 0; a < 3; ++a)
                            if (resistive)
                                e(a, i) += ElectromagneticReference::magnetic_resistivity * j(a, i);
                    });
        } else {
            m_dec.reduce_nodes(work, j);
            Kokkos::parallel_for(
                    "electric_ohm_law",
                    m_grid.cells,
                    KOKKOS_LAMBDA(int i) {
                        for (int a = 0; a < 3; ++a) {
                            e(a, i) = d(a, i) / ElectromagneticReference::permittivity;
                            if (resistive)
                                j(a,
                                  i) += (e(a, i) + (coupling == Coupling::FMHD ? motion(a, i) : 0))
                                        / ElectromagneticReference::electric_resistivity;
                        }
                    });
        }
        m_dec.velocity(j, current);
        m_dec.velocity(e, electric_field);
        Field const jn = current, acceleration = force;
        Kokkos::parallel_for(
                "lorentz_force",
                m_grid.cells,
                KOKKOS_LAMBDA(int i) {
                    std::array<double, 3> const current_value {jn(0, i), jn(1, i), jn(2, i)};
                    std::array<double, 3> const magnetic {bn(0, i), bn(1, i), bn(2, i)};
                    std::array<double, 3> const
                            electric_displacement {dn(0, i), dn(1, i), dn(2, i)};
                    similie::physics::electromagnetism::Electromagnetism const
                            model(ElectromagneticReference::permittivity,
                                  ElectromagneticReference::permeability);
                    std::array<double, 3> const force_value = model.lorentz_force(
                            charge(0, i),
                            electric_displacement,
                            magnetic,
                            current_value);
                    for (int a = 0; a < 3; ++a)
                        acceleration(a, i) = force_value[a];
                });
        m_dec.edge_values(j, m_current_edges);
        m_dec.edge_values(e, m_electric_edges);
        m_dec.edge_values(m_motional, m_motional_edges);
        Field const je = m_current_edges, ee = m_electric_edges, me = m_motional_edges,
                    power = fluid_power, heating = joule_heating;
        int const n = m_grid.n;
        Kokkos::parallel_for(
                "compatible_electric_work",
                m_grid.cells,
                KOKKOS_LAMBDA(int i) {
                    double energy = 0, heat = 0;
                    for (int a = 0; a < 3; ++a) {
                        int const previous = StructuredGrid::neighbor(i, a, -1, n);
                        energy += 0.5 * (je(a, i) * ee(a, i) + je(a, previous) * ee(a, previous));
                        if (resistive) {
                            if (coupling == Coupling::MHD)
                                heat += 0.5 * ElectromagneticReference::magnetic_resistivity
                                        * (je(a, i) * je(a, i) + je(a, previous) * je(a, previous));
                            else {
                                double const rest
                                        = ee(a, i) + (coupling == Coupling::FMHD ? me(a, i) : 0);
                                double const rest_previous
                                        = ee(a, previous)
                                          + (coupling == Coupling::FMHD ? me(a, previous) : 0);
                                heat += 0.5 * (rest * rest + rest_previous * rest_previous)
                                        / ElectromagneticReference::electric_resistivity;
                            }
                        }
                    }
                    power(0, i) = energy;
                    heating(0, i) = heat;
                });
        m_dec.divergence(j, m_work);
        if (magnetic())
            m_dec.exterior_curl(e, m_curl);
        else
            Kokkos::deep_copy(m_curl, 0.0);
        Field const rate = m_rate, div = m_work, curl = m_curl;
        Kokkos::parallel_for(
                "electromagnetic_evolution",
                m_grid.cells,
                KOKKOS_LAMBDA(int i) {
                    rate(0, i) = coupling == Coupling::MHD ? 0 : -div(0, i);
                    for (int a = 0; a < 3; ++a) {
                        rate(a + 1, i) = coupling == Coupling::FMHD ? ampere(a, i) - j(a, i) : 0;
                        rate(a + 4, i) = -curl(a, i);
                    }
                });
        m_dec.divergence(d, gauss_error);
        Field const residual = gauss_error;
        Kokkos::parallel_for(
                "gauss_constraint",
                m_grid.cells,
                KOKKOS_LAMBDA(int i) { residual(0, i) -= charge(0, i); });
        Kokkos::fence("electromagnetic_evaluation_complete");
        evaluation_seconds
                += std::chrono::duration<double>(std::chrono::steady_clock::now() - started)
                           .count();
    }
    void begin_step()
    {
        Kokkos::deep_copy(m_initial, state);
    }
    void advance_stage(int step, double dt)
    {
        Field const q = state, initial = m_initial, rate = m_rate, stage = m_stage;
        double const weight = step == 0 ? 0 : (step == 1 ? 0.75 : 1.0 / 3.0);
        Kokkos::parallel_for(
                "electromagnetic_ssprk3",
                m_grid.cells,
                KOKKOS_LAMBDA(int i) {
                    for (int c = 0; c < 7; ++c)
                        stage(c, i) = weight * initial(c, i)
                                      + (1 - weight) * (q(c, i) + dt * rate(c, i));
                });
        Kokkos::deep_copy(state, stage);
    }
    double timestep_limit() const
    {
        double limit = m_configuration.coupling == Coupling::FMHD
                               ? 0.15 * m_grid.h
                                         * std::sqrt(
                                                 ElectromagneticReference::permittivity
                                                 * ElectromagneticReference::permeability)
                               : std::numeric_limits<double>::infinity();
        if (m_configuration.transport == Transport::Resistive) {
            if (m_configuration.coupling == Coupling::MHD)
                limit = std::
                        min(limit,
                            0.1 * m_grid.h * m_grid.h * ElectromagneticReference::permeability
                                    / ElectromagneticReference::magnetic_resistivity);
            else
                limit = std::
                        min(limit,
                            0.3 * ElectromagneticReference::electric_resistivity
                                    * ElectromagneticReference::permittivity);
        }
        return limit;
    }

    double field_energy_rate()
    {
        Field const rate = m_rate, phi = potential;
        if (m_configuration.coupling == Coupling::EHD) {
            double value;
            Kokkos::parallel_reduce(
                    "electrostatic_energy_rate",
                    m_grid.cells,
                    KOKKOS_LAMBDA(int i, double& sum) { sum += phi(0, i) * rate(0, i); },
                    value);
            return value / m_grid.cells;
        }
        Field const electric_rate = m_edges, magnetic_rate = m_curl;
        Kokkos::parallel_for(
                "unpack_field_energy_rates",
                m_grid.cells,
                KOKKOS_LAMBDA(int i) {
                    for (int a = 0; a < 3; ++a) {
                        electric_rate(a, i) = rate(a + 1, i);
                        magnetic_rate(a, i) = rate(a + 4, i);
                    }
                });
        m_dec.edge_values(electric_rate, m_motional_edges);
        m_dec.face_values(magnetic_rate, m_current_edges);
        Field const d = m_displacement_edges, b = m_magnetic_faces, ddot = m_motional_edges,
                    bdot = m_current_edges;
        double value;
        Kokkos::parallel_reduce(
                "maxwell_magnetic_energy_rate",
                m_grid.cells,
                KOKKOS_LAMBDA(int i, double& sum) {
                    for (int a = 0; a < 3; ++a)
                        sum += d(a, i) * ddot(a, i) / ElectromagneticReference::permittivity
                               + b(a, i) * bdot(a, i) / ElectromagneticReference::permeability;
                },
                value);
        return value / m_grid.cells;
    }

    // time, magnetic energy, electric energy, J.E, heat, divB, Gauss, charge, energy exchange residual.
    std::array<double, 9> monitor(double time)
    {
        m_dec.edge_values(m_d, m_displacement_edges);
        m_dec.face_values(m_b, m_magnetic_faces);
        Field const d = m_displacement_edges, b = m_magnetic_faces, power = fluid_power,
                    heat = joule_heating, gauss = gauss_error, charge = charge_density;
        std::array<double, 9> values {time};
        for (int diagnostic = 1; diagnostic < 8; ++diagnostic) {
            if (diagnostic == 5) {
                values[5] = m_dec.closure_rms(m_b);
                continue;
            }
            double sum;
            Kokkos::parallel_reduce(
                    "electromagnetic_monitors",
                    m_grid.cells,
                    KOKKOS_LAMBDA(int i, double& value) {
                        if (diagnostic == 1 || diagnostic == 2) {
                            for (int a = 0; a < 3; ++a)
                                value += diagnostic == 1 ? b(a, i) * b(a, i)
                                                                   / (2
                                                                      * ElectromagneticReference::
                                                                              permeability)
                                                         : d(a, i) * d(a, i)
                                                                   / (2
                                                                      * ElectromagneticReference::
                                                                              permittivity);
                        } else if (diagnostic == 3)
                            value += power(0, i);
                        else if (diagnostic == 4)
                            value += heat(0, i);
                        else if (diagnostic == 6)
                            value += gauss(0, i) * gauss(0, i);
                        else
                            value += charge(0, i);
                    },
                    sum);
            values[diagnostic]
                    = diagnostic == 6 ? std::sqrt(sum / m_grid.cells) : sum / m_grid.cells;
        }
        values[8] = field_energy_rate() + values[3];
        for (double value : values)
            if (!std::isfinite(value))
                throw std::runtime_error("Nonfinite electromagnetic monitor");
        return values;
    }
};
} // namespace taylor_green_vortex
