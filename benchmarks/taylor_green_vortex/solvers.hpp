// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#pragma once
#include <array>
#include <cstdint>
#include <stdexcept>

#include <similie/physics/fluid/incompressible_navier_stokes.hpp>
#include <similie/solvers/minimize_strong_formulation_residual.hpp>

#include "parameters.hpp"
#include "vorticity_dec.hpp"

namespace taylor_green_vortex {
/** Conservative two-point kinetic-energy-preserving face transport, DEC divergence,
 * centered Newtonian stress and Fourier flux. Face-normal viscous derivatives
 * use exterior derivative/reconstruction, with centered tangential placement.
 * Mass flux is mean(rho)*mean(u_n); momentum flux is F_rho*mean(u)+mean(p)n.
 * Energy flux uses mean(internal energy)+u_L dot u_R/2 and mean(p)*mean(u_n).
 * The scheme is centered and second order, without reconstruction or limiter.
 * It is not positivity preserving; callers must reject invalid states.
 * \important This operator and documentation is fully AI-generated.
 */
class CompressibleSolver
{
    StructuredGrid& m_dec;
    CompressiblePhysics const m_physics;
    Field m_primitive, m_gradient, m_normal_gradient, m_flux, m_divergence;

public:
    explicit CompressibleSolver(StructuredGrid& dec)
        : m_dec(dec)
        , m_physics(Reference::model())
        , m_primitive("primitive", 5, dec.cells)
        , m_gradient("gradient", 9, dec.cells)
        , m_normal_gradient("face_normal_gradient", 4, dec.cells)
        , m_flux("flux", 5, dec.cells)
        , m_divergence("flux_divergence", 5, dec.cells)
    {
    }
    void primitives(Field state)
    {
        Field const primitive = m_primitive;
        CompressiblePhysics const physics = m_physics;
        Kokkos::parallel_for(
                "thermodynamic_closure",
                m_dec.cells,
                KOKKOS_LAMBDA(int i) {
                    CompressiblePhysics::PrimitiveVariables const values = physics.primitives(
                            {state(0, i), {state(1, i), state(2, i), state(3, i)}, state(4, i)});
                    primitive(0, i) = values.density;
                    for (int a = 0; a < 3; ++a)
                        primitive(a + 1, i) = values.velocity[a];
                    primitive(4, i) = values.temperature;
                });
    }
    void rhs(Field state, Field result)
    {
        primitives(state);
        for (int c = 1; c < 4; ++c)
            for (int a = 0; a < 3; ++a)
                m_dec.derivative(m_primitive, c, m_gradient, 3 * (c - 1) + a, a);
        Kokkos::deep_copy(result, 0.0);
        Field const primitive = m_primitive, gradient = m_gradient, normal = m_normal_gradient,
                    flux = m_flux, divergence = m_divergence;
        int const n = m_dec.n;
        CompressiblePhysics const physics = m_physics;
        for (int axis = 0; axis < 3; ++axis) {
            for (int c = 1; c < 5; ++c)
                m_dec.forward_derivative(primitive, c, normal, c - 1, axis);
            Kokkos::parallel_for(
                    "navier_stokes_face_flux",
                    m_dec.cells,
                    KOKKOS_LAMBDA(int i) {
                        int const j = StructuredGrid::neighbor(i, axis, 1, n);
                        CompressiblePhysics::ConservativeVariables const left
                                = Reference::state(state, i),
                                right = Reference::state(state, j);
                        double const pl = physics.pressure(left), pr = physics.pressure(right);
                        double const ul = primitive(axis + 1, i), ur = primitive(axis + 1, j);
                        double const mass_flux
                                = 0.25 * (primitive(0, i) + primitive(0, j)) * (ul + ur);
                        double const pressure = 0.5 * (pl + pr);
                        double kinetic_cross = 0;
                        for (int a = 0; a < 3; ++a)
                            kinetic_cross += 0.5 * primitive(a + 1, i) * primitive(a + 1, j);
                        double const internal = 0.5
                                                * (physics.specific_internal_energy(left)
                                                   + physics.specific_internal_energy(right));
                        CompressiblePhysics::VelocityGradient strain {};
                        for (int a = 0; a < 3; ++a)
                            for (int b = 0; b < 3; ++b)
                                strain[a][b] = b == axis ? normal(a, i)
                                                         : 0.5
                                                                   * (gradient(3 * a + b, i)
                                                                      + gradient(3 * a + b, j));
                        CompressiblePhysics::VelocityGradient const tau = physics.stress(strain);
                        std::array<double, 3> temperature_gradient {};
                        temperature_gradient[axis] = normal(3, i);
                        std::array<double, 3> const heat_flux
                                = physics.heat_flux(left, temperature_gradient);
                        std::array<double, 3> const right_heat_flux
                                = physics.heat_flux(right, temperature_gradient);
                        double work = 0;
                        for (int a = 0; a < 3; ++a)
                            work += tau[a][axis] * 0.5
                                    * (primitive(a + 1, i) + primitive(a + 1, j));
                        flux(0, i) = mass_flux;
                        for (int a = 0; a < 3; ++a)
                            flux(a + 1, i)
                                    = mass_flux * 0.5 * (primitive(a + 1, i) + primitive(a + 1, j))
                                      + (a == axis ? pressure : 0.0) - tau[a][axis];
                        flux(4, i) = mass_flux * (internal + kinetic_cross)
                                     + 0.5 * pressure * (ul + ur) - work
                                     + 0.5 * (heat_flux[axis] + right_heat_flux[axis]);
                    });
            for (int c = 0; c < 5; ++c)
                m_dec.flux_divergence(flux, c, divergence, c, axis);
            Kokkos::parallel_for(
                    "navier_stokes_conservation_laws",
                    m_dec.cells,
                    KOKKOS_LAMBDA(int i) {
                        CompressiblePhysics::ConservativeVariables const rate = physics(
                                {divergence(0, i),
                                 {divergence(1, i), divergence(2, i), divergence(3, i)},
                                 divergence(4, i)});
                        result(0, i) += rate.density;
                        for (int a = 0; a < 3; ++a)
                            result(a + 1, i) += rate.momentum[a];
                        result(4, i) += rate.total_energy;
                    });
        }
    }
};

/** Diagnostic pressure reconstruction, with a centered DEC gradient and
 * its negative adjoint divergence. Orthogonal periodic projection solves
 * -D G phi=-div u by matrix-free CG. The mean and even-grid checkerboard
 * null modes are untouched; divergence is in the range of this operator.
 * Viscosity uses the exterior staged Hodge Laplacian with its diffusion sign.
 * Applied only to diagnostic momentum balance; the evolved state is vorticity.
 * The elliptic potential is Bernoulli pressure p+|u|^2/2.
 * \important This operator and documentation is fully AI-generated.
 */
class PressureReconstruction
{
    StructuredGrid& m_dec;
    similie::physics::fluid::IncompressibleNavierStokes const m_physics;
    Field m_vorticity, m_lap, m_phi, m_r, m_p, m_ap, m_temp, m_projection_gradient,
            m_pressure_acceleration;
    void poisson(Field input, Field output)
    {
        for (int a = 0; a < 3; ++a)
            m_dec.derivative(input, 0, m_projection_gradient, a, a);
        m_dec.divergence(m_projection_gradient, output);
        Kokkos::parallel_for(
                "negative_div_gradient",
                m_dec.cells,
                KOKKOS_LAMBDA(int i) { output(0, i) *= -1; });
    }

    double dot(Field left, Field right)
    {
        double value;
        Kokkos::parallel_reduce(
                "projection_dot",
                m_dec.cells,
                KOKKOS_LAMBDA(int i, double& sum) { sum += left(0, i) * right(0, i); },
                value);
        return value;
    }

    void unprojected_rhs(Field velocity, Field result, Field supplied_vorticity = Field())
    {
        if (supplied_vorticity.data())
            Kokkos::deep_copy(m_vorticity, supplied_vorticity);
        else
            m_dec.curl(velocity, m_vorticity);
        for (int c = 0; c < 3; ++c)
            m_dec.laplacian(velocity, c, m_lap, c);
        Field const rotation = m_vorticity, lap = m_lap;
        similie::physics::fluid::IncompressibleNavierStokes const physics = m_physics;
        Kokkos::parallel_for(
                "incompressible_momentum",
                m_dec.cells,
                KOKKOS_LAMBDA(int i) {
                    std::array<double, 3> const rate = physics.momentum_rate(
                            {velocity(0, i), velocity(1, i), velocity(2, i)},
                            {rotation(0, i), rotation(1, i), rotation(2, i)},
                            {},
                            {lap(0, i), lap(1, i), lap(2, i)});
                    for (int a = 0; a < 3; ++a)
                        result(a, i) = rate[a];
                });
    }
    void recover_pressure(Field velocity, Field output, double mean_pressure)
    {
        Field const potential = m_phi;
        similie::physics::fluid::IncompressibleNavierStokes const physics = m_physics;
        Kokkos::parallel_for(
                "recover_hydrodynamic_pressure",
                m_dec.cells,
                KOKKOS_LAMBDA(int i) {
                    output(0, i) = physics.pressure(
                            potential(0, i),
                            {velocity(0, i), velocity(1, i), velocity(2, i)});
                });
        double sum;
        Kokkos::parallel_reduce(
                "pressure_mean",
                m_dec.cells,
                KOKKOS_LAMBDA(int i, double& value) { value += output(0, i); },
                sum);
        double const shift = mean_pressure - sum / m_dec.cells;
        Kokkos::parallel_for(
                "fix_pressure_gauge",
                m_dec.cells,
                KOKKOS_LAMBDA(int i) { output(0, i) += shift; });
    }

public:
    Field pressure;
    std::int64_t total_iterations = 0;
    explicit PressureReconstruction(StructuredGrid& dec)
        : m_dec(dec)
        , m_physics(1.0, Reference::viscosity)
        , m_vorticity("velocity_curl", 3, dec.cells)
        , m_lap("velocity_laplacian", 3, dec.cells)
        , m_phi("projection", 1, dec.cells)
        , m_r("cg_r", 1, dec.cells)
        , m_p("cg_p", 1, dec.cells)
        , m_ap("cg_ap", 1, dec.cells)
        , m_temp("cg_temp", 1, dec.cells)
        , m_projection_gradient("projection_gradient", 3, dec.cells)
        , m_pressure_acceleration("pressure_reconstruction_acceleration", 3, dec.cells)
        , pressure("pressure", 1, dec.cells)
    {
        Kokkos::deep_copy(pressure, 0.0);
    }
    /** Reconstruct periodic pressure for a supplied divergence-free velocity.
     * Solve -D G Pi=-D(u cross omega+nu Delta u), then recover
     * p=rho(Pi-|u|^2/2) and shift it to the requested volume mean.
     * Zero-start CG fixes unresolved checkerboard modes of Pi on even grids.
     * The supplied velocity is unchanged; output must be a separate scalar field.
     * This can be called independently of vorticity time integration.
     * \important This operator and documentation is fully AI-generated.
     */
    void compute_pressure(
            Field velocity,
            Field output,
            double mean_pressure = 0.0,
            Field supplied_vorticity = Field())
    {
        unprojected_rhs(velocity, m_pressure_acceleration, supplied_vorticity);
        project(m_pressure_acceleration);
        recover_pressure(velocity, output, mean_pressure);
    }

private:
    void project(Field velocity)
    {
        m_dec.divergence(velocity, m_r);
        Field const r = m_r, p = m_p, ap = m_ap, phi = m_phi, temp = m_temp;
        similie::physics::fluid::IncompressibleNavierStokes const physics = m_physics;
        Kokkos::parallel_for(
                "incompressibility_constraint",
                m_dec.cells,
                KOKKOS_LAMBDA(int i) { r(0, i) = -physics.incompressibility(r(0, i)); });
        Kokkos::deep_copy(phi, 0.0);
        Kokkos::deep_copy(p, r);
        double rr = dot(r, r);
        double const tolerance = 1e-20 * fmax(rr, 1.0);

        for (int iteration = 0; rr > tolerance && iteration < 4 * m_dec.n; ++iteration) {
            poisson(p, ap);
            double const pap = dot(p, ap);
            if (!(pap > 0))
                throw std::runtime_error("Incompressible projection CG breakdown");
            double const alpha = rr / pap;
            Kokkos::parallel_for(
                    "projection_cg_update",
                    m_dec.cells,
                    KOKKOS_LAMBDA(int i) {
                        phi(0, i) += alpha * p(0, i);
                        r(0, i) -= alpha * ap(0, i);
                    });
            double const next = dot(r, r), beta = next / rr;
            Kokkos::parallel_for(
                    "projection_cg_direction",
                    m_dec.cells,
                    KOKKOS_LAMBDA(int i) { p(0, i) = r(0, i) + beta * p(0, i); });
            rr = next;
            ++total_iterations;
        }
        if (rr > tolerance)
            throw std::runtime_error("Incompressible projection did not converge");
        for (int a = 0; a < 3; ++a) {
            m_dec.derivative(phi, 0, temp, 0, a);
            Kokkos::parallel_for(
                    "orthogonal_projection",
                    m_dec.cells,
                    KOKKOS_LAMBDA(int i) { velocity(a, i) -= temp(0, i); });
        }
    }
};

/** Matrix-free native two-form Hodge Laplacian; uniform periodic metric makes
 * it symmetric in the cochain Euclidean inner product. Mean-zero right-hand
 * sides and zero-start CG exclude its three harmonic constant modes.
 * \important This operator and documentation is fully AI-generated.
 */
class VorticityPoissonOperator
{
    VorticityDEC& m_dec;
    int const m_cells;
    Field m_input, m_output;

public:
    static constexpr bool IS_LINEAR = true;
    static constexpr bool IS_SYMMETRIC = true;
    VorticityPoissonOperator(VorticityDEC& dec, int cells)
        : m_dec(dec)
        , m_cells(cells)
        , m_input("poisson_input", 3, cells)
        , m_output("poisson_output", 3, cells)
    {
    }
    std::size_t size() const
    {
        return 3 * std::size_t(m_cells);
    }
    template <class Input, class Output>
    void apply(Kokkos::DefaultExecutionSpace exec, Input source, Output target) const
    {
        Field const input = m_input, output = m_output;
        int const cells = m_cells;
        Kokkos::parallel_for(
                "unpack_poisson_vector",
                Kokkos::RangePolicy<>(exec, 0, cells),
                KOKKOS_LAMBDA(int i) {
                    for (int c = 0; c < 3; ++c)
                        input(c, i) = source(c * cells + i, 0);
                });
        m_dec.positive_laplacian(input, output);
        Kokkos::parallel_for(
                "pack_poisson_vector",
                Kokkos::RangePolicy<>(exec, 0, cells),
                KOKKOS_LAMBDA(int i) {
                    for (int c = 0; c < 3; ++c)
                        target(c * cells + i, 0) = output(c, i);
                });
    }
};

/** Evolve W=dU, reconstruct B=*A by Delta_H B=W, and recover U=delta B.
 * The closed two-cochain gauge dB=0 is the Coulomb gauge div A=0.
 * The harmonic mean velocity is fixed to zero for Taylor-Green.
 * \important This operator and documentation is fully AI-generated.
 */
class IncompressibleSolver
{
    StructuredGrid& m_grid;
    VorticityDEC m_dec;
    VorticityPoissonOperator m_poisson;
    PressureReconstruction m_pressure;
    similie::physics::fluid::IncompressibleNavierStokes const m_physics;
    Field m_potential, m_one, m_transport, m_transport_one, m_lap, m_poisson_rhs,
            m_poisson_solution, m_check;

public:
    Field velocity, vorticity, vector_potential, pressure;
    std::int64_t total_iterations = 0;
    double reconstruction_seconds = 0, last_relative_residual = 0;
    double gauge_rms = 0, vorticity_divergence_rms = 0, curl_residual_rms = 0;
    explicit IncompressibleSolver(StructuredGrid& grid)
        : m_grid(grid)
        , m_dec(grid)
        , m_poisson(m_dec, grid.cells)
        , m_pressure(grid)
        , m_physics(1, Reference::viscosity)
        , m_potential("dual_vector_potential", 3, grid.cells)
        , m_one("velocity_cochain", 3, grid.cells)
        , m_transport("vorticity_transport", 3, grid.cells)
        , m_transport_one("transport_cochain", 3, grid.cells)
        , m_lap("vorticity_hodge_laplacian", 3, grid.cells)
        , m_poisson_rhs("vector_poisson_rhs", 3 * grid.cells, 1)
        , m_poisson_solution("vector_poisson_solution", 3 * grid.cells, 1)
        , m_check("curl_check", 3, grid.cells)
        , velocity("reconstructed_velocity", 3, grid.cells)
        , vorticity("reconstructed_vorticity", 3, grid.cells)
        , vector_potential("vector_potential", 3, grid.cells)
        , pressure(m_pressure.pressure)
    {
    }

    void initialize(Field state)
    {
        // This edge representation reconstructs the prescribed nodal velocity
        // exactly; it is closed under native codifferential on the uniform grid.
        int const n = m_grid.n;
        double const h = m_grid.h, scale = 1 / std::cos(h / 2);
        Field const edges = m_transport;
        Kokkos::parallel_for(
                "initial_velocity_edges",
                m_grid.cells,
                KOKKOS_LAMBDA(int i) {
                    double const x = -std::numbers::pi + (i / (n * n)) * h;
                    double const y = -std::numbers::pi + ((i / n) % n) * h;
                    double const z = -std::numbers::pi + (i % n) * h;
                    edges(0, i) = scale * sin(x + h / 2) * cos(y) * cos(z);
                    edges(1, i) = -scale * cos(x) * sin(y + h / 2) * cos(z);
                    edges(2, i) = 0;
                });
        m_dec.reduce_edges(edges, m_one);
        m_dec.exterior_curl(m_one, state);
    }
    void reconstruct_velocity(Field state, Field output)
    {
        std::chrono::steady_clock::time_point const start = std::chrono::steady_clock::now();
        // Remove only roundoff-sized harmonic means; a nonzero physical mean
        // vorticity is incompatible with a periodic vector potential.
        Field const rhs = m_poisson_rhs;
        int const cells = m_grid.cells;
        for (int c = 0; c < 3; ++c) {
            double sum, norm;
            Kokkos::parallel_reduce(
                    "vorticity_mean",
                    cells,
                    KOKKOS_LAMBDA(int i, double& value) { value += state(c, i); },
                    sum);
            Kokkos::parallel_reduce(
                    "vorticity_component_norm",
                    cells,
                    KOKKOS_LAMBDA(int i, double& value) { value += state(c, i) * state(c, i); },
                    norm);
            double const mean = sum / cells;
            if (std::abs(mean) > 1e-10 * std::max(std::sqrt(norm / cells), m_grid.h * m_grid.h))
                throw std::runtime_error("Periodic vorticity has a nonzero harmonic mean");
            Kokkos::parallel_for(
                    "mean_zero_vorticity_rhs",
                    cells,
                    KOKKOS_LAMBDA(int i) { rhs(c * cells + i, 0) = state(c, i) - mean; });
        }
        similie::solvers::StrongFormulationSolverSettings settings;
        settings.use_matrix_free = true;
        settings.preconditioner = similie::solvers::PreconditionerType::Identity;
        settings.relative_tolerance = 1e-11;
        settings.max_iterations = 8 * m_grid.n;
        similie::solvers::StrongFormulationSolverDiagnostics const diagnostics
                = similie::solvers::minimize_strong_formulation_residual(
                        Kokkos::DefaultExecutionSpace(),
                        m_poisson,
                        rhs,
                        m_poisson_solution,
                        settings);
        if (!std::isfinite(diagnostics.final_relative_residual)
            || diagnostics.final_relative_residual > 10 * settings.relative_tolerance)
            throw std::runtime_error("Vector-potential reconstruction did not converge");
        total_iterations += diagnostics.iterations;
        last_relative_residual = diagnostics.final_relative_residual;
        Field const solution = m_poisson_solution, potential = m_potential;
        Kokkos::parallel_for(
                "unpack_vector_potential",
                cells,
                KOKKOS_LAMBDA(int i) {
                    for (int c = 0; c < 3; ++c)
                        potential(c, i) = solution(c * cells + i, 0);
                });
        // Fix the remaining harmonic vector-potential freedom explicitly.
        for (int c = 0; c < 3; ++c) {
            double sum;
            Kokkos::parallel_reduce(
                    "potential_mean",
                    cells,
                    KOKKOS_LAMBDA(int i, double& value) { value += potential(c, i); },
                    sum);
            double const mean = sum / cells;
            Kokkos::parallel_for(
                    "zero_mean_vector_potential",
                    cells,
                    KOKKOS_LAMBDA(int i) { potential(c, i) -= mean; });
        }
        // Check the actual DEC residual, independently of the Krylov logger.
        m_dec.positive_laplacian(m_potential, m_check);
        Field const check = m_check;
        double residual_norm, rhs_norm;
        Kokkos::parallel_reduce(
                "vector_potential_true_residual",
                cells,
                KOKKOS_LAMBDA(int i, double& sum) {
                    for (int c = 0; c < 3; ++c) {
                        double const difference = check(c, i) - rhs(c * cells + i, 0);
                        sum += difference * difference;
                    }
                },
                residual_norm);
        Kokkos::parallel_reduce(
                "vector_potential_rhs_norm",
                cells,
                KOKKOS_LAMBDA(int i, double& sum) {
                    for (int c = 0; c < 3; ++c) {
                        double const value = rhs(c * cells + i, 0);
                        sum += value * value;
                    }
                },
                rhs_norm);
        last_relative_residual
                = rhs_norm == 0 ? std::sqrt(residual_norm) : std::sqrt(residual_norm / rhs_norm);
        if (!std::isfinite(last_relative_residual)
            || last_relative_residual > 10 * settings.relative_tolerance)
            throw std::runtime_error("Vector-potential true DEC residual is too large");
        m_dec.codifferential(m_potential, m_one);
        m_dec.velocity(m_one, output);
        Kokkos::fence("vector_potential_reconstruction");
        reconstruction_seconds
                += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    }
    void rhs(Field state, Field result, bool reconstruct_pressure = false)
    {
        reconstruct_velocity(state, velocity);
        m_dec.axial_nodes(state, vorticity);
        Field const u = velocity, w = vorticity, transport = m_transport;
        similie::physics::fluid::IncompressibleNavierStokes const physics = m_physics;
        Kokkos::parallel_for(
                "vorticity_nonlinear_transport",
                m_grid.cells,
                KOKKOS_LAMBDA(int i) {
                    std::array<double, 3> const values = physics.transport(
                            {u(0, i), u(1, i), u(2, i)},
                            {w(0, i), w(1, i), w(2, i)});
                    for (int a = 0; a < 3; ++a)
                        transport(a, i) = values[a];
                });
        m_dec.reduce_nodes(m_transport, m_transport_one);
        m_dec.exterior_curl(m_transport_one, result);
        m_dec.positive_laplacian(state, m_lap);
        Field const lap = m_lap;
        Kokkos::parallel_for(
                "vorticity_evolution",
                m_grid.cells,
                KOKKOS_LAMBDA(int i) {
                    for (int c = 0; c < 3; ++c)
                        result(c, i) = physics(result(c, i), -lap(c, i));
                });
        if (reconstruct_pressure) {
            m_pressure.compute_pressure(velocity, pressure, 0.0, vorticity);
            monitor_constraints(state);
        }
    }
    void compute_pressure(Field state, Field output, double mean_pressure = 0)
    {
        reconstruct_velocity(state, velocity);
        m_dec.axial_nodes(state, vorticity);
        m_pressure.compute_pressure(velocity, output, mean_pressure, vorticity);
    }
    void divergence(Field output)
    {
        m_dec.divergence(m_one, output);
    }
    void monitor_constraints(Field state)
    {
        vorticity_divergence_rms = m_dec.closure_rms(state);
        gauge_rms = m_dec.closure_rms(m_potential);
        m_dec.axial_nodes(m_potential, vector_potential);
        m_dec.exterior_curl(m_one, m_check);
        Field const check = m_check;
        double sum;
        Kokkos::parallel_reduce(
                "velocity_curl_residual",
                m_grid.cells,
                KOKKOS_LAMBDA(int i, double& value) {
                    for (int c = 0; c < 3; ++c) {
                        double const error = check(c, i) - state(c, i);
                        value += error * error;
                    }
                },
                sum);
        curl_residual_rms = std::sqrt(sum / m_grid.cells) / (m_grid.h * m_grid.h);
        if (vorticity_divergence_rms > 1e-8 || gauge_rms > 1e-8 || curl_residual_rms > 1e-8)
            throw std::runtime_error(
                    "Vorticity/vector-potential DEC constraints are not satisfied");
    }
};

} // namespace taylor_green_vortex
