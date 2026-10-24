// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#pragma once
#include <array>
#include <cstdint>
#include <stdexcept>

#include <similie/physics/fluid/incompressible_navier_stokes.hpp>
#include <similie/solvers/minimize_strong_formulation_residual.hpp>

#include <KokkosBlas1_axpby.hpp>

#include "parameters.hpp"
#include "poisson_solver.hpp"
#include "vorticity_dec.hpp"

namespace taylor_green_vortex {
/** Assemble the library fluid flux with reduction and codifferential.
 * Face gradients are reconstructed exterior derivatives, with arithmetic
 * tangential placement on the uniform grid. NavierStokesFourier supplies the
 * two-point flux, constitutive laws, and conservation rates.
 * \important This operator and documentation is fully AI-generated.
 */
class CompressibleSolver
{
    StructuredGrid& m_dec;
    CompressiblePhysics const m_physics;
    Field m_primitive, m_gradient, m_temperature_gradient, m_flux, m_divergence;

public:
    explicit CompressibleSolver(StructuredGrid& dec)
        : m_dec(dec)
        , m_physics(Reference::model())
        , m_primitive("primitive", 5, dec.cells)
        , m_gradient("gradient", 9, dec.cells)
        , m_temperature_gradient("temperature_gradient", 3, dec.cells)
        , m_flux("flux", 5, dec.cells)
        , m_divergence("flux_divergence", 5, dec.cells)
    {
    }
    void primitives(Field const& state)
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
    void rhs(Field const& state, Field const& result)
    {
        ProfilingRegion const region("equations/compressible-conservation");
        primitives(state);
        m_dec.velocity_gradient(m_primitive, m_gradient, 1);
        m_dec.gradient(m_primitive, m_temperature_gradient, 4);
        Kokkos::deep_copy(result, 0.0);
        Field const gradient = m_gradient, normal = m_temperature_gradient, flux = m_flux,
                    divergence = m_divergence;
        int const n = m_dec.n;
        CompressiblePhysics const physics = m_physics;
        for (int axis = 0; axis < 3; ++axis) {
            ProfilingRegion const axis_region("flux");
            Kokkos::parallel_for(
                    "navier_stokes_face_flux",
                    m_dec.cells,
                    KOKKOS_LAMBDA(int i) {
                        int const j = StructuredGrid::neighbor(i, axis, 1, n);
                        CompressiblePhysics::State const left = Reference::state(state, i),
                                                         right = Reference::state(state, j);
                        CompressiblePhysics::VelocityGradient strain {};
                        for (int a = 0; a < 3; ++a)
                            for (int b = 0; b < 3; ++b)
                                strain[a][b] = gradient(3 * a + b, i);
                        std::array<double, 3> const
                                temperature_gradient {normal(0, i), normal(1, i), normal(2, i)};
                        CompressiblePhysics::Flux const values = physics.two_point_flux(
                                left,
                                right,
                                strain,
                                temperature_gradient,
                                axis);
                        flux(0, i) = values.mass;
                        for (int a = 0; a < 3; ++a)
                            flux(a + 1, i) = values.momentum[a];
                        flux(4, i) = values.total_energy;
                    });
            std::array<char const*, 5> const
                    equations {"mass", "momentum-x", "momentum-y", "momentum-z", "total-energy"};
            for (int c = 0; c < 5; ++c) {
                ProfilingRegion const equation_region(std::string("equation/") + equations[c]);
                m_dec.negative_face_codifferential(flux, c, divergence, c, axis);
            }
            ProfilingRegion const assembly_region("shared-conservation-assembly");
            Kokkos::parallel_for(
                    "navier_stokes_conservation_laws",
                    m_dec.cells,
                    KOKKOS_LAMBDA(int i) {
                        CompressiblePhysics::State const rate = physics(
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

/** Matrix-free positive scalar Hodge Laplacian, with periodic constant gauge.
 * \important This operator and documentation is fully AI-generated.
 */
class ScalarPoissonOperator
{
    StructuredGrid& m_grid;
    Field m_input, m_output;

public:
    static constexpr bool IS_LINEAR = true, IS_SYMMETRIC = true;
    explicit ScalarPoissonOperator(StructuredGrid& grid)
        : m_grid(grid)
        , m_input("scalar_poisson_input", 1, grid.cells)
        , m_output("scalar_poisson_output", 1, grid.cells)
    {
    }
    std::size_t size() const
    {
        return m_grid.cells;
    }
    template <class Input, class Output>
    void apply(Kokkos::DefaultExecutionSpace const& exec, Input const& source, Output const& target)
            const
    {
        Field const input = m_input, output = m_output;
        Kokkos::parallel_for(
                "unpack_scalar_poisson",
                Kokkos::RangePolicy<>(exec, 0, m_grid.cells),
                KOKKOS_LAMBDA(int i) { input(0, i) = source(i, 0); });
        m_grid.laplacian(input, 0, output, 0);
        Kokkos::parallel_for(
                "pack_scalar_poisson",
                Kokkos::RangePolicy<>(exec, 0, m_grid.cells),
                KOKKOS_LAMBDA(int i) { target(i, 0) = -output(0, i); });
    }
};

/** Bernoulli pressure from the native scalar Hodge Laplacian.
 * With F=u cross omega+nu Delta u+force/rho, solve delta d Pi=delta R(F),
 * where R reduces the reconstructed force to one-cochains. This is a diagnostic
 * solve; velocity remains delta B plus its harmonic component. The warm-start
 * library Krylov solve retains the previous potential, then pressure is
 * p=rho(Pi-|u|^2/2), shifted to the requested mean.
 * \important This operator and documentation is fully AI-generated.
 */
class PressureReconstruction
{
    StructuredGrid& m_dec;
    VorticityDEC& m_forms;
    ScalarPoissonOperator m_poisson;
    PoissonSolver m_solver;
    similie::physics::fluid::IncompressibleNavierStokes const m_physics;
    Field m_vorticity, m_lap, m_phi, m_r, m_rhs, m_solution, m_pressure_acceleration, m_cochain,
            m_curl, m_form_diffusion, m_divergence;

public:
    // NVCC requires methods enclosing extended device lambdas to be public.
    void unprojected_rhs(
            Field const& velocity,
            Field const& result,
            Field const& supplied_vorticity = Field())
    {
        if (supplied_vorticity.data())
            Kokkos::deep_copy(m_vorticity, supplied_vorticity);
        else
            m_dec.reconstructed_derivative_one_form(velocity, m_vorticity);
        m_forms.reduce_one_form(velocity, m_cochain);
        m_forms.derivative_one_form(m_cochain, m_curl);
        m_forms.codifferential(m_curl, m_form_diffusion);
        m_forms.reconstruct_one_form(m_form_diffusion, m_lap);
        m_forms.negative_codifferential_one_form(m_cochain, m_divergence);
        m_dec.gradient(m_divergence, m_form_diffusion);
        KokkosBlas::axpby(1.0, m_form_diffusion, -1.0, m_lap);
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
    void recover_pressure(Field const& velocity, Field const& output, double mean_pressure)
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

    Field pressure;
    std::int64_t total_iterations = 0;
    PressureReconstruction(StructuredGrid& dec, VorticityDEC& forms)
        : m_dec(dec)
        , m_forms(forms)
        , m_poisson(dec)
        , m_solver(dec.n, 1, dec.h)
        , m_physics(1.0, Reference::viscosity)
        , m_vorticity("velocity_exterior_derivative", 3, dec.cells)
        , m_lap("velocity_laplacian", 3, dec.cells)
        , m_phi("bernoulli_potential", 1, dec.cells)
        , m_r("pressure_codifferential", 1, dec.cells)
        , m_rhs("pressure_poisson_rhs", dec.cells, 1)
        , m_solution("pressure_poisson_solution", dec.cells, 1)
        , m_pressure_acceleration("pressure_reconstruction_acceleration", 3, dec.cells)
        , m_cochain("pressure_velocity_cochain", 3, dec.cells)
        , m_curl("pressure_velocity_coboundary", 3, dec.cells)
        , m_form_diffusion("pressure_velocity_diffusion", 3, dec.cells)
        , m_divergence("pressure_velocity_divergence", 1, dec.cells)
        , pressure("pressure", 1, dec.cells)
    {
        Kokkos::deep_copy(pressure, 0.0);
    }

    void compute_pressure(
            Field const& velocity,
            Field const& output,
            double mean_pressure = 0.0,
            Field const& supplied_vorticity = Field(),
            Field const& acceleration = Field())
    {
        ProfilingRegion const region("pressure/reconstruction");
        unprojected_rhs(velocity, m_pressure_acceleration, supplied_vorticity);
        if (acceleration.data()) {
            Field const rate = m_pressure_acceleration;
            Kokkos::parallel_for(
                    "pressure_external_force",
                    m_dec.cells,
                    KOKKOS_LAMBDA(int i) {
                        for (int a = 0; a < 3; ++a)
                            rate(a, i) += acceleration(a, i);
                    });
        }
        m_dec.negative_reduced_codifferential(m_pressure_acceleration, m_r);
        Field const source = m_rhs, divergence = m_r;
        Kokkos::parallel_for(
                "pressure_poisson_rhs",
                m_dec.cells,
                KOKKOS_LAMBDA(int i) { source(i, 0) = -divergence(0, i); });
        similie::solvers::StrongFormulationSolverSettings const settings = m_solver.settings();
        similie::solvers::StrongFormulationSolverDiagnostics const diagnostics
                = profiled_poisson_solve(
                        "pressure",
                        Kokkos::DefaultExecutionSpace(),
                        m_poisson,
                        source,
                        m_solution,
                        settings);
        if (!std::isfinite(diagnostics.final_relative_residual)
            || diagnostics.final_relative_residual > 1e-10)
            throw std::runtime_error("Pressure reconstruction did not converge");
        total_iterations += diagnostics.iterations;
        Field const solution = m_solution, phi = m_phi;
        double sum;
        Kokkos::parallel_reduce(
                "bernoulli_potential_mean",
                m_dec.cells,
                KOKKOS_LAMBDA(int i, double& value) { value += solution(i, 0); },
                sum);
        double const mean = sum / m_dec.cells;
        Kokkos::parallel_for(
                "unpack_bernoulli_potential",
                m_dec.cells,
                KOKKOS_LAMBDA(int i) {
                    solution(i, 0) -= mean;
                    phi(0, i) = solution(i, 0);
                });
        recover_pressure(velocity, output, mean_pressure);
    }
};

/** Matrix-free native two-form Hodge Laplacian; uniform periodic metric makes
 * it symmetric in the cochain Euclidean inner product. Mean-zero right-hand
 * sides and explicit mean removal fix its three harmonic constant modes.
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
    void apply(Kokkos::DefaultExecutionSpace const& exec, Input const& source, Output const& target)
            const
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
 * Harmonic velocity is evolved separately when an external force has nonzero mean.
 * \important This operator and documentation is fully AI-generated.
 */
class IncompressibleSolver
{
    StructuredGrid& m_grid;
    VorticityDEC& m_dec;
    VorticityPoissonOperator m_poisson;
    PoissonSolver m_solver;
    PressureReconstruction m_pressure;
    similie::physics::fluid::IncompressibleNavierStokes const m_physics;
    Field m_potential, m_one, m_transport, m_transport_one, m_lap, m_poisson_rhs,
            m_poisson_solution, m_derivative_poisson_solution, m_check;

public:
    Field velocity, vorticity, vector_potential, pressure;
    std::int64_t total_iterations = 0;
    double reconstruction_seconds = 0, last_relative_residual = 0;
    double gauge_rms = 0, vorticity_divergence_rms = 0, curl_residual_rms = 0;
    std::array<double, 3> mean_velocity {}, mean_rate {};
    IncompressibleSolver(StructuredGrid& grid, VorticityDEC& forms)
        : m_grid(grid)
        , m_dec(forms)
        , m_poisson(m_dec, grid.cells)
        , m_solver(grid.n, 3, grid.h)
        , m_pressure(grid, m_dec)
        , m_physics(1, Reference::viscosity)
        , m_potential("dual_vector_potential", 3, grid.cells)
        , m_one("velocity_cochain", 3, grid.cells)
        , m_transport("vorticity_transport", 3, grid.cells)
        , m_transport_one("transport_cochain", 3, grid.cells)
        , m_lap("vorticity_hodge_laplacian", 3, grid.cells)
        , m_poisson_rhs("vector_poisson_rhs", 3 * grid.cells, 1)
        , m_poisson_solution("vector_poisson_solution", 3 * grid.cells, 1)
        , m_derivative_poisson_solution("vector_poisson_derivative_solution", 3 * grid.cells, 1)
        , m_check("curl_check", 3, grid.cells)
        , velocity("reconstructed_velocity", 3, grid.cells)
        , vorticity("reconstructed_vorticity", 3, grid.cells)
        , vector_potential("vector_potential", 3, grid.cells)
        , pressure(m_pressure.pressure)
    {
    }

    void initialize(Field const& state)
    {
        // The analytic vector potential A_z=sin(x) sin(y) cos(z) gives the
        // Taylor-Green velocity. Reduce *A, then initialize U=delta(*A) and W=dU.
        // This preserves both discrete constraints without shifted samples or
        // mesh-dependent amplitude corrections.
        int const n = m_grid.n;
        double const h = m_grid.h;
        Field const potential = m_transport;
        Kokkos::parallel_for(
                "initial_velocity_potential",
                m_grid.cells,
                KOKKOS_LAMBDA(int i) {
                    double const x = -std::numbers::pi + (i / (n * n)) * h;
                    double const y = -std::numbers::pi + ((i / n) % n) * h;
                    double const z = -std::numbers::pi + (i % n) * h;
                    potential(0, i) = 0;
                    potential(1, i) = 0;
                    potential(2, i) = sin(x) * sin(y) * cos(z);
                });
        m_dec.reduce_two_form(potential, m_potential);
        m_dec.codifferential(m_potential, m_one);
        m_dec.derivative_one_form(m_one, state);
    }
    void reconstruct_velocity(Field const& state, Field const& output, bool time_derivative = false)
    {
        ProfilingRegion const region(
                time_derivative ? "velocity-rate/reconstruction" : "velocity/reconstruction");
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
        similie::solvers::StrongFormulationSolverSettings const settings = m_solver.settings();
        // Monitor solves of dW/dt must not overwrite the state potential history.
        Field const solution = time_derivative ? m_derivative_poisson_solution : m_poisson_solution;
        similie::solvers::StrongFormulationSolverDiagnostics const diagnostics
                = profiled_poisson_solve(
                        time_derivative ? "vector-potential-rate" : "vector-potential",
                        Kokkos::DefaultExecutionSpace(),
                        m_poisson,
                        rhs,
                        solution,
                        settings);
        if (!std::isfinite(diagnostics.final_relative_residual)
            || diagnostics.final_relative_residual > 10 * settings.relative_tolerance)
            throw std::runtime_error("Vector-potential reconstruction did not converge");
        total_iterations += diagnostics.iterations;
        last_relative_residual = diagnostics.final_relative_residual;
        Field const potential = m_potential;
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
                    KOKKOS_LAMBDA(int i) {
                        potential(c, i) -= mean;
                        solution(c * cells + i, 0) = potential(c, i);
                    });
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
        m_dec.reconstruct_one_form(m_one, output);
        std::array<double, 3> const harmonic = time_derivative ? mean_rate : mean_velocity;
        Kokkos::parallel_for(
                "velocity_harmonic",
                cells,
                KOKKOS_LAMBDA(int i) {
                    for (int a = 0; a < 3; ++a)
                        output(a, i) += harmonic[a];
                });
        Kokkos::fence("vector_potential_reconstruction");
        reconstruction_seconds
                += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    }
    void rhs(
            Field const& state,
            Field const& result,
            bool reconstruct_pressure = false,
            Field const& acceleration = Field(),
            bool velocity_reconstructed = false)
    {
        ProfilingRegion const region("equations/incompressible-vorticity");
        if (!velocity_reconstructed)
            reconstruct_velocity(state, velocity);
        m_dec.reconstruct_two_form(state, vorticity);
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
                        transport(a, i)
                                = values[a] + (acceleration.data() ? acceleration(a, i) : 0);
                });
        m_dec.reduce_one_form(m_transport, m_transport_one);
        m_dec.derivative_one_form(m_transport_one, result);
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
            m_pressure.compute_pressure(velocity, pressure, 0.0, vorticity, acceleration);
            monitor_constraints(state);
        }
    }
    void compute_pressure(
            Field const& state,
            Field const& output,
            double mean_pressure = 0,
            Field const& acceleration = Field())
    {
        reconstruct_velocity(state, velocity);
        m_dec.reconstruct_two_form(state, vorticity);
        m_pressure.compute_pressure(velocity, output, mean_pressure, vorticity, acceleration);
    }
    void divergence(Field const& output)
    {
        m_dec.negative_codifferential_one_form(m_one, output);
    }
    void monitor_constraints(Field const& state)
    {
        vorticity_divergence_rms = m_dec.closure_rms(state);
        gauge_rms = m_dec.closure_rms(m_potential);
        m_dec.reconstruct_two_form(m_potential, vector_potential);
        m_dec.derivative_one_form(m_one, m_check);
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
