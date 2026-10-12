// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#pragma once
#include <array>
#include <cstdint>
#include <stdexcept>

#include <similie/physics/fluid/incompressible_navier_stokes.hpp>

#include "parameters.hpp"

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
    similie::physics::fluid::NavierStokesFourier const m_physics;
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
        similie::physics::fluid::NavierStokesFourier const physics = m_physics;
        Kokkos::parallel_for(
                "perfect_gas_closure",
                m_dec.cells,
                KOKKOS_LAMBDA(int i) {
                    similie::physics::fluid::NavierStokesFourier::PrimitiveVariables const values
                            = physics.primitives(
                                    {state(0, i),
                                     {state(1, i), state(2, i), state(3, i)},
                                     state(4, i)});
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
        similie::physics::fluid::NavierStokesFourier const physics = m_physics;
        for (int axis = 0; axis < 3; ++axis) {
            for (int c = 1; c < 5; ++c)
                m_dec.forward_derivative(primitive, c, normal, c - 1, axis);
            Kokkos::parallel_for(
                    "navier_stokes_face_flux",
                    m_dec.cells,
                    KOKKOS_LAMBDA(int i) {
                        int const j = StructuredGrid::neighbor(i, axis, 1, n);
                        double const pl
                                = primitive(0, i) * Reference::gas_constant * primitive(4, i),
                                pr = primitive(0, j) * Reference::gas_constant * primitive(4, j);
                        double const ul = primitive(axis + 1, i), ur = primitive(axis + 1, j);
                        double const mass_flux
                                = 0.25 * (primitive(0, i) + primitive(0, j)) * (ul + ur);
                        double const pressure = 0.5 * (pl + pr);
                        double kinetic_cross = 0;
                        for (int a = 0; a < 3; ++a)
                            kinetic_cross += 0.5 * primitive(a + 1, i) * primitive(a + 1, j);
                        double const internal = 0.5 * Reference::gas_constant
                                                * (primitive(4, i) + primitive(4, j))
                                                / (Reference::gamma - 1);
                        similie::physics::fluid::NavierStokesFourier::VelocityGradient strain {};
                        for (int a = 0; a < 3; ++a)
                            for (int b = 0; b < 3; ++b)
                                strain[a][b] = b == axis ? normal(a, i)
                                                         : 0.5
                                                                   * (gradient(3 * a + b, i)
                                                                      + gradient(3 * a + b, j));
                        similie::physics::fluid::NavierStokesFourier::VelocityGradient const tau
                                = physics.stress(strain);
                        std::array<double, 3> temperature_gradient {};
                        temperature_gradient[axis] = normal(3, i);
                        std::array<double, 3> const heat_flux
                                = physics.heat_flux(temperature_gradient);
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
                                     + 0.5 * pressure * (ul + ur) - work + heat_flux[axis];
                    });
            for (int c = 0; c < 5; ++c)
                m_dec.flux_divergence(flux, c, divergence, c, axis);
            Kokkos::parallel_for(
                    "navier_stokes_conservation_laws",
                    m_dec.cells,
                    KOKKOS_LAMBDA(int i) {
                        similie::physics::fluid::NavierStokesFourier::ConservativeVariables const
                                rate
                                = physics(
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

/** Rotational incompressible Navier-Stokes, with a centered DEC gradient and
 * its negative adjoint divergence. Orthogonal periodic projection solves
 * -D G phi=-div u by matrix-free CG. The mean and even-grid checkerboard
 * null modes are untouched; divergence is in the range of this operator.
 * Viscosity uses the exterior staged Hodge Laplacian with its diffusion sign.
 * Project the acceleration at every RK stage;
 * the projection potential is Bernoulli pressure p+|u|^2/2.
 * \important This operator and documentation is fully AI-generated.
 */
class IncompressibleSolver
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

    void unprojected_rhs(Field velocity, Field result)
    {
        m_dec.curl(velocity, m_vorticity);
        for (int c = 0; c < 3; ++c)
            m_dec.laplacian(velocity, c, m_lap, c);
        Field const rotation = m_vorticity, lap = m_lap;
        similie::physics::fluid::IncompressibleNavierStokes const physics = m_physics;
        Kokkos::parallel_for(
                "incompressible_momentum",
                m_dec.cells,
                KOKKOS_LAMBDA(int i) {
                    std::array<double, 3> const rate = physics(
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
    explicit IncompressibleSolver(StructuredGrid& dec)
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
    void rhs(Field velocity, Field result, bool reconstruct_pressure = false)
    {
        unprojected_rhs(velocity, result);
        project(result);
        if (reconstruct_pressure)
            recover_pressure(velocity, pressure, 0.0);
    }
    /** Reconstruct periodic pressure for a supplied divergence-free velocity.
     * Solve -D G Pi=-D(u cross omega+nu Delta u), then recover
     * p=rho(Pi-|u|^2/2) and shift it to the requested volume mean.
     * Zero-start CG fixes unresolved checkerboard modes of Pi on even grids.
     * The supplied velocity is unchanged; output must be a separate scalar field.
     * This can be called independently of rhs and time integration.
     * \important This operator and documentation is fully AI-generated.
     */
    void compute_pressure(Field velocity, Field output, double mean_pressure = 0.0)
    {
        unprojected_rhs(velocity, m_pressure_acceleration);
        project(m_pressure_acceleration);
        recover_pressure(velocity, output, mean_pressure);
    }
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

} // namespace taylor_green_vortex
