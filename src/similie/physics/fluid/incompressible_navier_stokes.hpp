// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#pragma once

#include <array>

#include <similie/physics/concepts.hpp>

#include <Kokkos_Core.hpp>

namespace similie::physics::fluid {
/** Constant-density incompressible Navier-Stokes in vorticity form.
 * dt omega = curl(u cross omega) + (mu/rho) Delta omega,
 * -Delta A=omega, div A=0, u=curl A. The vector potential has zero mean
 * on a periodic domain; any harmonic mean velocity must be supplied separately.
 * For pressure recovery, dt u = u cross omega - grad(Pi) + (mu/rho) Delta u,
 * Pi=p/rho+|u|^2/2. Gradients, curl, divergence, and the physical (negative
 * semidefinite) Laplacian are evaluated with the exterior module by the caller.
 * The pressure solve is a numerical constraint solve, not potential-flow physics.
 * \important This operator and documentation is fully AI-generated.
 */
class IncompressibleNavierStokes
{
public:
    static constexpr std::size_t N = 3;
    static constexpr bool IS_LINEAR = false;
    struct State
    {
        std::array<double, N> vorticity {};
    };
    double const density;
    double const viscosity;
    KOKKOS_FUNCTION explicit IncompressibleNavierStokes(
            double constant_density,
            double dynamic_viscosity)
        : density(constant_density)
        , viscosity(dynamic_viscosity)
    {
    }

    KOKKOS_FUNCTION double operator()(double transport_curl, double vorticity_laplacian) const
    {
        return transport_curl + viscosity / density * vorticity_laplacian;
    }
    KOKKOS_FUNCTION std::array<double, 3> transport(
            std::array<double, 3> const& velocity,
            std::array<double, 3> const& vorticity) const
    {
        std::array<double, 3> result {};
        for (int a = 0; a < 3; ++a) {
            int const b = (a + 1) % 3, c = (a + 2) % 3;
            result[a] = velocity[b] * vorticity[c] - velocity[c] * vorticity[b];
        }
        return result;
    }
    KOKKOS_FUNCTION std::array<double, 3> momentum_rate(
            std::array<double, 3> const& velocity,
            std::array<double, 3> const& vorticity,
            std::array<double, 3> const& bernoulli_gradient,
            std::array<double, 3> const& velocity_laplacian) const
    {
        std::array<double, 3> result = transport(velocity, vorticity);
        for (int a = 0; a < 3; ++a) {
            result[a] += -bernoulli_gradient[a] + viscosity / density * velocity_laplacian[a];
        }
        return result;
    }
    KOKKOS_FUNCTION double incompressibility(double velocity_divergence) const
    {
        return velocity_divergence;
    }
    KOKKOS_FUNCTION double pressure(double bernoulli, std::array<double, 3> const& velocity) const
    {
        double kinetic = 0;
        for (double value : velocity)
            kinetic += 0.5 * value * value;
        return density * (bernoulli - kinetic);
    }
};

static_assert(PhysicsConcept<IncompressibleNavierStokes>);
} // namespace similie::physics::fluid
