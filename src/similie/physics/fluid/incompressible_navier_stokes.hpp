// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#pragma once

#include <array>

#include <Kokkos_Core.hpp>

namespace similie::physics::fluid {
/** Constant-density incompressible Navier-Stokes in rotational form.
 * dt u = u cross omega - grad(Pi) + (mu/rho) Delta u, div u=0,
 * Pi=p/rho+|u|^2/2. Gradients, curl, divergence, and the physical (negative
 * semidefinite) Laplacian are evaluated with the exterior module by the caller.
 * The pressure solve is a numerical constraint solve, not potential-flow physics.
 * \important This operator and documentation is fully AI-generated.
 */
class IncompressibleNavierStokes
{
public:
    double const density;
    double const viscosity;
    KOKKOS_FUNCTION explicit IncompressibleNavierStokes(
            double constant_density,
            double dynamic_viscosity)
        : density(constant_density)
        , viscosity(dynamic_viscosity)
    {
    }

    KOKKOS_FUNCTION std::array<double, 3> operator()(
            std::array<double, 3> const& velocity,
            std::array<double, 3> const& vorticity,
            std::array<double, 3> const& bernoulli_gradient,
            std::array<double, 3> const& velocity_laplacian) const
    {
        std::array<double, 3> result {};
        for (int a = 0; a < 3; ++a) {
            int const b = (a + 1) % 3, c = (a + 2) % 3;
            result[a] = velocity[b] * vorticity[c] - velocity[c] * vorticity[b]
                        - bernoulli_gradient[a] + viscosity / density * velocity_laplacian[a];
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
} // namespace similie::physics::fluid
