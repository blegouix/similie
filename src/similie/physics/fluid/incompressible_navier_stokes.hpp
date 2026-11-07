// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#pragma once

#include <array>

#include <similie/exterior/form.hpp>
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
    using Scalar = sil::exterior::Form<double>;
    static constexpr std::size_t N = 3;
    static constexpr bool IS_LINEAR = false;
    struct State
    {
        sil::exterior::Form<std::array<double, N>> vorticity {};
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

    KOKKOS_FUNCTION sil::exterior::Form<std::array<double, N>> operator()(
            sil::exterior::Form<std::array<double, N>> const& transport_curl,
            sil::exterior::Form<std::array<double, N>> const& vorticity_laplacian) const
    {
        sil::exterior::Form<std::array<double, N>> result {};
        for (std::size_t component = 0; component < N; ++component)
            result[component] = transport_curl[component]
                                + viscosity / density * vorticity_laplacian[component];
        return result;
    }
    KOKKOS_FUNCTION sil::exterior::Form<std::array<double, 3>> transport(
            sil::exterior::Form<std::array<double, 3>> const& velocity,
            sil::exterior::Form<std::array<double, 3>> const& vorticity) const
    {
        sil::exterior::Form<std::array<double, 3>> result {};
        for (int a = 0; a < 3; ++a) {
            int const b = (a + 1) % 3, c = (a + 2) % 3;
            result[a] = velocity[b] * vorticity[c] - velocity[c] * vorticity[b];
        }
        return result;
    }
    KOKKOS_FUNCTION sil::exterior::Form<std::array<double, 3>> momentum_rate(
            sil::exterior::Form<std::array<double, 3>> const& velocity,
            sil::exterior::Form<std::array<double, 3>> const& vorticity,
            sil::exterior::Form<std::array<double, 3>> const& bernoulli_gradient,
            sil::exterior::Form<std::array<double, 3>> const& velocity_laplacian) const
    {
        sil::exterior::Form<std::array<double, 3>> result = transport(velocity, vorticity);
        for (int a = 0; a < 3; ++a) {
            result[a] += -bernoulli_gradient[a] + viscosity / density * velocity_laplacian[a];
        }
        return result;
    }
    KOKKOS_FUNCTION Scalar incompressibility(Scalar const& velocity_divergence) const
    {
        return Scalar(velocity_divergence);
    }
    KOKKOS_FUNCTION Scalar pressure(
            Scalar const& bernoulli,
            sil::exterior::Form<std::array<double, 3>> const& velocity) const
    {
        double kinetic = 0;
        for (double value : velocity)
            kinetic += 0.5 * value * value;
        return Scalar(density * (bernoulli - kinetic));
    }
};

static_assert(PhysicsConcept<IncompressibleNavierStokes>);
} // namespace similie::physics::fluid
