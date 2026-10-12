// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#pragma once

#include <array>
#include <cstddef>

#include <Kokkos_Core.hpp>

namespace similie::physics::fluid {

/** Compressible mass, momentum, and total-energy conservation with Fourier closure.
 *
 * dt rho = -div(rho u)
 * dt (rho u) = -div(rho u tensor u + p I - tau)
 * dt (rho E) = -div((rho E+p)u - tau.u + q)
 * p=(gamma-1)(rho E-rho |u|^2/2), T=p/(rho R)
 * tau=mu(grad u+grad u^T-2/3 div u I), q=-kappa grad T
 * kappa=mu cp/Pr, cp=gamma R/(gamma-1).
 *
 * This pointwise physics supplies constitutive fluxes and conservation laws.
 * The caller evaluates divergence, gradient, and Hodge operators in exterior;
 * no mesh, time scheme, benchmark constants, or solver selector belongs here.
 * Stokes' zero bulk-viscosity assumption is three-dimensional.
 * \important This operator and documentation is fully AI-generated.
 */
class NavierStokesFourier
{
public:
    struct ConservativeVariables
    {
        double density;
        std::array<double, 3> momentum;
        double total_energy;
    };
    struct PrimitiveVariables
    {
        double density;
        std::array<double, 3> velocity;
        double pressure;
        double temperature;
    };
    struct Flux
    {
        double mass;
        std::array<double, 3> momentum;
        double total_energy;
    };
    using VelocityGradient = std::array<std::array<double, 3>, 3>;

    double const gamma;
    double const prandtl;
    double const viscosity;
    double const gas_constant;

    KOKKOS_FUNCTION explicit NavierStokesFourier(
            double heat_capacity_ratio,
            double prandtl_number,
            double dynamic_viscosity,
            double specific_gas_constant)
        : gamma(heat_capacity_ratio)
        , prandtl(prandtl_number)
        , viscosity(dynamic_viscosity)
        , gas_constant(specific_gas_constant)
    {
    }

    KOKKOS_FUNCTION double heat_capacity_at_constant_volume() const
    {
        return gas_constant / (gamma - 1);
    }
    KOKKOS_FUNCTION double heat_capacity_at_constant_pressure() const
    {
        return gamma * gas_constant / (gamma - 1);
    }
    KOKKOS_FUNCTION double conductivity() const
    {
        return viscosity * heat_capacity_at_constant_pressure() / prandtl;
    }
    KOKKOS_FUNCTION double pressure(ConservativeVariables const& state) const
    {
        double kinetic = 0;
        for (int axis = 0; axis < 3; ++axis)
            kinetic += state.momentum[axis] * state.momentum[axis] / (2 * state.density);
        return (gamma - 1) * (state.total_energy - kinetic);
    }
    KOKKOS_FUNCTION PrimitiveVariables primitives(ConservativeVariables const& state) const
    {
        PrimitiveVariables result {state.density, {}, pressure(state), 0};
        for (int axis = 0; axis < 3; ++axis)
            result.velocity[axis] = state.momentum[axis] / state.density;
        result.temperature = result.pressure / (state.density * gas_constant);
        return result;
    }
    KOKKOS_FUNCTION VelocityGradient stress(VelocityGradient const& gradient) const
    {
        double const divergence = gradient[0][0] + gradient[1][1] + gradient[2][2];
        VelocityGradient result {};
        for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b)
                result[a][b]
                        = viscosity
                          * (gradient[a][b] + gradient[b][a] - (a == b ? 2 * divergence / 3 : 0));
        return result;
    }
    KOKKOS_FUNCTION std::array<double, 3> heat_flux(
            std::array<double, 3> const& temperature_gradient) const
    {
        std::array<double, 3> result {};
        for (int axis = 0; axis < 3; ++axis)
            result[axis] = -conductivity() * temperature_gradient[axis];
        return result;
    }
    KOKKOS_FUNCTION Flux
    flux(ConservativeVariables const& state,
         PrimitiveVariables const& primitive,
         VelocityGradient const& velocity_gradient,
         std::array<double, 3> const& temperature_gradient,
         int normal) const
    {
        VelocityGradient const tau = stress(velocity_gradient);
        std::array<double, 3> const q = heat_flux(temperature_gradient);
        Flux
                result {state.density * primitive.velocity[normal],
                        {},
                        (state.total_energy + primitive.pressure) * primitive.velocity[normal]
                                + q[normal]};
        for (int axis = 0; axis < 3; ++axis) {
            result.momentum[axis] = state.momentum[axis] * primitive.velocity[normal]
                                    + (axis == normal ? primitive.pressure : 0) - tau[axis][normal];
            result.total_energy -= tau[axis][normal] * primitive.velocity[axis];
        }
        return result;
    }
    /** Evaluate the three conservation laws after exterior divergence evaluation.
     * \important This operator and documentation is fully AI-generated.
     */
    KOKKOS_FUNCTION ConservativeVariables operator()(Flux const& divergence) const
    {
        ConservativeVariables result {-divergence.mass, {}, -divergence.total_energy};
        for (int axis = 0; axis < 3; ++axis)
            result.momentum[axis] = -divergence.momentum[axis];
        return result;
    }
};
} // namespace similie::physics::fluid
