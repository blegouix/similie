// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#pragma once

#include <array>

#include <similie/physics/combine_physics.hpp>
#include <similie/physics/coupling/electric_coupling_physics.hpp>
#include <similie/physics/coupling/magnetic_coupling_physics.hpp>

namespace similie::physics {

enum class EnergyConvention { Fluid, FluidAndField };

/** Assemble a fluid model and full Maxwell dynamics using shared exchange laws.
 * The state contains the two physical sectors; the stateless interactions own
 * no new degrees of freedom. get<0>() is the fluid, get<1>() is the field model,
 * and get<2/3>() are the generated electric/magnetic coupling objects.
 *
 * The caller computes spatial derivatives with exterior operators. In
 * particular, it supplies one reduced current to both the Maxwell update and
 * the force/power evaluation, and uses that current for the charge divergence.
 * Constitutive current laws and material parameters belong to the field model.
 *
 * For compressible flow, Fluid stores thermal+kinetic energy and adds J.E to
 * its rate. FluidAndField stores that energy plus W(D,B), subtracts W when
 * evaluating fluid primitives, and adds E cross H to the energy flux instead.
 * Its rate receives no internal exchange source. Joule heat must not be added
 * again. The field_energy_source diagnostic is already represented by -J in
 * the displacement update; it is not an additional field evolution equation.
 *
 * For incompressible flow, transport includes force/density before taking its
 * curl. The caller supplies that complete curl to the vorticity update; no
 * thermal or total-energy variable is evolved in this formulation.
 * \important This operator and documentation is fully AI-generated.
 */
template <
        PhysicsConcept FluidPhysics,
        PhysicsConcept ElectromagneticPhysics,
        EnergyConvention Energy = EnergyConvention::Fluid>
class CoupledFluidElectromagnetism
    : public CombinePhysics<
              FluidPhysics,
              ElectromagneticPhysics,
              coupling::ElectricCouplingPhysics,
              coupling::MagneticCouplingPhysics>
{
public:
    using Vector = std::array<double, 3>;
    static constexpr std::size_t N = FluidPhysics::N + ElectromagneticPhysics::N;
    static constexpr bool IS_LINEAR = false;
    struct State
    {
        typename FluidPhysics::State fluid;
        typename ElectromagneticPhysics::State electromagnetic;
    };

    KOKKOS_FUNCTION CoupledFluidElectromagnetism(
            FluidPhysics const& fluid,
            ElectromagneticPhysics const& electromagnetic)
        : CombinePhysics<
                  FluidPhysics,
                  ElectromagneticPhysics,
                  coupling::ElectricCouplingPhysics,
                  coupling::MagneticCouplingPhysics>(
                  fluid,
                  electromagnetic,
                  coupling::ElectricCouplingPhysics(),
                  coupling::MagneticCouplingPhysics())
    {
    }

    [[nodiscard]] KOKKOS_FUNCTION coupling::ElectromagneticExchange coupling_sources(
            State const& state,
            Vector const& total_current) const
    {
        coupling::ElectromagneticCouplingState const
                shared {state.electromagnetic.charge_density,
                        this->template get<1>().electric_field(state.electromagnetic.displacement),
                        state.electromagnetic.induction,
                        total_current};
        coupling::ElectromagneticExchange result = this->template get<2>()(shared);
        result += this->template get<3>()(shared);
        return result;
    }

    [[nodiscard]] KOKKOS_FUNCTION Vector current(State const& state, Vector const& velocity) const
    {
        return this->template get<1>().current(
                state.electromagnetic.charge_density,
                velocity,
                state.electromagnetic.displacement,
                state.electromagnetic.induction);
    }

    [[nodiscard]] KOKKOS_FUNCTION typename FluidPhysics::State fluid_state(State const& state) const
    {
        typename FluidPhysics::State result = state.fluid;
        if constexpr (Energy == EnergyConvention::FluidAndField) {
            result.total_energy -= this->template get<1>().hamiltonian(
                    state.electromagnetic.displacement,
                    state.electromagnetic.induction);
        }
        return result;
    }

    [[nodiscard]] KOKKOS_FUNCTION Vector current(State const& state) const
    {
        return current(state, this->template get<0>().primitives(fluid_state(state)).velocity);
    }

    template <class Fluid = FluidPhysics>
    [[nodiscard]] KOKKOS_FUNCTION typename Fluid::Flux flux(
            State const& state,
            typename Fluid::VelocityGradient const& velocity_gradient,
            Vector const& temperature_gradient,
            int normal) const
    {
        KOKKOS_ASSERT(normal >= 0 && normal < 3);
        typename Fluid::State const hydro = fluid_state(state);
        typename Fluid::PrimitiveVariables const primitive
                = this->template get<0>().primitives(hydro);
        typename Fluid::Flux result
                = this->template get<0>()
                          .flux(hydro, primitive, velocity_gradient, temperature_gradient, normal);
        if constexpr (Energy == EnergyConvention::FluidAndField) {
            Vector const poynting = this->template get<1>().poynting_flux(
                    state.electromagnetic.displacement,
                    state.electromagnetic.induction);
            result.total_energy += poynting[normal];
        }
        return result;
    }

    template <class FluidFlux>
    [[nodiscard]] KOKKOS_FUNCTION State operator()(
            State const& state,
            FluidFlux const& fluid_flux_divergence,
            double current_divergence,
            Vector const& curl_magnetic_field,
            Vector const& curl_electric_field,
            Vector const& total_current) const
    {
        typename FluidPhysics::State rate = this->template get<0>()(fluid_flux_divergence);
        coupling::ElectromagneticExchange const exchange = coupling_sources(state, total_current);
        for (int a = 0; a < 3; ++a)
            rate.momentum[a] += exchange.momentum_source[a];
        if constexpr (Energy == EnergyConvention::Fluid)
            rate.total_energy += exchange.fluid_energy_source;
        return {rate,
                this->template get<1>()(
                        current_divergence,
                        curl_magnetic_field,
                        curl_electric_field,
                        total_current)};
    }

    [[nodiscard]] KOKKOS_FUNCTION double combined_energy(State const& state) const
    {
        if constexpr (Energy == EnergyConvention::FluidAndField)
            return state.fluid.total_energy;
        else
            return state.fluid.total_energy
                   + this->template get<1>().hamiltonian(
                           state.electromagnetic.displacement,
                           state.electromagnetic.induction);
    }

    [[nodiscard]] KOKKOS_FUNCTION Vector
    transport(State const& state, Vector const& velocity, Vector const& total_current) const
    {
        Vector result = this->template get<0>().transport(velocity, state.fluid.vorticity);
        coupling::ElectromagneticExchange const exchange = coupling_sources(state, total_current);
        for (int a = 0; a < 3; ++a)
            result[a] += exchange.momentum_source[a] / this->template get<0>().density;
        return result;
    }

    [[nodiscard]] KOKKOS_FUNCTION Vector momentum_rate(
            State const& state,
            Vector const& velocity,
            Vector const& bernoulli_gradient,
            Vector const& velocity_laplacian,
            Vector const& total_current) const
    {
        Vector result = this->template get<0>().momentum_rate(
                velocity,
                state.fluid.vorticity,
                bernoulli_gradient,
                velocity_laplacian);
        coupling::ElectromagneticExchange const exchange = coupling_sources(state, total_current);
        for (int a = 0; a < 3; ++a)
            result[a] += exchange.momentum_source[a] / this->template get<0>().density;
        return result;
    }

    [[nodiscard]] KOKKOS_FUNCTION State operator()(
            Vector const& transport_curl,
            Vector const& vorticity_laplacian,
            double current_divergence,
            Vector const& curl_magnetic_field,
            Vector const& curl_electric_field,
            Vector const& total_current) const
    {
        typename FluidPhysics::State rate {};
        for (int a = 0; a < 3; ++a)
            rate.vorticity[a] = this->template get<0>()(transport_curl[a], vorticity_laplacian[a]);
        return {rate,
                this->template get<1>()(
                        current_divergence,
                        curl_magnetic_field,
                        curl_electric_field,
                        total_current)};
    }
};

} // namespace similie::physics
