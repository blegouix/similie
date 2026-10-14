// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#pragma once

#include <array>

#include <similie/physics/combine_physics.hpp>
#include <similie/physics/electrodynamics/electrodynamics.hpp>
#include <similie/physics/electrodynamics/resistive_electrodynamics.hpp>
#include <similie/physics/fluid/incompressible_navier_stokes.hpp>
#include <similie/physics/fluid/navier_stokes_fourier.hpp>

namespace similie::physics::ehd {
/** Electric-only charged-fluid vorticity: curl(u cross omega+charge E/rho).
 * Charge is evolved; displacement is supplied by the elliptic Gauss solve.
 * The physical Laplacian is negative semidefinite. No thermal state is evolved.
 * \important This operator and documentation is fully AI-generated.
 */
template <PhysicsConcept ElectricPhysics>
class IncompressibleEHDModel
    : public CombinePhysics<fluid::IncompressibleNavierStokes, ElectricPhysics>
{
public:
    using Vector = std::array<double, 3>;
    static constexpr std::size_t N = 4;
    static constexpr bool IS_LINEAR = false;
    struct State
    {
        Vector vorticity;
        double charge_density;
    };
    static_assert(PhysicsConcept<IncompressibleEHDModel>);
    KOKKOS_FUNCTION IncompressibleEHDModel(
            fluid::IncompressibleNavierStokes fluid_physics,
            ElectricPhysics electric_physics)
        : CombinePhysics<
                  fluid::IncompressibleNavierStokes,
                  ElectricPhysics>(fluid_physics, electric_physics)
    {
    }
    KOKKOS_FUNCTION Vector
    current(State const& state, Vector const& velocity, Vector const& displacement) const
    {
        return this->template get<1>().current(state.charge_density, velocity, displacement);
    }
    KOKKOS_FUNCTION Vector
    transport(State const& state, Vector const& velocity, Vector const& displacement) const
    {
        Vector result = this->template get<0>().transport(velocity, state.vorticity);
        Vector const force
                = this->template get<1>().electric_force(state.charge_density, displacement);
        for (int a = 0; a < 3; ++a)
            result[a] += force[a] / this->template get<0>().density;
        return result;
    }
    KOKKOS_FUNCTION State operator()(
            Vector const& transport_curl,
            Vector const& vorticity_laplacian,
            double current_divergence) const
    {
        State result {{}, this->template get<1>()(current_divergence)};
        for (int a = 0; a < 3; ++a)
            result.vorticity[a]
                    = this->template get<0>()(transport_curl[a], vorticity_laplacian[a]);
        return result;
    }
    KOKKOS_FUNCTION Vector momentum_rate(
            State const& state,
            Vector const& velocity,
            Vector const& bernoulli_gradient,
            Vector const& velocity_laplacian,
            Vector const& displacement) const
    {
        Vector result = this->template get<0>().momentum_rate(
                velocity,
                state.vorticity,
                bernoulli_gradient,
                velocity_laplacian);
        Vector const force
                = this->template get<1>().electric_force(state.charge_density, displacement);
        for (int a = 0; a < 3; ++a)
            result[a] += force[a] / this->template get<0>().density;
        return result;
    }
    KOKKOS_FUNCTION double pressure(double bernoulli, Vector const& velocity) const
    {
        return this->template get<0>().pressure(bernoulli, velocity);
    }
    KOKKOS_FUNCTION double incompressibility(double divergence) const
    {
        return this->template get<0>().incompressibility(divergence);
    }
    KOKKOS_FUNCTION double ohmic_heating(Vector const& displacement) const
    {
        return this->template get<1>().ohmic_heating(displacement);
    }
};

/** Electric-only compressible EHD. total_energy is FLUID energy (thermal+kinetic),
 * not electric field energy. Charge and fluid evolve explicitly; Gauss's law
 * reconstructs D separately. The fluid energy source is J.E, including Joule
 * heat once. Electric energy is monitored separately through its Hamiltonian.
 * \important This operator and documentation is fully AI-generated.
 */
template <class Thermodynamics, PhysicsConcept ElectricPhysics>
class CompressibleEHDModel
    : public CombinePhysics<fluid::NavierStokesFourier<Thermodynamics>, ElectricPhysics>
{
public:
    using FluidPhysics = fluid::NavierStokesFourier<Thermodynamics>;
    using Vector = std::array<double, 3>;
    static constexpr std::size_t N = 6;
    static constexpr bool IS_LINEAR = false;
    struct State
    {
        double density;
        Vector momentum;
        double total_energy;
        double charge_density;
    };
    static_assert(PhysicsConcept<CompressibleEHDModel>);
    struct Flux
    {
        typename FluidPhysics::Flux fluid;
        Vector electric_current;
    };
    KOKKOS_FUNCTION CompressibleEHDModel(
            FluidPhysics fluid_physics,
            ElectricPhysics electric_physics)
        : CombinePhysics<FluidPhysics, ElectricPhysics>(fluid_physics, electric_physics)
    {
    }
    KOKKOS_FUNCTION typename FluidPhysics::State fluid_state(State const& state) const
    {
        return {state.density, state.momentum, state.total_energy};
    }
    KOKKOS_FUNCTION typename FluidPhysics::PrimitiveVariables primitives(State const& state) const
    {
        return this->template get<0>().primitives(fluid_state(state));
    }
    KOKKOS_FUNCTION double pressure(State const& state) const
    {
        return this->template get<0>().pressure(fluid_state(state));
    }
    KOKKOS_FUNCTION Vector current(State const& state, Vector const& displacement) const
    {
        return this->template get<1>()
                .current(state.charge_density, primitives(state).velocity, displacement);
    }
    KOKKOS_FUNCTION Flux
    flux(State const& state,
         Vector const& displacement,
         typename FluidPhysics::VelocityGradient const& velocity_gradient,
         Vector const& temperature_gradient,
         int normal) const
    {
        KOKKOS_ASSERT(normal >= 0 && normal < 3);
        typename FluidPhysics::State const hydro = fluid_state(state);
        typename FluidPhysics::PrimitiveVariables const primitive
                = this->template get<0>().primitives(hydro);
        return {this->template get<0>()
                        .flux(hydro, primitive, velocity_gradient, temperature_gradient, normal),
                this->template get<1>()
                        .current(state.charge_density, primitive.velocity, displacement)};
    }
    KOKKOS_FUNCTION State operator()(
            State const& state,
            Vector const& displacement,
            typename FluidPhysics::Flux const& fluid_flux_divergence,
            double current_divergence,
            Vector const& total_current) const
    {
        typename FluidPhysics::State rate = this->template get<0>()(fluid_flux_divergence);
        Vector const force
                = this->template get<1>().electric_force(state.charge_density, displacement);
        for (int a = 0; a < 3; ++a)
            rate.momentum[a] += force[a];
        rate.total_energy
                += this->template get<1>().fluid_energy_source(displacement, total_current);
        return {rate.density,
                rate.momentum,
                rate.total_energy,
                this->template get<1>()(current_divergence)};
    }
    KOKKOS_FUNCTION double combined_energy(State const& state, Vector const& displacement) const
    {
        return state.total_energy + this->template get<1>().hamiltonian(displacement);
    }
    KOKKOS_FUNCTION double ohmic_heating(Vector const& displacement) const
    {
        return this->template get<1>().ohmic_heating(displacement);
    }
};

using IdealIncompressibleEHD = IncompressibleEHDModel<electrodynamics::Electrodynamics>;
template <class Thermodynamics>
using IdealCompressibleEHD = CompressibleEHDModel<Thermodynamics, electrodynamics::Electrodynamics>;
using ResistiveIncompressibleEHD
        = IncompressibleEHDModel<electrodynamics::ResistiveElectrodynamics>;
template <class Thermodynamics>
using ResistiveCompressibleEHD
        = CompressibleEHDModel<Thermodynamics, electrodynamics::ResistiveElectrodynamics>;
} // namespace similie::physics::ehd
