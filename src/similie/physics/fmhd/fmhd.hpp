// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#pragma once
#include <array>

#include <similie/physics/coupled_fluid_electromagnetism.hpp>
#include <similie/physics/electromagnetism/electromagnetism.hpp>
#include <similie/physics/electromagnetism/resistive_electromagnetism.hpp>
#include <similie/physics/fluid/incompressible_navier_stokes.hpp>
#include <similie/physics/fluid/navier_stokes_fourier.hpp>

namespace similie::physics::fmhd {
/** Incompressible vorticity with full Maxwell fields and advected charge.
 * omega_t=curl(u cross omega+(charge E+J cross B)/rho)+nu Delta omega.
 * Field/charge updates take one consistent current supplied by the caller.
 * \important This operator and documentation is fully AI-generated.
 */
template <PhysicsConcept ElectromagneticPhysics>
class IncompressibleFMHDModel
    : public CoupledFluidElectromagnetism<fluid::IncompressibleNavierStokes, ElectromagneticPhysics>
{
    using Assembly = CoupledFluidElectromagnetism<
            fluid::IncompressibleNavierStokes,
            ElectromagneticPhysics>;

public:
    using Vector = std::array<double, 3>;
    static constexpr std::size_t N = 10;
    static constexpr bool IS_LINEAR = false;
    struct State
    {
        Vector vorticity;
        double charge_density;
        Vector displacement;
        Vector induction;
    };
    static_assert(PhysicsConcept<IncompressibleFMHDModel>);
    KOKKOS_FUNCTION IncompressibleFMHDModel(
            fluid::IncompressibleNavierStokes const& fluid_physics,
            ElectromagneticPhysics const& electromagnetic_physics)
        : Assembly(fluid_physics, electromagnetic_physics)
    {
    }
    KOKKOS_FUNCTION Vector current(State const& state, Vector const& velocity) const
    {
        return Assembly::
                current({{state.vorticity},
                         {state.charge_density, state.displacement, state.induction}},
                        velocity);
    }
    KOKKOS_FUNCTION Vector
    transport(State const& state, Vector const& velocity, Vector const& total_current) const
    {
        return Assembly::transport(
                {{state.vorticity}, {state.charge_density, state.displacement, state.induction}},
                velocity,
                total_current);
    }
    KOKKOS_FUNCTION State operator()(
            Vector const& transport_curl,
            Vector const& vorticity_laplacian,
            double current_divergence,
            Vector const& curl_magnetic_field,
            Vector const& curl_electric_field,
            Vector const& total_current) const
    {
        typename Assembly::State const rate = Assembly::operator()(
                transport_curl,
                vorticity_laplacian,
                current_divergence,
                curl_magnetic_field,
                curl_electric_field,
                total_current);
        return {rate.fluid.vorticity,
                rate.electromagnetic.charge_density,
                rate.electromagnetic.displacement,
                rate.electromagnetic.induction};
    }
    KOKKOS_FUNCTION Vector momentum_rate(
            State const& state,
            Vector const& velocity,
            Vector const& bernoulli_gradient,
            Vector const& velocity_laplacian,
            Vector const& total_current) const
    {
        return Assembly::momentum_rate(
                {{state.vorticity}, {state.charge_density, state.displacement, state.induction}},
                velocity,
                bernoulli_gradient,
                velocity_laplacian,
                total_current);
    }
    KOKKOS_FUNCTION double pressure(double bernoulli, Vector const& velocity) const
    {
        return this->template get<0>().pressure(bernoulli, velocity);
    }
    KOKKOS_FUNCTION double incompressibility(double divergence) const
    {
        return this->template get<0>().incompressibility(divergence);
    }
    KOKKOS_FUNCTION double ohmic_heating(State const& state, Vector const& velocity) const
    {
        return this->template get<1>().ohmic_heating(velocity, state.displacement, state.induction);
    }
};

/** Full Maxwell compressible flow with caller-supplied thermodynamics.
 * total_energy includes both field Hamiltonians. Momentum is fluid momentum
 * with a Lorentz source; energy uses fluid flux plus E cross H with no duplicate
 * internal exchange source. Fields and charge evolve explicitly through DEC.
 * \important This operator and documentation is fully AI-generated.
 */
template <class Thermodynamics, PhysicsConcept ElectromagneticPhysics>
class CompressibleFMHDModel
    : public CoupledFluidElectromagnetism<
              fluid::NavierStokesFourier<Thermodynamics>,
              ElectromagneticPhysics,
              EnergyConvention::FluidAndField>
{
    using Assembly = CoupledFluidElectromagnetism<
            fluid::NavierStokesFourier<Thermodynamics>,
            ElectromagneticPhysics,
            EnergyConvention::FluidAndField>;

public:
    using FluidPhysics = fluid::NavierStokesFourier<Thermodynamics>;
    using Vector = std::array<double, 3>;
    static constexpr std::size_t N = 12;
    static constexpr bool IS_LINEAR = false;
    struct State
    {
        double density;
        Vector momentum;
        double total_energy;
        double charge_density;
        Vector displacement;
        Vector induction;
    };
    static_assert(PhysicsConcept<CompressibleFMHDModel>);
    struct Flux
    {
        typename FluidPhysics::Flux conservation;
        Vector electric_current;
    };
    KOKKOS_FUNCTION CompressibleFMHDModel(
            FluidPhysics const& fluid_physics,
            ElectromagneticPhysics const& electromagnetic_physics)
        : Assembly(fluid_physics, electromagnetic_physics)
    {
    }
    KOKKOS_FUNCTION typename FluidPhysics::State fluid_state(State const& state) const
    {
        return Assembly::fluid_state(
                {{state.density, state.momentum, state.total_energy},
                 {state.charge_density, state.displacement, state.induction}});
    }
    KOKKOS_FUNCTION typename FluidPhysics::PrimitiveVariables primitives(State const& state) const
    {
        return this->template get<0>().primitives(fluid_state(state));
    }
    KOKKOS_FUNCTION double pressure(State const& state) const
    {
        return this->template get<0>().pressure(fluid_state(state));
    }
    KOKKOS_FUNCTION Vector current(State const& state) const
    {
        return Assembly::current(
                {{state.density, state.momentum, state.total_energy},
                 {state.charge_density, state.displacement, state.induction}});
    }
    KOKKOS_FUNCTION Flux
    flux(State const& state,
         typename FluidPhysics::VelocityGradient const& velocity_gradient,
         Vector const& temperature_gradient,
         int normal) const
    {
        return {Assembly::
                        flux({{state.density, state.momentum, state.total_energy},
                              {state.charge_density, state.displacement, state.induction}},
                             velocity_gradient,
                             temperature_gradient,
                             normal),
                current(state)};
    }
    KOKKOS_FUNCTION State operator()(
            State const& state,
            typename FluidPhysics::Flux const& total_flux_divergence,
            double current_divergence,
            Vector const& curl_magnetic_field,
            Vector const& curl_electric_field,
            Vector const& total_current) const
    {
        typename Assembly::State const rate = Assembly::operator()(
                {{state.density, state.momentum, state.total_energy},
                 {state.charge_density, state.displacement, state.induction}},
                total_flux_divergence,
                current_divergence,
                curl_magnetic_field,
                curl_electric_field,
                total_current);
        return {rate.fluid.density,
                rate.fluid.momentum,
                rate.fluid.total_energy,
                rate.electromagnetic.charge_density,
                rate.electromagnetic.displacement,
                rate.electromagnetic.induction};
    }
    KOKKOS_FUNCTION double fluid_energy_source(State const& state, Vector const& total_current)
            const
    {
        return Assembly::coupling_sources(
                       {{state.density, state.momentum, state.total_energy},
                        {state.charge_density, state.displacement, state.induction}},
                       total_current)
                .fluid_energy_source;
    }
    KOKKOS_FUNCTION double ohmic_heating(State const& state) const
    {
        return this->template get<1>()
                .ohmic_heating(primitives(state).velocity, state.displacement, state.induction);
    }
};

using IdealIncompressibleFMHD = IncompressibleFMHDModel<electromagnetism::Electromagnetism>;
template <class Thermodynamics>
using IdealCompressibleFMHD
        = CompressibleFMHDModel<Thermodynamics, electromagnetism::Electromagnetism>;
using ResistiveIncompressibleFMHD
        = IncompressibleFMHDModel<electromagnetism::ResistiveElectromagnetism>;
template <class Thermodynamics>
using ResistiveCompressibleFMHD
        = CompressibleFMHDModel<Thermodynamics, electromagnetism::ResistiveElectromagnetism>;
} // namespace similie::physics::fmhd
