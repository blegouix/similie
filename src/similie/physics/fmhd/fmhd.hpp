// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#pragma once
#include <array>

#include <similie/physics/combine_physics.hpp>
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
    : public CombinePhysics<fluid::IncompressibleNavierStokes, ElectromagneticPhysics>
{
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
        : CombinePhysics<
                  fluid::IncompressibleNavierStokes,
                  ElectromagneticPhysics>(fluid_physics, electromagnetic_physics)
    {
    }
    KOKKOS_FUNCTION Vector current(State const& state, Vector const& velocity) const
    {
        return this->template get<1>()
                .current(state.charge_density, velocity, state.displacement, state.induction);
    }
    KOKKOS_FUNCTION Vector
    transport(State const& state, Vector const& velocity, Vector const& total_current) const
    {
        Vector result = this->template get<0>().transport(velocity, state.vorticity);
        Vector const force = this->template get<1>().lorentz_force(
                state.charge_density,
                state.displacement,
                state.induction,
                total_current);
        for (int a = 0; a < 3; ++a)
            result[a] += force[a] / this->template get<0>().density;
        return result;
    }
    KOKKOS_FUNCTION State operator()(
            Vector const& transport_curl,
            Vector const& vorticity_laplacian,
            double current_divergence,
            Vector const& curl_magnetic_field,
            Vector const& curl_electric_field,
            Vector const& total_current) const
    {
        typename ElectromagneticPhysics::State const field_rate = this->template get<
                1>()(current_divergence, curl_magnetic_field, curl_electric_field, total_current);
        State result {{}, field_rate.charge_density, field_rate.displacement, field_rate.induction};
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
            Vector const& total_current) const
    {
        Vector result = this->template get<0>().momentum_rate(
                velocity,
                state.vorticity,
                bernoulli_gradient,
                velocity_laplacian);
        Vector const force = this->template get<1>().lorentz_force(
                state.charge_density,
                state.displacement,
                state.induction,
                total_current);
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
    : public CombinePhysics<fluid::NavierStokesFourier<Thermodynamics>, ElectromagneticPhysics>
{
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
        : CombinePhysics<
                  FluidPhysics,
                  ElectromagneticPhysics>(fluid_physics, electromagnetic_physics)
    {
    }
    KOKKOS_FUNCTION typename FluidPhysics::State fluid_state(State const& state) const
    {
        return {state.density,
                state.momentum,
                state.total_energy
                        - this->template get<1>().hamiltonian(state.displacement, state.induction)};
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
        return this->template get<1>().current(
                state.charge_density,
                primitives(state).velocity,
                state.displacement,
                state.induction);
    }
    KOKKOS_FUNCTION Flux
    flux(State const& state,
         typename FluidPhysics::VelocityGradient const& velocity_gradient,
         Vector const& temperature_gradient,
         int normal) const
    {
        KOKKOS_ASSERT(normal >= 0 && normal < 3);
        typename FluidPhysics::State const hydro = fluid_state(state);
        typename FluidPhysics::PrimitiveVariables const primitive
                = this->template get<0>().primitives(hydro);
        Flux
                result {this->template get<0>()
                                .flux(hydro,
                                      primitive,
                                      velocity_gradient,
                                      temperature_gradient,
                                      normal),
                        this->template get<1>().current(
                                state.charge_density,
                                primitive.velocity,
                                state.displacement,
                                state.induction)};
        Vector const poynting
                = this->template get<1>().poynting_flux(state.displacement, state.induction);
        result.conservation.total_energy += poynting[normal];
        return result;
    }
    KOKKOS_FUNCTION State operator()(
            State const& state,
            typename FluidPhysics::Flux const& total_flux_divergence,
            double current_divergence,
            Vector const& curl_magnetic_field,
            Vector const& curl_electric_field,
            Vector const& total_current) const
    {
        typename FluidPhysics::State rate = this->template get<0>()(total_flux_divergence);
        Vector const force = this->template get<1>().lorentz_force(
                state.charge_density,
                state.displacement,
                state.induction,
                total_current);
        for (int a = 0; a < 3; ++a)
            rate.momentum[a] += force[a];
        typename ElectromagneticPhysics::State const field_rate = this->template get<
                1>()(current_divergence, curl_magnetic_field, curl_electric_field, total_current);
        return {rate.density,
                rate.momentum,
                rate.total_energy,
                field_rate.charge_density,
                field_rate.displacement,
                field_rate.induction};
    }
    KOKKOS_FUNCTION double fluid_energy_source(State const& state, Vector const& total_current)
            const
    {
        return this->template get<1>().fluid_energy_source(state.displacement, total_current);
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
