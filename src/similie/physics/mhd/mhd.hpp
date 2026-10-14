// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#pragma once

#include <array>
#include <cmath>

#include <similie/physics/combine_physics.hpp>
#include <similie/physics/fluid/incompressible_navier_stokes.hpp>
#include <similie/physics/fluid/navier_stokes_fourier.hpp>
#include <similie/physics/magnetodynamics/magnetodynamics.hpp>
#include <similie/physics/magnetodynamics/resistive_magnetodynamics.hpp>

namespace similie::physics::mhd {
/** Shared quasistatic momentum and energy exchange, evaluated pointwise.
 * u.J cross B + eta J.J = J.E. Total MHD energy instead uses conservative
 * Poynting flux and must not receive this exchange a second time.
 * \important This operator and documentation is fully AI-generated.
 */
template <PhysicsConcept MagneticPhysics>
KOKKOS_FUNCTION double fluid_energy_source(
        MagneticPhysics const& magnetic,
        std::array<double, 3> const& velocity,
        std::array<double, 3> const& induction,
        std::array<double, 3> const& current)
{
    return MagneticPhysics::dot(current, magnetic.ohm_electric_field(velocity, induction, current));
}

/** Vorticity formulation coupled to closed magnetic-induction two-cochains.
 * omega_t=curl(u cross omega+J cross B/rho)+nu Delta omega,
 * B_t=-curl E. The caller reconstructs u with the fluid Coulomb-gauge solve
 * and evaluates J=curl(H) with the native exterior module.
 * \important This operator and documentation is fully AI-generated.
 */
template <PhysicsConcept MagneticPhysics>
class IncompressibleMHDModel
    : public CombinePhysics<fluid::IncompressibleNavierStokes, MagneticPhysics>
{
public:
    using Vector = std::array<double, 3>;
    static constexpr std::size_t N = 6;
    static constexpr bool IS_LINEAR = false;
    struct State
    {
        Vector vorticity;
        Vector induction;
    };
    static_assert(PhysicsConcept<IncompressibleMHDModel>);
    KOKKOS_FUNCTION IncompressibleMHDModel(
            fluid::IncompressibleNavierStokes fluid_physics,
            MagneticPhysics magnetic_physics)
        : CombinePhysics<
                  fluid::IncompressibleNavierStokes,
                  MagneticPhysics>(fluid_physics, magnetic_physics)
    {
    }
    KOKKOS_FUNCTION Vector transport(
            Vector const& velocity,
            Vector const& vorticity,
            Vector const& induction,
            Vector const& current) const
    {
        Vector result = this->template get<0>().transport(velocity, vorticity);
        Vector const force = this->template get<1>().lorentz_force(current, induction);
        for (int a = 0; a < 3; ++a)
            result[a] += force[a] / this->template get<0>().density;
        return result;
    }
    KOKKOS_FUNCTION Vector
    electric_field(Vector const& velocity, Vector const& induction, Vector const& current) const
    {
        return this->template get<1>().ohm_electric_field(velocity, induction, current);
    }
    KOKKOS_FUNCTION State operator()(
            Vector const& transport_curl,
            Vector const& vorticity_laplacian,
            Vector const& curl_electric_field) const
    {
        State result {{}, this->template get<1>().induction_rate(curl_electric_field)};
        for (int a = 0; a < 3; ++a)
            result.vorticity[a]
                    = this->template get<0>()(transport_curl[a], vorticity_laplacian[a]);
        return result;
    }
    KOKKOS_FUNCTION Vector momentum_rate(
            Vector const& velocity,
            Vector const& vorticity,
            Vector const& bernoulli_gradient,
            Vector const& velocity_laplacian,
            Vector const& induction,
            Vector const& current) const
    {
        Vector result = this->template get<0>().momentum_rate(
                velocity,
                vorticity,
                bernoulli_gradient,
                velocity_laplacian);
        Vector const force = this->template get<1>().lorentz_force(current, induction);
        for (int a = 0; a < 3; ++a)
            result[a] += force[a] / this->template get<0>().density;
        return result;
    }
    KOKKOS_FUNCTION double ohmic_heating(Vector const& current) const
    {
        return this->template get<1>().ohmic_heating(current);
    }
    KOKKOS_FUNCTION double pressure(double bernoulli, Vector const& velocity) const
    {
        return this->template get<0>().pressure(bernoulli, velocity);
    }
    KOKKOS_FUNCTION double incompressibility(double divergence) const
    {
        return this->template get<0>().incompressibility(divergence);
    }
};

/** Conservative compressible MHD with caller-supplied thermodynamics.
 * State total_energy includes B.H/2. Fluid primitives subtract this magnetic
 * energy before evaluating p(rho,e) and T(rho,e). Momentum adds magnetic stress;
 * total-energy flux adds E cross H, including resistive transport. Faraday's
 * curl supplies induction evolution. No optimizer and no fluid Hamiltonian.
 * \important This operator and documentation is fully AI-generated.
 */
template <class Thermodynamics, PhysicsConcept MagneticPhysics>
class CompressibleMHDModel
    : public CombinePhysics<fluid::NavierStokesFourier<Thermodynamics>, MagneticPhysics>
{
public:
    using FluidPhysics = fluid::NavierStokesFourier<Thermodynamics>;
    using Vector = std::array<double, 3>;
    static constexpr std::size_t N = 8;
    static constexpr bool IS_LINEAR = false;
    struct State
    {
        double density;
        Vector momentum;
        double total_energy;
        Vector induction;
    };
    static_assert(PhysicsConcept<CompressibleMHDModel>);
    struct Flux
    {
        typename FluidPhysics::Flux conservation;
        Vector electric_field;
    };
    KOKKOS_FUNCTION CompressibleMHDModel(
            FluidPhysics fluid_physics,
            MagneticPhysics magnetic_physics)
        : CombinePhysics<FluidPhysics, MagneticPhysics>(fluid_physics, magnetic_physics)
    {
    }
    KOKKOS_FUNCTION typename FluidPhysics::State fluid_state(State const& state) const
    {
        return {state.density,
                state.momentum,
                state.total_energy - this->template get<1>().magnetic_energy(state.induction)};
    }
    KOKKOS_FUNCTION typename FluidPhysics::PrimitiveVariables primitives(State const& state) const
    {
        return this->template get<0>().primitives(fluid_state(state));
    }
    KOKKOS_FUNCTION double pressure(State const& state) const
    {
        return this->template get<0>().pressure(fluid_state(state));
    }
    KOKKOS_FUNCTION Vector electric_field(State const& state, Vector const& current) const
    {
        return this->template get<1>()
                .ohm_electric_field(primitives(state).velocity, state.induction, current);
    }
    KOKKOS_FUNCTION Flux
    flux(State const& state,
         typename FluidPhysics::VelocityGradient const& velocity_gradient,
         Vector const& temperature_gradient,
         Vector const& current,
         int normal) const
    {
        KOKKOS_ASSERT(normal >= 0 && normal < 3);
        typename FluidPhysics::State const hydrodynamic = fluid_state(state);
        typename FluidPhysics::PrimitiveVariables const primitive
                = this->template get<0>().primitives(hydrodynamic);
        Vector const electric
                = this->template get<1>()
                          .ohm_electric_field(primitive.velocity, state.induction, current);
        Flux
                result {this->template get<0>()
                                .flux(hydrodynamic,
                                      primitive,
                                      velocity_gradient,
                                      temperature_gradient,
                                      normal),
                        electric};
        typename MagneticPhysics::Tensor const stress
                = this->template get<1>().momentum_flux(state.induction);
        Vector const poynting = this->template get<1>().poynting_flux(electric, state.induction);
        for (int a = 0; a < 3; ++a)
            result.conservation.momentum[a] += stress[a][normal];
        result.conservation.total_energy += poynting[normal];
        return result;
    }
    KOKKOS_FUNCTION State operator()(
            typename FluidPhysics::Flux const& total_flux_divergence,
            Vector const& curl_electric_field) const
    {
        typename FluidPhysics::State const rate = this->template get<0>()(total_flux_divergence);
        return {rate.density,
                rate.momentum,
                rate.total_energy,
                this->template get<1>().induction_rate(curl_electric_field)};
    }
    KOKKOS_FUNCTION double ohmic_heating(Vector const& current) const
    {
        return this->template get<1>().ohmic_heating(current);
    }
    KOKKOS_FUNCTION double fast_magnetosonic_speed_squared(State const& state, int normal) const
    {
        KOKKOS_ASSERT(normal >= 0 && normal < 3);
        double const sound = this->template get<0>().sound_speed_squared(fluid_state(state));
        Vector const field = this->template get<1>().magnetic_field(state.induction);
        double const alfven = MagneticPhysics::dot(state.induction, field) / state.density;
        double const normal_alfven = state.induction[normal] * field[normal] / state.density;
        double const sum = sound + alfven;
        double const discriminant = sum * sum - 4 * sound * normal_alfven;
        return 0.5 * (sum + std::sqrt(discriminant > 0 ? discriminant : 0));
    }
};

using IdealIncompressibleMHD = IncompressibleMHDModel<magnetodynamics::Magnetodynamics>;
template <class Thermodynamics>
using IdealCompressibleMHD = CompressibleMHDModel<Thermodynamics, magnetodynamics::Magnetodynamics>;
using ResistiveIncompressibleMHD
        = IncompressibleMHDModel<magnetodynamics::ResistiveMagnetodynamics>;
template <class Thermodynamics>
using ResistiveCompressibleMHD
        = CompressibleMHDModel<Thermodynamics, magnetodynamics::ResistiveMagnetodynamics>;
} // namespace similie::physics::mhd
