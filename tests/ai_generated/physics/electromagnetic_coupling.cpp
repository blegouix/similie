// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#include <gtest/gtest.h>
#include <similie/physics/coupled_fluid_electromagnetism.hpp>
#include <similie/physics/electromagnetism/resistive_electromagnetism.hpp>
#include <similie/physics/fluid/incompressible_navier_stokes.hpp>
#include <similie/physics/fluid/navier_stokes_fourier.hpp>

using Scalar = sil::exterior::Form<double>;

namespace {
struct CaloricClosure
{
    KOKKOS_FUNCTION Scalar pressure(Scalar const& rho, Scalar const& e) const
    {
        return Scalar(0.4 * rho * e);
    }
    KOKKOS_FUNCTION Scalar temperature([[maybe_unused]] Scalar const& rho, Scalar const& e) const
    {
        return Scalar(0.4 * e);
    }
    KOKKOS_FUNCTION double heat_capacity_at_constant_pressure(
            [[maybe_unused]] Scalar const& rho,
            [[maybe_unused]] Scalar const& e) const
    {
        return 3.5;
    }
};

using Fluid = similie::physics::fluid::NavierStokesFourier<CaloricClosure>;
using Fields = similie::physics::electromagnetism::ResistiveElectromagnetism;
using Assembly = similie::physics::CoupledFluidElectromagnetism<Fluid, Fields>;
using CombinedAssembly = similie::physics::CoupledFluidElectromagnetism<
        Fluid,
        Fields,
        similie::physics::EnergyConvention::FluidAndField>;

static_assert(similie::physics::PhysicsConcept<Assembly>);
static_assert(Assembly::N == Fluid::N + Fields::N);
static_assert(Assembly::size == 4);

void check_device_evaluation()
{
    Assembly const model(Fluid(CaloricClosure(), 1, 0), Fields(2, 4, 2));
    Assembly::State const
            state {{Scalar(2), {2, 0, 0}, Scalar(11)}, {Scalar(-2), {6, -8, 4}, {-2, 3, 4}}};
    Kokkos::View<double*> result("coupled_energy_rate", 1);
    Kokkos::parallel_for(
            "evaluate_coupled_rate",
            1,
            KOKKOS_LAMBDA(int) {
                Fluid::Flux const divergence {Scalar(0), {}, Scalar(0)};
                result(0) = model(state, divergence, Scalar(0), {}, {}, {5, -1, 7})
                                    .fluid.total_energy;
            });
    Kokkos::View<double*, Kokkos::HostSpace> const host
            = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), result);
    EXPECT_DOUBLE_EQ(host(0), 33);
}

} // namespace

TEST(ElectromagneticCoupling, NeutralConductionStillTransfersMomentumAndEnergy)
{
    similie::physics::coupling::ElectromagneticCouplingState const
            state {Scalar(0), {3, -4, 2}, {-2, 3, 4}, {5, -1, 7}};
    similie::physics::coupling::ElectromagneticExchange const electric
            = similie::physics::coupling::ElectricCouplingPhysics()(state);
    similie::physics::coupling::ElectromagneticExchange const magnetic
            = similie::physics::coupling::MagneticCouplingPhysics()(state);
    EXPECT_EQ(electric.momentum_source, (sil::exterior::Form<std::array<double, 3>> {0, 0, 0}));
    EXPECT_EQ(
            magnetic.momentum_source,
            (sil::exterior::Form<std::array<double, 3>> {-25, -34, 13}));
    EXPECT_DOUBLE_EQ(electric.fluid_energy_source, 33);
    EXPECT_DOUBLE_EQ(electric.field_energy_source, -33);
    EXPECT_DOUBLE_EQ(magnetic.fluid_energy_source, 0);
    EXPECT_DOUBLE_EQ(magnetic.field_energy_source, 0);
}

TEST(FluidElectromagneticAssembly, SharedCurrentAndLocalEnergyBalance)
{
    Assembly const model(Fluid(CaloricClosure(), 1, 0), Fields(2, 4, 2));
    Assembly::State const
            state {{Scalar(2), {2, 0, 0}, Scalar(11)}, {Scalar(-2), {6, -8, 4}, {-2, 3, 4}}};
    // A spatially reduced current can differ from a pointwise constitutive
    // evaluation. Every assembled exchange and field update must use this one.
    sil::exterior::Form<std::array<double, 3>> const current {5, -1, 7};
    Fluid::Flux const divergence {Scalar(0.2), {1, 2, 3}, Scalar(4)};
    Assembly::State const rate = model(state, divergence, Scalar(2), {}, {}, current);
    EXPECT_DOUBLE_EQ(rate.fluid.density, -0.2);
    EXPECT_EQ(rate.fluid.momentum, (sil::exterior::Form<std::array<double, 3>> {-32, -28, 6}));
    EXPECT_DOUBLE_EQ(rate.fluid.total_energy, 29);
    EXPECT_DOUBLE_EQ(rate.electromagnetic.charge_density, -2);
    EXPECT_EQ(
            rate.electromagnetic.displacement,
            (sil::exterior::Form<std::array<double, 3>> {-5, 1, -7}));
    EXPECT_EQ(rate.electromagnetic.induction, (sil::exterior::Form<std::array<double, 3>> {}));
    double const field_power = Fields::
            dot(model.get<1>().electric_field(state.electromagnetic.displacement),
                rate.electromagnetic.displacement);
    EXPECT_DOUBLE_EQ(field_power, model.coupling_sources(state, current).field_energy_source);
    EXPECT_DOUBLE_EQ(rate.fluid.total_energy + field_power, -divergence.total_energy);
    EXPECT_DOUBLE_EQ(model.combined_energy(state), 43.625);
    EXPECT_DOUBLE_EQ(model.get<0>().pressure(model.fluid_state(state)), 4);
}

TEST(FluidElectromagneticAssembly, EnergyConventionsAgreeWithoutDuplicateHeating)
{
    Assembly const fluid_energy(Fluid(CaloricClosure(), 1, 0), Fields(2, 4, 2));
    CombinedAssembly const combined_energy(Fluid(CaloricClosure(), 1, 0), Fields(2, 4, 2));
    Assembly::State const
            state {{Scalar(2), {2, 0, 0}, Scalar(11)}, {Scalar(-2), {6, -8, 4}, {-2, 3, 4}}};
    CombinedAssembly::State const combined {
            {Scalar(2), {2, 0, 0}, fluid_energy.combined_energy(state)},
            state.electromagnetic};
    sil::exterior::Form<std::array<double, 3>> const current = fluid_energy.current(state);
    EXPECT_EQ(current, combined_energy.current(combined));
    EXPECT_DOUBLE_EQ(combined_energy.fluid_state(combined).total_energy, state.fluid.total_energy);
    Fluid::Flux const divergence {Scalar(0.2), {1, 2, 3}, Scalar(4)};
    CombinedAssembly::State const rate
            = combined_energy(combined, divergence, Scalar(2), {}, {}, current);
    EXPECT_DOUBLE_EQ(rate.fluid.total_energy, -4);
    EXPECT_DOUBLE_EQ(
            combined_energy.combined_energy(combined),
            fluid_energy.combined_energy(state));
    for (int normal = 0; normal < 3; ++normal) {
        Fluid::Flux const fluid_flux = fluid_energy.flux(state, {}, {}, normal);
        Fluid::Flux const combined_flux = combined_energy.flux(combined, {}, {}, normal);
        EXPECT_EQ(fluid_flux.momentum, combined_flux.momentum);
        EXPECT_DOUBLE_EQ(
                combined_flux.total_energy - fluid_flux.total_energy,
                combined_energy.get<1>().poynting_flux(
                        combined.electromagnetic.displacement,
                        combined.electromagnetic.induction)[normal]);
    }
    // Both electric and magnetic mechanical work enter bulk kinetic energy.
    // The remaining J.E is heat; adding this heat again would break balance.
    double const heat = fluid_energy.get<1>().ohmic_heating(
            {1, 0, 0},
            state.electromagnetic.displacement,
            state.electromagnetic.induction);
    similie::physics::coupling::ElectromagneticExchange const exchange
            = fluid_energy.coupling_sources(state, current);
    EXPECT_DOUBLE_EQ(exchange.fluid_energy_source, exchange.momentum_source[0] + heat);
}

TEST(FluidElectromagneticAssembly, IncompressibleForcePrecedesExteriorCurl)
{
    similie::physics::CoupledFluidElectromagnetism<
            similie::physics::fluid::IncompressibleNavierStokes,
            Fields> const
            model(similie::physics::fluid::IncompressibleNavierStokes(2, 0.5), Fields(2, 4, 2));
    similie::physics::CoupledFluidElectromagnetism<
            similie::physics::fluid::IncompressibleNavierStokes,
            Fields>::State const state {{{0, 0, 2}}, {Scalar(-2), {6, -8, 4}, {-2, 3, 4}}};
    sil::exterior::Form<std::array<double, 3>> const current {5, -1, 7};
    EXPECT_EQ(
            model.transport(state, {1, 0, 0}, current),
            (sil::exterior::Form<std::array<double, 3>> {-15.5, -15, 4.5}));
    EXPECT_EQ(
            model.momentum_rate(state, {1, 0, 0}, {}, {}, current),
            model.transport(state, {1, 0, 0}, current));
    sil::exterior::Form<std::array<double, 3>> const curl {1, 2, 3}, laplacian {4, -4, 8};
    EXPECT_EQ(
            model(curl, laplacian, Scalar(2), {}, {}, current).fluid.vorticity,
            (sil::exterior::Form<std::array<double, 3>> {2, 1, 5}));
}

TEST(FluidElectromagneticAssembly, DeviceEvaluation)
{
    check_device_evaluation();
}

TEST(NavierStokesFourier, TwoPointFluxIsSymmetricAndConsistent)
{
    Fluid const model(CaloricClosure(), 0.71, 0.01);
    Fluid::State const left {Scalar(2), {1, -2, 3}, Scalar(12)},
            right {Scalar(1), {-1, 2, 0}, Scalar(10)};
    Fluid::VelocityGradient const gradient {{{1, 2, 3}, {-2, 1, 0}, {0, 4, -2}}};
    sil::exterior::Form<std::array<double, 3>> const temperature_gradient {2, -1, 3};
    for (int normal = 0; normal < 3; ++normal) {
        Fluid::Flux const physical
                = model.flux(left, model.primitives(left), gradient, temperature_gradient, normal);
        Fluid::Flux const identical
                = model.two_point_flux(left, left, gradient, temperature_gradient, normal);
        EXPECT_NEAR(physical.mass, identical.mass, 1e-14);
        EXPECT_NEAR(physical.total_energy, identical.total_energy, 1e-14);
        Fluid::Flux const forward
                = model.two_point_flux(left, right, gradient, temperature_gradient, normal);
        Fluid::Flux const backward
                = model.two_point_flux(right, left, gradient, temperature_gradient, normal);
        EXPECT_DOUBLE_EQ(forward.mass, backward.mass);
        EXPECT_DOUBLE_EQ(forward.total_energy, backward.total_energy);
        for (int a = 0; a < 3; ++a) {
            EXPECT_NEAR(physical.momentum[a], identical.momentum[a], 1e-14);
            EXPECT_DOUBLE_EQ(forward.momentum[a], backward.momentum[a]);
        }
    }
}
