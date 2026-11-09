// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#include <gtest/gtest.h>
#include <similie/physics/coupled_fluid_electromagnetism.hpp>
#include <similie/physics/electrodynamics/resistive_electrodynamics.hpp>
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
} // namespace

TEST(Electrodynamics, ElectricHamiltonianAndPotentialSign)
{
    similie::physics::electrodynamics::Electrodynamics const electric(2);
    sil::exterior::Form<std::array<double, 3>> const d {4, 0, 0}, u {0, 1, 0};
    EXPECT_DOUBLE_EQ(electric.hamiltonian(d), 4);
    EXPECT_DOUBLE_EQ(electric.electric_field(d)[0], 2);
    EXPECT_DOUBLE_EQ(electric.field_from_potential_gradient({-2, 0, 0})[0], 2);
    EXPECT_DOUBLE_EQ(electric.displacement({2, 0, 0})[0], 4);
    EXPECT_DOUBLE_EQ(electric.current(Scalar(3), u, d)[1], 3);
    EXPECT_DOUBLE_EQ(electric.electric_force(Scalar(3), d)[0], 6);
    EXPECT_DOUBLE_EQ(electric.ohmic_heating(d), 0);
    EXPECT_DOUBLE_EQ(electric.gauss_residual(Scalar(3), Scalar(3)), 0);
    EXPECT_DOUBLE_EQ(electric(Scalar(7)), -7);
}

TEST(ResistiveElectrodynamics, CurrentRelaxationAndHeat)
{
    similie::physics::electrodynamics::ResistiveElectrodynamics const electric(2, 4);
    sil::exterior::Form<std::array<double, 3>> const d {4, 0, 0}, u {0, 1, 0};
    sil::exterior::Form<std::array<double, 3>> const current = electric.current(Scalar(3), u, d);
    EXPECT_DOUBLE_EQ(current[0], 0.5);
    EXPECT_DOUBLE_EQ(current[1], 3);
    EXPECT_DOUBLE_EQ(electric.ohmic_heating(d), 1);
    EXPECT_DOUBLE_EQ(electric.fluid_energy_source(d, current), 1);
    EXPECT_DOUBLE_EQ(electric.charge_relaxation_rate(), 0.125);
}

TEST(IncompressibleEHD, ElectricForceAndChargeEvolution)
{
    using Ideal = similie::physics::CoupledFluidElectromagnetism<
            similie::physics::fluid::IncompressibleNavierStokes,
            similie::physics::electrodynamics::Electrodynamics>;
    using Resistive = similie::physics::CoupledFluidElectromagnetism<
            similie::physics::fluid::IncompressibleNavierStokes,
            similie::physics::electrodynamics::ResistiveElectrodynamics>;
    Ideal const
            ideal(similie::physics::fluid::IncompressibleNavierStokes(2, 0),
                  similie::physics::electrodynamics::Electrodynamics(2));
    Resistive const resistive(
            similie::physics::fluid::IncompressibleNavierStokes(2, 0),
            similie::physics::electrodynamics::ResistiveElectrodynamics(2, 4));
    Ideal::State const state {{{0, 0, 2}}, {Scalar(3), {4, 0, 0}}};
    Resistive::State const resistive_state {state.fluid, state.electromagnetic};
    sil::exterior::Form<std::array<double, 3>> const velocity {0, 1, 0};
    EXPECT_DOUBLE_EQ(ideal.transport(state, velocity, ideal.current(state, velocity))[0], 5);
    EXPECT_DOUBLE_EQ(ideal.momentum_rate(state, velocity, {}, {}, {})[0], 5);
    EXPECT_DOUBLE_EQ(ideal.current(state, velocity)[0], 0);
    EXPECT_DOUBLE_EQ(resistive.current(resistive_state, velocity)[0], 0.5);
    EXPECT_DOUBLE_EQ(resistive.get<1>().ohmic_heating(state.electromagnetic.displacement), 1);
    Resistive::State const rate = resistive({1, 2, 3}, {}, Scalar(9), {}, {}, {});
    EXPECT_DOUBLE_EQ(rate.fluid.vorticity[1], 2);
    EXPECT_DOUBLE_EQ(rate.electromagnetic.charge_density, -9);
    EXPECT_EQ(rate.electromagnetic.displacement, (sil::exterior::Form<std::array<double, 3>> {}));
}

TEST(CompressibleEHD, FluidEnergyAndElectricalWork)
{
    using Fluid = similie::physics::fluid::NavierStokesFourier<CaloricClosure>;
    using Model = similie::physics::
            CoupledFluidElectromagnetism<Fluid, similie::physics::electrodynamics::Electrodynamics>;
    Model const
            ideal(Fluid(CaloricClosure(), 1, 0),
                  similie::physics::electrodynamics::Electrodynamics(2));
    Model::State const state {{Scalar(1), {1, 0, 0}, Scalar(3)}, {Scalar(2), {2, 0, 0}}};
    EXPECT_NEAR(ideal.get<0>().pressure(ideal.fluid_state(state)), 1, 1e-14);
    EXPECT_DOUBLE_EQ(ideal.combined_energy(state), 4);
    EXPECT_NEAR(ideal.flux(state, {}, {}, 0).total_energy, 4, 1e-14);
    Model::State const rate
            = ideal(state,
                    Fluid::Flux {Scalar(0), {}, Scalar(0)},
                    Scalar(0),
                    {},
                    {},
                    ideal.current(state));
    EXPECT_DOUBLE_EQ(rate.fluid.momentum[0], 2);
    EXPECT_DOUBLE_EQ(rate.fluid.total_energy, 2);
}

TEST(ResistiveCompressibleEHD, JouleHeatingIsIncludedOnce)
{
    using Fluid = similie::physics::fluid::NavierStokesFourier<CaloricClosure>;
    using Model = similie::physics::CoupledFluidElectromagnetism<
            Fluid,
            similie::physics::electrodynamics::ResistiveElectrodynamics>;
    Model const resistive(
            Fluid(CaloricClosure(), 1, 0),
            similie::physics::electrodynamics::ResistiveElectrodynamics(2, 2));
    Model::State const state {{Scalar(1), {1, 0, 0}, Scalar(3)}, {Scalar(2), {2, 0, 0}}};
    EXPECT_DOUBLE_EQ(resistive.current(state)[0], 2.5);
    EXPECT_DOUBLE_EQ(resistive.get<1>().ohmic_heating(state.electromagnetic.displacement), 0.5);
    Model::State const rate = resistive(
            state,
            Fluid::Flux {Scalar(0), {}, Scalar(0)},
            Scalar(0),
            {},
            {},
            resistive.current(state));
    EXPECT_DOUBLE_EQ(rate.fluid.momentum[0], 2);
    EXPECT_DOUBLE_EQ(rate.fluid.total_energy, 2.5);
}
