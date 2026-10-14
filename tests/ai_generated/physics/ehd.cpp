// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#include <gtest/gtest.h>
#include <similie/physics/ehd/ehd.hpp>

namespace {
struct CaloricClosure
{
    KOKKOS_FUNCTION double pressure(double rho, double e) const
    {
        return 0.4 * rho * e;
    }
    KOKKOS_FUNCTION double temperature([[maybe_unused]] double rho, double e) const
    {
        return 0.4 * e;
    }
    KOKKOS_FUNCTION double heat_capacity_at_constant_pressure(
            [[maybe_unused]] double rho,
            [[maybe_unused]] double e) const
    {
        return 3.5;
    }
};
} // namespace

TEST(Electrodynamics, ElectricHamiltonianAndPotentialSign)
{
    similie::physics::electrodynamics::Electrodynamics const electric(2);
    std::array<double, 3> const d {4, 0, 0}, u {0, 1, 0};
    EXPECT_DOUBLE_EQ(electric.hamiltonian(d), 4);
    EXPECT_DOUBLE_EQ(electric.electric_field(d)[0], 2);
    EXPECT_DOUBLE_EQ(electric.field_from_potential_gradient({-2, 0, 0})[0], 2);
    EXPECT_DOUBLE_EQ(electric.displacement({2, 0, 0})[0], 4);
    EXPECT_DOUBLE_EQ(electric.current(3, u, d)[1], 3);
    EXPECT_DOUBLE_EQ(electric.electric_force(3, d)[0], 6);
    EXPECT_DOUBLE_EQ(electric.ohmic_heating(d), 0);
    EXPECT_DOUBLE_EQ(electric.gauss_residual(3, 3), 0);
    EXPECT_DOUBLE_EQ(electric(7), -7);
}

TEST(ResistiveElectrodynamics, CurrentRelaxationAndHeat)
{
    similie::physics::electrodynamics::ResistiveElectrodynamics const electric(2, 4);
    std::array<double, 3> const d {4, 0, 0}, u {0, 1, 0};
    std::array<double, 3> const current = electric.current(3, u, d);
    EXPECT_DOUBLE_EQ(current[0], 0.5);
    EXPECT_DOUBLE_EQ(current[1], 3);
    EXPECT_DOUBLE_EQ(electric.ohmic_heating(d), 1);
    EXPECT_DOUBLE_EQ(electric.fluid_energy_source(d, current), 1);
    EXPECT_DOUBLE_EQ(electric.charge_relaxation_rate(), 0.125);
}

TEST(IncompressibleEHD, ElectricForceAndChargeEvolution)
{
    similie::physics::fluid::IncompressibleNavierStokes const fluid(2, 0);
    similie::physics::ehd::IdealIncompressibleEHD const
            ideal(fluid, similie::physics::electrodynamics::Electrodynamics(2));
    similie::physics::ehd::ResistiveIncompressibleEHD const
            resistive(fluid, similie::physics::electrodynamics::ResistiveElectrodynamics(2, 4));
    similie::physics::ehd::IdealIncompressibleEHD::State const state {{0, 0, 2}, 3};
    similie::physics::ehd::ResistiveIncompressibleEHD::State const
            resistive_state {state.vorticity, state.charge_density};
    std::array<double, 3> const u {0, 1, 0}, d {4, 0, 0};
    EXPECT_DOUBLE_EQ(ideal.transport(state, u, d)[0], 5);
    EXPECT_DOUBLE_EQ(ideal.momentum_rate(state, u, {}, {}, d)[0], 5);
    EXPECT_DOUBLE_EQ(ideal.current(state, u, d)[0], 0);
    EXPECT_DOUBLE_EQ(resistive.current(resistive_state, u, d)[0], 0.5);
    EXPECT_DOUBLE_EQ(resistive.ohmic_heating(d), 1);
    similie::physics::ehd::ResistiveIncompressibleEHD::State const rate
            = resistive({1, 2, 3}, {}, 9);
    EXPECT_DOUBLE_EQ(rate.vorticity[1], 2);
    EXPECT_DOUBLE_EQ(rate.charge_density, -9);
}

TEST(CompressibleEHD, FluidEnergyAndElectricalWork)
{
    similie::physics::fluid::NavierStokesFourier<CaloricClosure> const
            fluid(CaloricClosure(), 1, 0);
    similie::physics::ehd::IdealCompressibleEHD<CaloricClosure> const
            ideal(fluid, similie::physics::electrodynamics::Electrodynamics(2));
    similie::physics::ehd::IdealCompressibleEHD<CaloricClosure>::State const
            state {1, {1, 0, 0}, 3, 2};
    std::array<double, 3> const d {2, 0, 0};
    EXPECT_NEAR(ideal.pressure(state), 1, 1e-14);
    EXPECT_DOUBLE_EQ(ideal.combined_energy(state, d), 4);
    EXPECT_NEAR(ideal.flux(state, d, {}, {}, 0).fluid.total_energy, 4, 1e-14);
    similie::physics::ehd::IdealCompressibleEHD<CaloricClosure>::State const rate
            = ideal(state, d, {0, {}, 0}, 0, ideal.current(state, d));
    EXPECT_DOUBLE_EQ(rate.momentum[0], 2);
    EXPECT_DOUBLE_EQ(rate.total_energy, 2);
}

TEST(ResistiveCompressibleEHD, JouleHeatingIsIncludedOnce)
{
    similie::physics::fluid::NavierStokesFourier<CaloricClosure> const
            fluid(CaloricClosure(), 1, 0);
    similie::physics::ehd::ResistiveCompressibleEHD<CaloricClosure> const
            resistive(fluid, similie::physics::electrodynamics::ResistiveElectrodynamics(2, 2));
    similie::physics::ehd::ResistiveCompressibleEHD<CaloricClosure>::State const
            state {1, {1, 0, 0}, 3, 2};
    std::array<double, 3> const d {2, 0, 0};
    EXPECT_DOUBLE_EQ(resistive.current(state, d)[0], 2.5);
    EXPECT_DOUBLE_EQ(resistive.ohmic_heating(d), 0.5);
    similie::physics::ehd::ResistiveCompressibleEHD<CaloricClosure>::State const rate
            = resistive(state, d, {0, {}, 0}, 0, resistive.current(state, d));
    EXPECT_DOUBLE_EQ(rate.momentum[0], 2);
    EXPECT_DOUBLE_EQ(rate.total_energy, 2.5);
}
