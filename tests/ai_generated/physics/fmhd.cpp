// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#include <gtest/gtest.h>
#include <similie/physics/coupled_fluid_electromagnetism.hpp>
#include <similie/physics/electromagnetism/resistive_electromagnetism.hpp>
#include <similie/physics/fluid/incompressible_navier_stokes.hpp>
#include <similie/physics/fluid/navier_stokes_fourier.hpp>

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

TEST(Electromagnetism, CombinedHamiltonianAndFullMaxwellRates)
{
    similie::physics::electromagnetism::Electromagnetism const fields(2, 1);
    std::array<double, 3> const d {2, 0, 0}, b {0, 1, 0}, u {1, 0, 0};
    EXPECT_DOUBLE_EQ(fields.hamiltonian(d, b), 1.5);
    EXPECT_DOUBLE_EQ(fields.electric_field(d)[0], 1);
    EXPECT_DOUBLE_EQ(fields.magnetic_field(b)[1], 1);
    EXPECT_DOUBLE_EQ(fields.poynting_flux(d, b)[2], 1);
    EXPECT_NEAR(fields.wave_speed() * fields.wave_speed(), 0.5, 1e-14);
    std::array<double, 3> const current = fields.current(2, u, d, b);
    EXPECT_DOUBLE_EQ(current[0], 2);
    EXPECT_DOUBLE_EQ(fields.lorentz_force(2, d, b, current)[0], 2);
    EXPECT_DOUBLE_EQ(fields.lorentz_force(2, d, b, current)[2], 2);
    similie::physics::electromagnetism::Electromagnetism::State const rate
            = fields(7, {1, 2, 3}, {3, 2, 1}, current);
    EXPECT_DOUBLE_EQ(rate.charge_density, -7);
    EXPECT_DOUBLE_EQ(rate.displacement[0], -1);
    EXPECT_DOUBLE_EQ(rate.displacement[1], 2);
    EXPECT_DOUBLE_EQ(rate.induction[0], -3);
}

TEST(ResistiveElectromagnetism, MotionalConductionAndJouleHeating)
{
    similie::physics::electromagnetism::ResistiveElectromagnetism const fields(2, 1, 2);
    std::array<double, 3> const d {2, 0, 0}, b {0, 1, 0}, u {1, 0, 0};
    std::array<double, 3> const current = fields.current(2, u, d, b);
    EXPECT_DOUBLE_EQ(current[0], 2.5);
    EXPECT_DOUBLE_EQ(current[2], 0.5);
    EXPECT_DOUBLE_EQ(fields.ohmic_heating(u, d, b), 1);
    EXPECT_DOUBLE_EQ(fields.fluid_energy_source(d, current), 2.5);
    EXPECT_DOUBLE_EQ(fields.lorentz_force(2, d, b, current)[0], 1.5);
    EXPECT_DOUBLE_EQ(
            fields.fluid_energy_source(d, current),
            fields.dot(u, fields.lorentz_force(2, d, b, current)) + fields.ohmic_heating(u, d, b));
    EXPECT_DOUBLE_EQ(fields.charge_relaxation_rate(), 0.25);
}

TEST(IncompressibleFMHD, BothFieldForcesAndSharedCurrent)
{
    using Model = similie::physics::CoupledFluidElectromagnetism<
            similie::physics::fluid::IncompressibleNavierStokes,
            similie::physics::electromagnetism::ResistiveElectromagnetism>;
    Model const resistive(
            similie::physics::fluid::IncompressibleNavierStokes(2, 0),
            similie::physics::electromagnetism::ResistiveElectromagnetism(2, 1, 4));
    Model::State const state {{{0, 0, 2}}, {3, {4, 0, 0}, {1, 0, 0}}};
    std::array<double, 3> const velocity {0, 1, 0}, ideal_current {0, 3, 0};
    EXPECT_DOUBLE_EQ(resistive.transport(state, velocity, ideal_current)[0], 5);
    EXPECT_DOUBLE_EQ(resistive.transport(state, velocity, ideal_current)[2], -1.5);
    EXPECT_DOUBLE_EQ(resistive.momentum_rate(state, velocity, {}, {}, ideal_current)[2], -1.5);
    std::array<double, 3> const current = resistive.current(state, velocity);
    EXPECT_DOUBLE_EQ(current[0], 0.5);
    EXPECT_DOUBLE_EQ(current[2], -0.25);
    EXPECT_DOUBLE_EQ(
            resistive.get<1>().ohmic_heating(
                    velocity,
                    state.electromagnetic.displacement,
                    state.electromagnetic.induction),
            1.25);
    Model::State const rate = resistive({1, 2, 3}, {}, 9, {}, {}, current);
    EXPECT_DOUBLE_EQ(rate.fluid.vorticity[1], 2);
    EXPECT_DOUBLE_EQ(rate.electromagnetic.charge_density, -9);
    EXPECT_DOUBLE_EQ(rate.electromagnetic.displacement[2], 0.25);
}

TEST(CompressibleFMHD, CombinedEnergyThermodynamicsAndPoyntingFlux)
{
    using Fluid = similie::physics::fluid::NavierStokesFourier<CaloricClosure>;
    using Model = similie::physics::CoupledFluidElectromagnetism<
            Fluid,
            similie::physics::electromagnetism::Electromagnetism,
            similie::physics::EnergyConvention::FluidAndField>;
    Model const
            ideal(Fluid(CaloricClosure(), 1, 0),
                  similie::physics::electromagnetism::Electromagnetism(2, 1));
    Model::State const state {{1, {1, 0, 0}, 4.5}, {2, {2, 0, 0}, {0, 1, 0}}};
    EXPECT_DOUBLE_EQ(ideal.fluid_state(state).total_energy, 3);
    EXPECT_NEAR(ideal.get<0>().pressure(ideal.fluid_state(state)), 1, 1e-14);
    EXPECT_NEAR(ideal.flux(state, {}, {}, 0).total_energy, 4, 1e-14);
    EXPECT_DOUBLE_EQ(ideal.flux(state, {}, {}, 2).total_energy, 1);
    std::array<double, 3> const current = ideal.current(state);
    Model::State const rate = ideal(state, Fluid::Flux {0, {}, 0}, 0, {}, {}, current);
    EXPECT_DOUBLE_EQ(rate.fluid.momentum[0], 2);
    EXPECT_DOUBLE_EQ(rate.fluid.momentum[2], 2);
    EXPECT_DOUBLE_EQ(rate.electromagnetic.displacement[0], -2);
    EXPECT_DOUBLE_EQ(rate.fluid.total_energy, 0);
    EXPECT_DOUBLE_EQ(ideal.coupling_sources(state, current).fluid_energy_source, 2);
}

TEST(ResistiveCompressibleFMHD, PowerExchangeWithoutDoubleCounting)
{
    using Fluid = similie::physics::fluid::NavierStokesFourier<CaloricClosure>;
    using Model = similie::physics::CoupledFluidElectromagnetism<
            Fluid,
            similie::physics::electromagnetism::ResistiveElectromagnetism,
            similie::physics::EnergyConvention::FluidAndField>;
    Model const resistive(
            Fluid(CaloricClosure(), 1, 0),
            similie::physics::electromagnetism::ResistiveElectromagnetism(2, 1, 2));
    Model::State const state {{1, {1, 0, 0}, 4.5}, {2, {2, 0, 0}, {0, 1, 0}}};
    std::array<double, 3> const current = resistive.current(state);
    EXPECT_DOUBLE_EQ(current[0], 2.5);
    EXPECT_DOUBLE_EQ(current[2], 0.5);
    EXPECT_DOUBLE_EQ(
            resistive.get<1>().ohmic_heating(
                    {1, 0, 0},
                    state.electromagnetic.displacement,
                    state.electromagnetic.induction),
            1);
    EXPECT_DOUBLE_EQ(resistive.coupling_sources(state, current).fluid_energy_source, 2.5);
    Model::State const rate = resistive(state, Fluid::Flux {0, {}, 0}, 0, {}, {}, current);
    EXPECT_DOUBLE_EQ(rate.fluid.momentum[0], 1.5);
    EXPECT_DOUBLE_EQ(rate.fluid.momentum[2], 2.5);
    EXPECT_DOUBLE_EQ(rate.electromagnetic.displacement[0], -2.5);
    EXPECT_DOUBLE_EQ(rate.fluid.total_energy, 0);
}
