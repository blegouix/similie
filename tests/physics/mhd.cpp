// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#include <array>

#include <gtest/gtest.h>
#include <similie/physics/mhd/mhd.hpp>

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
    KOKKOS_FUNCTION double heat_capacity_at_constant_volume(
            [[maybe_unused]] double rho,
            [[maybe_unused]] double e) const
    {
        return 2.5;
    }
    KOKKOS_FUNCTION double temperature_density_derivative(
            [[maybe_unused]] double rho,
            [[maybe_unused]] double e) const
    {
        return 0;
    }
    KOKKOS_FUNCTION double temperature_energy_derivative(
            [[maybe_unused]] double rho,
            [[maybe_unused]] double e) const
    {
        return 0.4;
    }
    KOKKOS_FUNCTION double sound_speed_squared([[maybe_unused]] double rho, double e) const
    {
        return 0.56 * e;
    }
};
} // namespace

TEST(Magnetodynamics, MagneticHamiltonianAndFaraday)
{
    similie::physics::magnetodynamics::Magnetodynamics const magnetic(2);
    std::array<double, 3> const b {2, 0, 0};
    EXPECT_DOUBLE_EQ(magnetic.hamiltonian(b), 1);
    EXPECT_DOUBLE_EQ(magnetic.magnetic_energy(b), 1);
    EXPECT_DOUBLE_EQ(magnetic.magnetic_field(b)[0], 1);
    EXPECT_DOUBLE_EQ(magnetic({3, 2, 1})[0], -3);
}

TEST(CombinePhysics, ThreeSectorsAndConstAccess)
{
    similie::physics::CombinePhysics<
            similie::physics::fluid::IncompressibleNavierStokes,
            similie::physics::magnetodynamics::Magnetodynamics,
            similie::physics::magnetodynamics::ResistiveMagnetodynamics> const
            combined(
                    similie::physics::fluid::IncompressibleNavierStokes(2, 0),
                    similie::physics::magnetodynamics::Magnetodynamics(3),
                    similie::physics::magnetodynamics::ResistiveMagnetodynamics(4, 0.5));
    static_assert(decltype(combined)::size == 3);
    EXPECT_DOUBLE_EQ(combined.get<0>().density, 2);
    EXPECT_DOUBLE_EQ(combined.get<1>().permeability, 3);
    EXPECT_DOUBLE_EQ(combined.get<2>().magnetic_diffusivity(), 0.125);
}

TEST(IncompressibleMHD, LorentzVorticityTransportAndResistiveInduction)
{
    similie::physics::mhd::IdealIncompressibleMHD const
            ideal(similie::physics::fluid::IncompressibleNavierStokes(2, 0),
                  similie::physics::magnetodynamics::Magnetodynamics(1));
    similie::physics::mhd::ResistiveIncompressibleMHD const resistive(
            similie::physics::fluid::IncompressibleNavierStokes(2, 0),
            similie::physics::magnetodynamics::ResistiveMagnetodynamics(1, 0.5));
    std::array<double, 3> const b {1, 0, 0}, j {0, 0, 2}, u {0, 1, 0};
    EXPECT_DOUBLE_EQ(ideal.transport({}, {}, b, j)[1], 1);
    EXPECT_DOUBLE_EQ(ideal.electric_field(u, b, j)[2], 1);
    EXPECT_DOUBLE_EQ(resistive.electric_field(u, b, j)[2], 2);
    EXPECT_DOUBLE_EQ(resistive.ohmic_heating(j), 2);
    similie::physics::mhd::IdealIncompressibleMHD::Variables const rate
            = ideal({1, 2, 3}, {0, 0, 0}, {3, 2, 1});
    EXPECT_DOUBLE_EQ(rate.vorticity[1], 2);
    EXPECT_DOUBLE_EQ(rate.induction[0], -3);
}

TEST(CompressibleMHD, MagneticEnergyIsRemovedBeforeThermodynamics)
{
    similie::physics::fluid::NavierStokesFourier<CaloricClosure> const
            fluid(CaloricClosure(), 1, 0);
    similie::physics::mhd::IdealCompressibleMHD<CaloricClosure> const
            ideal(fluid, similie::physics::magnetodynamics::Magnetodynamics(1));
    // rho=1, u=(1,0,0), thermal energy=2.5, magnetic energy=2.
    similie::physics::mhd::IdealCompressibleMHD<CaloricClosure>::ConservativeVariables const
            state {1, {1, 0, 0}, 5, {0, 2, 0}};
    EXPECT_DOUBLE_EQ(ideal.fluid_state(state).total_energy, 3);
    EXPECT_NEAR(ideal.pressure(state), 1, 1e-14);
    similie::physics::mhd::IdealCompressibleMHD<CaloricClosure>::Flux const flux
            = ideal.flux(state, {}, {}, {}, 0);
    EXPECT_DOUBLE_EQ(flux.conservation.mass, 1);
    EXPECT_NEAR(flux.conservation.momentum[0], 4, 1e-14);
    EXPECT_NEAR(flux.conservation.total_energy, 8, 1e-14);
    EXPECT_DOUBLE_EQ(flux.electric_field[2], -2);
    EXPECT_NEAR(ideal.fast_magnetosonic_speed_squared(state, 0), 5.4, 1e-14);
    similie::physics::mhd::IdealCompressibleMHD<CaloricClosure>::ConservativeVariables const rate
            = ideal({1, {2, 3, 4}, 5}, {6, 7, 8});
    EXPECT_DOUBLE_EQ(rate.total_energy, -5);
    EXPECT_DOUBLE_EQ(rate.induction[0], -6);
}

TEST(ResistiveCompressibleMHD, PoyntingFluxAndJouleHeatingAreNotDoubleCounted)
{
    similie::physics::fluid::NavierStokesFourier<CaloricClosure> const
            fluid(CaloricClosure(), 1, 0);
    similie::physics::mhd::ResistiveCompressibleMHD<CaloricClosure> const
            resistive(fluid, similie::physics::magnetodynamics::ResistiveMagnetodynamics(1, 0.5));
    similie::physics::mhd::ResistiveCompressibleMHD<CaloricClosure>::ConservativeVariables const
            state {1, {1, 0, 0}, 5, {0, 2, 0}};
    std::array<double, 3> const current {0, 0, 2};
    similie::physics::mhd::ResistiveCompressibleMHD<CaloricClosure>::Flux const flux
            = resistive.flux(state, {}, {}, current, 0);
    EXPECT_DOUBLE_EQ(flux.electric_field[2], -1);
    EXPECT_NEAR(flux.conservation.total_energy, 6, 1e-14);
    EXPECT_DOUBLE_EQ(resistive.ohmic_heating(current), 2);
    EXPECT_DOUBLE_EQ(
            similie::physics::mhd::
                    fluid_energy_source(resistive.get<1>(), {1, 0, 0}, state.induction, current),
            -2);
    EXPECT_DOUBLE_EQ(resistive({0, {}, 0}, {}).total_energy, 0);
}
