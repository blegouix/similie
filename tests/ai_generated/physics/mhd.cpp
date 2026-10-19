// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#include <array>

#include <gtest/gtest.h>
#include <similie/physics/coupled_fluid_electromagnetism.hpp>
#include <similie/physics/fluid/incompressible_navier_stokes.hpp>
#include <similie/physics/fluid/navier_stokes_fourier.hpp>
#include <similie/physics/magnetodynamics/resistive_magnetodynamics.hpp>

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
    using Model = similie::physics::CoupledFluidElectromagnetism<
            similie::physics::fluid::IncompressibleNavierStokes,
            similie::physics::magnetodynamics::ResistiveMagnetodynamics>;
    Model const
            model(similie::physics::fluid::IncompressibleNavierStokes(2, 0),
                  similie::physics::magnetodynamics::ResistiveMagnetodynamics(1, 0.5));
    Model::State const state {{}, {{1, 0, 0}}};
    std::array<double, 3> const current {0, 0, 2}, velocity {0, 1, 0};
    EXPECT_DOUBLE_EQ(model.transport(state, velocity, current)[1], 1);
    EXPECT_DOUBLE_EQ(
            model.get<1>()
                    .ohm_electric_field(velocity, state.electromagnetic.induction, current)[2],
            2);
    EXPECT_DOUBLE_EQ(model.get<1>().ohmic_heating(current), 2);
    Model::State const rate = model({1, 2, 3}, {}, 0, {}, {3, 2, 1}, current);
    EXPECT_DOUBLE_EQ(rate.fluid.vorticity[1], 2);
    EXPECT_DOUBLE_EQ(rate.electromagnetic.induction[0], -3);
}

TEST(CompressibleMHD, MagneticEnergyIsRemovedBeforeThermodynamics)
{
    using Fluid = similie::physics::fluid::NavierStokesFourier<CaloricClosure>;
    using Model = similie::physics::CoupledFluidElectromagnetism<
            Fluid,
            similie::physics::magnetodynamics::Magnetodynamics,
            similie::physics::EnergyConvention::FluidAndField>;
    Model const
            ideal(Fluid(CaloricClosure(), 1, 0),
                  similie::physics::magnetodynamics::Magnetodynamics(1));
    Model::State const state {{1, {1, 0, 0}, 5}, {{0, 2, 0}}};
    EXPECT_DOUBLE_EQ(ideal.fluid_state(state).total_energy, 3);
    EXPECT_NEAR(ideal.get<0>().pressure(ideal.fluid_state(state)), 1, 1e-14);
    Fluid::Flux const flux = ideal.flux(state, {}, {}, 0, {});
    EXPECT_DOUBLE_EQ(flux.mass, 1);
    EXPECT_NEAR(flux.total_energy, 8, 1e-14);
    // Momentum is fluid momentum with a Lorentz source. Magnetic stress is
    // available from the sector for callers choosing conservative momentum.
    EXPECT_NEAR(flux.momentum[0], 2, 1e-14);
    EXPECT_DOUBLE_EQ(ideal.get<1>().momentum_flux(state.electromagnetic.induction)[0][0], 2);
    Model::State const rate = ideal(state, Fluid::Flux {1, {2, 3, 4}, 5}, 0, {}, {6, 7, 8}, {});
    EXPECT_DOUBLE_EQ(rate.fluid.total_energy, -5);
    EXPECT_DOUBLE_EQ(rate.electromagnetic.induction[0], -6);
}

TEST(ResistiveCompressibleMHD, PoyntingFluxAndJouleHeatingAreNotDoubleCounted)
{
    using Fluid = similie::physics::fluid::NavierStokesFourier<CaloricClosure>;
    using Model = similie::physics::CoupledFluidElectromagnetism<
            Fluid,
            similie::physics::magnetodynamics::ResistiveMagnetodynamics,
            similie::physics::EnergyConvention::FluidAndField>;
    Model const resistive(
            Fluid(CaloricClosure(), 1, 0),
            similie::physics::magnetodynamics::ResistiveMagnetodynamics(1, 0.5));
    Model::State const state {{1, {1, 0, 0}, 5}, {{0, 2, 0}}};
    std::array<double, 3> const current {0, 0, 2};
    EXPECT_NEAR(resistive.flux(state, {}, {}, 0, current).total_energy, 6, 1e-14);
    EXPECT_DOUBLE_EQ(resistive.get<1>().ohmic_heating(current), 2);
    EXPECT_DOUBLE_EQ(resistive.coupling_sources(state, current, {1, 0, 0}).fluid_energy_source, -2);
    Model::State const rate = resistive(state, Fluid::Flux {0, {}, 0}, 0, {}, {}, current);
    EXPECT_DOUBLE_EQ(rate.fluid.total_energy, 0);
    EXPECT_DOUBLE_EQ(rate.fluid.momentum[0], -4);
}
