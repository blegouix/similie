// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#pragma once

#include <array>

#include <Kokkos_Core.hpp>

namespace similie::physics::coupling {

struct ElectromagneticCouplingState
{
    double charge_density {};
    std::array<double, 3> electric_field {};
    std::array<double, 3> induction {};
    std::array<double, 3> current {};
};

struct ElectromagneticExchange
{
    std::array<double, 3> momentum_source {};
    double fluid_energy_source {};
    double field_energy_source {};

    KOKKOS_FUNCTION ElectromagneticExchange& operator+=(ElectromagneticExchange const& other)
    {
        for (int i = 0; i < 3; ++i)
            momentum_source[i] += other.momentum_source[i];
        fluid_energy_source += other.fluid_energy_source;
        field_energy_source += other.field_energy_source;
        return *this;
    }
};

} // namespace similie::physics::coupling
