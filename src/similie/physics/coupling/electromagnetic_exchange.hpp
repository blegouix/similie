// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#pragma once

#include <array>

#include <similie/exterior/form.hpp>

#include <Kokkos_Core.hpp>

namespace similie::physics::coupling {

struct ElectromagneticCouplingState
{
    sil::exterior::Form<double> charge_density {};
    sil::exterior::Form<std::array<double, 3>> electric_field {};
    sil::exterior::Form<std::array<double, 3>> induction {};
    sil::exterior::Form<std::array<double, 3>> current {};
};

struct ElectromagneticExchange
{
    sil::exterior::Form<std::array<double, 3>> momentum_source {};
    sil::exterior::Form<double> fluid_energy_source {};
    sil::exterior::Form<double> field_energy_source {};

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
