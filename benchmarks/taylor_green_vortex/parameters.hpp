// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#pragma once
#include <similie/physics/fluid/navier_stokes_fourier.hpp>

#include "structured_grid.hpp"
#include "taylor_green_ideal_gas.hpp"

namespace taylor_green_vortex {
using CompressiblePhysics = similie::physics::fluid::NavierStokesFourier<IdealGas>;
/** Fixed workshop reference values. N is the only numerical resolution input.
 * \important This documentation is fully AI-generated.
 */
struct Reference
{
    static constexpr double gamma = 1.4;
    static constexpr double prandtl = 0.71;
    static constexpr double viscosity = 1.0 / 1600;
    static constexpr double gas_constant = 1;
    static constexpr double reference_pressure = 1.0 / (gamma * 0.1 * 0.1);
    KOKKOS_FUNCTION static CompressiblePhysics model()
    {
        return CompressiblePhysics(IdealGas(gamma, gas_constant), prandtl, viscosity);
    }
    KOKKOS_FUNCTION static double conductivity()
    {
        return model().conductivity({1, {}, reference_pressure / (gamma - 1)});
    }
    KOKKOS_FUNCTION static CompressiblePhysics::State state(Field const& field, int i)
    {
        return {field(0, i), {field(1, i), field(2, i), field(3, i)}, field(4, i)};
    }
    KOKKOS_FUNCTION static double pressure(Field const& state, int i)
    {
        return model().pressure(Reference::state(state, i));
    }
};
} // namespace taylor_green_vortex
