// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#pragma once
#include <similie/physics/fluid/navier_stokes_fourier.hpp>

#include "structured_grid.hpp"

namespace taylor_green_vortex {
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
    KOKKOS_FUNCTION static similie::physics::fluid::NavierStokesFourier model()
    {
        return similie::physics::fluid::
                NavierStokesFourier(gamma, prandtl, viscosity, gas_constant);
    }
    KOKKOS_FUNCTION static double conductivity()
    {
        return model().conductivity();
    }
    KOKKOS_FUNCTION static double pressure(Field state, int i)
    {
        return model().pressure(
                {state(0, i), {state(1, i), state(2, i), state(3, i)}, state(4, i)});
    }
};
} // namespace taylor_green_vortex
