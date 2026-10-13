// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#pragma once
#include "electromagnetism.hpp"

namespace similie::physics::electromagnetism {
/** Full Maxwell Ohm law: j=(E+u cross B)/eta, J=charge*u+j.
 * Heat is j.(E+u cross B), not eta times the square of advective total current.
 * eta>0; nonconducting ideal FMHD is the eta->infinity limit.
 * \important This operator and documentation is fully AI-generated.
 */
class ResistiveElectromagnetism : public Electromagnetism
{
public:
    double const resistivity;
    KOKKOS_FUNCTION ResistiveElectromagnetism(
            double electric_permittivity,
            double magnetic_permeability,
            double electrical_resistivity)
        : Electromagnetism(electric_permittivity, magnetic_permeability)
        , resistivity(electrical_resistivity)
    {
        KOKKOS_ASSERT(resistivity > 0);
    }
    KOKKOS_FUNCTION Vector conduction_current(
            Vector const& velocity,
            Vector const& displacement,
            Vector const& induction) const
    {
        Vector result = rest_frame_electric_field(velocity, displacement, induction);
        for (double& value : result)
            value /= resistivity;
        return result;
    }
    KOKKOS_FUNCTION Vector
    current(double charge_density,
            Vector const& velocity,
            Vector const& displacement,
            Vector const& induction) const
    {
        Vector result = conduction_current(velocity, displacement, induction);
        for (int a = 0; a < 3; ++a)
            result[a] += charge_density * velocity[a];
        return result;
    }
    KOKKOS_FUNCTION double ohmic_heating(
            Vector const& velocity,
            Vector const& displacement,
            Vector const& induction) const
    {
        Vector const rest_electric = rest_frame_electric_field(velocity, displacement, induction);
        return dot(rest_electric, rest_electric) / resistivity;
    }
    KOKKOS_FUNCTION double charge_relaxation_rate() const
    {
        return 1 / (resistivity * get<0>().permittivity);
    }
};
} // namespace similie::physics::electromagnetism
