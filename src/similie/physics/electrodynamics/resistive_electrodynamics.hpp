// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#pragma once
#include <similie/physics/electrodynamics/electrodynamics.hpp>

namespace similie::physics::electrodynamics {
/** Electric-only conduction: j=E/eta, J=charge*u+j, Q=E.E/eta>=0.
 * eta>0 is electrical resistivity; ideal advected-charge EHD has zero
 * conductivity (eta->infinity), rather than the singular eta=0 limit.
 * \important This operator and documentation is fully AI-generated.
 */
class ResistiveElectrodynamics : public Electrodynamics
{
public:
    double const resistivity;
    KOKKOS_FUNCTION ResistiveElectrodynamics(
            double electric_permittivity,
            double electrical_resistivity)
        : Electrodynamics(electric_permittivity)
        , resistivity(electrical_resistivity)
    {
        KOKKOS_ASSERT(resistivity > 0);
    }
    KOKKOS_FUNCTION Vector conduction_current(Vector const& displacement) const
    {
        Vector result = electric_field(displacement);
        for (double& value : result)
            value /= resistivity;
        return result;
    }
    KOKKOS_FUNCTION Vector
    current(double charge_density, Vector const& velocity, Vector const& displacement) const
    {
        Vector result = conduction_current(displacement);
        for (int a = 0; a < 3; ++a)
            result[a] += charge_density * velocity[a];
        return result;
    }
    KOKKOS_FUNCTION double ohmic_heating(Vector const& displacement) const
    {
        Vector const electric = electric_field(displacement);
        return dot(electric, electric) / resistivity;
    }
    KOKKOS_FUNCTION double charge_relaxation_rate() const
    {
        return 1 / (resistivity * permittivity);
    }
};

static_assert(PhysicsConcept<ResistiveElectrodynamics>);
} // namespace similie::physics::electrodynamics
