// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#pragma once
#include "magnetodynamics.hpp"

namespace similie::physics::magnetodynamics {
/** Dissipative Ohm-law extension of the Hamiltonian magnetic sector.
 * E=-u cross B+eta J, J=curl H, Q_J=eta J.J, magnetic diffusivity=eta/mu.
 * eta is electrical resistivity, not magnetic diffusivity. Faraday must act
 * on the entire E, including eta J; this also supports locally varying eta
 * represented by separate pointwise physics objects.
 * \important This operator and documentation is fully AI-generated.
 */
class ResistiveMagnetodynamics : public Magnetodynamics
{
public:
    double const resistivity;
    KOKKOS_FUNCTION ResistiveMagnetodynamics(
            double magnetic_permeability,
            double electrical_resistivity)
        : Magnetodynamics(magnetic_permeability)
        , resistivity(electrical_resistivity)
    {
        KOKKOS_ASSERT(resistivity >= 0);
    }
    KOKKOS_FUNCTION Vector
    ohm_electric_field(Vector const& velocity, Vector const& induction, Vector const& current) const
    {
        Vector result = Magnetodynamics::ohm_electric_field(velocity, induction, current);
        for (int a = 0; a < 3; ++a)
            result[a] += resistivity * current[a];
        return result;
    }
    KOKKOS_FUNCTION double ohmic_heating(Vector const& current) const
    {
        return resistivity * dot(current, current);
    }
    KOKKOS_FUNCTION double magnetic_diffusivity() const
    {
        return resistivity / permeability;
    }
};

static_assert(PhysicsConcept<ResistiveMagnetodynamics>);
} // namespace similie::physics::magnetodynamics
