// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#pragma once

#include <array>
#include <span>

#include <similie/physics/magnetodynamics/magnetodynamics_hamiltonian.hpp>

#include <Kokkos_Core.hpp>

namespace similie::physics::magnetodynamics {
/** Magnetic-only Hamiltonian sector H(B)=B.B/(2 mu), H_field=H_B.
 * J=curl H_field, E=-u cross B, B_t=-curl E. E is an Ohm-law closure,
 * not an independent electric state. Compatible native exterior d preserves
 * closed magnetic induction. No electric displacement or electric energy.
 * \important This operator and documentation is fully AI-generated.
 */
class Magnetodynamics
{
    struct Component
    {
    };
    MagnetodynamicsHamiltonian const m_hamiltonian;

public:
    using Vector = std::array<double, 3>;
    using Tensor = std::array<std::array<double, 3>, 3>;
    double const permeability;
    KOKKOS_FUNCTION explicit Magnetodynamics(double magnetic_permeability)
        : m_hamiltonian(magnetic_permeability)
        , permeability(magnetic_permeability)
    {
        KOKKOS_ASSERT(permeability > 0);
    }
    KOKKOS_FUNCTION static Vector cross(Vector const& left, Vector const& right)
    {
        Vector result {};
        for (int a = 0; a < 3; ++a) {
            int const b = (a + 1) % 3, c = (a + 2) % 3;
            result[a] = left[b] * right[c] - left[c] * right[b];
        }
        return result;
    }
    KOKKOS_FUNCTION static double dot(Vector const& left, Vector const& right)
    {
        double result = 0;
        for (int a = 0; a < 3; ++a)
            result += left[a] * right[a];
        return result;
    }
    KOKKOS_FUNCTION double hamiltonian(Vector const& induction) const
    {
        return m_hamiltonian.hamiltonian(0.0, std::span<double const, 3>(induction));
    }
    KOKKOS_FUNCTION double magnetic_energy(Vector const& induction) const
    {
        return hamiltonian(induction);
    }
    KOKKOS_FUNCTION Vector magnetic_field(Vector const& induction) const
    {
        Vector result {};
        for (int a = 0; a < 3; ++a)
            result[a] = m_hamiltonian.template dhamiltonian_dmoments<Component>(induction[a]);
        return result;
    }
    KOKKOS_FUNCTION Vector operator()(Vector const& curl_electric_field) const
    {
        return induction_rate(curl_electric_field);
    }
    KOKKOS_FUNCTION Vector ohm_electric_field(
            Vector const& velocity,
            Vector const& induction,
            [[maybe_unused]] Vector const& current) const
    {
        Vector result = cross(velocity, induction);
        for (double& value : result)
            value = -value;
        return result;
    }
    KOKKOS_FUNCTION Vector induction_rate(Vector const& curl_electric_field) const
    {
        Vector result {};
        for (int a = 0; a < 3; ++a)
            result[a] = -curl_electric_field[a];
        return result;
    }
    KOKKOS_FUNCTION Vector lorentz_force(Vector const& current, Vector const& induction) const
    {
        return cross(current, induction);
    }
    KOKKOS_FUNCTION Tensor momentum_flux(Vector const& induction) const
    {
        Vector const field = magnetic_field(induction);
        double const energy = magnetic_energy(induction);
        Tensor result {};
        for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b)
                result[a][b] = (a == b ? energy : 0) - induction[a] * field[b];
        return result;
    }
    KOKKOS_FUNCTION Vector poynting_flux(Vector const& electric, Vector const& induction) const
    {
        return cross(electric, magnetic_field(induction));
    }
    KOKKOS_FUNCTION double ohmic_heating([[maybe_unused]] Vector const& current) const
    {
        return 0;
    }
    KOKKOS_FUNCTION double magnetic_diffusivity() const
    {
        return 0;
    }
};
} // namespace similie::physics::magnetodynamics
