// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#pragma once

#include <array>
#include <span>

#include <similie/physics/magnetodynamics/magnetodynamics_hamiltonian.hpp>

#include <Kokkos_Core.hpp>

namespace similie::physics::magnetodynamics {
/** Hamiltonian electromagnetism and its quasistatic ideal-MHD magnetic sector.
 * H(D,B)=D.D/(2 epsilon)+B.B/(2 mu), E=H_D, H=H_B.
 * D_t=curl H-J, B_t=-curl E. In MHD, J=curl H and E=-u cross B;
 * displacement current and electric energy are excluded from the MHD limit.
 * Spatial curls/divergences must be evaluated with the compatible exterior
 * complex. Storing B as a closed two-cochain preserves dB=0 under Faraday dE.
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
    struct MaxwellVariables
    {
        Vector displacement;
        Vector induction;
    };
    double const permeability, permittivity;
    KOKKOS_FUNCTION explicit Magnetodynamics(
            double magnetic_permeability,
            double electric_permittivity = 1.0)
        : m_hamiltonian(electric_permittivity, magnetic_permeability)
        , permeability(magnetic_permeability)
        , permittivity(electric_permittivity)
    {
        KOKKOS_ASSERT(permeability > 0 && permittivity > 0);
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
    KOKKOS_FUNCTION double hamiltonian(MaxwellVariables const& state) const
    {
        return m_hamiltonian.hamiltonian(
                std::span<double const, 3>(state.displacement),
                std::span<double const, 3>(state.induction));
    }
    KOKKOS_FUNCTION double magnetic_energy(Vector const& induction) const
    {
        return hamiltonian({{}, induction});
    }
    KOKKOS_FUNCTION Vector magnetic_field(Vector const& induction) const
    {
        Vector result {};
        for (int a = 0; a < 3; ++a)
            result[a] = m_hamiltonian.template dhamiltonian_dmoments<Component>(induction[a]);
        return result;
    }
    KOKKOS_FUNCTION Vector electric_field(Vector const& displacement) const
    {
        Vector result {};
        for (int a = 0; a < 3; ++a)
            result[a] = m_hamiltonian.dhamiltonian_ddisplacement(displacement[a]);
        return result;
    }
    KOKKOS_FUNCTION MaxwellVariables maxwell_rate(
            Vector const& curl_magnetic_field,
            Vector const& curl_electric_field,
            Vector const& current) const
    {
        MaxwellVariables result {};
        for (int a = 0; a < 3; ++a) {
            result.displacement[a] = curl_magnetic_field[a] - current[a];
            result.induction[a] = -curl_electric_field[a];
        }
        return result;
    }
    KOKKOS_FUNCTION MaxwellVariables operator()(
            Vector const& curl_magnetic_field,
            Vector const& curl_electric_field,
            Vector const& current) const
    {
        return maxwell_rate(curl_magnetic_field, curl_electric_field, current);
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
