// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#pragma once

#include <array>
#include <cmath>

#include <similie/physics/combine_physics.hpp>
#include <similie/physics/electrodynamics/electrodynamics.hpp>
#include <similie/physics/magnetodynamics/magnetodynamics.hpp>

namespace similie::physics::electromagnetism {
/** Full Maxwell sector, combining the generated electric and magnetic energies.
 * W(D,B)=D.D/(2 epsilon)+B.B/(2 mu), E=W_D, H=W_B.
 * D_t=curl H-J, B_t=-curl E, charge_t=-div J; ideal current=charge*u.
 * The same reduced current must enter charge and displacement updates.
 * \important This operator and documentation is fully AI-generated.
 */
class Electromagnetism
    : public CombinePhysics<electrodynamics::Electrodynamics, magnetodynamics::Magnetodynamics>
{
public:
    using Vector = std::array<double, 3>;
    struct Variables
    {
        double charge_density;
        Vector displacement;
        Vector induction;
    };
    KOKKOS_FUNCTION Electromagnetism(double electric_permittivity, double magnetic_permeability)
        : CombinePhysics<electrodynamics::Electrodynamics, magnetodynamics::Magnetodynamics>(
                  electrodynamics::Electrodynamics(electric_permittivity),
                  magnetodynamics::Magnetodynamics(magnetic_permeability))
    {
    }
    KOKKOS_FUNCTION static Vector cross(Vector const& left, Vector const& right)
    {
        return magnetodynamics::Magnetodynamics::cross(left, right);
    }
    KOKKOS_FUNCTION static double dot(Vector const& left, Vector const& right)
    {
        return electrodynamics::Electrodynamics::dot(left, right);
    }
    KOKKOS_FUNCTION double hamiltonian(Vector const& displacement, Vector const& induction) const
    {
        return get<0>().hamiltonian(displacement) + get<1>().hamiltonian(induction);
    }
    KOKKOS_FUNCTION Vector electric_field(Vector const& displacement) const
    {
        return get<0>().electric_field(displacement);
    }
    KOKKOS_FUNCTION Vector magnetic_field(Vector const& induction) const
    {
        return get<1>().magnetic_field(induction);
    }
    KOKKOS_FUNCTION Vector rest_frame_electric_field(
            Vector const& velocity,
            Vector const& displacement,
            Vector const& induction) const
    {
        Vector result = electric_field(displacement);
        Vector const motional = cross(velocity, induction);
        for (int a = 0; a < 3; ++a)
            result[a] += motional[a];
        return result;
    }
    KOKKOS_FUNCTION Vector conduction_current(
            [[maybe_unused]] Vector const& velocity,
            [[maybe_unused]] Vector const& displacement,
            [[maybe_unused]] Vector const& induction) const
    {
        return {};
    }
    KOKKOS_FUNCTION Vector
    current(double charge_density,
            Vector const& velocity,
            Vector const& displacement,
            [[maybe_unused]] Vector const& induction) const
    {
        return get<0>().current(charge_density, velocity, displacement);
    }
    KOKKOS_FUNCTION Variables operator()(
            double current_divergence,
            Vector const& curl_magnetic_field,
            Vector const& curl_electric_field,
            Vector const& total_current) const
    {
        Variables result {-current_divergence, {}, get<1>()(curl_electric_field)};
        for (int a = 0; a < 3; ++a)
            result.displacement[a] = curl_magnetic_field[a] - total_current[a];
        return result;
    }
    KOKKOS_FUNCTION Vector lorentz_force(
            double charge_density,
            Vector const& displacement,
            Vector const& induction,
            Vector const& total_current) const
    {
        Vector result = get<0>().electric_force(charge_density, displacement);
        Vector const magnetic = get<1>().lorentz_force(total_current, induction);
        for (int a = 0; a < 3; ++a)
            result[a] += magnetic[a];
        return result;
    }
    KOKKOS_FUNCTION Vector poynting_flux(Vector const& displacement, Vector const& induction) const
    {
        return get<1>().poynting_flux(electric_field(displacement), induction);
    }
    KOKKOS_FUNCTION double fluid_energy_source(
            Vector const& displacement,
            Vector const& total_current) const
    {
        return get<0>().fluid_energy_source(displacement, total_current);
    }
    KOKKOS_FUNCTION double ohmic_heating(
            [[maybe_unused]] Vector const& velocity,
            [[maybe_unused]] Vector const& displacement,
            [[maybe_unused]] Vector const& induction) const
    {
        return 0;
    }
    KOKKOS_FUNCTION double wave_speed() const
    {
        return 1 / std::sqrt(get<0>().permittivity * get<1>().permeability);
    }
    KOKKOS_FUNCTION double charge_relaxation_rate() const
    {
        return 0;
    }
    KOKKOS_FUNCTION double gauss_residual(double displacement_divergence, double charge_density)
            const
    {
        return get<0>().gauss_residual(displacement_divergence, charge_density);
    }
};
} // namespace similie::physics::electromagnetism
