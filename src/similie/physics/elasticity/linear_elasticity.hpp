// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

#include <array>
#include <cmath>

#include <Kokkos_Core.hpp>

namespace similie::physics::elasticity {

struct PlaneStressMaterial
{
    double young_modulus = 1.0;
    double poisson_ratio = 0.3;

    [[nodiscard]] KOKKOS_FUNCTION constexpr double lambda() const
    {
        return young_modulus * poisson_ratio / (1.0 - poisson_ratio * poisson_ratio);
    }

    [[nodiscard]] KOKKOS_FUNCTION constexpr double mu() const
    {
        return young_modulus / (2.0 * (1.0 + poisson_ratio));
    }

    [[nodiscard]] KOKKOS_FUNCTION constexpr double c11() const
    {
        return young_modulus / (1.0 - poisson_ratio * poisson_ratio);
    }

    [[nodiscard]] KOKKOS_FUNCTION constexpr double c12() const
    {
        return young_modulus * poisson_ratio / (1.0 - poisson_ratio * poisson_ratio);
    }
};

struct SmallStrain2D
{
    double xx = 0.0;
    double yy = 0.0;
    double xy = 0.0;
};

struct CauchyStress2D
{
    double xx = 0.0;
    double yy = 0.0;
    double xy = 0.0;

    [[nodiscard]] KOKKOS_FUNCTION double von_mises() const
    {
        return Kokkos::sqrt(xx * xx - xx * yy + yy * yy + 3.0 * xy * xy);
    }
};

[[nodiscard]] KOKKOS_FUNCTION constexpr CauchyStress2D hooke_plane_stress(
        PlaneStressMaterial material,
        SmallStrain2D strain)
{
    return {
            material.c11() * strain.xx + material.c12() * strain.yy,
            material.c12() * strain.xx + material.c11() * strain.yy,
            2.0 * material.mu() * strain.xy,
    };
}

[[nodiscard]] KOKKOS_FUNCTION constexpr double strain_energy_density_plane_stress(
        PlaneStressMaterial material,
        SmallStrain2D strain)
{
    CauchyStress2D const stress = hooke_plane_stress(material, strain);
    return 0.5 * (stress.xx * strain.xx + stress.yy * strain.yy + 2.0 * stress.xy * strain.xy);
}

} // namespace similie::physics::elasticity
