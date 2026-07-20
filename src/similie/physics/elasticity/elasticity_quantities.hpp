// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

#include <cmath>
#include <cstddef>
#include <type_traits>

#include <Kokkos_Core.hpp>

namespace similie::physics::elasticity {

template <std::size_t I, std::size_t J>
struct StrainTensorIndex
{
    static constexpr std::size_t FIRST = I;
    static constexpr std::size_t SECOND = J;
};

using StrainXX = StrainTensorIndex<0, 0>;
using StrainXY = StrainTensorIndex<0, 1>;
using StrainYY = StrainTensorIndex<1, 1>;

struct SmallStrain2D
{
    double xx = 0.0;
    double yy = 0.0;
    double xy = 0.0;

    template <class Index>
    [[nodiscard]] KOKKOS_FUNCTION constexpr double get() const
    {
        if constexpr (std::is_same_v<Index, StrainXX>) {
            return xx;
        } else if constexpr (std::is_same_v<Index, StrainXY>) {
            return xy;
        } else if constexpr (std::is_same_v<Index, StrainYY>) {
            return yy;
        } else {
            static_assert(
                    std::is_same_v<Index, StrainXX> || std::is_same_v<Index, StrainYY>
                            || std::is_same_v<Index, StrainXY>,
                    "unsupported elasticity strain component index");
        }
    }
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

} // namespace similie::physics::elasticity
