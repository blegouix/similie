// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

#include <cmath>
#include <cstddef>
#include <type_traits>

#include <ddc/ddc.hpp>

#include <similie/exterior/covariant_derivative.hpp>

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

struct Strain2D
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

struct DisplacementToStrain
{
    [[nodiscard]] KOKKOS_FUNCTION static constexpr Strain2D from_gradient(
            double dux_dx,
            double duy_dy,
            double dux_dy,
            double duy_dx)
    {
        return {
                .xx = dux_dx,
                .yy = duy_dy,
                .xy = 0.5 * (dux_dy + duy_dx),
        };
    }

    template <
            class StrainIndex,
            class DisplacementComponent,
            class... SpatialIndex,
            class Elem,
            class PositionType>
    [[nodiscard]] KOKKOS_FUNCTION static auto forward_value(Elem elem, PositionType position)
    {
        using SpatialIndexSeq = ddc::detail::TypeSeq<SpatialIndex...>;
        using X = ddc::type_seq_element_t<0, SpatialIndexSeq>;
        using Y = ddc::type_seq_element_t<1, SpatialIndexSeq>;
        using Derivative = sil::exterior::CovariantDerivative<SpatialIndex...>;

        if constexpr (std::is_same_v<StrainIndex, StrainXX>) {
            return Derivative::template value<X, X, DisplacementComponent>(elem, position);
        } else if constexpr (std::is_same_v<StrainIndex, StrainYY>) {
            return Derivative::template value<Y, Y, DisplacementComponent>(elem, position);
        } else if constexpr (std::is_same_v<StrainIndex, StrainXY>) {
            if constexpr (std::is_same_v<DisplacementComponent, X>) {
                auto stencil
                        = Derivative::template value<X, Y, DisplacementComponent>(elem, position);
                stencil *= 0.5;
                return stencil;
            } else {
                auto stencil
                        = Derivative::template value<Y, X, DisplacementComponent>(elem, position);
                stencil *= 0.5;
                return stencil;
            }
        } else {
            static_assert(
                    std::is_same_v<StrainIndex, StrainXX> || std::is_same_v<StrainIndex, StrainYY>
                            || std::is_same_v<StrainIndex, StrainXY>,
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
