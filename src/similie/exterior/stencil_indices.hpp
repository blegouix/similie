// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#pragma once

#include <array>
#include <cstddef>

#include <Kokkos_Macros.hpp>

namespace sil::exterior {

template <std::size_t Dimensions>
struct StencilIndex
{
    std::size_t output_component;
    std::array<int, Dimensions> spatial_offset;
    std::size_t input_component;
};

namespace detail {

// One incidence changes one coordinate. Two incidences have Manhattan radius two.
// Both signs are retained to cover reflected physical boundary closures as well.
template <class Index, std::size_t Radius>
struct StencilShape
{
    static_assert(Radius <= 2);
    static constexpr std::size_t size = Radius == 0   ? 1
                                        : Radius == 1 ? 1 + 2 * Index::size()
                                                      : 1 + 2 * Index::size() * (Index::size() + 1);
    KOKKOS_FUNCTION static constexpr std::array<int, Index::size()> offset(std::size_t point)
    {
        std::array<int, Index::size()> result {};
        if (point == 0)
            return result;
        --point;
        if (point < 2 * Index::size()) {
            result[point / 2] = point % 2 == 0 ? -1 : 1;
            return result;
        }
        point -= 2 * Index::size();
        if constexpr (Radius == 2) {
            if (point < 2 * Index::size()) {
                result[point / 2] = point % 2 == 0 ? -2 : 2;
                return result;
            }
            point -= 2 * Index::size();
            for (std::size_t first = 0; first < Index::size(); ++first)
                for (std::size_t second = first + 1; second < Index::size(); ++second) {
                    if (point < 4) {
                        result[first] = point % 2 == 0 ? -1 : 1;
                        result[second] = point / 2 == 0 ? -1 : 1;
                        return result;
                    }
                    point -= 4;
                }
        }
        return result;
    }
};


} // namespace detail

/** Decode the standard ordering: output component, spatial point, input component.
 * The input component varies fastest. The spatial ordering is StencilShape::offset.
 * \important This documentation is fully AI-generated.
 */
template <class Index, class Source, class Target, std::size_t Radius>
KOKKOS_FUNCTION constexpr StencilIndex<Index::size()> stencil_reindex(std::size_t flat)
{
    return {flat / (detail::StencilShape<Index, Radius>::size * Source::mem_size()),
            detail::StencilShape<Index, Radius>::offset(
                    (flat / Source::mem_size()) % detail::StencilShape<Index, Radius>::size),
            flat % Source::mem_size()};
}

template <class Index, class Source, class Target, std::size_t Radius>
KOKKOS_FUNCTION constexpr std::size_t stencil_flatten(StencilIndex<Index::size()> index)
{
    for (std::size_t point = 0; point < detail::StencilShape<Index, Radius>::size; ++point)
        if (detail::StencilShape<Index, Radius>::offset(point) == index.spatial_offset)
            return (index.output_component * detail::StencilShape<Index, Radius>::size + point)
                           * Source::mem_size()
                   + index.input_component;
    return Target::mem_size() * detail::StencilShape<Index, Radius>::size * Source::mem_size();
}

namespace detail {

template <class Index, std::size_t Count>
KOKKOS_FUNCTION constexpr std::size_t stencil_component_index(
        std::array<std::size_t, Count> const& ids)
{
    if constexpr (Index::rank() == 0)
        return 0;
    else if constexpr (Index::rank() == 1)
        return ids[0];
    else
        return Index::mem_id(ids);
}

} // namespace detail
} // namespace sil::exterior
