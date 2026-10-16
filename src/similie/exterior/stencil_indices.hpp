// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#pragma once

#include <array>
#include <cstddef>
#include <type_traits>

#include <Kokkos_Macros.hpp>

namespace sil::exterior {

/** The general pattern permits curved geometry and arbitrary supported closures.
 * CartesianPeriodicStencil requires a uniform orthogonal grid, identity physical
 * metric and periodic closures. Its Laplacian is componentwise seven-point in 3D.
 * \important This operator and documentation is fully AI-generated.
 */
struct GeneralStencil
{
};
struct CartesianPeriodicStencil
{
};

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


template <std::size_t Size>
KOKKOS_FUNCTION constexpr std::array<std::size_t, Size> dense_stencil_indices()
{
    std::array<std::size_t, Size> result {};
    for (std::size_t i = 0; i < Size; ++i)
        result[i] = i;
    return result;
}

template <auto Mask>
KOKKOS_FUNCTION constexpr auto nonzero_stencil_indices()
{
    constexpr std::size_t count = [] {
        std::size_t result = 0;
        for (bool entry : Mask)
            result += entry;
        return result;
    }();
    std::array<std::size_t, count> result {};
    std::size_t entry = 0;
    for (std::size_t i = 0; i < Mask.size(); ++i)
        if (Mask[i])
            result[entry++] = i;
    return result;
}

template <class T>
struct IsStencilIndices : std::false_type
{
};
template <std::size_t Size>
struct IsStencilIndices<std::array<std::size_t, Size>> : std::true_type
{
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

template <class Operator, class Source, class Target, std::size_t SpatialPoints>
KOKKOS_FUNCTION constexpr std::array<std::size_t, Target::mem_size() + 1> stencil_component_rows()
{
    std::array<std::size_t, Target::mem_size() + 1> rows {};
    for (std::size_t entry : Operator::stencil_nonzero_indices())
        ++rows[entry / (SpatialPoints * Source::mem_size()) + 1];
    for (std::size_t row = 0; row < Target::mem_size(); ++row)
        rows[row + 1] += rows[row];
    return rows;
}

template <class Operator, class Source, class Target, class Index, std::size_t Radius>
KOKKOS_FUNCTION constexpr std::array<std::size_t, Target::mem_size() + 1> stencil_row_offsets()
{
    return stencil_component_rows<Operator, Source, Target, StencilShape<Index, Radius>::size>();
}

template <class Index, class Source, class Target, bool Transposed>
KOKKOS_FUNCTION constexpr auto incidence_nonzero_indices()
{
    constexpr auto mask = [] {
        std::array<bool, Target::mem_size() * StencilShape<Index, 1>::size * Source::mem_size()>
                result {};
        for (std::size_t flat = 0; flat < result.size(); ++flat) {
            auto const entry = stencil_reindex<Index, Source, Target, 1>(flat);
            auto const source = Source::mem_id_to_canonical_natural_ids(entry.input_component);
            auto const target = Target::mem_id_to_canonical_natural_ids(entry.output_component);
            bool subset = true;
            for (std::size_t axis : source) {
                bool found = false;
                for (std::size_t other : target)
                    found |= axis == other;
                subset &= found;
            }
            std::size_t added = 0;
            for (std::size_t axis : target) {
                bool found = false;
                for (std::size_t other : source)
                    found |= axis == other;
                if (!found)
                    added = axis;
            }
            bool center = true, neighbor = true;
            for (std::size_t axis = 0; axis < Index::size(); ++axis) {
                center &= entry.spatial_offset[axis] == 0;
                neighbor &= entry.spatial_offset[axis]
                            == (axis == added ? (Transposed ? -1 : 1) : 0);
            }
            result[flat] = subset && (center || neighbor);
        }
        return result;
    }();
    return nonzero_stencil_indices<mask>();
}

template <class Operator, std::size_t FullSize>
constexpr bool valid_stencil_indices()
{
    constexpr auto indices = Operator::stencil_nonzero_indices();
    for (std::size_t i = 0; i < indices.size(); ++i)
        if (indices[i] >= FullSize || (i && indices[i - 1] >= indices[i]))
            return false;
    return true;
}

template <class Index, class Cochain, class Pattern>
KOKKOS_FUNCTION constexpr auto laplacian_nonzero_indices()
{
    constexpr auto mask = [] {
        std::array<bool, Cochain::mem_size() * StencilShape<Index, 2>::size * Cochain::mem_size()>
                result {};
        for (std::size_t i = 0; i < result.size(); ++i) {
            auto const entry = stencil_reindex<Index, Cochain, Cochain, 2>(i);
            if constexpr (std::is_same_v<Pattern, CartesianPeriodicStencil>) {
                std::size_t distance = 0;
                for (int coordinate : entry.spatial_offset)
                    distance += coordinate < 0 ? -coordinate : coordinate;
                result[i] = entry.output_component == entry.input_component && distance <= 1;
            } else {
                static_assert(std::is_same_v<Pattern, GeneralStencil>);
                result[i] = true;
            }
        }
        return result;
    }();
    return nonzero_stencil_indices<mask>();
}

} // namespace detail
} // namespace sil::exterior
