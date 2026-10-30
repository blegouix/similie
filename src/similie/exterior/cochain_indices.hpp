// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#pragma once

#include <array>
#include <cstddef>
#include <tuple>
#include <type_traits>

#include <ddc/ddc.hpp>

namespace sil::exterior::detail {

template <class ElemType>
struct FlatNaturalElemRank;

template <class... Tags>
struct FlatNaturalElemRank<ddc::DiscreteElement<Tags...>>
{
    static constexpr std::size_t value = (FlatNaturalElemRank<Tags>::value + ... + 0);
};

template <class Tag>
struct FlatNaturalElemRank
{
    static constexpr std::size_t value = 1;
};

template <class ElemType, std::size_t N>
KOKKOS_FUNCTION void copy_flat_natural_elem_ids(
        std::array<std::size_t, N>& ids,
        std::size_t& offset,
        ElemType const& elem)
{
    auto const entries = ddc::detail::array(elem);
    if constexpr (std::tuple_size_v<decltype(entries)> == 0) {
        return;
    } else {
        using EntryType = std::remove_cvref_t<decltype(entries[0])>;
        if constexpr (std::is_same_v<EntryType, std::size_t>) {
            for (auto const& entry : entries) {
                ids[offset++] = entry;
            }
        } else {
            for (auto const& entry : entries) {
                copy_flat_natural_elem_ids(ids, offset, entry);
            }
        }
    }
}

template <class ElemType>
KOKKOS_FUNCTION auto flat_natural_elem_ids(ElemType const& elem)
{
    std::array<std::size_t, FlatNaturalElemRank<ElemType>::value> ids {};
    std::size_t offset = 0;
    copy_flat_natural_elem_ids(ids, offset, elem);
    return ids;
}

template <class ElemType, std::size_t N>
KOKKOS_FUNCTION void assign_flat_natural_elem_ids(
        ElemType& elem,
        std::array<std::size_t, N> const& ids,
        std::size_t& offset)
{
    auto entries = ddc::detail::array(elem);
    if constexpr (std::tuple_size_v<decltype(entries)> == 0) {
        return;
    } else {
        using EntryType = std::remove_cvref_t<decltype(entries[0])>;
        if constexpr (std::is_same_v<EntryType, std::size_t>) {
            for (auto& entry : entries) {
                entry = ids[offset++];
            }
            ddc::detail::array(elem) = entries;
        } else {
            for (auto& entry : entries) {
                assign_flat_natural_elem_ids(entry, ids, offset);
            }
            ddc::detail::array(elem) = entries;
        }
    }
}

template <class ElemType, std::size_t N>
KOKKOS_FUNCTION ElemType natural_elem_from_flat_ids(std::array<std::size_t, N> const& ids)
{
    static_assert(
            FlatNaturalElemRank<ElemType>::value == N,
            "Flat natural element ids must match the element rank.");
    ElemType elem;
    std::size_t offset = 0;
    assign_flat_natural_elem_ids(elem, ids, offset);
    return elem;
}

} // namespace sil::exterior::detail
