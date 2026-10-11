// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#pragma once

#include <array>
#include <cassert>
#include <cstddef>

#include <ddc/ddc.hpp>

#include "type_seq_conversion.hpp"

namespace sil::misc {
/** Absolute logical coordinates with signed storage, including exterior nodes.
 * \important This operator and documentation is fully AI-generated.
 * DDC DiscreteElement stores size_t, so it cannot represent negative coordinates.
 * This element owns signed coordinates; it is neither a displacement nor a DDC
 * storage element. Convert only after a boundary rule resolves negative indices.
 */
template <class... Dimensions>
class SignedDiscreteElement
{
    std::array<std::ptrdiff_t, sizeof...(Dimensions)> m_coordinates {};

    template <class Dimension, class Head, class... Tail>
    KOKKOS_FUNCTION static std::ptrdiff_t coordinate(Head const& head, Tail const&... tail)
    {
        if constexpr (ddc::type_seq_contains_v<ddc::TypeSeq<Dimension>, to_type_seq_t<Head>>)
            return static_cast<std::ptrdiff_t>(head.template uid<Dimension>());
        else
            return coordinate<Dimension>(tail...);
    }

public:
    KOKKOS_DEFAULTED_FUNCTION SignedDiscreteElement() = default;
    template <class... Coordinates>
        requires(
                sizeof...(Coordinates) == sizeof...(Dimensions)
                && (std::is_integral_v<Coordinates> && ...))
    KOKKOS_FUNCTION explicit SignedDiscreteElement(Coordinates... coordinates)
        : m_coordinates {static_cast<std::ptrdiff_t>(coordinates)...}
    {
    }
    template <class... Elements>
        requires(sizeof...(Elements) > 0 && (!std::is_integral_v<Elements> && ...))
    KOKKOS_FUNCTION explicit SignedDiscreteElement(Elements const&... elements)
        : m_coordinates {coordinate<Dimensions>(elements...)...}
    {
    }
    template <class Dimension>
        requires(ddc::type_seq_contains_v<ddc::TypeSeq<Dimension>, ddc::TypeSeq<Dimensions...>>)
    KOKKOS_FUNCTION std::ptrdiff_t& uid()
    {
        return m_coordinates[ddc::type_seq_rank_v<Dimension, ddc::TypeSeq<Dimensions...>>];
    }
    template <class Dimension>
        requires(ddc::type_seq_contains_v<ddc::TypeSeq<Dimension>, ddc::TypeSeq<Dimensions...>>)
    KOKKOS_FUNCTION std::ptrdiff_t uid() const
    {
        return m_coordinates[ddc::type_seq_rank_v<Dimension, ddc::TypeSeq<Dimensions...>>];
    }
    KOKKOS_FUNCTION std::array<std::ptrdiff_t, sizeof...(Dimensions)>& coordinates()
    {
        return m_coordinates;
    }
    KOKKOS_FUNCTION std::array<std::ptrdiff_t, sizeof...(Dimensions)> const& coordinates() const
    {
        return m_coordinates;
    }
    KOKKOS_FUNCTION ddc::DiscreteElement<Dimensions...> storage_element() const
    {
        assert(((uid<Dimensions>() >= 0) && ...));
        return ddc::DiscreteElement<Dimensions...>(static_cast<std::size_t>(uid<Dimensions>())...);
    }
};
} // namespace sil::misc

namespace ddc::detail {
template <class... Dimensions>
KOKKOS_FUNCTION auto& array(sil::misc::SignedDiscreteElement<Dimensions...>& element)
{
    return element.coordinates();
}
template <class... Dimensions>
KOKKOS_FUNCTION auto const& array(sil::misc::SignedDiscreteElement<Dimensions...> const& element)
{
    return element.coordinates();
}
} // namespace ddc::detail
