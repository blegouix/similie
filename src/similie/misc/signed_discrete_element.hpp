// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#pragma once

#include <cassert>
#include <cstddef>

#include <ddc/ddc.hpp>

namespace sil::misc {

/** Tagged logical coordinates, including zero and negative exterior indices.
 * \important This operator and documentation is fully AI-generated.
 * DDC vectors provide signed storage; uid<Tag>() exposes absolute coordinates
 * here rather than a displacement. Conversion to a storage element is explicit
 * and requires nonnegative coordinates. Extrapolation must resolve exterior
 * coordinates before that conversion, since DDC elements store std::size_t.
 */
template <class... Dimensions>
class SignedDiscreteElement : public ddc::DiscreteVector<Dimensions...>
{
public:
    using ddc::DiscreteVector<Dimensions...>::DiscreteVector;
    KOKKOS_DEFAULTED_FUNCTION SignedDiscreteElement() = default;

    template <class Element>
        requires(requires(Element elem) { (elem.template uid<Dimensions>(), ...); })
    KOKKOS_FUNCTION explicit SignedDiscreteElement(Element elem)
        : ddc::DiscreteVector<Dimensions...>(
                  static_cast<std::ptrdiff_t>(elem.template uid<Dimensions>())...)
    {
    }

    template <class Dimension>
    KOKKOS_FUNCTION std::ptrdiff_t& uid()
    {
        return this->template get<Dimension>();
    }
    template <class Dimension>
    KOKKOS_FUNCTION std::ptrdiff_t uid() const
    {
        return this->template get<Dimension>();
    }

    KOKKOS_FUNCTION ddc::DiscreteElement<Dimensions...> storage_element() const
    {
        assert(((this->template uid<Dimensions>() >= 0) && ...));
        return ddc::DiscreteElement<Dimensions...>(
                static_cast<std::size_t>(this->template uid<Dimensions>())...);
    }
};

} // namespace sil::misc
