// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: MIT

#pragma once

#include <ddc/ddc.hpp>

#if defined(__cpp_constexpr) && __cpp_constexpr >= 202211L
#define SIL_CONSTEXPR_IF_CXX23 constexpr
#else
#define SIL_CONSTEXPR_IF_CXX23
#endif

namespace sil::detail {

template <class ElementType>
inline constexpr ElementType static_zero = ElementType(0);

template <class ElementType>
inline constexpr ElementType static_one = ElementType(1);

template <class ElementType>
KOKKOS_FUNCTION SIL_CONSTEXPR_IF_CXX23 inline ElementType const& static_value(ElementType value)
{
    static ElementType storage {};
    storage = value;
    return storage;
}

} // namespace sil::detail
