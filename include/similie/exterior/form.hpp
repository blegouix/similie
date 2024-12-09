// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <ddc/ddc.hpp>

#include "cochain.hpp"
#include "cosimplex.hpp"

namespace sil {

namespace exterior {

namespace detail {

template <class T, class ElementType, class LayoutStridedPolicy>
struct FormWrapper;

template <misc::NotSpecialization<Chain> T, class ElementType, class LayoutStridedPolicy>
struct FormWrapper<T, ElementType, LayoutStridedPolicy>
{
    using type = Cosimplex<T, ElementType>;
};

template <misc::Specialization<Chain> T, class ElementType, class LayoutStridedPolicy>
struct FormWrapper<T, ElementType, LayoutStridedPolicy>
{
    using type = Cochain<T, ElementType, LayoutStridedPolicy>;
};

/*
template <misc::NotSpecialization<Chain> Head, class ElementType, class LayoutStridedPoliy >
FormWrapper(Head, ElementType) -> FormWrapper<Head, ElementType, LayoutStridedPolicy, Kokkos::DefaultHostExecutionSpace>;

template <misc::Specialization<Chain> Head, class ElementType, class LayoutStridedPolicy, class ExecSpace>
FormWrapper(Head, ElementType, Allocator) -> FormWrapper<Head, ElementType, LayoutStridedPolicy, ExecSpace>;
*/

} // namespace detail

// Usage should be avoided because CTAD cannot go through it
template <class T, class ElementType = double, class LayoutStridedPolicy = Kokkos::LayoutRight>
using Form = typename detail::FormWrapper<T, ElementType, LayoutStridedPolicy>::type;

} // namespace exterior

} // namespace sil
