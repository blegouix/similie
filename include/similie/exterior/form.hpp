// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <ddc/ddc.hpp>

#include "cochain.hpp"
#include "cosimplex.hpp"

namespace sil {

namespace exterior {

namespace detail {

template <class T, class ElementType, class ExecSpace>
struct FormWrapper;

template <misc::NotSpecialization<Chain> T, class ElementType, class ExecSpace>
struct FormWrapper<T, ElementType, ExecSpace>
{
    using type = Cosimplex<T, ElementType>;
};

template <misc::Specialization<Chain> T, class ElementType, class ExecSpace>
struct FormWrapper<T, ElementType, ExecSpace>
{
    using type = Cochain<T, ElementType, ExecSpace>;
};

/*
template <misc::NotSpecialization<Chain> Head, class ElementType>
FormWrapper(Head, ElementType) -> FormWrapper<Head, ElementType, Kokkos::DefaultHostExecutionSpace>;

template <misc::Specialization<Chain> Head, class ElementType, class ExecSpace>
FormWrapper(Head, ElementType, Allocator) -> FormWrapper<Head, ElementType, ExecSpace>;
*/

} // namespace detail

// Usage should be avoided because CTAD cannot go through it
template <class T, class ElementType = double, class ExecSpace = Kokkos::DefaultHostExecutionSpace>
using Form = typename detail::FormWrapper<T, ElementType, ExecSpace>::type;

} // namespace exterior

} // namespace sil
