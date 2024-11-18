// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <ddc/ddc.hpp>

#include "cochain.hpp"
#include "cosimplex.hpp"

namespace sil {

namespace exterior {

namespace detail {

template <class T, class ElementType, class Allocator>
struct FormWrapper;

template <misc::NotSpecialization<Chain> T, class ElementType, class Allocator>
struct FormWrapper<T, ElementType, Allocator>
{
    using type = Cosimplex<T, ElementType>;
};

template <misc::Specialization<Chain> T, class ElementType, class Allocator>
struct FormWrapper<T, ElementType, Allocator>
{
    using type = Cochain<T, ElementType, Allocator>;
};

/*
template <misc::NotSpecialization<Chain> Head, class ElementType>
FormWrapper(Head, ElementType) -> FormWrapper<Head, ElementType, std::allocator<ElementType>>;

template <misc::Specialization<Chain> Head, class ElementType, class Allocator>
FormWrapper(Head, ElementType, Allocator) -> FormWrapper<Head, ElementType, Allocator>;
*/

} // namespace detail

// Usage should be avoided because CTAD cannot go through it
template <class T, class ElementType = double, class Allocator = std::allocator<double>>
using Form = typename detail::FormWrapper<T, ElementType, Allocator>::type;

} // namespace exterior

} // namespace sil
