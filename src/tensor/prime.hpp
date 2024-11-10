// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <ddc/ddc.hpp>

#include "character.hpp"
#include "specialization.hpp"

namespace sil {

namespace tensor {

template <TensorNatIndex Index, std::size_t I = 1>
struct prime : Index
{
};

template <TensorNatIndex Index>
using second = prime<Index, 2>;

namespace detail {

template <class Indices, std::size_t I>
struct Primes;

template <class... Index, std::size_t I>
struct Primes<ddc::detail::TypeSeq<Index...>, I>
{
    using type = ddc::detail::TypeSeq<prime<Index, I>...>;
};

template <class... Index, std::size_t I>
struct Primes<ddc::detail::TypeSeq<TensorContravariantNaturalIndex<Index>...>, I>
{
    using type = ddc::detail::TypeSeq<TensorContravariantNaturalIndex<prime<Index, I>>...>;
};

template <class... Index, std::size_t I>
struct Primes<ddc::detail::TypeSeq<TensorCovariantNaturalIndex<Index>...>, I>
{
    using type = ddc::detail::TypeSeq<TensorCovariantNaturalIndex<prime<Index, I>>...>;
};

} // namespace detail

template <misc::Specialization<ddc::detail::TypeSeq> Indices, std::size_t I = 1>
using primes = detail::Primes<Indices, I>::type;

template <misc::Specialization<ddc::detail::TypeSeq> Indices>
using seconds = primes<Indices, 2>;

} // namespace tensor

} // namespace sil
