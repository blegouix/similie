// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

#include <ddc/ddc.hpp>

#include <similie/misc/specialization.hpp>

#include "character.hpp"

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

template <std::size_t I>
struct Primes<ddc::TypeSeq<>, I>
{
    using type = ddc::TypeSeq<>;
};

template <class... Index, std::size_t I>
struct Primes<ddc::TypeSeq<Index...>, I>
{
    using type = ddc::TypeSeq<prime<Index, I>...>;
};

template <class... Index, std::size_t I>
struct Primes<ddc::TypeSeq<Contravariant<Index>...>, I>
{
    using type = ddc::TypeSeq<Contravariant<prime<Index, I>>...>;
};

template <class... Index, std::size_t I>
struct Primes<ddc::TypeSeq<Covariant<Index>...>, I>
{
    using type = ddc::TypeSeq<Covariant<prime<Index, I>>...>;
};

} // namespace detail

template <misc::Specialization<ddc::TypeSeq> Indices, std::size_t I = 1>
using primes = detail::Primes<Indices, I>::type;

template <misc::Specialization<ddc::TypeSeq> Indices>
using seconds = primes<Indices, 2>;

} // namespace tensor

} // namespace sil
