// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <ddc/ddc.hpp>

#include <similie/misc/specialization.hpp>
#include <similie/tensor/full_tensor.hpp>

namespace sil {

namespace tensor {

// Dummy tag indexed by I
template <std::size_t I>
struct Dummy
{
};

namespace detail {

template <class Ids>
struct NaturalIndex;

template <std::size_t... Id>
struct NaturalIndex<std::index_sequence<Id...>>
{
    template <std::size_t RankId>
    struct type : tensor::TensorNaturalIndex<Dummy<Id>...>
    {
    };
};

template <class NaturalIds, class RankIds>
struct DummyIndex;

template <std::size_t... Id, std::size_t... RankId>
struct DummyIndex<std::index_sequence<Id...>, std::index_sequence<RankId...>>
{
    using type = tensor::TensorFullIndex<
            typename NaturalIndex<std::index_sequence<Id...>>::template type<RankId>...>;
};

} // namespace detail

template <std::size_t Dimension, std::size_t Rank>
using dummy_index_t = detail::
        DummyIndex<std::make_index_sequence<Dimension>, std::make_index_sequence<Rank>>::type;

} // namespace tensor

} // namespace sil
