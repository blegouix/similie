// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <ddc/ddc.hpp>

namespace sil {

namespace misc {

namespace detail {

// Helpers to compute the strides of a right layout.
template <std::size_t max_rank, class OTensorNaturalIndex, class... TensorNaturalIndex>
static constexpr std::size_t stride_factor()
{
    if constexpr (
            ddc::type_seq_rank_v < OTensorNaturalIndex,
            ddc::detail::TypeSeq < TensorNaturalIndex... >>> max_rank) {
        return OTensorNaturalIndex::mem_size();
    } else {
        return 1;
    }
}

template <class OTensorNaturalIndex, class... TensorNaturalIndex>
static constexpr std::size_t stride()
{
    return (stride_factor<
                    ddc::type_seq_rank_v<
                            OTensorNaturalIndex,
                            ddc::detail::TypeSeq<TensorNaturalIndex...>>,
                    TensorNaturalIndex,
                    TensorNaturalIndex...>()
            * ...);
}

template <class OTensorNaturalIndex, class... TensorNaturalIndex>
static constexpr std::size_t next_stride()
{
    if constexpr (
            ddc::type_seq_rank_v<OTensorNaturalIndex, ddc::detail::TypeSeq<TensorNaturalIndex...>>
            == 0) {
        return std::numeric_limits<std::size_t>::max();
    } else {
        return (stride_factor<
                        ddc::type_seq_rank_v<
                                OTensorNaturalIndex,
                                ddc::detail::TypeSeq<TensorNaturalIndex...>>
                                - 1,
                        TensorNaturalIndex,
                        TensorNaturalIndex...>()
                * ...);
    }
}

// Helpers to compute the strides of a symmetric tensor.
template <class OTensorNaturalIndex, class... TensorNaturalIndex>
static constexpr std::size_t symmetric_stride()
{
    return misc::binomial_coefficient(
            OTensorNaturalIndex::mem_size(),
            ddc::type_seq_rank_v<OTensorNaturalIndex, ddc::detail::TypeSeq<TensorNaturalIndex...>>);
}

template <class OTensorNaturalIndex, class... TensorNaturalIndex>
static constexpr std::size_t symmetric_next_stride()
{
    if constexpr (
            ddc::type_seq_rank_v<OTensorNaturalIndex, ddc::detail::TypeSeq<TensorNaturalIndex...>>
            == 0) {
        return std::numeric_limits<std::size_t>::max();
    } else {
        return misc::binomial_coefficient(
                OTensorNaturalIndex::mem_size(),
                ddc::type_seq_rank_v<
                        OTensorNaturalIndex,
                        ddc::detail::TypeSeq<TensorNaturalIndex...>>
                        - 1);
    }
}

// Helpers to compute the strides of an antisymmetric tensor.
template <std::size_t i, class OTensorNaturalIndex, class... TensorNaturalIndex>
static constexpr std::size_t antisymmetric_stride()
{
    return misc::binomial_coefficient(
            OTensorNaturalIndex::mem_size(),
            sizeof...(TensorNaturalIndex) - ddc::type_seq_rank_v<OTensorNaturalIndex, ddc::detail::TypeSeq<TensorNaturalIndex...>>);
}

template <std::size_t i, class OTensorNaturalIndex, class... TensorNaturalIndex>
static constexpr std::size_t antisymmetric_next_stride()
{
    if constexpr (
            ddc::type_seq_rank_v<OTensorNaturalIndex, ddc::detail::TypeSeq<TensorNaturalIndex...>>
            == 0) {
        return std::numeric_limits<std::size_t>::max();
    } else {
        return misc::binomial_coefficient(
                OTensorNaturalIndex::mem_size(),
                sizeof...(TensorNaturalIndex) - ddc::type_seq_rank_v<
                        OTensorNaturalIndex,
                        ddc::detail::TypeSeq<TensorNaturalIndex...>>
                        - 1);
    }
}

} // namespace detail

} // namespace misc

} // namespace sil
