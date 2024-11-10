// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: GPL-3.0

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

} // namespace detail

} // namespace misc

} // namespace sil
