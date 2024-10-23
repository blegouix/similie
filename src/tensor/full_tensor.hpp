// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: GPL-3.0

#pragma once

#include <ddc/ddc.hpp>

#include "tensor.hpp"

namespace sil {

namespace tensor {

// Helpers to compute the strides of a right layout. This is necessary to support non-squared full tensors.
namespace detail {
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
            ddc::type_seq_rank_v<
                    OTensorNaturalIndex,
                    ddc::detail::TypeSeq<TensorNaturalIndex...>> == 0) {
        return std::numeric_limits<std::size_t>::max();
    } else {
        return (stride_factor<
                        ddc::type_seq_rank_v<
                                OTensorNaturalIndex,
                                ddc::detail::TypeSeq<TensorNaturalIndex...>> - 1,
                        TensorNaturalIndex,
                        TensorNaturalIndex...>()
                * ...);
    }
}

} // namespace detail

// struct representing an abstract unique index sweeping on all possible combination of natural indexes, for a full tensor (dense with no particular structure).
template <class... TensorIndex>
struct FullTensorIndex
{
    static constexpr std::size_t rank()
    {
        return (TensorIndex::rank() + ...);
    }

    static constexpr std::size_t size()
    {
        return (TensorIndex::size() * ...);
    }

    static constexpr std::size_t mem_size()
    {
        return (TensorIndex::mem_size() * ...);
    }

    static constexpr std::size_t access_size()
    {
        return (TensorIndex::access_size() * ...);
    }

    template <class... CDim>
    static constexpr std::pair<std::vector<double>, std::vector<std::size_t>> mem_id()
    {
        //static_assert(rank() == sizeof...(CDim));
        return std::pair<std::vector<double>, std::vector<std::size_t>>(
                std::vector<double> {},
                std::vector<std::size_t> {access_id<CDim...>()});
    }

    template <class... CDim>
    static constexpr std::size_t access_id()
    {
        return ((detail::stride<TensorIndex, TensorIndex...>()
                 * detail::access_id<TensorIndex, ddc::detail::TypeSeq<TensorIndex...>, CDim...>())
                + ...);
    }

    static constexpr std::pair<std::vector<double>, std::vector<std::size_t>> access_id_to_mem_id(
            std::size_t access_id)
    {
        return std::pair<
                std::vector<double>,
                std::vector<
                        std::size_t>>(std::vector<double> {}, std::vector<std::size_t> {access_id});
    }

    template <class Tensor, class Elem, class Id>
    static constexpr Tensor::element_type process_access(
            std::function<typename Tensor::element_type(Tensor, Elem)> access,
            Tensor tensor,
            Elem elem)
    {
        return access(tensor, elem);
    }
};

namespace detail {
template <class... SubIndex>
struct IsTensorIndex<FullTensorIndex<SubIndex...>>
{
    using type = std::true_type;
};

} // namespace detail

} // namespace tensor

} // namespace sil
