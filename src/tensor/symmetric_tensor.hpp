// SPDX-License-Identifier: GPL-3.0

#pragma once

#include <ddc/ddc.hpp>

#include <boost/math/special_functions/binomial.hpp>

#include "tensor.hpp"

namespace sil {

namespace tensor {

// struct representing an abstract unique index sweeping on all possible combination of natural indexes, for a summetric tensor.
template <class... TensorIndex>
struct SymmetricTensorIndex
{
    static constexpr std::size_t rank()
    {
        return (TensorIndex::rank() + ...);
    }

    static constexpr std::size_t dim_size()
    {
        return boost::math::binomial_coefficient<double>(
                std::min({TensorIndex::dim_size()...}) + sizeof...(TensorIndex) - 1,
                sizeof...(TensorIndex));
    }

    template <class... CDim>
    static constexpr std::size_t mem_id()
    {
        // static_assert(rank() == sizeof...(CDim));
        std::array<int, sizeof...(TensorIndex)> sorted_ids {
                detail::access_id<TensorIndex, ddc::detail::TypeSeq<TensorIndex...>, CDim...>()...};
        std::sort(sorted_ids.begin(), sorted_ids.end());
        return boost::math::binomial_coefficient<double>(
                       std::min({TensorIndex::dim_size()...}) + sizeof...(TensorIndex) - 1,
                       sizeof...(TensorIndex))
               - ((sorted_ids[ddc::type_seq_rank_v<
                           TensorIndex,
                           ddc::detail::TypeSeq<TensorIndex...>>]
                                   == TensorIndex::dim_size() - 1
                           ? 0
                           : boost::math::binomial_coefficient<double>(
                                   TensorIndex::dim_size()
                                           - sorted_ids[ddc::type_seq_rank_v<
                                                   TensorIndex,
                                                   ddc::detail::TypeSeq<TensorIndex...>>]
                                           + sizeof...(TensorIndex)
                                           - ddc::type_seq_rank_v<
                                                   TensorIndex,
                                                   ddc::detail::TypeSeq<TensorIndex...>> - 2,
                                   sizeof...(TensorIndex)
                                           - ddc::type_seq_rank_v<
                                                   TensorIndex,
                                                   ddc::detail::TypeSeq<TensorIndex...>>))
                  + ...)
               - 1;
    }

    template <class... CDim>
    static constexpr std::size_t access_id()
    {
        return mem_id<CDim...>();
    }

    static constexpr std::size_t access_id_to_mem_id(std::size_t access_id)
    {
        return access_id;
    }

    template <class Tensor, class Elem>
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
struct IsTensorIndex<SymmetricTensorIndex<SubIndex...>>
{
    using type = std::true_type;
};

} // namespace detail

} // namespace tensor

} // namespace sil
