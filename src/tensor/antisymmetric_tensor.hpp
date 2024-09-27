// SPDX-License-Identifier: GPL-3.0

#pragma once

#include <ddc/ddc.hpp>

#include <boost/math/special_functions/binomial.hpp>

#include "tensor_impl.hpp"

namespace sil {

namespace tensor {

// struct representing an abstract unique index sweeping on all possible combination of natural indexes, for an antisummetric tensor.
template <class... TensorIndex>
struct AntisymmetricTensorIndex
{
    static constexpr std::size_t rank()
    {
        return (TensorIndex::rank() + ...);
    }

    static constexpr std::size_t dim_size()
    {
        return (TensorIndex::dim_size() * ...);
    }

    static constexpr std::size_t mem_dim_size()
    {
        return boost::math::binomial_coefficient<
                double>(std::min({TensorIndex::mem_dim_size()...}), sizeof...(TensorIndex));
    }

    static constexpr std::size_t access_dim_size()
    {
        return mem_dim_size() + 1;
    }

    template <class... CDim>
    static constexpr std::size_t mem_id()
    {
        // static_assert(rank() == sizeof...(CDim));
        std::array<int, sizeof...(TensorIndex)> sorted_ids {
                detail::access_id<TensorIndex, ddc::detail::TypeSeq<TensorIndex...>, CDim...>()...};
        std::sort(sorted_ids.begin(), sorted_ids.end());
        return boost::math::binomial_coefficient<
                       double>(std::min({TensorIndex::mem_dim_size()...}), sizeof...(TensorIndex))
               - ((sorted_ids[ddc::type_seq_rank_v<
                           TensorIndex,
                           ddc::detail::TypeSeq<TensorIndex...>>]
                                   == TensorIndex::mem_dim_size() - sizeof...(TensorIndex)
                                              + ddc::type_seq_rank_v<
                                                      TensorIndex,
                                                      ddc::detail::TypeSeq<TensorIndex...>>
                           ? 0
                           : boost::math::binomial_coefficient<double>(
                                   TensorIndex::mem_dim_size()
                                           - sorted_ids[ddc::type_seq_rank_v<
                                                   TensorIndex,
                                                   ddc::detail::TypeSeq<TensorIndex...>>]
                                           - 1,
                                   sizeof...(TensorIndex)
                                           - ddc::type_seq_rank_v<
                                                   TensorIndex,
                                                   ddc::detail::TypeSeq<TensorIndex...>>))
                  + ...)
               - 1;
    }


private:
    template <class Head, class... Tail>
    inline static constexpr bool are_all_same = (std::is_same_v<Head, Tail> && ...);

    template <class... CDim>
    static constexpr bool permutation_parity()
    {
        std::array<int, sizeof...(TensorIndex)> ids {
                detail::access_id<TensorIndex, ddc::detail::TypeSeq<TensorIndex...>, CDim...>()...};
        bool cnt = false;
        for (int i = 0; i < sizeof...(CDim); i++)
            for (int j = i + 1; j < sizeof...(CDim); j++)
                if (ids[i] > ids[j])
                    cnt = !cnt;
        return cnt;
    }

public:
    template <class... CDim>
    static constexpr std::size_t access_id()
    {
        if constexpr (are_all_same<CDim...>) {
            return 0;
        } else if (!permutation_parity<CDim...>()) {
            return 1 + mem_id<CDim...>();
        } else {
            return access_dim_size() + mem_id<CDim...>();
        }
    }

    static constexpr std::size_t access_id_to_mem_id(std::size_t access_id)
    {
        assert(access_id != 0 && "There is no mem_id associated to access_id=0");
        return (access_id - 1) % mem_dim_size();
    }

    template <class Tensor, class Elem>
    static constexpr Tensor::element_type process_access(
            std::function<typename Tensor::element_type(Tensor, Elem)> access,
            Tensor tensor,
            Elem elem)
    {
        if (elem.uid() == 0) {
            return 0.;
        } else if (elem.uid() < access_dim_size()) {
            return access(tensor, elem);
        } else {
            return -access(tensor, elem);
        }
    }
};

namespace detail {
template <class... SubIndex>
struct IsTensorIndex<AntisymmetricTensorIndex<SubIndex...>>
{
    using type = std::true_type;
};

} // namespace detail

} // namespace tensor

} // namespace sil
