// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <ddc/ddc.hpp>

#include <similie/misc/binomial_coefficient.hpp>

#include "tensor_impl.hpp"

namespace sil {

namespace tensor {

// struct representing an abstract unique index sweeping on all possible combination of natural indices, for a summetric tensor.
template <class... TensorIndex>
struct TensorSymmetricIndex
{
    static constexpr bool is_tensor_index = true;

    using subindices_domain_t = ddc::DiscreteDomain<TensorIndex...>;

    static constexpr subindices_domain_t subindices_domain()
    {
        return ddc::DiscreteDomain<TensorIndex...>(
                ddc::DiscreteElement<TensorIndex...>(ddc::DiscreteElement<TensorIndex>(0)...),
                ddc::DiscreteVector<TensorIndex...>(
                        ddc::DiscreteVector<TensorIndex>(TensorIndex::size())...));
    }

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
        return misc::binomial_coefficient(
                std::min({TensorIndex::mem_size()...}) + sizeof...(TensorIndex) - 1,
                sizeof...(TensorIndex));
    }

    static constexpr std::size_t access_size()
    {
        return mem_size();
    }

    static constexpr std::pair<std::vector<double>, std::vector<std::size_t>> mem_lin_comb(
            std::array<std::size_t, sizeof...(TensorIndex)> const ids)
    {
        std::array<std::size_t, sizeof...(TensorIndex)> sorted_ids(ids);
        std::sort(sorted_ids.begin(), sorted_ids.end());
        return std::pair<std::vector<double>, std::vector<std::size_t>>(
                std::vector<double> {1.},
                std::vector<std::size_t> {static_cast<std::size_t>(
                        misc::binomial_coefficient(
                                std::min({TensorIndex::mem_size()...}) + sizeof...(TensorIndex) - 1,
                                sizeof...(TensorIndex))
                        - ((sorted_ids[ddc::type_seq_rank_v<
                                    TensorIndex,
                                    ddc::detail::TypeSeq<TensorIndex...>>]
                                            == TensorIndex::mem_size() - 1
                                    ? 0
                                    : misc::binomial_coefficient(
                                              TensorIndex::mem_size()
                                                      - sorted_ids[ddc::type_seq_rank_v<
                                                              TensorIndex,
                                                              ddc::detail::TypeSeq<TensorIndex...>>]
                                                      + sizeof...(TensorIndex)
                                                      - ddc::type_seq_rank_v<
                                                              TensorIndex,
                                                              ddc::detail::TypeSeq<TensorIndex...>>
                                                      - 2,
                                              sizeof...(TensorIndex)
                                                      - ddc::type_seq_rank_v<
                                                              TensorIndex,
                                                              ddc::detail::TypeSeq<
                                                                      TensorIndex...>>))
                           + ...)
                        - 1)});
    }

    static constexpr std::size_t access_id(
            std::array<std::size_t, sizeof...(TensorIndex)> const ids)
    {
        return std::get<1>(mem_lin_comb(ids))[0];
    }

    static constexpr std::pair<std::vector<double>, std::vector<std::size_t>>
    access_id_to_mem_lin_comb(std::size_t access_id)
    {
        return std::pair<std::vector<double>, std::vector<std::size_t>>(
                std::vector<double> {1.},
                std::vector<std::size_t> {access_id});
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

} // namespace tensor

} // namespace sil
