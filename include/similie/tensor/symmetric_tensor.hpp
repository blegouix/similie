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
                ddc::type_seq_element_t<0, ddc::detail::TypeSeq<TensorIndex...>>::mem_size()
                        + rank() - 1,
                rank());
    }

    static constexpr std::size_t access_size()
    {
        return mem_size();
    }

    static constexpr std::pair<std::vector<double>, std::vector<std::size_t>> mem_lin_comb(
            std::array<std::size_t, rank()> const ids)
    {
        std::array<std::size_t, rank()> sorted_ids(ids);
        std::sort(sorted_ids.begin(), sorted_ids.end());
        return std::pair<std::vector<double>, std::vector<std::size_t>>(
                std::vector<double> {1.},
                std::vector<std::size_t> {static_cast<std::size_t>(
                        misc::binomial_coefficient(
                                ddc::type_seq_element_t<0, ddc::detail::TypeSeq<TensorIndex...>>::
                                                mem_size()
                                        + rank() - 1,
                                rank())
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
                                                      + rank()
                                                      - ddc::type_seq_rank_v<
                                                              TensorIndex,
                                                              ddc::detail::TypeSeq<TensorIndex...>>
                                                      - 2,
                                              rank()
                                                      - ddc::type_seq_rank_v<
                                                              TensorIndex,
                                                              ddc::detail::TypeSeq<
                                                                      TensorIndex...>>))
                           + ...)
                        - 1)});
    }

    static constexpr std::size_t access_id(std::array<std::size_t, rank()> const ids)
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

    static constexpr std::vector<std::size_t> mem_id_to_canonical_natural_ids(std::size_t mem_id)
    {
        assert(mem_id < mem_size());
        std::vector<std::size_t> ids(rank());
        std::size_t d
                = ddc::type_seq_element_t<0, ddc::detail::TypeSeq<TensorIndex...>>::mem_size();
        std::size_t r = rank();
        for (std::size_t i = 0; i < rank(); ++i) {
            const std::size_t triangle_size = misc::binomial_coefficient(d + r - i - 1, r - i);
            for (std::size_t j = 0; j < d; ++j) {
                const std::size_t subtriangle_size
                        = misc::binomial_coefficient(d - j + r - i - 2, r - i);
                if (triangle_size - subtriangle_size > mem_id) {
                    ids[i] = ddc::type_seq_element_t<0, ddc::detail::TypeSeq<TensorIndex...>>::
                                     mem_size()
                             - d + j;
                    mem_id -= triangle_size - misc::binomial_coefficient(d - j + r - i - 1, r - i);
                    d -= j;
                    break;
                }
                ids[i] = 0;
            }
        }
        return ids;
    }
};

} // namespace tensor

} // namespace sil
