// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <ddc/ddc.hpp>

#include <similie/misc/binomial_coefficient.hpp>
#include <similie/misc/portable_stl.hpp>

#include "tensor_impl.hpp"

namespace sil {

namespace tensor {

// struct representing an abstract unique index sweeping on all possible combination of natural indices, for a summetric tensor.
template <TensorNatIndex... TensorIndex>
struct TensorSymmetricIndex
{
    static constexpr bool is_tensor_index = true;
    static constexpr bool is_explicitely_stored_tensor = true;

    using subindices_domain_t = ddc::DiscreteDomain<TensorIndex...>;

    KOKKOS_FUNCTION static constexpr subindices_domain_t subindices_domain()
    {
        return ddc::DiscreteDomain<TensorIndex...>(
                ddc::DiscreteElement<TensorIndex...>(ddc::DiscreteElement<TensorIndex>(0)...),
                ddc::DiscreteVector<TensorIndex...>(
                        ddc::DiscreteVector<TensorIndex>(TensorIndex::size())...));
    }

    KOKKOS_FUNCTION static constexpr std::size_t rank()
    {
        return (TensorIndex::rank() + ...);
    }

    KOKKOS_FUNCTION static constexpr std::size_t size()
    {
        return (TensorIndex::size() * ...);
    }

    KOKKOS_FUNCTION static constexpr std::size_t mem_size()
    {
        return misc::binomial_coefficient(
                ddc::type_seq_element_t<0, ddc::detail::TypeSeq<TensorIndex...>>::mem_size()
                        + rank() - 1,
                rank());
    }

    KOKKOS_FUNCTION static constexpr std::size_t access_size()
    {
        return mem_size();
    }

    KOKKOS_FUNCTION static constexpr std::size_t mem_id(
            std::array<std::size_t, rank()> const natural_ids)
    {
        std::array<std::size_t, rank()> sorted_ids(natural_ids);
        misc::detail::sort(sorted_ids.begin(), sorted_ids.end());
        return misc::binomial_coefficient(
                       ddc::type_seq_element_t<0, ddc::detail::TypeSeq<TensorIndex...>>::mem_size()
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
                                                     ddc::detail::TypeSeq<TensorIndex...>>))
                  + ...)
               - 1;
    }

    KOKKOS_FUNCTION static constexpr std::size_t access_id(
            std::array<std::size_t, rank()> const natural_ids)
    {
        return mem_id(natural_ids);
    }

    KOKKOS_FUNCTION static constexpr std::size_t access_id_to_mem_id(std::size_t access_id)
    {
        return access_id;
    }

    template <class Tensor, class Elem, class Id, class FunctorType>
    KOKKOS_FUNCTION static constexpr Tensor::element_type process_access(
            const FunctorType& access,
            Tensor tensor,
            Elem elem)
    {
        return access(tensor, elem);
    }

    KOKKOS_FUNCTION static constexpr std::array<std::size_t, rank()>
    mem_id_to_canonical_natural_ids(std::size_t mem_id)
    {
        assert(mem_id < mem_size());
        if constexpr (rank() == 0) {
            return std::array<std::size_t, rank()> {};
        } else {
            std::array<std::size_t, rank()> ids;
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
                        mem_id -= triangle_size
                                  - misc::binomial_coefficient(d - j + r - i - 1, r - i);
                        d -= j;
                        break;
                    }
                    ids[i] = 0;
                }
            }
            return ids;
        }
    }
};

} // namespace tensor

} // namespace sil
