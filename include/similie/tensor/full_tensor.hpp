// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <ddc/ddc.hpp>

#include <similie/misc/stride.hpp>

#include "tensor_impl.hpp"

namespace sil {

namespace tensor {

// struct representing an abstract unique index sweeping on all possible combination of natural indices, for a full tensor (dense with no particular structure).
template <TensorNatIndex... TensorIndex>
struct TensorFullIndex
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
        if constexpr (sizeof...(TensorIndex) == 0) {
            return 0;
        } else {
            return (TensorIndex::rank() + ...);
        }
    }

    KOKKOS_FUNCTION static constexpr std::size_t size()
    {
        if constexpr (rank() == 0) {
            return 1;
        } else {
            return (TensorIndex::size() * ...);
        }
    }

    KOKKOS_FUNCTION static constexpr std::size_t mem_size()
    {
        if constexpr (rank() == 0) {
            return 1;
        } else {
            return (TensorIndex::mem_size() * ...);
        }
    }

    KOKKOS_FUNCTION static constexpr std::size_t access_size()
    {
        return (TensorIndex::access_size() * ...);
    }

    KOKKOS_FUNCTION static constexpr std::size_t mem_id(
            std::array<std::size_t, sizeof...(TensorIndex)> const natural_ids)
    {
        return access_id(natural_ids);
    }

    KOKKOS_FUNCTION static constexpr std::size_t access_id(
            std::array<std::size_t, sizeof...(TensorIndex)> const natural_ids)
    {
        if constexpr (rank() == 0) {
            return 0;
        } else {
            return ((misc::detail::stride<TensorIndex, TensorIndex...>()
                     * natural_ids[ddc::type_seq_rank_v<
                             TensorIndex,
                             ddc::detail::TypeSeq<TensorIndex...>>])
                    + ...);
        }
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
            return std::array<std::size_t, rank()> {
                    (mem_id % misc::detail::next_stride<TensorIndex, TensorIndex...>())
                    / misc::detail::stride<TensorIndex, TensorIndex...>()...};
        }
    }
};

} // namespace tensor

} // namespace sil
