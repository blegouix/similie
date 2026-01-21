// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: MIT

#pragma once

#include <ddc/ddc.hpp>

#include "tensor_impl.hpp"

namespace sil {

namespace tensor {

// struct representing an identity tensor (no storage).
template <TensorNatIndex... TensorIndex>
struct TensorIdentityIndex
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
        return 0;
    }

    KOKKOS_FUNCTION static constexpr std::size_t access_size()
    {
        return 2;
    }

    KOKKOS_FUNCTION static constexpr std::size_t mem_id(
            std::array<std::size_t, sizeof...(TensorIndex)> const natural_ids)
    {
        return std::numeric_limits<std::size_t>::max();
    }

    KOKKOS_FUNCTION static constexpr std::size_t access_id(
            std::array<std::size_t, sizeof...(TensorIndex)> const natural_ids)
    {
        if (!std::all_of(natural_ids.begin(), natural_ids.end(), [&](const std::size_t id) {
                return id == *natural_ids.begin();
            })) {
            return 0;
        } else {
            return 1;
        }
    }

    KOKKOS_FUNCTION static constexpr std::size_t access_id_to_mem_id(
            [[maybe_unused]] std::size_t access_id)
    {
        return std::numeric_limits<std::size_t>::max();
    }

    template <class Tensor, class Elem, class Id, class FunctorType>
    KOKKOS_FUNCTION static typename Tensor::element_type const& process_access(
            const FunctorType& access,
            Tensor tensor,
            Elem elem)
    {
        if (elem.template uid<Id>() == 0) {
            return detail::StaticValue<typename Tensor::element_type>::zero();
        } else {
            return access(tensor, elem);
        }
    }

    KOKKOS_FUNCTION static constexpr std::array<std::size_t, rank()>
    mem_id_to_canonical_natural_ids(std::size_t mem_id)
    {
        assert(false);
        std::array<std::size_t, rank()> ids;
        for (auto i = ids.begin(); i < ids.end(); ++i) {
            *i = 0;
        }
        return ids;
    }
};

} // namespace tensor

} // namespace sil
