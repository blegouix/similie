// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: MIT

#pragma once

#include <ddc/ddc.hpp>

#include <similie/misc/filled_struct.hpp>

#include "tensor_impl.hpp"

namespace sil {

namespace tensor {

// struct representing and index for a diagonal tensor (only diagonal is stored).
template <TensorNatIndex... TensorIndex>
struct TensorDiagonalIndex
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
        return ddc::type_seq_element_t<0, ddc::detail::TypeSeq<TensorIndex...>>::mem_size();
    }

    KOKKOS_FUNCTION static constexpr std::size_t access_size()
    {
        return 1 + mem_size();
    }

    KOKKOS_FUNCTION static constexpr std::size_t mem_id(
            std::array<std::size_t, sizeof...(TensorIndex)> const natural_ids)
    {
        assert(std::all_of(natural_ids.begin(), natural_ids.end(), [&](const std::size_t id) {
            return id == *natural_ids.begin();
        }));
        return natural_ids[0];
    }

    KOKKOS_FUNCTION static constexpr std::size_t access_id(
            std::array<std::size_t, sizeof...(TensorIndex)> const natural_ids)
    {
        if (!std::all_of(natural_ids.begin(), natural_ids.end(), [&](const std::size_t id) {
                return id == *natural_ids.begin();
            })) {
            return 0;
        } else {
            return 1 + mem_id(natural_ids);
        }
    }

    KOKKOS_FUNCTION static constexpr std::size_t access_id_to_mem_id(std::size_t access_id)
    {
        assert(access_id != 0 && "There is no mem_id associated to access_id=0");
        return access_id - 1;
    }

    template <class Tensor, class Elem, class Id, class FunctorType>
    KOKKOS_FUNCTION static SIL_CONSTEXPR_IF_CXX23 typename Tensor::element_type const&
    process_access(const FunctorType& access, Tensor tensor, Elem elem)
    {
        if (elem.template uid<Id>() == 0) {
            return ::sil::detail::static_zero<typename Tensor::element_type>();
        } else {
            return access(tensor, elem);
        }
    }

    KOKKOS_FUNCTION static constexpr std::array<std::size_t, rank()>
    mem_id_to_canonical_natural_ids(std::size_t mem_id)
    {
        assert(mem_id < mem_size());
        std::array<std::size_t, rank()> ids;
        std::fill(ids.begin(), ids.end(), mem_id);
        return ids;
    }
};

} // namespace tensor

} // namespace sil
