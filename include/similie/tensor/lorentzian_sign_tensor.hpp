// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <ddc/ddc.hpp>

#include "tensor_impl.hpp"

namespace sil {

namespace tensor {

// struct representing an identity tensor (no storage).
template <class Q, class... TensorIndex>
struct TensorLorentzianSignIndex
{
    static constexpr bool is_tensor_index = true;
    static constexpr bool is_explicitely_stored_tensor = true;

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
        return 0;
    }

    static constexpr std::size_t access_size()
    {
        return 3;
    }

    static constexpr std::size_t mem_id(
            std::array<std::size_t, sizeof...(TensorIndex)> const natural_ids)
    {
        return std::numeric_limits<std::size_t>::infinity();
    }

    static constexpr std::size_t access_id(
            std::array<std::size_t, sizeof...(TensorIndex)> const natural_ids)
    {
        if (!std::all_of(natural_ids.begin(), natural_ids.end(), [&](const std::size_t id) {
                return id == *natural_ids.begin();
            })) {
            return 0;
        } else {
            if (natural_ids[0] < Q::value) {
                return 1;
            } else {
                return 2;
            }
        }
    }

    static constexpr std::size_t access_id_to_mem_id(std::size_t access_id)
    {
        return std::numeric_limits<std::size_t>::infinity();
    }

    template <class Tensor, class Elem, class Id>
    static constexpr Tensor::element_type process_access(
            std::function<typename Tensor::element_type(Tensor, Elem)> access,
            Tensor tensor,
            Elem elem)
    {
        if (elem.template uid<Id>() == 0) {
            return 0.;
        } else if (elem.template uid<Id>() == 1) {
            return -access(tensor, elem);
        } else {
            return access(tensor, elem);
        }
    }

    static constexpr std::array<std::size_t, rank()> mem_id_to_canonical_natural_ids(
            std::size_t mem_id)
    {
        assert(mem_id < mem_size()); // Always false
        std::array<std::size_t, rank()> ids;
        return ids;
    }
};

} // namespace tensor

} // namespace sil
