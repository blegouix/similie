// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: GPL-3.0

#pragma once

#include <ddc/ddc.hpp>

#include "stride.hpp"
#include "tensor.hpp"

namespace sil {

namespace tensor {

// struct representing an abstract unique index sweeping on all possible combination of natural indices, for a full tensor (dense with no particular structure).
template <class... TensorIndex>
struct TensorFullIndex
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
        return (TensorIndex::mem_size() * ...);
    }

    static constexpr std::size_t access_size()
    {
        return (TensorIndex::access_size() * ...);
    }

    static constexpr std::pair<std::vector<double>, std::vector<std::size_t>> mem_id(
            std::array<std::size_t, sizeof...(TensorIndex)> const ids)
    {
        return std::pair<std::vector<double>, std::vector<std::size_t>>(
                std::vector<double> {1.},
                std::vector<std::size_t> {access_id(ids)});
    }

    static constexpr std::size_t access_id(
            std::array<std::size_t, sizeof...(TensorIndex)> const ids)
    {
        return ((sil::misc::detail::stride<TensorIndex, TensorIndex...>()
                 * ids[ddc::type_seq_rank_v<TensorIndex, ddc::detail::TypeSeq<TensorIndex...>>])
                + ...);
    }

    static constexpr std::pair<std::vector<double>, std::vector<std::size_t>> access_id_to_mem_id(
            std::size_t access_id)
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
