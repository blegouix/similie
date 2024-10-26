// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: GPL-3.0

#pragma once

#include <ddc/ddc.hpp>

#include "tensor_impl.hpp"

namespace sil {

namespace tensor {

// struct representing and index for a diagonal tensor (only diagonal is stored).
template <class... TensorIndex>
struct DiagonalTensorIndex
{
    static constexpr bool is_natural_tensor_index = false;

    using subindexes_domain_t = ddc::DiscreteDomain<TensorIndex...>;

    static constexpr subindexes_domain_t subindexes_domain()
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
        return std::min({TensorIndex::mem_size()...});
    }

    static constexpr std::size_t access_size()
    {
        return 1 + mem_size();
    }

    static constexpr std::pair<std::vector<double>, std::vector<std::size_t>> mem_id(
            ddc::DiscreteElement<TensorIndex...> elem)
    {
        std::array<std::size_t, sizeof...(TensorIndex)> ids {
                elem.template uid<TensorIndex>()...}; // better with std::initialize_list ?
        static_assert(std::all_of(ids.begin(), ids.end(), [&](const std::size_t id) {
            return id == *ids.begin();
        }));
        return std::pair<
                std::vector<double>,
                std::vector<
                        std::size_t>>(std::vector<double> {}, std::vector<std::size_t> {ids[0]});
    }

    static constexpr std::size_t access_id(ddc::DiscreteElement<TensorIndex...> elem)
    {
        std::array<std::size_t, sizeof...(TensorIndex)> ids {
                elem.template uid<TensorIndex>()...}; // better with std::initialize_list ?
        if constexpr (!std::all_of(ids.begin(), ids.end(), [&](const std::size_t id) {
                          return id == *ids.begin();
                      })) {
            return 0;
        } else {
            return 1 + std::get<1>(mem_id(elem))[0];
        }
    }

    static constexpr std::pair<std::vector<double>, std::vector<std::size_t>> access_id_to_mem_id(
            std::size_t access_id)
    {
        assert(access_id != 0 && "There is no mem_id associated to access_id=0");
        return std::pair<std::vector<double>, std::vector<std::size_t>>(
                std::vector<double> {},
                std::vector<std::size_t> {access_id - 1});
    }

    template <class Tensor, class Elem, class Id>
    static constexpr Tensor::element_type process_access(
            std::function<typename Tensor::element_type(Tensor, Elem)> access,
            Tensor tensor,
            Elem elem)
    {
        if (elem.template uid<Id>() == 0) {
            return 0.;
        } else {
            return access(tensor, elem);
        }
    }
};

namespace detail {
template <class... SubIndex>
struct IsTensorIndex<DiagonalTensorIndex<SubIndex...>>
{
    using type = std::true_type;
};

} // namespace detail

} // namespace tensor

} // namespace sil
