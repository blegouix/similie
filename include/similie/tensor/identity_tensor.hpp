// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <ddc/ddc.hpp>

#include "tensor_impl.hpp"

namespace sil {

namespace tensor {

// struct representing an identity tensor (no storage).
template <class... TensorIndex>
struct TensorIdentityIndex
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
        return 0;
    }

    static constexpr std::size_t access_size()
    {
        return 2;
    }

    static constexpr std::pair<std::vector<double>, std::vector<std::size_t>> mem_lin_comb(
            std::array<std::size_t, sizeof...(TensorIndex)> const ids)
    {
        assert(false);
        return std::pair<
                std::vector<double>,
                std::vector<std::size_t>>(std::vector<double> {1.}, std::vector<std::size_t> {});
    }

    static constexpr std::size_t access_id(
            std::array<std::size_t, sizeof...(TensorIndex)> const ids)
    {
        if (!std::all_of(ids.begin(), ids.end(), [&](const std::size_t id) {
                return id == *ids.begin();
            })) {
            return 0;
        } else {
            return 1;
        }
    }

    static constexpr std::pair<std::vector<double>, std::vector<std::size_t>>
    access_id_to_mem_lin_comb(std::size_t access_id)
    {
        assert(access_id != 0 && "There is no mem_lin_comb associated to access_id=0");
        return std::pair<
                std::vector<double>,
                std::vector<std::size_t>>(std::vector<double> {}, std::vector<std::size_t> {});
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

} // namespace tensor

} // namespace sil