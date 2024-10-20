// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: GPL-3.0

#pragma once

#include <ddc/ddc.hpp>

#include "tensor.hpp"
#include "young_tableau.hpp"

namespace sil {

namespace tensor {

// struct representing an abstract unique index sweeping on all possible combination of natural indexes, for a summetric tensor.
template <class YoungTableau, class... TensorIndex>
struct YoungTableauTensorIndex
{
    static constexpr YoungTableau s_young_tableau = YoungTableau();

public:
    static constexpr YoungTableau young_tableau()
    {
        return s_young_tableau;
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
        return YoungTableau::irrep_dim();
    }

    static constexpr std::size_t access_size()
    {
        return mem_size() + 1;
    }

    template <class... CDim>
    static constexpr std::pair<std::vector<double>, std::vector<std::size_t>> mem_id()
    {
        // static_assert(rank() == sizeof...(CDim));
        return std::pair<
                std::vector<double>,
                std::vector<std::size_t>>(std::vector<double> {}, std::vector<std::size_t> {0});
    }

    template <class... CDim>
    static constexpr std::size_t access_id()
    {
        return std::get<1>(mem_id<CDim...>())[0];
    }

    static constexpr std::pair<std::vector<double>, std::vector<std::size_t>> access_id_to_mem_id(
            std::size_t access_id)
    {
        return std::pair<
                std::vector<double>,
                std::vector<
                        std::size_t>>(std::vector<double> {}, std::vector<std::size_t> {access_id});
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

namespace detail {

template <class YoungTableau, class... SubIndex>
struct IsTensorIndex<YoungTableauTensorIndex<YoungTableau, SubIndex...>>
{
    using type = std::true_type;
};

} // namespace detail

} // namespace tensor

} // namespace sil
