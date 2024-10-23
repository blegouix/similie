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
    using young_tableau = YoungTableau;

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
        if (elem.template uid<Id>() == 0) {
            return 0.;
        } else {
            return access(tensor, elem);
        }
    }
};

namespace detail {

template <class YoungTableau, class... SubIndex>
struct IsTensorIndex<YoungTableauTensorIndex<YoungTableau, SubIndex...>>
{
    using type = std::true_type;
};

} // namespace detail

// Compress & uncompress (multiply by young_tableau.u or young_tableau.v
template <class YoungTableauIndex, class... Id>
sil::tensor::Tensor<
        double,
        ddc::DiscreteDomain<YoungTableauIndex>,
        std::experimental::layout_right,
        Kokkos::DefaultHostExecutionSpace::memory_space>
compress(
        sil::tensor::Tensor<
                double,
                ddc::DiscreteDomain<YoungTableauIndex>,
                std::experimental::layout_right,
                Kokkos::DefaultHostExecutionSpace::memory_space> compressed,
        sil::tensor::Tensor<
                double,
                ddc::DiscreteDomain<Id...>,
                std::experimental::layout_right,
                Kokkos::DefaultHostExecutionSpace::memory_space> tensor)
{
    typename YoungTableauIndex::young_tableau young_tableau;
    sil::csr::Csr u = young_tableau.template u<YoungTableauIndex, Id...>(tensor.domain());

    return sil::csr::tensor_prod(compressed, u, tensor);
}

template <class YoungTableauIndex, class... Id>
sil::tensor::Tensor<
        double,
        ddc::DiscreteDomain<Id...>,
        std::experimental::layout_right,
        Kokkos::DefaultHostExecutionSpace::memory_space>
uncompress(
        sil::tensor::Tensor<
                double,
                ddc::DiscreteDomain<Id...>,
                std::experimental::layout_right,
                Kokkos::DefaultHostExecutionSpace::memory_space> uncompressed,
        sil::tensor::Tensor<
                double,
                ddc::DiscreteDomain<YoungTableauIndex>,
                std::experimental::layout_right,
                Kokkos::DefaultHostExecutionSpace::memory_space> tensor)
{
    typename YoungTableauIndex::young_tableau young_tableau;
    sil::csr::Csr v = young_tableau.template v<YoungTableauIndex, Id...>(uncompressed.domain());

    return sil::csr::tensor_prod(uncompressed, tensor, v);
}

} // namespace tensor

} // namespace sil
