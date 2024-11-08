// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: GPL-3.0

#pragma once

#include <ddc/ddc.hpp>

#include "csr.hpp"
#include "stride.hpp"
#include "tensor.hpp"
#include "young_tableau.hpp"

namespace sil {

namespace tensor {

// struct representing an abstract unique index sweeping on all possible combination of natural indices, for a summetric tensor.
template <class YoungTableau, class... TensorIndex>
struct YoungTableauTensorIndex
{
    static constexpr bool is_natural_tensor_index = false;

    using young_tableau = YoungTableau;

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
        return YoungTableau::irrep_dim();
    }

    static constexpr std::size_t access_size()
    {
        return size();
    }

    static constexpr std::pair<std::vector<double>, std::vector<std::size_t>> mem_id(
            std::array<std::size_t, sizeof...(TensorIndex)> const ids)
    {
        std::pair<std::vector<double>, std::vector<std::size_t>> result {};
        constexpr sil::csr::Csr v = young_tableau::template v<
                YoungTableauTensorIndex<YoungTableau, TensorIndex...>,
                TensorIndex...>(ddc::DiscreteDomain<TensorIndex...>(
                ddc::DiscreteElement<TensorIndex...>(ddc::DiscreteElement<TensorIndex>(0)...),
                ddc::DiscreteVector<TensorIndex...>(
                        ddc::DiscreteVector<TensorIndex>(TensorIndex::size())...)));
        for (std::size_t j = 0; j < v.values().size(); ++j) {
            if (((v.idx()[ddc::type_seq_rank_v<TensorIndex, ddc::detail::TypeSeq<TensorIndex...>>]
                         [j]
                  == TensorIndex::access_id(ids))
                 && ...)) {
                std::get<0>(result).push_back(v.values()[j]);
                std::size_t k = 0;
                while (k < v.coalesc_idx().size() - 1 && v.coalesc_idx()[k + 1] <= j) {
                    k++;
                }
                std::get<1>(result).push_back(k);
            }
        }
        return result;
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
        std::pair<std::vector<double>, std::vector<std::size_t>> result {};
        constexpr sil::csr::Csr v = young_tableau::template v<
                YoungTableauTensorIndex<YoungTableau, TensorIndex...>,
                TensorIndex...>(ddc::DiscreteDomain<TensorIndex...>(
                ddc::DiscreteElement<TensorIndex...>(ddc::DiscreteElement<TensorIndex>(0)...),
                ddc::DiscreteVector<TensorIndex...>(
                        ddc::DiscreteVector<TensorIndex>(TensorIndex::size())...)));
        for (std::size_t j = 0; j < v.values().size(); ++j) {
            if (((v.idx()[ddc::type_seq_rank_v<TensorIndex, ddc::detail::TypeSeq<TensorIndex...>>]
                         [j]
                  == ((access_id % sil::misc::detail::next_stride<TensorIndex, TensorIndex...>())
                      / sil::misc::detail::stride<TensorIndex, TensorIndex...>()))
                 && ...)) {
                std::get<0>(result).push_back(v.values()[j]);
                std::size_t k = 0;
                while (k < v.coalesc_idx().size() - 1 && v.coalesc_idx()[k + 1] <= j) {
                    k++;
                }
                std::get<1>(result).push_back(k);
            }
        }
        return result;
    }

    template <class Tensor, class Elem, class Id>
    static constexpr Tensor::element_type process_access(
            std::function<typename Tensor::element_type(Tensor, Elem)> access,
            Tensor tensor,
            Elem const elem)
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

// Compress & uncompress (multiply by young_tableau.u or young_tableau.v
template <class YoungTableauIndex, class... Id>
sil::tensor::Tensor<
        double,
        ddc::DiscreteDomain<YoungTableauIndex>,
        Kokkos::layout_right,
        Kokkos::DefaultHostExecutionSpace::memory_space>
compress(
        sil::tensor::Tensor<
                double,
                ddc::DiscreteDomain<YoungTableauIndex>,
                Kokkos::layout_right,
                Kokkos::DefaultHostExecutionSpace::memory_space> compressed,
        sil::tensor::Tensor<
                double,
                ddc::DiscreteDomain<Id...>,
                Kokkos::layout_right,
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
        Kokkos::layout_right,
        Kokkos::DefaultHostExecutionSpace::memory_space>
uncompress(
        sil::tensor::Tensor<
                double,
                ddc::DiscreteDomain<Id...>,
                Kokkos::layout_right,
                Kokkos::DefaultHostExecutionSpace::memory_space> uncompressed,
        sil::tensor::Tensor<
                double,
                ddc::DiscreteDomain<YoungTableauIndex>,
                Kokkos::layout_right,
                Kokkos::DefaultHostExecutionSpace::memory_space> tensor)
{
    typename YoungTableauIndex::young_tableau young_tableau;
    sil::csr::Csr v = young_tableau.template v<YoungTableauIndex, Id...>(uncompressed.domain());

    return sil::csr::tensor_prod(uncompressed, tensor, v);
}

} // namespace tensor

} // namespace sil
