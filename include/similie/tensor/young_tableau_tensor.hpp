// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: MIT

#pragma once

#include <ddc/ddc.hpp>

#include <similie/csr/csr.hpp>
#include <similie/misc/stride.hpp>
#include <similie/young_tableau/young_tableau.hpp>

#include "tensor_impl.hpp"

namespace sil {

namespace tensor {

// struct representing an abstract unique index sweeping on all possible combination of natural indices, for a summetric tensor.
template <class YoungTableau, TensorNatIndex... TensorIndex>
struct TensorYoungTableauIndex
{
    static constexpr bool is_tensor_index = true;
    static constexpr bool is_explicitely_stored_tensor = false;

    using young_tableau = YoungTableau;

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
        return YoungTableau::irrep_dim();
    }

    KOKKOS_FUNCTION static constexpr std::size_t access_size()
    {
        return size();
    }

    KOKKOS_FUNCTION static constexpr std::pair<std::vector<double>, std::vector<std::size_t>>
    mem_lin_comb(std::array<std::size_t, sizeof...(TensorIndex)> const natural_ids)
    {
        std::pair<std::vector<double>, std::vector<std::size_t>> result {};
        constexpr csr::Csr v = young_tableau::template v<
                TensorYoungTableauIndex<YoungTableau, TensorIndex...>,
                TensorIndex...>(ddc::DiscreteDomain<TensorIndex...>(
                ddc::DiscreteElement<TensorIndex...>(ddc::DiscreteElement<TensorIndex>(0)...),
                ddc::DiscreteVector<TensorIndex...>(
                        ddc::DiscreteVector<TensorIndex>(TensorIndex::size())...)));
        for (std::size_t j = 0; j < v.values().size(); ++j) {
            if (((v.idx()[ddc::type_seq_rank_v<TensorIndex, ddc::detail::TypeSeq<TensorIndex...>>]
                         [j]
                  == TensorIndex::access_id(natural_ids))
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

    KOKKOS_FUNCTION static constexpr std::size_t access_id(
            std::array<std::size_t, sizeof...(TensorIndex)> const natural_ids)
    {
        return ((misc::detail::stride<TensorIndex, TensorIndex...>()
                 * natural_ids
                         [ddc::type_seq_rank_v<TensorIndex, ddc::detail::TypeSeq<TensorIndex...>>])
                + ...);
    }

    KOKKOS_FUNCTION static constexpr std::pair<std::vector<double>, std::vector<std::size_t>>
    access_id_to_mem_lin_comb(std::size_t access_id)
    {
        std::pair<std::vector<double>, std::vector<std::size_t>> result {};
        constexpr csr::Csr v = young_tableau::template v<
                TensorYoungTableauIndex<YoungTableau, TensorIndex...>,
                TensorIndex...>(ddc::DiscreteDomain<TensorIndex...>(
                ddc::DiscreteElement<TensorIndex...>(ddc::DiscreteElement<TensorIndex>(0)...),
                ddc::DiscreteVector<TensorIndex...>(
                        ddc::DiscreteVector<TensorIndex>(TensorIndex::size())...)));
        for (std::size_t j = 0; j < v.values().size(); ++j) {
            if (((v.idx()[ddc::type_seq_rank_v<TensorIndex, ddc::detail::TypeSeq<TensorIndex...>>]
                         [j]
                  == ((access_id % misc::detail::next_stride<TensorIndex, TensorIndex...>())
                      / misc::detail::stride<TensorIndex, TensorIndex...>()))
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

    template <class Tensor, class Elem, class Id, class FunctorType>
    KOKKOS_FUNCTION static SIL_CONSTEXPR_IF_CXX23 typename Tensor::element_type const&
    process_access(const FunctorType& access, Tensor tensor, Elem elem)
    {
        return access(tensor, elem);
    }
};

// Compress & uncompress (multiply by young_tableau.u or young_tableau.v
template <class YoungTableauIndex, class... Id>
tensor::Tensor<
        double,
        ddc::DiscreteDomain<YoungTableauIndex>,
        Kokkos::layout_right,
        Kokkos::DefaultHostExecutionSpace::memory_space>
compress(
        tensor::Tensor<
                double,
                ddc::DiscreteDomain<YoungTableauIndex>,
                Kokkos::layout_right,
                Kokkos::DefaultHostExecutionSpace::memory_space> compressed,
        tensor::Tensor<
                double,
                ddc::DiscreteDomain<Id...>,
                Kokkos::layout_right,
                Kokkos::DefaultHostExecutionSpace::memory_space> tensor)
{
    typename YoungTableauIndex::young_tableau young_tableau;
    csr::Csr u = young_tableau.template u<YoungTableauIndex, Id...>(tensor.domain());

    return csr::tensor_prod(compressed, u, tensor);
}

template <class YoungTableauIndex, class... Id>
tensor::Tensor<
        double,
        ddc::DiscreteDomain<Id...>,
        Kokkos::layout_right,
        Kokkos::DefaultHostExecutionSpace::memory_space>
uncompress(
        tensor::Tensor<
                double,
                ddc::DiscreteDomain<Id...>,
                Kokkos::layout_right,
                Kokkos::DefaultHostExecutionSpace::memory_space> uncompressed,
        tensor::Tensor<
                double,
                ddc::DiscreteDomain<YoungTableauIndex>,
                Kokkos::layout_right,
                Kokkos::DefaultHostExecutionSpace::memory_space> tensor)
{
    typename YoungTableauIndex::young_tableau young_tableau;
    csr::Csr v = young_tableau.template v<YoungTableauIndex, Id...>(uncompressed.domain());

    return csr::tensor_prod(uncompressed, tensor, v);
}

} // namespace tensor

} // namespace sil
