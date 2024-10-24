// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: GPL-3.0

#pragma once

#include <ddc/ddc.hpp>

#include "young_tableau_tensor.hpp"

namespace sil {

namespace tensor {

namespace detail {
/*
// Domain of a tensor result of product between two tensors
template <class Dom1, class Dom2>
struct NaturalTensorProdDomain;

template <class... DDim1, class... DDim2>
struct NaturalTensorProdDomain<ddc::DiscreteDomain<DDim1...>, ddc::DiscreteDomain<DDim2...>>
{
    using type = ddc::detail::convert_type_seq_to_discrete_domain_t<ddc::type_seq_merge_t<
            ddc::type_seq_remove_t<ddc::detail::TypeSeq<DDim1...>, ddc::detail::TypeSeq<DDim2...>>,
            ddc::type_seq_remove_t<
                    ddc::detail::TypeSeq<DDim2...>,
                    ddc::detail::TypeSeq<DDim1...>>>>;
};

} // namespace detail

template <class Dom1, class Dom2>
using natural_tensor_prod_domain_t = detail::NaturalTensorProdDomain<Dom1, Dom2>::type;

template <class Tensor1, class Tensor2>
natural_tensor_prod_domain_t<
        typename Tensor1::discrete_domain_type,
        typename Tensor2::discrete_domain_type>
natural_tensor_prod_domain(Tensor1 tensor1, Tensor2 tensor2)
{
    return natural_tensor_prod_domain_t<
            typename Tensor1::discrete_domain_type,
            typename Tensor2::discrete_domain_type>(tensor1.domain(), tensor2.domain());
}

// Domain of a tensor result of product between two tensors
template <class T>
struct DomainFromSubindexes;

template <class... Subindex>
struct DomainFromSubindexes<ddc::DiscreteDomain<DDim1...>, ddc::DiscreteDomain<DDim2...>>
{
    using type = ddc::detail::convert_type_seq_to_discrete_domain_t<ddc::type_seq_merge_t<
            ddc::type_seq_remove_t<ddc::detail::TypeSeq<DDim1...>, ddc::detail::TypeSeq<DDim2...>>,
            ddc::type_seq_remove_t<
                    ddc::detail::TypeSeq<DDim2...>,
                    ddc::detail::TypeSeq<DDim1...>>>>;
};

} // namespace detail
*/

template <class Dom1, class Dom2>
using natural_tensor_prod_domain_t = detail::NaturalTensorProdDomain<Dom1, Dom2>::type;

template <class Tensor1, class Tensor2>
natural_tensor_prod_domain_t<
        typename Tensor1::discrete_domain_type,
        typename Tensor2::discrete_domain_type>
natural_tensor_prod_domain(Tensor1 tensor1, Tensor2 tensor2)
{
    return natural_tensor_prod_domain_t<
            typename Tensor1::discrete_domain_type,
            typename Tensor2::discrete_domain_type>(tensor1.domain(), tensor2.domain());
}
template <
        class YoungTableauIndex,
        class HeadDDim1TypeSeq,
        class ContractDDimTypeSeq,
        class TailDDim2TypeSeq>
struct TensorProd;

template <class YoungTableau, class... HeadDDim1, class... ContractDDim, class... TailDDim2>
struct TensorProd<
        YoungTableau,
        ddc::detail::TypeSeq<HeadDDim1...>,
        ddc::detail::TypeSeq<ContractDDim...>,
        ddc::detail::TypeSeq<TailDDim2...>>
{
    template <class ElementType, class LayoutStridedPolicy, class MemorySpace>
    static Tensor<
            ElementType,
            ddc::DiscreteDomain<HeadDDim1..., TailDDim2...>,
            LayoutStridedPolicy,
            MemorySpace>
    run(Tensor<ElementType,
               ddc::DiscreteDomain<HeadDDim1..., TailDDim2...>,
               std::experimental::layout_right,
               Kokkos::DefaultHostExecutionSpace::memory_space> prod_tensor,
        Tensor<ElementType,
               ddc::DiscreteDomain<YoungTableauIndex typename sil::tensor::YoungTableauTensorIndex<
                       YoungTableau,
                       HeadDDim1...,
                       ContractDDim...>>,
               LayoutStridedPolicy,
               MemorySpace> tensor1,
        Tensor<ElementType,
               ddc::DiscreteDomain<ContractDDim..., TailDDim2...>,
               LayoutStridedPolicy,
               MemorySpace> tensor2)
    {
        /*
    typename YoungTableauTensorIndex<DDim1...>::young_tableau young_tableau;
    sil::csr::Csr u = young_tableau.template u<YoungTableauIndex, DDim2...>(tensor2.domain());
*/
        ddc::Chunk uncompressed_tensor1_alloc(tensor1.domain(), ddc::HostAllocator<double>());
        sil::tensor::Tensor<
                double,
                ddc::DiscreteDomain<HeadDDim1..., ContractDDim...>,
                std::experimental::layout_right,
                Kokkos::DefaultHostExecutionSpace::memory_space>
                uncompressed_tensor1(uncompressed_tensor1_alloc);

        sil::tensor::uncompress(uncompressed_tensor1, tensor1);

        return natural_tensor_prod(prod_tensor, uncompressed_tensor1, tensor1);
    }
};

} // namespace detail

template <
        class... ProdDDim,
        class... DDim1,
        class... DDim2,
        class ElementType,
        class LayoutStridedPolicy,
        class MemorySpace,
        class YoungTableau>
Tensor<ElementType,
       ddc::DiscreteDomain<ProdDDim...>,
       std::experimental::layout_right,
       Kokkos::DefaultHostExecutionSpace::memory_space>
tensor_prod(
        Tensor<ElementType,
               ddc::DiscreteDomain<ProdDDim...>,
               std::experimental::layout_right,
               Kokkos::DefaultHostExecutionSpace::memory_space> prod_tensor,
        Tensor<ElementType,
               ddc::DiscreteDomain<YoungTableauTensorIndex<YoungTableau, DDim1...>>,
               LayoutStridedPolicy,
               MemorySpace> tensor1,
        Tensor<ElementType, ddc::DiscreteDomain<DDim2...>, LayoutStridedPolicy, MemorySpace>
                tensor2)
{
    static_assert(std::is_same_v<
                  ddc::type_seq_remove_t<
                          ddc::detail::TypeSeq<DDim1...>,
                          ddc::detail::TypeSeq<ProdDDim...>>,
                  ddc::type_seq_remove_t<
                          ddc::detail::TypeSeq<DDim2...>,
                          ddc::detail::TypeSeq<ProdDDim...>>>);
    return detail::TensorProd<
            YoungTableau,
            ddc::type_seq_remove_t<
                    ddc::detail::TypeSeq<ProdDDim...>,
                    ddc::detail::TypeSeq<DDim2...>>,
            ddc::type_seq_remove_t<
                    ddc::detail::TypeSeq<DDim1...>,
                    ddc::detail::TypeSeq<ProdDDim...>>,
            ddc::type_seq_remove_t<
                    ddc::detail::TypeSeq<ProdDDim...>,
                    ddc::detail::TypeSeq<DDim1...>>>::run(prod_tensor, tensor1, tensor2);
}

} // namespace tensor

} // namespace sil
