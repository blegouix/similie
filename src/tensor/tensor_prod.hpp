// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: GPL-3.0

#pragma once

#include <ddc/ddc.hpp>

namespace sil {

namespace tensor {

// Domain of a tensor result of product between two tensors
template <class Dom1, class Dom2>
struct TensorProdDomain;

template <class... DDim1, class... DDim2>
struct TensorProdDomain<ddc::DiscreteDomain<DDim1...>, ddc::DiscreteDomain<DDim2...>>
{
    using type = ddc::detail::convert_type_seq_to_discrete_domain_t<ddc::type_seq_merge_t<
            ddc::type_seq_remove_t<ddc::detail::TypeSeq<DDim1...>, ddc::detail::TypeSeq<DDim2...>>,
            ddc::type_seq_remove_t<
                    ddc::detail::TypeSeq<DDim2...>,
                    ddc::detail::TypeSeq<DDim1...>>>>;
};

template <class Dom1, class Dom2>
using tensor_prod_domain_t = TensorProdDomain<Dom1, Dom2>::type;

template <class Tensor1, class Tensor2>
tensor_prod_domain_t<typename Tensor1::discrete_domain_type, typename Tensor2::discrete_domain_type>
tensor_prod_domain(Tensor1 tensor1, Tensor2 tensor2)
{
    return sil::tensor::tensor_prod_domain_t<
            typename Tensor1::discrete_domain_type,
            typename Tensor2::discrete_domain_type>(tensor1.domain(), tensor2.domain());
}

// Product between two tensors. Only natural indexing supported atm.
template <class HeadDDim1TypeSeq, class ContractDDimTypeSeq, class TailDDim2TypeSeq>
struct TensorProd;

template <class... HeadDDim1, class... ContractDDim, class... TailDDim2>
struct TensorProd<
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
               ddc::DiscreteDomain<HeadDDim1..., ContractDDim...>,
               LayoutStridedPolicy,
               MemorySpace> tensor1,
        Tensor<ElementType,
               ddc::DiscreteDomain<ContractDDim..., TailDDim2...>,
               LayoutStridedPolicy,
               MemorySpace> tensor2)
    {
        ddc::for_each(
                prod_tensor.domain(),
                [&](ddc::DiscreteElement<HeadDDim1..., TailDDim2...> elem) {
                    prod_tensor(elem) = ddc::transform_reduce(
                            tensor1.template domain<ContractDDim...>(),
                            0.,
                            ddc::reducer::sum<ElementType>(),
                            [&](ddc::DiscreteElement<ContractDDim...> contract_elem) {
                                return tensor1(ddc::select<HeadDDim1...>(elem), contract_elem)
                                       * tensor2(ddc::select<TailDDim2...>(elem), contract_elem);
                            });
                });
        return prod_tensor;
    }
};

template <
        class... ProdDDim,
        class... DDim1,
        class... DDim2,
        class ElementType,
        class LayoutStridedPolicy,
        class MemorySpace>
Tensor<ElementType,
       ddc::DiscreteDomain<ProdDDim...>,
       std::experimental::layout_right,
       Kokkos::DefaultHostExecutionSpace::memory_space>
tensor_prod(
        Tensor<ElementType,
               ddc::DiscreteDomain<ProdDDim...>,
               std::experimental::layout_right,
               Kokkos::DefaultHostExecutionSpace::memory_space> prod_tensor,
        Tensor<ElementType, ddc::DiscreteDomain<DDim1...>, LayoutStridedPolicy, MemorySpace>
                tensor1,
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
    return TensorProd<
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
