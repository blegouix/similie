// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: GPL-3.0

#pragma once

#include <ddc/ddc.hpp>

#include "young_tableau_tensor.hpp"

namespace sil {

namespace tensor {

namespace detail {

template <class Index>
struct SubindexesDomain;

template <template <class...> class T, class... SubIndex>
struct SubindexesDomain<T<SubIndex...>>
{
    using type = ddc::DiscreteDomain<SubIndex...>;

    static constexpr type run()
    {
        return ddc::DiscreteDomain<SubIndex...>(
                ddc::DiscreteElement<SubIndex...>(ddc::DiscreteElement<SubIndex>(0)...),
                ddc::DiscreteVector<SubIndex...>(
                        ddc::DiscreteVector<SubIndex>(SubIndex::size())...));
    }
};

} // namespace detail

template <class T>
using subindexes_domain_t = detail::SubindexesDomain<T>::type;

template <class T>
static constexpr subindexes_domain_t<T> subindexes_domain()
{
    return detail::SubindexesDomain<T>::run();
};

namespace detail {

template <class Index1, class HeadDDim1TypeSeq, class ContractDDimTypeSeq, class TailDDim2TypeSeq>
struct TensorProd;

template <class Index1, class... HeadDDim1, class... ContractDDim, class... TailDDim2>
struct TensorProd<
        Index1,
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
        Tensor<ElementType, ddc::DiscreteDomain<Index1>, LayoutStridedPolicy, MemorySpace> tensor1,
        Tensor<ElementType,
               ddc::DiscreteDomain<ContractDDim..., TailDDim2...>,
               LayoutStridedPolicy,
               MemorySpace> tensor2)
    {
        /*
    typename YoungTableauTensorIndex<DDim1...>::young_tableau young_tableau;
    sil::csr::Csr u = young_tableau.template u<YoungTableauIndex, DDim2...>(tensor2.domain());
*/
        ddc::Chunk uncompressed_tensor1_alloc(
                Index1::subindexes_domain(),
                ddc::HostAllocator<double>());
        sil::tensor::Tensor<
                double,
                typename Index1::subindexes_domain_t,
                std::experimental::layout_right,
                Kokkos::DefaultHostExecutionSpace::memory_space>
                uncompressed_tensor1(uncompressed_tensor1_alloc);

        sil::tensor::uncompress(uncompressed_tensor1, tensor1);

        return natural_tensor_prod(prod_tensor, uncompressed_tensor1, tensor2);
    }
};

} // namespace detail

template <
        class... ProdDDim,
        class Index1,
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
        Tensor<ElementType, ddc::DiscreteDomain<Index1>, LayoutStridedPolicy, MemorySpace> tensor1,
        Tensor<ElementType, ddc::DiscreteDomain<DDim2...>, LayoutStridedPolicy, MemorySpace>
                tensor2)
{
    static_assert(std::is_same_v<
                  ddc::type_seq_remove_t<
                          ddc::to_type_seq_t<typename Index1::subindexes_domain_t>,
                          ddc::detail::TypeSeq<ProdDDim...>>,
                  ddc::type_seq_remove_t<
                          ddc::detail::TypeSeq<DDim2...>,
                          ddc::detail::TypeSeq<ProdDDim...>>>);
    return detail::TensorProd<
            Index1,
            ddc::type_seq_remove_t<
                    ddc::detail::TypeSeq<ProdDDim...>,
                    ddc::detail::TypeSeq<DDim2...>>,
            ddc::type_seq_remove_t<
                    ddc::to_type_seq_t<typename Index1::subindexes_domain_t>,
                    ddc::detail::TypeSeq<ProdDDim...>>,
            ddc::type_seq_remove_t<
                    ddc::detail::TypeSeq<ProdDDim...>,
                    ddc::to_type_seq_t<typename Index1::subindexes_domain_t>>>::
            run(prod_tensor, tensor1, tensor2);
}

namespace detail {

template <
        class Index1,
        class Index2,
        class HeadDDim1TypeSeq,
        class ContractDDimTypeSeq,
        class TailDDim2TypeSeq>
struct TensorProd2;

template <class Index1, class Index2, class... HeadDDim1, class... ContractDDim, class... TailDDim2>
struct TensorProd2<
        Index1,
        Index2,
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
        Tensor<ElementType, ddc::DiscreteDomain<Index1>, LayoutStridedPolicy, MemorySpace> tensor1,
        Tensor<ElementType, ddc::DiscreteDomain<Index2>, LayoutStridedPolicy, MemorySpace> tensor2)
    {
        /*
    typename YoungTableauTensorIndex<DDim1...>::young_tableau young_tableau;
    sil::csr::Csr u = young_tableau.template u<YoungTableauIndex, DDim2...>(tensor2.domain());
*/
        ddc::Chunk uncompressed_tensor1_alloc(
                Index1::subindexes_domain(),
                ddc::HostAllocator<double>());
        sil::tensor::Tensor<
                double,
                typename Index1::subindexes_domain_t,
                std::experimental::layout_right,
                Kokkos::DefaultHostExecutionSpace::memory_space>
                uncompressed_tensor1(uncompressed_tensor1_alloc);

        ddc::Chunk uncompressed_tensor2_alloc(
                Index2::subindexes_domain(),
                ddc::HostAllocator<double>());
        sil::tensor::Tensor<
                double,
                typename Index2::subindexes_domain_t,
                std::experimental::layout_right,
                Kokkos::DefaultHostExecutionSpace::memory_space>
                uncompressed_tensor2(uncompressed_tensor2_alloc);

        sil::tensor::uncompress(uncompressed_tensor2, tensor2);

        return natural_tensor_prod(prod_tensor, uncompressed_tensor1, uncompressed_tensor2);
    }
};

} // namespace detail

template <
        class... ProdDDim,
        class Index1,
        class Index2,
        class ElementType,
        class LayoutStridedPolicy,
        class MemorySpace>
Tensor<ElementType,
       ddc::DiscreteDomain<ProdDDim...>,
       std::experimental::layout_right,
       Kokkos::DefaultHostExecutionSpace::memory_space>
tensor_prod2(
        Tensor<ElementType,
               ddc::DiscreteDomain<ProdDDim...>,
               std::experimental::layout_right,
               Kokkos::DefaultHostExecutionSpace::memory_space> prod_tensor,
        Tensor<ElementType, ddc::DiscreteDomain<Index1>, LayoutStridedPolicy, MemorySpace> tensor1,
        Tensor<ElementType, ddc::DiscreteDomain<Index2>, LayoutStridedPolicy, MemorySpace> tensor2)
{
    static_assert(std::is_same_v<
                  ddc::type_seq_remove_t<
                          ddc::to_type_seq_t<typename Index1::subindexes_domain_t>,
                          ddc::detail::TypeSeq<ProdDDim...>>,
                  ddc::type_seq_remove_t<
                          ddc::to_type_seq_t<typename Index2::subindexes_domain_t>,
                          ddc::detail::TypeSeq<ProdDDim...>>>);
    return detail::TensorProd2<
            Index1,
            Index2,
            ddc::type_seq_remove_t<
                    ddc::detail::TypeSeq<ProdDDim...>,
                    ddc::to_type_seq_t<typename Index2::subindexes_domain_t>>,
            ddc::type_seq_remove_t<
                    ddc::to_type_seq_t<typename Index1::subindexes_domain_t>,
                    ddc::detail::TypeSeq<ProdDDim...>>,
            ddc::type_seq_remove_t<
                    ddc::detail::TypeSeq<ProdDDim...>,
                    ddc::to_type_seq_t<typename Index1::subindexes_domain_t>>>::
            run(prod_tensor, tensor1, tensor2);
}
} // namespace tensor

} // namespace sil
