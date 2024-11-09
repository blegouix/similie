// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: GPL-3.0

#pragma once

#include <ddc/ddc.hpp>

#include "character.hpp"
#include "specialization.hpp"
#include "young_tableau_tensor.hpp"

namespace sil {

namespace tensor {

namespace detail {

template <class Index>
struct SubindicesDomain;

template <template <class...> class T, class... SubIndex>
struct SubindicesDomain<T<SubIndex...>>
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
using subindices_domain_t = detail::SubindicesDomain<T>::type;

template <class T>
static constexpr subindices_domain_t<T> subindices_domain()
{
    return detail::SubindicesDomain<T>::run();
};

// Young-dense product
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
               Kokkos::layout_right,
               Kokkos::DefaultHostExecutionSpace::memory_space> prod_tensor,
        Tensor<ElementType, ddc::DiscreteDomain<Index1>, LayoutStridedPolicy, MemorySpace> tensor1,
        Tensor<ElementType,
               ddc::DiscreteDomain<ContractDDim..., TailDDim2...>,
               LayoutStridedPolicy,
               MemorySpace> tensor2)
    {
        /*
    typename TensorYoungTableauIndex<DDim1...>::young_tableau young_tableau;
    sil::csr::Csr u = young_tableau.template u<YoungTableauIndex, DDim2...>(tensor2.domain());
*/
        ddc::Chunk uncompressed_tensor1_alloc(
                Index1::subindices_domain(),
                ddc::HostAllocator<double>());
        sil::tensor::Tensor<
                double,
                typename Index1::subindices_domain_t,
                Kokkos::layout_right,
                Kokkos::DefaultHostExecutionSpace::memory_space>
                uncompressed_tensor1(uncompressed_tensor1_alloc);

        sil::tensor::uncompress(uncompressed_tensor1, tensor1);

        return natural_tensor_prod(prod_tensor, uncompressed_tensor1, tensor2);
    }
};

} // namespace detail

template <
        class... ProdDDim,
        misc::Specialization<TensorYoungTableauIndex> Index1,
        class... DDim2,
        class ElementType,
        class LayoutStridedPolicy,
        class MemorySpace>
Tensor<ElementType,
       ddc::DiscreteDomain<ProdDDim...>,
       Kokkos::layout_right,
       Kokkos::DefaultHostExecutionSpace::memory_space>
tensor_prod(
        Tensor<ElementType,
               ddc::DiscreteDomain<ProdDDim...>,
               Kokkos::layout_right,
               Kokkos::DefaultHostExecutionSpace::memory_space> prod_tensor,
        Tensor<ElementType, ddc::DiscreteDomain<Index1>, LayoutStridedPolicy, MemorySpace> tensor1,
        Tensor<ElementType, ddc::DiscreteDomain<DDim2...>, LayoutStridedPolicy, MemorySpace>
                tensor2)
{
    static_assert(std::is_same_v<
                  ddc::type_seq_remove_t<
                          ddc::to_type_seq_t<typename Index1::subindices_domain_t>,
                          ddc::detail::TypeSeq<ProdDDim...>>,
                  ddc::type_seq_remove_t<
                          ddc::detail::TypeSeq<DDim2...>,
                          ddc::detail::TypeSeq<ProdDDim...>>>);
    std::cout << "test";
    return detail::TensorProd<
            Index1,
            ddc::type_seq_remove_t<
                    ddc::detail::TypeSeq<ProdDDim...>,
                    ddc::detail::TypeSeq<DDim2...>>,
            ddc::type_seq_remove_t<
                    ddc::to_type_seq_t<typename Index1::subindices_domain_t>,
                    ddc::detail::TypeSeq<ProdDDim...>>,
            ddc::type_seq_remove_t<
                    ddc::detail::TypeSeq<ProdDDim...>,
                    ddc::to_type_seq_t<typename Index1::subindices_domain_t>>>::
            run(prod_tensor, tensor1, tensor2);
}

// Young-young product
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
               Kokkos::layout_right,
               Kokkos::DefaultHostExecutionSpace::memory_space> prod_tensor,
        Tensor<ElementType, ddc::DiscreteDomain<Index1>, LayoutStridedPolicy, MemorySpace> tensor1,
        Tensor<ElementType, ddc::DiscreteDomain<Index2>, LayoutStridedPolicy, MemorySpace> tensor2)
    {
        /*
    typename TensorYoungTableauIndex<DDim1...>::young_tableau young_tableau;
    sil::csr::Csr u = young_tableau.template u<YoungTableauIndex, DDim2...>(tensor2.domain());
*/
        ddc::Chunk uncompressed_tensor1_alloc(
                Index1::subindices_domain(),
                ddc::HostAllocator<double>());
        sil::tensor::Tensor<
                double,
                typename Index1::subindices_domain_t,
                Kokkos::layout_right,
                Kokkos::DefaultHostExecutionSpace::memory_space>
                uncompressed_tensor1(uncompressed_tensor1_alloc);

        ddc::Chunk uncompressed_tensor2_alloc(
                Index2::subindices_domain(),
                ddc::HostAllocator<double>());
        sil::tensor::Tensor<
                double,
                typename Index2::subindices_domain_t,
                Kokkos::layout_right,
                Kokkos::DefaultHostExecutionSpace::memory_space>
                uncompressed_tensor2(uncompressed_tensor2_alloc);

        sil::tensor::uncompress(uncompressed_tensor1, tensor1);
        sil::tensor::uncompress(uncompressed_tensor2, tensor2);

        return natural_tensor_prod(prod_tensor, uncompressed_tensor1, uncompressed_tensor2);
    }
};

} // namespace detail

template <
        class... ProdDDim,
        misc::Specialization<TensorYoungTableauIndex> Index1,
        misc::Specialization<TensorYoungTableauIndex> Index2,
        class ElementType,
        class LayoutStridedPolicy,
        class MemorySpace>
Tensor<ElementType,
       ddc::DiscreteDomain<ProdDDim...>,
       Kokkos::layout_right,
       Kokkos::DefaultHostExecutionSpace::memory_space>
tensor_prod(
        Tensor<ElementType,
               ddc::DiscreteDomain<ProdDDim...>,
               Kokkos::layout_right,
               Kokkos::DefaultHostExecutionSpace::memory_space> prod_tensor,
        Tensor<ElementType, ddc::DiscreteDomain<Index1>, LayoutStridedPolicy, MemorySpace> tensor1,
        Tensor<ElementType, ddc::DiscreteDomain<Index2>, LayoutStridedPolicy, MemorySpace> tensor2)
{
    static_assert(std::is_same_v<
                  ddc::type_seq_remove_t<
                          uncharacterize<ddc::to_type_seq_t<typename Index1::subindices_domain_t>>,
                          uncharacterize<ddc::detail::TypeSeq<ProdDDim...>>>,
                  ddc::type_seq_remove_t<
                          uncharacterize<ddc::to_type_seq_t<typename Index2::subindices_domain_t>>,
                          uncharacterize<ddc::detail::TypeSeq<ProdDDim...>>>>);
    static_assert(
            std::is_same_v < ddc::type_seq_remove_t<
                    ddc::to_type_seq_t<typename Index1::subindices_domain_t>,
                    ddc::to_type_seq_t<typename Index2::subindices_domain_t>>,
            ddc::to_type_seq_t<
                    typename Index1::
                            subindices_domain_t>> && std::is_same_v < ddc::type_seq_remove_t<
                    ddc::to_type_seq_t<typename Index2::subindices_domain_t>,
                    ddc::to_type_seq_t<typename Index1::subindices_domain_t>>,
            ddc::to_type_seq_t<
                    typename Index2::
                            subindices_domain_t>>); // tensor1 and tensor2 should not have any subindex in common because their characters are different
    detail::TensorProd2<
            uncharacterize<Index1>,
            uncharacterize<Index2>,
            ddc::type_seq_remove_t<
                    uncharacterize<ddc::detail::TypeSeq<ProdDDim...>>,
                    uncharacterize<ddc::to_type_seq_t<typename Index2::subindices_domain_t>>>,
            ddc::type_seq_remove_t<
                    uncharacterize<ddc::to_type_seq_t<typename Index1::subindices_domain_t>>,
                    uncharacterize<ddc::detail::TypeSeq<ProdDDim...>>>,
            ddc::type_seq_remove_t<
                    uncharacterize<ddc::detail::TypeSeq<ProdDDim...>>,
                    uncharacterize<ddc::to_type_seq_t<typename Index1::subindices_domain_t>>>>::
            run(uncharacterize_tensor(prod_tensor),
                uncharacterize_tensor(tensor1),
                uncharacterize_tensor(tensor2));
    return prod_tensor;
}

// Any-any product into Young (do not support fields atm)
namespace detail {

template <
        class ProdIndices,
        class Indices1,
        class Indices2,
        class HeadDDim1TypeSeq,
        class ContractDDimTypeSeq,
        class TailDDim2TypeSeq>
struct TensorProd3;

template <
        class... ProdIndex,
        class... Index1,
        class... Index2,
        class... HeadDDim1,
        class... ContractDDim,
        class... TailDDim2>
struct TensorProd3<
        ddc::detail::TypeSeq<ProdIndex...>,
        ddc::detail::TypeSeq<Index1...>,
        ddc::detail::TypeSeq<Index2...>,
        ddc::detail::TypeSeq<HeadDDim1...>,
        ddc::detail::TypeSeq<ContractDDim...>,
        ddc::detail::TypeSeq<TailDDim2...>>
{
    template <class ElementType, class LayoutStridedPolicy, class MemorySpace>
    static Tensor<ElementType, ddc::DiscreteDomain<ProdIndex...>, LayoutStridedPolicy, MemorySpace>
    run(Tensor<ElementType,
               ddc::DiscreteDomain<ProdIndex...>,
               Kokkos::layout_right,
               Kokkos::DefaultHostExecutionSpace::memory_space> prod_tensor,
        Tensor<ElementType, ddc::DiscreteDomain<Index1...>, LayoutStridedPolicy, MemorySpace>
                tensor1,
        Tensor<ElementType, ddc::DiscreteDomain<Index2...>, LayoutStridedPolicy, MemorySpace>
                tensor2)
    {
        /*
    typename TensorYoungTableauIndex<DDim1...>::young_tableau young_tableau;
    sil::csr::Csr u = young_tableau.template u<YoungTableauIndex, DDim2...>(tensor2.domain());
*/
        ddc::Chunk uncompressed_prod_alloc(
                ddc::DiscreteDomain(ProdIndex::subindices_domain()...),
                ddc::HostAllocator<double>());
        sil::tensor::Tensor<
                double,
                ddc::cartesian_prod_t<typename ProdIndex::subindices_domain_t...>,
                Kokkos::layout_right,
                Kokkos::DefaultHostExecutionSpace::memory_space>
                uncompressed_prod(uncompressed_prod_alloc);

        sil::tensor::TensorAccessor<ContractDDim...> contract_accessor;
        ddc::DiscreteDomain<ContractDDim...> contract_dom = contract_accessor.natural_domain();

        ddc::for_each(
                uncompressed_prod.domain(),
                [&](ddc::cartesian_prod_t<
                        typename ProdIndex::subindices_domain_t...>::discrete_element_type elem) {
                    uncompressed_prod(elem) = ddc::transform_reduce(
                            contract_dom,
                            0.,
                            ddc::reducer::sum<ElementType>(),
                            [&](ddc::DiscreteElement<ContractDDim...> contract_elem) {
                                return tensor1.get(tensor1.access_element(
                                               ddc::DiscreteElement<HeadDDim1..., ContractDDim...>(
                                                       ddc::select<HeadDDim1...>(elem),
                                                       contract_accessor.access_element(
                                                               contract_elem))))
                                       * tensor2.get(tensor2.access_element(
                                               ddc::DiscreteElement<ContractDDim..., TailDDim2...>(
                                                       contract_elem,
                                                       ddc::select<TailDDim2...>(elem))));
                            });
                });
        sil::tensor::compress(prod_tensor, uncompressed_prod);
        return prod_tensor;
    }
};

} // namespace detail

template <
        class... ProdIndex, // TODO Align convention with tensor_prod
        class... Index1,
        class... Index2,
        class ElementType,
        class LayoutStridedPolicy,
        class MemorySpace>
Tensor<ElementType,
       ddc::DiscreteDomain<ProdIndex...>,
       Kokkos::layout_right,
       Kokkos::DefaultHostExecutionSpace::memory_space>
tensor_prod(
        Tensor<ElementType,
               ddc::DiscreteDomain<ProdIndex...>,
               Kokkos::layout_right,
               Kokkos::DefaultHostExecutionSpace::memory_space> prod_tensor,
        Tensor<ElementType, ddc::DiscreteDomain<Index1...>, LayoutStridedPolicy, MemorySpace>
                tensor1,
        Tensor<ElementType, ddc::DiscreteDomain<Index2...>, LayoutStridedPolicy, MemorySpace>
                tensor2)
{
    static_assert(std::is_same_v<
                  ddc::type_seq_remove_t<
                          uncharacterize<ddc::to_type_seq_t<
                                  ddc::cartesian_prod_t<typename Index1::subindices_domain_t...>>>,
                          uncharacterize<ddc::to_type_seq_t<ddc::cartesian_prod_t<
                                  typename ProdIndex::subindices_domain_t...>>>>,
                  ddc::type_seq_remove_t<
                          uncharacterize<ddc::to_type_seq_t<
                                  ddc::cartesian_prod_t<typename Index2::subindices_domain_t...>>>,
                          uncharacterize<ddc::to_type_seq_t<ddc::cartesian_prod_t<
                                  typename ProdIndex::subindices_domain_t...>>>>>);
    static_assert(
            std::is_same_v < ddc::type_seq_remove_t<
                    ddc::to_type_seq_t<ddc::cartesian_prod_t<typename Index1::subindices_domain_t...>>,
                    ddc::to_type_seq_t<ddc::cartesian_prod_t<typename Index2::subindices_domain_t...>>>,
            ddc::to_type_seq_t<
                    ddc::cartesian_prod_t<typename Index1::
                            subindices_domain_t...>>> && std::is_same_v < ddc::type_seq_remove_t<
                    ddc::to_type_seq_t<ddc::cartesian_prod_t<typename Index2::subindices_domain_t...>>,
                    ddc::to_type_seq_t<ddc::cartesian_prod_t<typename Index1::subindices_domain_t...>>>,
            ddc::to_type_seq_t<
                    ddc::cartesian_prod_t<typename Index2::
                            subindices_domain_t...>>>); // tensor1 and tensor2 should not have any subindex in common because their characters are different

    detail::TensorProd3<
            uncharacterize<ddc::detail::TypeSeq<ProdIndex...>>,
            uncharacterize<ddc::detail::TypeSeq<Index1...>>,
            uncharacterize<ddc::detail::TypeSeq<Index2...>>,
            ddc::type_seq_remove_t<
                    uncharacterize<ddc::to_type_seq_t<
                            ddc::cartesian_prod_t<typename ProdIndex::subindices_domain_t...>>>,
                    uncharacterize<ddc::to_type_seq_t<
                            ddc::cartesian_prod_t<typename Index2::subindices_domain_t...>>>>,
            ddc::type_seq_remove_t<
                    uncharacterize<ddc::to_type_seq_t<
                            ddc::cartesian_prod_t<typename Index1::subindices_domain_t...>>>,
                    uncharacterize<ddc::to_type_seq_t<
                            ddc::cartesian_prod_t<typename ProdIndex::subindices_domain_t...>>>>,
            ddc::type_seq_remove_t<
                    uncharacterize<ddc::to_type_seq_t<
                            ddc::cartesian_prod_t<typename ProdIndex::subindices_domain_t...>>>,
                    uncharacterize<ddc::to_type_seq_t<
                            ddc::cartesian_prod_t<typename Index1::subindices_domain_t...>>>>>::
            run(uncharacterize_tensor(prod_tensor),
                uncharacterize_tensor(tensor1),
                uncharacterize_tensor(tensor2));
    return prod_tensor;
}
} // namespace tensor

} // namespace sil
