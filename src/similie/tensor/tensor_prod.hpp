// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: MIT

#pragma once

#include <ddc/ddc.hpp>

#include <similie/misc/specialization.hpp>

#include "character.hpp"
#if defined BUILD_YOUNG_TABLEAU
#include "young_tableau_tensor.hpp"
#endif

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

// Check tensor compatibility
namespace detail {

template <class ProdDDims, class Indices1, class Indices2>
struct CheckTensorsCompatibility;

template <class... ProdDDim, class... Index1, class... Index2>
struct CheckTensorsCompatibility<
        ddc::detail::TypeSeq<ProdDDim...>,
        ddc::detail::TypeSeq<Index1...>,
        ddc::detail::TypeSeq<Index2...>>
{
    KOKKOS_FUNCTION static constexpr void run()
    {
        static_assert(std::is_same_v<
                      ddc::type_seq_remove_t<
                              uncharacterize_t<ddc::to_type_seq_t<
                                      ddc::cartesian_prod_t<natural_domain_t<Index1>...>>>,
                              uncharacterize_t<ddc::to_type_seq_t<
                                      ddc::cartesian_prod_t<natural_domain_t<ProdDDim>...>>>>,
                      ddc::type_seq_remove_t<
                              uncharacterize_t<ddc::to_type_seq_t<
                                      ddc::cartesian_prod_t<natural_domain_t<Index2>...>>>,
                              uncharacterize_t<ddc::to_type_seq_t<
                                      ddc::cartesian_prod_t<natural_domain_t<ProdDDim>...>>>>>);
        static_assert(are_different_characters_v<
                      ddc::type_seq_remove_t<
                              ddc::to_type_seq_t<
                                      ddc::cartesian_prod_t<natural_domain_t<Index1>...>>,
                              ddc::to_type_seq_t<
                                      ddc::cartesian_prod_t<natural_domain_t<ProdDDim>...>>>,
                      ddc::type_seq_remove_t<
                              ddc::to_type_seq_t<
                                      ddc::cartesian_prod_t<natural_domain_t<Index2>...>>,
                              ddc::to_type_seq_t<
                                      ddc::cartesian_prod_t<natural_domain_t<ProdDDim>...>>>>);
    }
};

} // namespace detail

template <class ProdDDims, class Indices1, class Indices2>
constexpr void check_tensors_compatibility()
{
    return detail::CheckTensorsCompatibility<ProdDDims, Indices1, Indices2>::run();
}

// Any-any product into Any (general case not optimized)
namespace detail {

template <
        class ProdDDims,
        class Indices1,
        class Indices2,
        class HeadDDim1TypeSeq,
        class ContractDDimTypeSeq,
        class TailDDim2TypeSeq>
struct TensorProdAnyAnyAny;

template <
        class... ProdDDim,
        class... Index1,
        class... Index2,
        class... HeadDDim1,
        class... ContractDDim,
        class... TailDDim2>
struct TensorProdAnyAnyAny<
        ddc::detail::TypeSeq<ProdDDim...>,
        ddc::detail::TypeSeq<Index1...>,
        ddc::detail::TypeSeq<Index2...>,
        ddc::detail::TypeSeq<HeadDDim1...>,
        ddc::detail::TypeSeq<ContractDDim...>,
        ddc::detail::TypeSeq<TailDDim2...>>
{
    template <class ElementType, class LayoutStridedPolicy, class MemorySpace>
    KOKKOS_FUNCTION static Tensor<
            ElementType,
            ddc::DiscreteDomain<ProdDDim...>,
            LayoutStridedPolicy,
            MemorySpace>
    run(Tensor<ElementType, ddc::DiscreteDomain<ProdDDim...>, LayoutStridedPolicy, MemorySpace>
                prod_tensor,
        Tensor<ElementType, ddc::DiscreteDomain<Index1...>, LayoutStridedPolicy, MemorySpace>
                tensor1,
        Tensor<ElementType, ddc::DiscreteDomain<Index2...>, LayoutStridedPolicy, MemorySpace>
                tensor2)
    {
        tensor::TensorAccessor<ContractDDim...> contract_accessor;
        ddc::DiscreteDomain<ContractDDim...> contract_dom = contract_accessor.natural_domain();

        ddc::device_for_each(prod_tensor.domain(), [&](ddc::DiscreteElement<ProdDDim...> mem_elem) {
            auto elem = prod_tensor.canonical_natural_element(mem_elem);
            prod_tensor.mem(mem_elem) = ddc::device_transform_reduce(
                    contract_dom,
                    0.,
                    ddc::reducer::sum<ElementType>(),
                    [&](ddc::DiscreteElement<ContractDDim...> contract_elem) {
                        return tensor1.get(tensor1.access_element(
                                       ddc::DiscreteElement<HeadDDim1..., ContractDDim...>(
                                               ddc::select<HeadDDim1...>(elem),
                                               contract_accessor.access_element(contract_elem))))
                               * tensor2.get(tensor2.access_element(
                                       ddc::DiscreteElement<ContractDDim..., TailDDim2...>(
                                               contract_accessor.access_element(contract_elem),
                                               ddc::select<TailDDim2...>(elem))));
                    });
        });
        return prod_tensor;
    }
};

} // namespace detail

template <
        class... ProdDDim,
        class... Index1,
        class... Index2,
        class ElementType,
        class LayoutStridedPolicy,
        class MemorySpace>
    requires(
            ((!TensorNatIndex<ProdDDim> || ...) || (!TensorNatIndex<Index2> || ...)
             || (!TensorNatIndex<Index1> || ...))
#if defined BUILD_YOUNG_TABLEAU
            && (!misc::Specialization<ProdDDim, TensorYoungTableauIndex> && ...)
            && (!misc::Specialization<Index1, TensorYoungTableauIndex> && ...)
            && (!misc::Specialization<Index2, TensorYoungTableauIndex> && ...)
#endif
                    )
Tensor<ElementType, ddc::DiscreteDomain<ProdDDim...>, LayoutStridedPolicy, MemorySpace>
        KOKKOS_FUNCTION tensor_prod(
                Tensor<ElementType,
                       ddc::DiscreteDomain<ProdDDim...>,
                       LayoutStridedPolicy,
                       MemorySpace> prod_tensor,
                Tensor<ElementType,
                       ddc::DiscreteDomain<Index1...>,
                       LayoutStridedPolicy,
                       MemorySpace> tensor1,
                Tensor<ElementType,
                       ddc::DiscreteDomain<Index2...>,
                       LayoutStridedPolicy,
                       MemorySpace> tensor2)
{
    check_tensors_compatibility<
            ddc::detail::TypeSeq<ProdDDim...>,
            ddc::detail::TypeSeq<Index1...>,
            ddc::detail::TypeSeq<Index2...>>();

    detail::TensorProdAnyAnyAny<
            uncharacterize_t<ddc::detail::TypeSeq<ProdDDim...>>,
            uncharacterize_t<ddc::detail::TypeSeq<Index1...>>,
            uncharacterize_t<ddc::detail::TypeSeq<Index2...>>,
            ddc::type_seq_remove_t<
                    uncharacterize_t<ddc::to_type_seq_t<
                            ddc::cartesian_prod_t<natural_domain_t<ProdDDim>...>>>,
                    uncharacterize_t<ddc::to_type_seq_t<
                            ddc::cartesian_prod_t<natural_domain_t<Index2>...>>>>,
            ddc::type_seq_remove_t<
                    uncharacterize_t<
                            ddc::to_type_seq_t<ddc::cartesian_prod_t<natural_domain_t<Index1>...>>>,
                    uncharacterize_t<ddc::to_type_seq_t<
                            ddc::cartesian_prod_t<natural_domain_t<ProdDDim>...>>>>,
            ddc::type_seq_remove_t<
                    uncharacterize_t<ddc::to_type_seq_t<
                            ddc::cartesian_prod_t<natural_domain_t<ProdDDim>...>>>,
                    uncharacterize_t<ddc::to_type_seq_t<
                            ddc::cartesian_prod_t<natural_domain_t<Index1>...>>>>>::
            run(uncharacterize_tensor(prod_tensor),
                uncharacterize_tensor(tensor1),
                uncharacterize_tensor(tensor2));
    return prod_tensor;
}

#if defined BUILD_YOUNG_TABLEAU
// Young-dense product
namespace detail {

template <class Index1, class HeadDDim1TypeSeq, class ContractDDimTypeSeq, class TailDDim2TypeSeq>
struct TensorProdNatYoungNat;

template <class Index1, class... HeadDDim1, class... ContractDDim, class... TailDDim2>
struct TensorProdNatYoungNat<
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
    csr::Csr u = young_tableau.template u<YoungTableauIndex, DDim2...>(tensor2.domain());
*/
        ddc::Chunk uncompressed_tensor1_alloc(
                Index1::subindices_domain(),
                ddc::HostAllocator<double>());
        tensor::Tensor uncompressed_tensor1(uncompressed_tensor1_alloc);

        tensor::uncompress(uncompressed_tensor1, tensor1);

        return tensor_prod(prod_tensor, uncompressed_tensor1, tensor2);
    }
};

} // namespace detail

template <
        TensorNatIndex... ProdDDim,
        misc::Specialization<TensorYoungTableauIndex> Index1,
        TensorNatIndex... DDim2,
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
    /*
    check_tensors_compatibility<
            ddc::detail::TypeSeq<ProdDDim...>,
            ddc::detail::TypeSeq<Index1>,
            ddc::detail::TypeSeq<DDim2...>>();
    */
    // TODO DDim2 -> Index2, characterize
    return detail::TensorProdNatYoungNat<
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
struct TensorProdNatYoungYoung;

template <class Index1, class Index2, class... HeadDDim1, class... ContractDDim, class... TailDDim2>
struct TensorProdNatYoungYoung<
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
    csr::Csr u = young_tableau.template u<YoungTableauIndex, DDim2...>(tensor2.domain());
*/
        ddc::Chunk uncompressed_tensor1_alloc(
                Index1::subindices_domain(),
                ddc::HostAllocator<double>());
        tensor::Tensor uncompressed_tensor1(uncompressed_tensor1_alloc);

        ddc::Chunk uncompressed_tensor2_alloc(
                Index2::subindices_domain(),
                ddc::HostAllocator<double>());
        tensor::Tensor uncompressed_tensor2(uncompressed_tensor2_alloc);

        tensor::uncompress(uncompressed_tensor1, tensor1);
        tensor::uncompress(uncompressed_tensor2, tensor2);

        return tensor_prod(prod_tensor, uncompressed_tensor1, uncompressed_tensor2);
    }
};

} // namespace detail

template <
        TensorNatIndex... ProdDDim,
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
    check_tensors_compatibility<
            ddc::detail::TypeSeq<ProdDDim...>,
            ddc::detail::TypeSeq<Index1>,
            ddc::detail::TypeSeq<Index2>>();

    detail::TensorProdNatYoungYoung<
            uncharacterize_t<Index1>,
            uncharacterize_t<Index2>,
            ddc::type_seq_remove_t<
                    uncharacterize_t<ddc::to_type_seq_t<
                            ddc::cartesian_prod_t<natural_domain_t<ProdDDim>...>>>,
                    uncharacterize_t<
                            ddc::to_type_seq_t<ddc::cartesian_prod_t<natural_domain_t<Index2>>>>>,
            ddc::type_seq_remove_t<
                    uncharacterize_t<
                            ddc::to_type_seq_t<ddc::cartesian_prod_t<natural_domain_t<Index1>>>>,
                    uncharacterize_t<ddc::to_type_seq_t<
                            ddc::cartesian_prod_t<natural_domain_t<ProdDDim>...>>>>,
            ddc::type_seq_remove_t<
                    uncharacterize_t<ddc::to_type_seq_t<
                            ddc::cartesian_prod_t<natural_domain_t<ProdDDim>...>>>,
                    uncharacterize_t<
                            ddc::to_type_seq_t<ddc::cartesian_prod_t<natural_domain_t<Index1>>>>>>::
            run(uncharacterize_tensor(prod_tensor),
                uncharacterize_tensor(tensor1),
                uncharacterize_tensor(tensor2));

    return prod_tensor;
}

// Any-any product into Young
namespace detail {

template <
        class ProdDDims,
        class Indices1,
        class Indices2,
        class HeadDDim1TypeSeq,
        class ContractDDimTypeSeq,
        class TailDDim2TypeSeq>
struct TensorProdYoungAnyAny;

template <
        class ProdDDim,
        class... Index1,
        class... Index2,
        class... HeadDDim1,
        class... ContractDDim,
        class... TailDDim2>
struct TensorProdYoungAnyAny<
        ddc::detail::TypeSeq<ProdDDim>,
        ddc::detail::TypeSeq<Index1...>,
        ddc::detail::TypeSeq<Index2...>,
        ddc::detail::TypeSeq<HeadDDim1...>,
        ddc::detail::TypeSeq<ContractDDim...>,
        ddc::detail::TypeSeq<TailDDim2...>>
{
    template <class Indices>
    struct YoungIndexIn;

    template <class Fallback, class... TensorIndex>
    struct FirstYoungIndex
    {
        using type = Fallback;
    };

    template <class Fallback, class HeadIndex, class... TailIndex>
    struct FirstYoungIndex<Fallback, HeadIndex, TailIndex...>
    {
        using tail_type = typename FirstYoungIndex<Fallback, TailIndex...>::type;
        using type = std::conditional_t<
                misc::Specialization<HeadIndex, TensorYoungTableauIndex>,
                HeadIndex,
                tail_type>;
    };

    template <class... TensorIndex>
    struct YoungIndexIn<ddc::detail::TypeSeq<TensorIndex...>>
    {
        using type = typename FirstYoungIndex<void, TensorIndex...>::type;
    };

    template <class NaturalIndices>
    struct NaturalElementFromCsr;

    template <class... NaturalIndex>
    struct NaturalElementFromCsr<ddc::detail::TypeSeq<NaturalIndex...>>
    {
        template <class CsrType>
        KOKKOS_FUNCTION static ddc::DiscreteElement<NaturalIndex...> run(
                CsrType const& csr,
                std::size_t j)
        {
            return ddc::DiscreteElement<NaturalIndex...>(csr.idx()[ddc::type_seq_rank_v<
                    NaturalIndex,
                    ddc::detail::TypeSeq<NaturalIndex...>>][j]...);
        }
    };

    template <class TensorIndex, class YoungIndex, class Elem>
    KOKKOS_FUNCTION static std::size_t mem_uid(Elem elem, std::size_t young_mem_uid)
    {
        if constexpr (std::is_same_v<TensorIndex, YoungIndex>) {
            return young_mem_uid;
        } else {
            static_assert(TensorNatIndex<TensorIndex>);
            return elem.template uid<TensorIndex>();
        }
    }

    template <class... TensorIndex, class TensorType, class Elem, class YoungIndex>
    KOKKOS_FUNCTION static typename TensorType::element_type tensor_mem(
            TensorType tensor,
            Elem elem,
            std::size_t young_mem_uid,
            std::type_identity<ddc::detail::TypeSeq<TensorIndex...>>,
            YoungIndex)
    {
        return tensor.mem(
                ddc::DiscreteElement<TensorIndex...>(ddc::DiscreteElement<TensorIndex>(
                        mem_uid<TensorIndex, YoungIndex>(elem, young_mem_uid))...));
    }

    template <class TensorType, class Elem, class YoungIndex, class... NaturalIndex>
    KOKKOS_FUNCTION static typename TensorType::element_type young_tensor_value_impl(
            TensorType tensor,
            Elem elem,
            YoungIndex,
            std::type_identity<ddc::detail::TypeSeq<NaturalIndex...>>)
    {
        constexpr csr::Csr v = YoungIndex::young_tableau::template v<YoungIndex>(
                YoungIndex::subindices_domain());
        typename TensorType::element_type tensor_value = 0.;
        for (std::size_t j = 0; j < v.values().size(); ++j) {
            if (((v.idx()[ddc::type_seq_rank_v<NaturalIndex, ddc::detail::TypeSeq<NaturalIndex...>>]
                         [j]
                  == elem.template uid<NaturalIndex>())
                 && ...)) {
                std::size_t young_mem_uid = 0;
                while (young_mem_uid < v.coalesc_idx().size() - 1
                       && v.coalesc_idx()[young_mem_uid + 1] <= j) {
                    ++young_mem_uid;
                }
                tensor_value += v.values()[j]
                                * tensor_mem(
                                        tensor,
                                        elem,
                                        young_mem_uid,
                                        std::type_identity<ddc::to_type_seq_t<
                                                typename TensorType::indices_domain_t>>(),
                                        YoungIndex());
            }
        }
        return tensor_value;
    }

    template <class TensorType, class Elem, class Indices>
    KOKKOS_FUNCTION static typename TensorType::element_type tensor_value(
            TensorType tensor,
            Elem elem,
            std::type_identity<Indices>)
    {
        using young_index = typename YoungIndexIn<Indices>::type;
        if constexpr (std::is_void_v<young_index>) {
            return tensor.get(tensor.access_element(elem));
        } else {
            return young_tensor_value_impl(
                    tensor,
                    elem,
                    young_index(),
                    std::type_identity<
                            ddc::to_type_seq_t<typename young_index::subindices_domain_t>>());
        }
    }

    template <class ElementType, class LayoutStridedPolicy, class MemorySpace>
    KOKKOS_FUNCTION static Tensor<
            ElementType,
            ddc::DiscreteDomain<ProdDDim>,
            LayoutStridedPolicy,
            MemorySpace>
    run(Tensor<ElementType,
               ddc::DiscreteDomain<ProdDDim>,
               Kokkos::layout_right,
               Kokkos::DefaultHostExecutionSpace::memory_space> prod_tensor,
        Tensor<ElementType, ddc::DiscreteDomain<Index1...>, LayoutStridedPolicy, MemorySpace>
                tensor1,
        Tensor<ElementType, ddc::DiscreteDomain<Index2...>, LayoutStridedPolicy, MemorySpace>
                tensor2)
    {
        constexpr csr::Csr u
                = ProdDDim::young_tableau::template u<ProdDDim>(ProdDDim::subindices_domain());
        tensor::TensorAccessor<ContractDDim...> contract_accessor;
        ddc::DiscreteDomain<ContractDDim...> contract_dom = contract_accessor.natural_domain();

        using natural_indices = ddc::to_type_seq_t<typename ProdDDim::subindices_domain_t>;
        ddc::device_for_each(
                prod_tensor.domain(),
                [&](ddc::DiscreteElement<ProdDDim> prod_mem_elem) {
                    ElementType prod_value = 0.;
                    std::size_t const j_begin
                            = u.coalesc_idx()[prod_mem_elem.template uid<ProdDDim>()];
                    std::size_t const j_end
                            = u.coalesc_idx()[prod_mem_elem.template uid<ProdDDim>() + 1];
                    for (std::size_t j = j_begin; j < j_end; ++j) {
                        auto const elem = NaturalElementFromCsr<natural_indices>::run(u, j);
                        ElementType const uncompressed_value = ddc::device_transform_reduce(
                                contract_dom,
                                0.,
                                ddc::reducer::sum<ElementType>(),
                                [&](ddc::DiscreteElement<ContractDDim...> contract_elem) {
                                    return tensor_value(
                                                   tensor1,
                                                   ddc::DiscreteElement<
                                                           HeadDDim1...,
                                                           ContractDDim...>(
                                                           ddc::select<HeadDDim1...>(elem),
                                                           contract_accessor.access_element(
                                                                   contract_elem)),
                                                   std::type_identity<
                                                           ddc::detail::TypeSeq<Index1...>>())
                                           * tensor_value(
                                                   tensor2,
                                                   ddc::DiscreteElement<
                                                           ContractDDim...,
                                                           TailDDim2...>(
                                                           contract_accessor.access_element(
                                                                   contract_elem),
                                                           ddc::select<TailDDim2...>(elem)),
                                                   std::type_identity<
                                                           ddc::detail::TypeSeq<Index2...>>());
                                });
                        prod_value += u.values()[j] * uncompressed_value;
                    }
                    prod_tensor.mem(prod_mem_elem) = prod_value;
                });
        return prod_tensor;
    }
};

} // namespace detail

template <
        misc::Specialization<TensorYoungTableauIndex> ProdDDim,
        class... Index1,
        class... Index2,
        class ElementType,
        class LayoutStridedPolicy,
        class MemorySpace>
Tensor<ElementType,
       ddc::DiscreteDomain<ProdDDim>,
       Kokkos::layout_right,
       Kokkos::DefaultHostExecutionSpace::memory_space>
tensor_prod(
        Tensor<ElementType,
               ddc::DiscreteDomain<ProdDDim>,
               Kokkos::layout_right,
               Kokkos::DefaultHostExecutionSpace::memory_space> prod_tensor,
        Tensor<ElementType, ddc::DiscreteDomain<Index1...>, LayoutStridedPolicy, MemorySpace>
                tensor1,
        Tensor<ElementType, ddc::DiscreteDomain<Index2...>, LayoutStridedPolicy, MemorySpace>
                tensor2)
{
    check_tensors_compatibility<
            ddc::detail::TypeSeq<ProdDDim>,
            ddc::detail::TypeSeq<Index1...>,
            ddc::detail::TypeSeq<Index2...>>();

    detail::TensorProdYoungAnyAny<
            uncharacterize_t<ddc::detail::TypeSeq<ProdDDim>>,
            uncharacterize_t<ddc::detail::TypeSeq<Index1...>>,
            uncharacterize_t<ddc::detail::TypeSeq<Index2...>>,
            ddc::type_seq_remove_t<
                    uncharacterize_t<
                            ddc::to_type_seq_t<ddc::cartesian_prod_t<natural_domain_t<ProdDDim>>>>,
                    uncharacterize_t<ddc::to_type_seq_t<
                            ddc::cartesian_prod_t<natural_domain_t<Index2>...>>>>,
            ddc::type_seq_remove_t<
                    uncharacterize_t<
                            ddc::to_type_seq_t<ddc::cartesian_prod_t<natural_domain_t<Index1>...>>>,
                    uncharacterize_t<
                            ddc::to_type_seq_t<ddc::cartesian_prod_t<natural_domain_t<ProdDDim>>>>>,
            ddc::type_seq_remove_t<
                    uncharacterize_t<
                            ddc::to_type_seq_t<ddc::cartesian_prod_t<natural_domain_t<ProdDDim>>>>,
                    uncharacterize_t<ddc::to_type_seq_t<
                            ddc::cartesian_prod_t<natural_domain_t<Index1>...>>>>>::
            run(uncharacterize_tensor(prod_tensor),
                uncharacterize_tensor(tensor1),
                uncharacterize_tensor(tensor2));
    return prod_tensor;
}
#endif

} // namespace tensor

} // namespace sil
