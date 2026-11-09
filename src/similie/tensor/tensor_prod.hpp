// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#pragma once

#include <array>
#include <utility>

#include <ddc/ddc.hpp>

#include <similie/misc/specialization.hpp>
#include <similie/misc/type_seq_ext.hpp>
#include <similie/tensor/tensor_indices/character.hpp>
#if defined BUILD_YOUNG_TABLEAU
#include <similie/tensor/tensor_indices/young_tableau_tensor.hpp>
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

    KOKKOS_FUNCTION static constexpr type operator()()
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
KOKKOS_FUNCTION static constexpr subindices_domain_t<T> subindices_domain()
{
    return detail::SubindicesDomain<T>::operator()();
};

// Check tensor compatibility
namespace detail {

template <class ProdDDims, class Indices1, class Indices2>
struct CheckTensorsCompatibility;

template <class... ProdDDim, class... Index1, class... Index2>
struct CheckTensorsCompatibility<
        ddc::TypeSeq<ProdDDim...>,
        ddc::TypeSeq<Index1...>,
        ddc::TypeSeq<Index2...>>
{
    static constexpr void operator()()
    {
        static_assert(std::is_same_v<
                      ddc::type_seq_remove_t<
                              uncharacterize_t<ddc::to_type_seq_t<
                                      sil::misc::cartesian_prod_t<natural_domain_t<Index1>...>>>,
                              uncharacterize_t<ddc::to_type_seq_t<
                                      sil::misc::cartesian_prod_t<natural_domain_t<ProdDDim>...>>>>,
                      ddc::type_seq_remove_t<
                              uncharacterize_t<ddc::to_type_seq_t<
                                      sil::misc::cartesian_prod_t<natural_domain_t<Index2>...>>>,
                              uncharacterize_t<ddc::to_type_seq_t<sil::misc::cartesian_prod_t<
                                      natural_domain_t<ProdDDim>...>>>>>);
        static_assert(are_different_characters_v<
                      ddc::type_seq_remove_t<
                              ddc::to_type_seq_t<
                                      sil::misc::cartesian_prod_t<natural_domain_t<Index1>...>>,
                              ddc::to_type_seq_t<
                                      sil::misc::cartesian_prod_t<natural_domain_t<ProdDDim>...>>>,
                      ddc::type_seq_remove_t<
                              ddc::to_type_seq_t<
                                      sil::misc::cartesian_prod_t<natural_domain_t<Index2>...>>,
                              ddc::to_type_seq_t<sil::misc::cartesian_prod_t<
                                      natural_domain_t<ProdDDim>...>>>>);
    }
};

} // namespace detail

template <class ProdDDims, class Indices1, class Indices2>
constexpr void check_tensors_compatibility()
{
    return detail::CheckTensorsCompatibility<ProdDDims, Indices1, Indices2>::operator()();
}

namespace detail {

// Evaluate the index's existing access transformation with a unit coefficient.
// This resolves structural zeros, ones and signs without reading tensor storage.
// In particular, access ID zero need not mean a structural zero for every index.
struct TensorProductUnitAccess
{
    using element_type = double;

    template <class Tensor, class Element>
    KOKKOS_FUNCTION constexpr double operator()(Tensor const&, Element const&) const
    {
        return 1.;
    }
};

template <class Index, class Element>
KOKKOS_FUNCTION constexpr double tensor_product_access_coefficient(Element const& element)
{
    return Index::template process_access<
            TensorProductUnitAccess,
            Element,
            Index>(TensorProductUnitAccess(), TensorProductUnitAccess(), element);
}

// A structural zero is a property of the access transformation, independent of
// the index family and of the values stored in a tensor.
template <class Index>
KOKKOS_FUNCTION constexpr bool tensor_product_has_structural_zeros()
{
    for (std::size_t id = 0; id < Index::access_size(); ++id) {
        if (tensor_product_access_coefficient<Index>(ddc::DiscreteElement<Index>(id)) == 0.)
            return true;
    }
    return false;
}

template <class... Free, class... Contracted>
KOKKOS_FUNCTION constexpr ddc::DiscreteElement<Free..., Contracted...> tensor_product_known_element(
        ddc::DiscreteElement<Free...> free,
        ddc::DiscreteElement<Contracted...> contracted)
{
    return ddc::DiscreteElement<Free..., Contracted...>(free, contracted);
}

template <class ProductIndices, class LeftIndices, class RightIndices, class ContractedIndices>
class TensorProdUnrolled;

// Unfold the canonical output components and contractions for any index types.
// Accessor conversions see constant indices and resolve storage offsets and signs
// at compile time. The index access transformations identify structural zeros.
template <class... Product, class... Left, class... Right, class... Contracted>
class TensorProdUnrolled<
        ddc::TypeSeq<Product...>,
        ddc::TypeSeq<Left...>,
        ddc::TypeSeq<Right...>,
        ddc::TypeSeq<Contracted...>>
{
    template <std::size_t Output>
    KOKKOS_FUNCTION static constexpr ddc::DiscreteElement<Product...> output_element()
    {
        std::array<std::size_t, sizeof...(Product)> const sizes {Product::mem_size()...};
        std::array<std::size_t, sizeof...(Product)> ids {};
        std::size_t remaining = Output;
        for (std::size_t i = sizes.size(); i > 0; --i) {
            ids[i - 1] = remaining % sizes[i - 1];
            remaining /= sizes[i - 1];
        }
        ddc::DiscreteElement<Product...> element;
        ddc::detail::array(element) = ids;
        return element;
    }

    template <std::size_t... Values, std::size_t... Positions>
    KOKKOS_FUNCTION static constexpr ddc::DiscreteElement<
            ddc::type_seq_element_t<Positions, ddc::TypeSeq<Contracted...>>...>
    contracted_element(std::index_sequence<Values...>, std::index_sequence<Positions...>)
    {
        return ddc::DiscreteElement<
                ddc::type_seq_element_t<Positions, ddc::TypeSeq<Contracted...>>...>(Values...);
    }

    template <
            std::size_t Output,
            class Tensor1,
            class Tensor2,
            std::size_t... Values,
            std::size_t... Next>
    KOKKOS_FORCEINLINE_FUNCTION static void extend(
            Tensor1 const& left,
            Tensor2 const& right,
            std::remove_const_t<typename Tensor1::element_type>& sum,
            bool& initialized,
            std::index_sequence<Values...>,
            std::index_sequence<Next...>)
    {
        (contract<Output>(left, right, sum, initialized, std::index_sequence<Values..., Next>()),
         ...);
    }

    template <std::size_t Output, class Tensor1, class Tensor2, std::size_t... Values>
    KOKKOS_FORCEINLINE_FUNCTION static void contract(
            Tensor1 const& left,
            Tensor2 const& right,
            std::remove_const_t<typename Tensor1::element_type>& sum,
            bool& initialized,
            std::index_sequence<Values...> assigned)
    {
        constexpr auto known = tensor_product_known_element(
                TensorAccessor<Product...>::canonical_natural_element(output_element<Output>()),
                contracted_element(
                        std::index_sequence<Values...>(),
                        std::make_index_sequence<sizeof...(Values)>()));
        if constexpr (sizeof...(Values) == sizeof...(Contracted)) {
            constexpr auto left_element = TensorAccessor<Left...>::access_element(known);
            constexpr auto right_element = TensorAccessor<Right...>::access_element(known);
            if constexpr (
                    ((tensor_product_access_coefficient<Left>(left_element) != 0.) && ...)
                    && ((tensor_product_access_coefficient<Right>(right_element) != 0.) && ...)) {
                if (initialized)
                    sum += left.get(left_element) * right.get(right_element);
                else {
                    sum = left.get(left_element) * right.get(right_element);
                    initialized = true;
                }
            }
        } else {
            extend<Output>(
                    left,
                    right,
                    sum,
                    initialized,
                    assigned,
                    std::make_index_sequence<ddc::type_seq_element_t<
                            sizeof...(Values),
                            ddc::TypeSeq<Contracted...>>::size()>());
        }
    }

    template <std::size_t Output, class ProductTensor, class Tensor1, class Tensor2>
    KOKKOS_FORCEINLINE_FUNCTION static void write(
            ProductTensor const& product,
            Tensor1 const& left,
            Tensor2 const& right)
    {
        constexpr ddc::DiscreteElement<Product...> element = output_element<Output>();
        // Tensor views may select only a subset of the stored output components.
        if (product.domain().contains(element)) {
            typename ProductTensor::element_type sum = 0.;
            // The general reduction starts from zero. Preserve that addition
            // (including signed zeros). Products with structural zeros start from
            // their first retained term, preserving copies and signed zeros.
            bool initialized = sizeof...(Contracted) > 0
                               && !((tensor_product_has_structural_zeros<Left>() || ...)
                                    || (tensor_product_has_structural_zeros<Right>() || ...));
            contract<Output>(left, right, sum, initialized, std::index_sequence<>());
            product.mem(element) = sum;
        }
    }

    template <class ProductTensor, class Tensor1, class Tensor2, std::size_t... Outputs>
    KOKKOS_FORCEINLINE_FUNCTION static void write(
            ProductTensor const& product,
            Tensor1 const& left,
            Tensor2 const& right,
            std::index_sequence<Outputs...>)
    {
        (write<Outputs>(product, left, right), ...);
    }

public:
    template <class ProductTensor, class Tensor1, class Tensor2>
    KOKKOS_FUNCTION static void operator()(
            ProductTensor const& product,
            Tensor1 const& left,
            Tensor2 const& right)
    {
        write(product,
              left,
              right,
              std::make_index_sequence<(Product::mem_size() * ... * std::size_t(1))>());
    }
};

} // namespace detail

// Any-any product into Any; unfold small contractions at compile time.
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
        ddc::TypeSeq<ProdDDim...>,
        ddc::TypeSeq<Index1...>,
        ddc::TypeSeq<Index2...>,
        ddc::TypeSeq<HeadDDim1...>,
        ddc::TypeSeq<ContractDDim...>,
        ddc::TypeSeq<TailDDim2...>>
{
    template <
            class ElementType,
            class LeftElement,
            class RightElement,
            class LayoutStridedPolicy,
            class MemorySpace>
    KOKKOS_FUNCTION static Tensor<
            ElementType,
            ddc::DiscreteDomain<ProdDDim...>,
            LayoutStridedPolicy,
            MemorySpace>
    operator()(
            Tensor<ElementType,
                   ddc::DiscreteDomain<ProdDDim...>,
                   LayoutStridedPolicy,
                   MemorySpace> const& prod_tensor,
            Tensor<LeftElement,
                   ddc::DiscreteDomain<Index1...>,
                   LayoutStridedPolicy,
                   MemorySpace> const& tensor1,
            Tensor<RightElement,
                   ddc::DiscreteDomain<Index2...>,
                   LayoutStridedPolicy,
                   MemorySpace> const& tensor2)
    {
        // Bound expansion of the general case by the number of scalar products.
        // Larger products retain the loop implementation to limit generated code.
        if constexpr (
                (tensor_product_has_structural_zeros<Index1>() || ...)
                || (tensor_product_has_structural_zeros<Index2>() || ...)
                || ((ProdDDim::mem_size() * ... * 1) <= 128
                    && (ContractDDim::size() * ... * 1) <= 128
                    && (ProdDDim::mem_size() * ... * 1) * (ContractDDim::size() * ... * 1)
                               <= 128)) {
            TensorProdUnrolled<
                    ddc::TypeSeq<ProdDDim...>,
                    ddc::TypeSeq<Index1...>,
                    ddc::TypeSeq<Index2...>,
                    ddc::TypeSeq<ContractDDim...>>::operator()(prod_tensor, tensor1, tensor2);
        } else {
            tensor::TensorAccessor<ContractDDim...> const contract_accessor;
            ddc::DiscreteDomain<ContractDDim...> const contract_dom
                    = contract_accessor.natural_domain();

            ddc::device_for_each(
                    prod_tensor.domain(),
                    [&](ddc::DiscreteElement<ProdDDim...> mem_elem) {
                        auto const elem = prod_tensor.canonical_natural_element(mem_elem);
                        prod_tensor.mem(mem_elem) = ddc::device_transform_reduce(
                                contract_dom,
                                0.,
                                ddc::reducer::sum<ElementType>(),
                                [&](ddc::DiscreteElement<ContractDDim...> contract_elem) {
                                    return tensor1.get(tensor1.access_element(
                                                   ddc::DiscreteElement<
                                                           HeadDDim1...,
                                                           ContractDDim...>(
                                                           ddc::select<HeadDDim1...>(elem),
                                                           contract_accessor.access_element(
                                                                   contract_elem))))
                                           * tensor2.get(tensor2.access_element(
                                                   ddc::DiscreteElement<
                                                           ContractDDim...,
                                                           TailDDim2...>(
                                                           contract_accessor.access_element(
                                                                   contract_elem),
                                                           ddc::select<TailDDim2...>(elem))));
                                });
                    });
        }
        return prod_tensor;
    }
};

} // namespace detail

template <
        class... ProdDDim,
        class... Index1,
        class... Index2,
        class ElementType,
        class LeftElement,
        class RightElement,
        class LayoutStridedPolicy,
        class MemorySpace>
    requires(
            std::is_same_v<ElementType, std::remove_const_t<LeftElement>>
            && std::is_same_v<ElementType, std::remove_const_t<RightElement>>
            && ((!TensorNatIndex<ProdDDim> || ...) || (!TensorNatIndex<Index2> || ...)
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
                       MemorySpace> const& prod_tensor,
                Tensor<LeftElement,
                       ddc::DiscreteDomain<Index1...>,
                       LayoutStridedPolicy,
                       MemorySpace> const& tensor1,
                Tensor<RightElement,
                       ddc::DiscreteDomain<Index2...>,
                       LayoutStridedPolicy,
                       MemorySpace> const& tensor2)
{
    check_tensors_compatibility<
            ddc::TypeSeq<ProdDDim...>,
            ddc::TypeSeq<Index1...>,
            ddc::TypeSeq<Index2...>>();

    detail::TensorProdAnyAnyAny<
            uncharacterize_t<ddc::TypeSeq<ProdDDim...>>,
            uncharacterize_t<ddc::TypeSeq<Index1...>>,
            uncharacterize_t<ddc::TypeSeq<Index2...>>,
            ddc::type_seq_remove_t<
                    uncharacterize_t<ddc::to_type_seq_t<
                            sil::misc::cartesian_prod_t<natural_domain_t<ProdDDim>...>>>,
                    uncharacterize_t<ddc::to_type_seq_t<
                            sil::misc::cartesian_prod_t<natural_domain_t<Index2>...>>>>,
            ddc::type_seq_remove_t<
                    uncharacterize_t<ddc::to_type_seq_t<
                            sil::misc::cartesian_prod_t<natural_domain_t<Index1>...>>>,
                    uncharacterize_t<ddc::to_type_seq_t<
                            sil::misc::cartesian_prod_t<natural_domain_t<ProdDDim>...>>>>,
            ddc::type_seq_remove_t<
                    uncharacterize_t<ddc::to_type_seq_t<
                            sil::misc::cartesian_prod_t<natural_domain_t<ProdDDim>...>>>,
                    uncharacterize_t<ddc::to_type_seq_t<
                            sil::misc::cartesian_prod_t<natural_domain_t<Index1>...>>>>>::
    operator()(
            uncharacterize_tensor(prod_tensor),
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
        ddc::TypeSeq<HeadDDim1...>,
        ddc::TypeSeq<ContractDDim...>,
        ddc::TypeSeq<TailDDim2...>>
{
    template <class ElementType, class LayoutStridedPolicy, class MemorySpace>
    static Tensor<
            ElementType,
            ddc::DiscreteDomain<HeadDDim1..., TailDDim2...>,
            LayoutStridedPolicy,
            MemorySpace>
    operator()(
            Tensor<ElementType,
                   ddc::DiscreteDomain<HeadDDim1..., TailDDim2...>,
                   Kokkos::layout_right,
                   Kokkos::DefaultHostExecutionSpace::memory_space> const& prod_tensor,
            Tensor<ElementType,
                   ddc::DiscreteDomain<Index1>,
                   LayoutStridedPolicy,
                   MemorySpace> const& tensor1,
            Tensor<ElementType,
                   ddc::DiscreteDomain<ContractDDim..., TailDDim2...>,
                   LayoutStridedPolicy,
                   MemorySpace> const& tensor2)
    {
        std::array<ElementType, Index1::subindices_domain().size()> uncompressed_tensor1_alloc;
        tensor::Tensor uncompressed_tensor1(
                uncompressed_tensor1_alloc.data(),
                Index1::subindices_domain());

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
               Kokkos::DefaultHostExecutionSpace::memory_space> const& prod_tensor,
        Tensor<ElementType, ddc::DiscreteDomain<Index1>, LayoutStridedPolicy, MemorySpace> const&
                tensor1,
        Tensor<ElementType, ddc::DiscreteDomain<DDim2...>, LayoutStridedPolicy, MemorySpace> const&
                tensor2)
{
    // TODO DDim2 -> Index2, characterize
    return detail::TensorProdNatYoungNat<
            Index1,
            ddc::type_seq_remove_t<ddc::TypeSeq<ProdDDim...>, ddc::TypeSeq<DDim2...>>,
            ddc::type_seq_remove_t<
                    ddc::to_type_seq_t<typename Index1::subindices_domain_t>,
                    ddc::TypeSeq<ProdDDim...>>,
            ddc::type_seq_remove_t<
                    ddc::TypeSeq<ProdDDim...>,
                    ddc::to_type_seq_t<typename Index1::subindices_domain_t>>>::
    operator()(prod_tensor, tensor1, tensor2);
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
        ddc::TypeSeq<HeadDDim1...>,
        ddc::TypeSeq<ContractDDim...>,
        ddc::TypeSeq<TailDDim2...>>
{
    template <class ElementType, class LayoutStridedPolicy, class MemorySpace>
    static Tensor<
            ElementType,
            ddc::DiscreteDomain<HeadDDim1..., TailDDim2...>,
            LayoutStridedPolicy,
            MemorySpace>
    operator()(
            Tensor<ElementType,
                   ddc::DiscreteDomain<HeadDDim1..., TailDDim2...>,
                   Kokkos::layout_right,
                   Kokkos::DefaultHostExecutionSpace::memory_space> const& prod_tensor,
            Tensor<ElementType,
                   ddc::DiscreteDomain<Index1>,
                   LayoutStridedPolicy,
                   MemorySpace> const& tensor1,
            Tensor<ElementType,
                   ddc::DiscreteDomain<Index2>,
                   LayoutStridedPolicy,
                   MemorySpace> const& tensor2)
    {
        std::array<ElementType, Index1::subindices_domain().size()> uncompressed_tensor1_alloc;
        tensor::Tensor uncompressed_tensor1(
                uncompressed_tensor1_alloc.data(),
                Index1::subindices_domain());

        std::array<ElementType, Index2::subindices_domain().size()> uncompressed_tensor2_alloc;
        tensor::Tensor uncompressed_tensor2(
                uncompressed_tensor2_alloc.data(),
                Index2::subindices_domain());

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
               Kokkos::DefaultHostExecutionSpace::memory_space> const& prod_tensor,
        Tensor<ElementType, ddc::DiscreteDomain<Index1>, LayoutStridedPolicy, MemorySpace> const&
                tensor1,
        Tensor<ElementType, ddc::DiscreteDomain<Index2>, LayoutStridedPolicy, MemorySpace> const&
                tensor2)
{
    check_tensors_compatibility<
            ddc::TypeSeq<ProdDDim...>,
            ddc::TypeSeq<Index1>,
            ddc::TypeSeq<Index2>>();

    detail::TensorProdNatYoungYoung<
            uncharacterize_t<Index1>,
            uncharacterize_t<Index2>,
            ddc::type_seq_remove_t<
                    uncharacterize_t<ddc::to_type_seq_t<
                            sil::misc::cartesian_prod_t<natural_domain_t<ProdDDim>...>>>,
                    uncharacterize_t<ddc::to_type_seq_t<
                            sil::misc::cartesian_prod_t<natural_domain_t<Index2>>>>>,
            ddc::type_seq_remove_t<
                    uncharacterize_t<ddc::to_type_seq_t<
                            sil::misc::cartesian_prod_t<natural_domain_t<Index1>>>>,
                    uncharacterize_t<ddc::to_type_seq_t<
                            sil::misc::cartesian_prod_t<natural_domain_t<ProdDDim>...>>>>,
            ddc::type_seq_remove_t<
                    uncharacterize_t<ddc::to_type_seq_t<
                            sil::misc::cartesian_prod_t<natural_domain_t<ProdDDim>...>>>,
                    uncharacterize_t<ddc::to_type_seq_t<
                            sil::misc::cartesian_prod_t<natural_domain_t<Index1>>>>>>::
    operator()(
            uncharacterize_tensor(prod_tensor),
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
        class... ProdDDim,
        class... Index1,
        class... Index2,
        class... HeadDDim1,
        class... ContractDDim,
        class... TailDDim2>
struct TensorProdYoungAnyAny<
        ddc::TypeSeq<ProdDDim...>,
        ddc::TypeSeq<Index1...>,
        ddc::TypeSeq<Index2...>,
        ddc::TypeSeq<HeadDDim1...>,
        ddc::TypeSeq<ContractDDim...>,
        ddc::TypeSeq<TailDDim2...>>
{
    template <class ElementType, class LayoutStridedPolicy, class MemorySpace>
    static Tensor<ElementType, ddc::DiscreteDomain<ProdDDim...>, LayoutStridedPolicy, MemorySpace>
    operator()(
            Tensor<ElementType,
                   ddc::DiscreteDomain<ProdDDim...>,
                   Kokkos::layout_right,
                   Kokkos::DefaultHostExecutionSpace::memory_space> const& prod_tensor,
            Tensor<ElementType,
                   ddc::DiscreteDomain<Index1...>,
                   LayoutStridedPolicy,
                   MemorySpace> const& tensor1,
            Tensor<ElementType,
                   ddc::DiscreteDomain<Index2...>,
                   LayoutStridedPolicy,
                   MemorySpace> const& tensor2)
    {
        std::array<ElementType, ddc::DiscreteDomain(ProdDDim::subindices_domain()...).size()>
                uncompressed_prod_alloc;
        tensor::Tensor uncompressed_prod(
                uncompressed_prod_alloc.data(),
                ddc::DiscreteDomain(ProdDDim::subindices_domain()...));

        tensor::TensorAccessor<ContractDDim...> const contract_accessor;
        ddc::DiscreteDomain<ContractDDim...> const contract_dom
                = contract_accessor.natural_domain();

        ddc::host_for_each(
                uncompressed_prod.domain(),
                [&](sil::misc::cartesian_prod_t<
                        typename ProdDDim::subindices_domain_t...>::discrete_element_type elem) {
                    uncompressed_prod(elem) = ddc::host_transform_reduce(
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
                                                       contract_accessor.access_element(
                                                               contract_elem),
                                                       ddc::select<TailDDim2...>(elem))));
                            });
                });
        tensor::compress(prod_tensor, uncompressed_prod);
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
               Kokkos::DefaultHostExecutionSpace::memory_space> const& prod_tensor,
        Tensor<ElementType, ddc::DiscreteDomain<Index1...>, LayoutStridedPolicy, MemorySpace> const&
                tensor1,
        Tensor<ElementType, ddc::DiscreteDomain<Index2...>, LayoutStridedPolicy, MemorySpace> const&
                tensor2)
{
    check_tensors_compatibility<
            ddc::TypeSeq<ProdDDim>,
            ddc::TypeSeq<Index1...>,
            ddc::TypeSeq<Index2...>>();

    detail::TensorProdYoungAnyAny<
            uncharacterize_t<ddc::TypeSeq<ProdDDim>>,
            uncharacterize_t<ddc::TypeSeq<Index1...>>,
            uncharacterize_t<ddc::TypeSeq<Index2...>>,
            ddc::type_seq_remove_t<
                    uncharacterize_t<ddc::to_type_seq_t<
                            sil::misc::cartesian_prod_t<natural_domain_t<ProdDDim>>>>,
                    uncharacterize_t<ddc::to_type_seq_t<
                            sil::misc::cartesian_prod_t<natural_domain_t<Index2>...>>>>,
            ddc::type_seq_remove_t<
                    uncharacterize_t<ddc::to_type_seq_t<
                            sil::misc::cartesian_prod_t<natural_domain_t<Index1>...>>>,
                    uncharacterize_t<ddc::to_type_seq_t<
                            sil::misc::cartesian_prod_t<natural_domain_t<ProdDDim>>>>>,
            ddc::type_seq_remove_t<
                    uncharacterize_t<ddc::to_type_seq_t<
                            sil::misc::cartesian_prod_t<natural_domain_t<ProdDDim>>>>,
                    uncharacterize_t<ddc::to_type_seq_t<
                            sil::misc::cartesian_prod_t<natural_domain_t<Index1>...>>>>>::
    operator()(
            uncharacterize_tensor(prod_tensor),
            uncharacterize_tensor(tensor1),
            uncharacterize_tensor(tensor2));
    return prod_tensor;
}
#endif

} // namespace tensor

} // namespace sil
