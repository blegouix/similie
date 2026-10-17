// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#pragma once

#include <array>
#include <utility>

#include <similie/misc/specialization.hpp>
#include <similie/misc/type_seq_ext.hpp>
#include <similie/tensor/tensor_indices/character.hpp>
#include <similie/tensor/tensor_indices/diagonal_tensor.hpp>
#include <similie/tensor/tensor_indices/identity_tensor.hpp>
#include <similie/tensor/tensor_indices/levi_civita_tensor.hpp>

#include "diagonal.hpp"
#include "identity.hpp"
#include "levi_civita.hpp"

namespace sil::tensor::detail {

template <class Index>
concept StructuredTensorProductIndex = misc::Specialization<Index, TensorIdentityIndex>
                                       || misc::Specialization<Index, TensorDiagonalIndex>
                                       || misc::Specialization<Index, TensorLeviCivitaIndex>;

template <class Index, class... Tags>
KOKKOS_FUNCTION constexpr bool tensor_product_indices_match(ddc::DiscreteElement<Tags...> element)
{
    if constexpr (misc::Specialization<Index, TensorIdentityIndex>) {
        return IdentityTensorProductMatcher<Index>::operator()(
                misc::type_seq_intersect_t<
                        ddc::to_type_seq_t<natural_domain_t<Index>>,
                        ddc::TypeSeq<Tags...>>(),
                element);
    } else if constexpr (misc::Specialization<Index, TensorDiagonalIndex>) {
        return DiagonalTensorProductMatcher<Index>::operator()(
                misc::type_seq_intersect_t<
                        ddc::to_type_seq_t<natural_domain_t<Index>>,
                        ddc::TypeSeq<Tags...>>(),
                element);
    } else if constexpr (misc::Specialization<Index, TensorLeviCivitaIndex>) {
        return LeviCivitaTensorProductMatcher<Index>::operator()(
                misc::type_seq_intersect_t<
                        ddc::to_type_seq_t<natural_domain_t<Index>>,
                        ddc::TypeSeq<Tags...>>(),
                element);
    } else {
        return true;
    }
}

template <class... Free, class... Contracted>
KOKKOS_FUNCTION constexpr ddc::DiscreteElement<Free..., Contracted...> tensor_product_known_element(
        ddc::DiscreteElement<Free...> free,
        ddc::DiscreteElement<Contracted...> contracted)
{
    return ddc::DiscreteElement<Free..., Contracted...>(free, contracted);
}

template <class ProductIndices, class LeftIndices, class RightIndices, class ContractedIndices>
class TensorProdStructured;

// Unfold the canonical output components and structurally possible contractions.
// Accessor conversions then see constant indices, including antisymmetric signs.
// This works for a structured index in either operand, alongside other indices.
template <class... Product, class... Left, class... Right, class... Contracted>
class TensorProdStructured<
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
            Tensor1 left,
            Tensor2 right,
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
            Tensor1 left,
            Tensor2 right,
            std::remove_const_t<typename Tensor1::element_type>& sum,
            bool& initialized,
            std::index_sequence<Values...> assigned)
    {
        constexpr auto known = tensor_product_known_element(
                TensorAccessor<Product...>::canonical_natural_element(output_element<Output>()),
                contracted_element(
                        std::index_sequence<Values...>(),
                        std::make_index_sequence<sizeof...(Values)>()));
        if constexpr (
                (tensor_product_indices_match<Left>(known) && ...)
                && (tensor_product_indices_match<Right>(known) && ...)) {
            if constexpr (sizeof...(Values) == sizeof...(Contracted)) {
                constexpr auto left_element = TensorAccessor<Left...>::access_element(known);
                constexpr auto right_element = TensorAccessor<Right...>::access_element(known);
                if (initialized)
                    sum += left.get(left_element) * right.get(right_element);
                else {
                    sum = left.get(left_element) * right.get(right_element);
                    initialized = true;
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
    }

    template <std::size_t Output, class ProductTensor, class Tensor1, class Tensor2>
    KOKKOS_FORCEINLINE_FUNCTION static void write(
            ProductTensor product,
            Tensor1 left,
            Tensor2 right)
    {
        constexpr ddc::DiscreteElement<Product...> element = output_element<Output>();
        // Tensor views may select only a subset of the stored output components.
        if (product.domain().contains(element)) {
            typename ProductTensor::element_type sum = 0.;
            bool initialized = false;
            contract<Output>(left, right, sum, initialized, std::index_sequence<>());
            product.mem(element) = sum;
        }
    }

    template <class ProductTensor, class Tensor1, class Tensor2, std::size_t... Outputs>
    KOKKOS_FORCEINLINE_FUNCTION static void write(
            ProductTensor product,
            Tensor1 left,
            Tensor2 right,
            std::index_sequence<Outputs...>)
    {
        (write<Outputs>(product, left, right), ...);
    }

public:
    template <class ProductTensor, class Tensor1, class Tensor2>
    KOKKOS_FUNCTION static void operator()(ProductTensor product, Tensor1 left, Tensor2 right)
    {
        write(product,
              left,
              right,
              std::make_index_sequence<(Product::mem_size() * ... * std::size_t(1))>());
    }
};

} // namespace sil::tensor::detail
