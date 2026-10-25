// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#pragma once

#include <array>
#include <concepts>
#include <cstddef>
#include <type_traits>
#include <utility>
#include <vector>

#include <ddc/ddc.hpp>

namespace sil::tensor {

template <class ElementType, class SupportType, class LayoutStridedPolicy, class MemorySpace>
class Tensor;

namespace detail {
struct DimensionConcept
{
};

struct GridConcept : ddc::UniformPointSampling<DimensionConcept>
{
};

template <class Index>
using FieldConcept = Tensor<
        double,
        ddc::DiscreteDomain<GridConcept, Index>,
        Kokkos::layout_right,
        Kokkos::HostSpace>;

struct IndexTensorConcept
{
    using element_type = double;
};

struct IndexAccessConcept
{
    template <class Tensor, class Element>
    double operator()(Tensor const& tensor, Element element) const;
};
} // namespace detail

// All indices expose the same metadata and access operation. Explicit storage
// additionally provides a direct map between access IDs and memory IDs.
template <class Index>
concept TensorIndexConcept = requires(
        detail::IndexTensorConcept tensor,
        ddc::DiscreteElement<Index> element,
        detail::IndexAccessConcept access) {
    requires Index::is_tensor_index;
    typename std::bool_constant<Index::is_explicitely_stored_tensor>;
    typename Index::subindices_domain_t;
    { Index::subindices_domain() } -> std::same_as<typename Index::subindices_domain_t>;
    { Index::rank() } -> std::same_as<std::size_t>;
    { Index::size() } -> std::same_as<std::size_t>;
    { Index::mem_size() } -> std::same_as<std::size_t>;
    { Index::access_size() } -> std::same_as<std::size_t>;
    typename std::integral_constant<std::size_t, Index::rank()>;
    typename std::integral_constant<std::size_t, Index::size()>;
    typename std::integral_constant<std::size_t, Index::mem_size()>;
    typename std::integral_constant<std::size_t, Index::access_size()>;
    {
        Index::template process_access<
                detail::IndexTensorConcept,
                ddc::DiscreteElement<Index>,
                Index>(access, tensor, element)
    } -> std::same_as<double>;
    requires(!Index::is_explicitely_stored_tensor || requires(std::size_t id) {
        { Index::access_id_to_mem_id(id) } -> std::same_as<std::size_t>;
        {
            Index::mem_id_to_canonical_natural_ids(id)
        } -> std::same_as<std::array<std::size_t, Index::rank()>>;
        // Natural indices take one ID; composite indices take subindex IDs.
        requires (requires(std::array<std::size_t, Index::subindices_domain_t::rank()> const& ids) {
            { Index::mem_id(ids) } -> std::same_as<std::size_t>;
            { Index::access_id(ids) } -> std::same_as<std::size_t>;
        } || requires {
            requires Index::is_tensor_natural_index;
            { Index::mem_id(id) } -> std::same_as<std::size_t>;
            { Index::access_id(id) } -> std::same_as<std::size_t>;
        });
    });
};

template <class Index, class Dimension>
concept NaturalTensorIndexConcept = TensorIndexConcept<Index> && requires(std::size_t id) {
    requires Index::is_explicitely_stored_tensor;
    requires Index::is_tensor_natural_index;
    typename Index::type_seq_dimensions;
    { Index::template mem_id<Dimension>() } -> std::same_as<std::size_t>;
    { Index::mem_id(id) } -> std::same_as<std::size_t>;
    { Index::access_id(id) } -> std::same_as<std::size_t>;
};

template <class Index>
concept YoungTableauTensorIndexConcept
        = TensorIndexConcept<Index>
          && requires(std::array<std::size_t, Index::subindices_domain_t::rank()> const& ids) {
                 requires(!Index::is_explicitely_stored_tensor);
                 { Index::access_id(ids) } -> std::same_as<std::size_t>;
                 {
                     Index::mem_lin_comb(ids)
                 } -> std::same_as<std::pair<std::vector<double>, std::vector<std::size_t>>>;
             };

template <class Rule, class Field, class Element>
concept ExtrapolationRuleConcept = requires(
        Rule const& rule,
        Field const& field,
        typename Field::non_indices_domain_t domain,
        Element element) {
    rule.value(domain, element).domain();
    { rule(field, element).domain() } -> std::same_as<typename Field::indices_domain_t>;
};

template <class Rules, class Field, class Element>
concept ExtrapolationRulesConcept
        = ExtrapolationRuleConcept<Rules, Field, Element> && requires(Rules const& rules) {
              requires std::is_lvalue_reference_v<
                      decltype(rules.template boundary_rule<0, false>())>;
              requires std::is_const_v<
                      std::remove_reference_t<decltype(rules.template boundary_rule<0, false>())>>;
              requires std::is_lvalue_reference_v<
                      decltype(rules.template boundary_rule<0, true>())>;
              requires std::is_const_v<
                      std::remove_reference_t<decltype(rules.template boundary_rule<0, true>())>>;
          };

} // namespace sil::tensor
