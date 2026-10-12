// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#pragma once

#include <array>

#include <similie/misc/clamp_to_domain.hpp>
#include <similie/misc/type_seq_conversion.hpp>

#include "owning_tensor.hpp"

namespace sil::tensor {
namespace detail {
template <class Domain, class Memory, class Evaluate>
KOKKOS_FUNCTION auto pointwise_tensor(Evaluate evaluate)
{
    using Accessor = tensor_accessor_for_domain_t<Domain>;
    std::array<double, Accessor::domain().size()> storage {};
    Tensor<double, Domain, Kokkos::layout_right, Memory> tensor(
            ddc::ChunkSpan<
                    double,
                    Domain,
                    Kokkos::layout_right,
                    Memory>(storage.data(), Accessor::domain()));
    ddc::device_for_each(tensor.domain(), [&](auto component) {
        tensor.mem(component) = evaluate(component);
    });
    return OwningTensor(tensor, std::move(storage));
}

template <class Field, class Evaluate>
KOKKOS_FUNCTION auto pointwise_tensor(Evaluate evaluate)
{
    return pointwise_tensor<typename Field::indices_domain_t, typename Field::memory_space>(
            evaluate);
}

// Source zero denotes stored samples. For a single rule, source one denotes
// its evaluation at the requested exterior point. For per-face rules, sources
// 1 + 2*axis and 2 + 2*axis denote the lower and upper face evaluations.
// The spatial domain contains the nearest node and its inward neighbours;
// rule values occupy the nearest node in their source plane.
struct ExtrapolationSource
{
};

template <class Domain, class Element>
KOKKOS_FUNCTION auto extrapolation_sample(Domain domain, Element element)
{
    misc::convert_type_seq_to_t<misc::SignedDiscreteElement, ddc::to_type_seq_t<Domain>> const
            sample(element);
    return misc::clamp_to_domain(domain, sample).storage_element();
}

template <std::size_t Sources = 2, class... Grid>
KOKKOS_FUNCTION auto make_extrapolation_basis(ddc::DiscreteDomain<Grid...> domain, auto element)
{
    typename ddc::DiscreteDomain<Grid...>::discrete_element_type front
            = extrapolation_sample(domain, element);
    typename ddc::DiscreteDomain<Grid...>::discrete_vector_type extent = domain.extents();
    for (std::size_t dim = 0; dim < sizeof...(Grid); ++dim) {
        if (ddc::detail::array(extent)[dim] > 1) {
            ddc::detail::array(extent)[dim] = 2;
            if (ddc::detail::array(front)[dim] == ddc::detail::array(domain.back())[dim])
                --ddc::detail::array(front)[dim];
        }
    }
    ddc::DiscreteDomain<Grid..., ExtrapolationSource> const
            patch(ddc::DiscreteDomain<Grid...>(front, extent),
                  ddc::DiscreteDomain<ExtrapolationSource>(
                          ddc::DiscreteElement<ExtrapolationSource>(0),
                          ddc::DiscreteVector<ExtrapolationSource>(Sources)));
    std::array<double, (1UL << sizeof...(Grid)) * Sources> storage {};
    Tensor<double,
           ddc::DiscreteDomain<Grid..., ExtrapolationSource>,
           Kokkos::layout_right,
           Kokkos::HostSpace>
            tensor(ddc::ChunkSpan<
                    double,
                    ddc::DiscreteDomain<Grid..., ExtrapolationSource>,
                    Kokkos::layout_right,
                    Kokkos::HostSpace>(storage.data(), patch));
    return OwningTensor(tensor, std::move(storage));
}

template <class Basis, class Stored, class Evaluated>
KOKKOS_FUNCTION double contract_extrapolation_basis(
        Basis const& basis,
        Stored stored,
        Evaluated evaluated)
{
    using Spatial = misc::convert_type_seq_to_t<
            ddc::DiscreteDomain,
            ddc::type_seq_remove_t<
                    ddc::to_type_seq_t<typename Basis::discrete_domain_type>,
                    ddc::TypeSeq<ExtrapolationSource>>>;
    double result = 0.0;
    ddc::device_for_each(basis.domain(), [&](auto entry) {
        double const weight = basis.mem(entry);
        if (weight != 0.0) {
            std::size_t const source = entry.template uid<ExtrapolationSource>();
            result += weight
                      * (source == 0 ? stored(typename Spatial::discrete_element_type(entry))
                                     : evaluated(source));
        }
    });
    return result;
}
template <class Element>
KOKKOS_FUNCTION auto storage_element(Element element)
{
    if constexpr (misc::Specialization<Element, misc::SignedDiscreteElement>)
        return element.storage_element();
    else
        return element;
}

template <class Rule>
KOKKOS_FUNCTION bool is_interface(Rule const& rule)
{
    if constexpr (requires { rule.is_interface(); })
        return rule.is_interface();
    else if constexpr (requires { Rule::IS_INTERFACE; })
        return Rule::IS_INTERFACE;
    else
        return false;
}
template <class Rule>
KOKKOS_FUNCTION bool is_dirichlet(Rule const& rule)
{
    if constexpr (requires { rule.is_dirichlet(); })
        return rule.is_dirichlet();
    else if constexpr (requires { Rule::IS_DIRICHLET; })
        return Rule::IS_DIRICHLET;
    else
        return false;
}
template <class Rule>
KOKKOS_FUNCTION bool shared_boundary(Rule const& rule)
{
    if constexpr (requires { rule.shared_boundary(); })
        return rule.shared_boundary();
    else if constexpr (requires { rule.boundary_map.shared_boundary; })
        return rule.boundary_map.shared_boundary;
    else
        return true;
}
template <class... Grid, class Entry>
KOKKOS_FUNCTION ddc::DiscreteElement<Grid...> stencil_sample(
        ddc::DiscreteElement<Grid...> center,
        Entry entry)
{
    ((center.template uid<Grid>() += entry.template uid<Grid>() - 1), ...);
    return center;
}

template <class Rule, class Field, class Element, class Component>
KOKKOS_FUNCTION double extrapolation_source_value(
        Rule const& rule,
        Field const& field,
        Element element,
        Component component)
{
    if constexpr (requires { rule.source_value(); })
        return rule.source_value();
    else if constexpr (requires { rule.prescribed_value; })
        return rule.prescribed_value;
    else if constexpr (requires { rule.exterior_value; })
        return rule.exterior_value;
    else if constexpr (requires { Rule::STORED_SAMPLES_ONLY; })
        return 0.0;
    else
        return rule(field, element).mem(component);
}
} // namespace detail
} // namespace sil::tensor
