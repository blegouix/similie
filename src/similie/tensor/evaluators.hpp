// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

#include <similie/misc/clamp_to_domain.hpp>
#include <similie/misc/domain_contains.hpp>
#include <similie/misc/specialization.hpp>

#include <Kokkos_Core.hpp>

#include "extrapolation_stencil.hpp"

namespace sil {

namespace tensor {

namespace detail {

struct IdentityStencilEvaluator
{
    KOKKOS_FUNCTION double operator()(auto, auto) const
    {
        return 0.0;
    }

    KOKKOS_FUNCTION double value(auto sampler, auto sampled_elem, auto component) const
    {
        return sampler(sampled_elem, component);
    }
};

template <class TensorType>
struct ClampedTensorEvaluator
{
    TensorType tensor;

    KOKKOS_FUNCTION double operator()(auto sampled_elem, auto component) const
    {
        auto const clamped_elem = misc::clamp_to_domain(tensor.non_indices_domain(), sampled_elem);
        return tensor.mem(clamped_elem, component);
    }

    KOKKOS_FUNCTION double value(auto sampler, auto sampled_elem, auto component) const
    {
        auto const clamped_elem = misc::clamp_to_domain(tensor.non_indices_domain(), sampled_elem);
        return sampler(clamped_elem, component);
    }
};

template <class TensorType>
struct ZeroOutsideTensorEvaluator
{
    TensorType tensor;

    KOKKOS_FUNCTION double operator()(auto sampled_elem, auto component) const
    {
        if (!misc::domain_contains(tensor.non_indices_domain(), sampled_elem)) {
            return 0.0;
        }
        return tensor.mem(sampled_elem, component);
    }

    KOKKOS_FUNCTION double value(auto sampler, auto sampled_elem, auto component) const
    {
        if (!misc::domain_contains(tensor.non_indices_domain(), sampled_elem)) {
            return 0.0;
        }
        return sampler(sampled_elem, component);
    }
};

} // namespace detail

} // namespace tensor

} // namespace sil

namespace sil::tensor {

struct StoredTensorSampler
{
    template <class Field, class Element, class Component>
    KOKKOS_FUNCTION double operator()(Field const& field, Element element, Component component)
            const
    {
        if constexpr (misc::Specialization<Element, misc::SignedDiscreteElement>)
            return field.mem(element.storage_element(), component);
        else
            return field.mem(element, component);
    }
};

namespace detail {
template <class Element>
KOKKOS_FUNCTION auto storage_element(Element element)
{
    if constexpr (misc::Specialization<Element, misc::SignedDiscreteElement>)
        return element.storage_element();
    else
        return element;
}

template <class Rule, class Field, class Sampler, class Element>
KOKKOS_FUNCTION auto evaluate_rule_basis(
        Rule const& rule,
        Field const& field,
        Sampler sampler,
        Element element)
{
    if constexpr (std::is_same_v<Sampler, StoredTensorSampler>)
        return rule(field, element);
    else if constexpr (requires { rule.evaluate_basis(field, sampler, element); })
        return rule.evaluate_basis(field, sampler, element);
    else {
        auto const stencil = rule.value(field.non_indices_domain(), element);
        return pointwise_tensor<Field>([&](auto component) {
            return stencil.evaluate(
                    [&](auto const& term) { return sampler(field, term.element, component); });
        });
    }
}

template <class Field, class Sampler, class Element, class Component>
KOKKOS_FUNCTION double evaluate_tensor_basis(
        Field const& field,
        Sampler sampler,
        Element element,
        Component component)
{
    if constexpr (requires {
                      field.rules;
                      field.field;
                      field.fixed;
                  }) {
        misc::convert_type_seq_to_t<
                misc::SignedDiscreteElement,
                ddc::to_type_seq_t<typename decltype(field.field)::non_indices_domain_t>> const
                sample(field.fixed, element);
        return evaluate_rule_basis(field.rules, field.field, sampler, sample).mem(component);
    } else
        return sampler(field, element, component);
}

// Basis substitution is an assembly detail, not part of the extrapolation API.
template <class Sampler>
struct EvaluateRuleValue
{
    Sampler sampler;
    template <class Rule, class Field, class Element, class Component>
    KOKKOS_FUNCTION double operator()(
            Rule const& rule,
            Field const& field,
            Element element,
            Component component) const
    {
        return evaluate_rule_basis(rule, field, sampler, element).mem(component);
    }
};
} // namespace detail
} // namespace sil::tensor

namespace sil::tensor::detail {
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
} // namespace sil::tensor::detail
