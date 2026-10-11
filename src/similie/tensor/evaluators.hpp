// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

#include <similie/misc/clamp_to_domain.hpp>
#include <similie/misc/domain_contains.hpp>
#include <similie/misc/specialization.hpp>

#include <Kokkos_Core.hpp>

#include "extrapolation_basis.hpp"

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

template <class Rule, class Field, class Element, class Component>
KOKKOS_FUNCTION double evaluate_extrapolation_source(
        Rule const& rule,
        Field const& field,
        Element element,
        Component component)
{
    if constexpr (requires { Rule::STORED_SAMPLES_ONLY; })
        return 0.0;
    else if constexpr (requires { rule.prescribed_value; })
        return rule.prescribed_value;
    else if constexpr (requires { rule.exterior_value; })
        return rule.exterior_value;
    else
        return rule(field, element).mem(component);
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
            return contract_extrapolation_basis(
                    stencil,
                    [&](auto sample) { return sampler(field, sample, component); },
                    [&](std::size_t) {
                        return evaluate_extrapolation_source(rule, field, element, component);
                    });
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

namespace sil::tensor::detail {
template <class... Grid, class Entry>
KOKKOS_FUNCTION ddc::DiscreteElement<Grid...> stencil_sample(
        ddc::DiscreteElement<Grid...> center,
        Entry entry)
{
    ((center.template uid<Grid>() += entry.template uid<Grid>() - 1), ...);
    return center;
}

// Expand native coefficient tensors into stored samples and independent sources.
// Donor maps and affine jumps remain properties of the extrapolation policies.
template <class Field, class Element, class Component, class Sample, class Constant>
KOKKOS_FUNCTION void visit_tensor_basis(
        Field const& field,
        Element element,
        Component component,
        double weight,
        Sample sample,
        Constant constant);

template <class Rule, class Field, class Element, class Component, class Sample, class Constant>
KOKKOS_FUNCTION void visit_rule_basis(
        Rule const& rule,
        Field const& field,
        Element element,
        Component component,
        double weight,
        Sample sample,
        Constant constant);

template <class Rule, class Field, class Element, class Component, class Sample, class Constant>
KOKKOS_FUNCTION void visit_rule_source(
        Rule const& rule,
        Field const& field,
        Element element,
        Component component,
        double weight,
        Sample sample,
        Constant constant)
{
    if constexpr (requires {
                      rule.interface;
                      rule.external;
                      rule.boundary;
                  }) {
        if (rule.interface)
            visit_rule_source(rule.external, field, element, component, weight, sample, constant);
        else
            visit_rule_source(rule.boundary, field, element, component, weight, sample, constant);
    } else if constexpr (requires {
                             rule.neighbor;
                             rule.boundary_map;
                             rule.target_rule;
                         }) {
        visit_rule_basis(
                rule.target_rule,
                rule.neighbor,
                rule.boundary_map(
                        field.non_indices_domain(),
                        rule.neighbor.non_indices_domain(),
                        element),
                component,
                weight * rule.orientation,
                sample,
                constant);
        constant(weight * rule.jump);
    } else
        constant(
                weight
                * evaluate_rule_basis(
                          rule,
                          field,
                          [](auto const&, auto, auto) { return 0.0; },
                          element)
                          .mem(component));
}

template <
        class Rule,
        class Field,
        class Element,
        class Component,
        class Sample,
        class Constant,
        std::size_t... Axis>
KOKKOS_FUNCTION void visit_rule_faces(
        Rule const& rule,
        Field const& field,
        Element element,
        Component component,
        std::size_t source,
        double coefficient,
        Sample sample,
        Constant constant,
        std::index_sequence<Axis...>)
{
    ((source == 1 + 2 * Axis ? visit_rule_source(
                                       rule.template boundary_rule<Axis, false>(),
                                       field,
                                       element,
                                       component,
                                       coefficient,
                                       sample,
                                       constant)
                             : void()),
     ...);
    ((source == 2 + 2 * Axis ? visit_rule_source(
                                       rule.template boundary_rule<Axis, true>(),
                                       field,
                                       element,
                                       component,
                                       coefficient,
                                       sample,
                                       constant)
                             : void()),
     ...);
}

template <class Rule, class Field, class Element, class Component, class Sample, class Constant>
KOKKOS_FUNCTION void visit_rule_basis(
        Rule const& rule,
        Field const& field,
        Element element,
        Component component,
        double weight,
        Sample sample,
        Constant constant)
{
    if constexpr (requires {
                      rule.interface;
                      rule.external;
                      rule.boundary;
                  }) {
        if (rule.interface)
            visit_rule_basis(rule.external, field, element, component, weight, sample, constant);
        else
            visit_rule_basis(rule.boundary, field, element, component, weight, sample, constant);
    } else {
        auto const coefficients = rule.value(field.non_indices_domain(), element);
        ddc::device_for_each(coefficients.domain(), [&](auto entry) {
            double const coefficient = weight * coefficients.mem(entry);
            if (coefficient == 0.0)
                return;
            std::size_t const source = entry.template uid<ExtrapolationSource>();
            if (source == 0)
                visit_tensor_basis(
                        field,
                        typename Field::non_indices_domain_t::discrete_element_type(entry),
                        component,
                        coefficient,
                        sample,
                        constant);
            else if constexpr (requires { rule.template boundary_rule<0, false>(); }) {
                visit_rule_faces(
                        rule,
                        field,
                        element,
                        component,
                        source,
                        coefficient,
                        sample,
                        constant,
                        std::make_index_sequence<ddc::type_seq_size_v<
                                ddc::to_type_seq_t<typename Field::non_indices_domain_t>>> {});
            } else
                visit_rule_source(rule, field, element, component, coefficient, sample, constant);
        });
    }
}

template <class Field, class Element, class Component, class Sample, class Constant>
KOKKOS_FUNCTION void visit_tensor_basis(
        Field const& field,
        Element element,
        Component component,
        double weight,
        Sample sample,
        Constant constant)
{
    if constexpr (requires {
                      *field.field;
                      field.domain_id;
                  })
        visit_tensor_basis(*field.field, element, component, weight, sample, constant);
    else if constexpr (requires {
                           field.rules;
                           field.field;
                           field.fixed;
                       }) {
        misc::convert_type_seq_to_t<
                misc::SignedDiscreteElement,
                ddc::to_type_seq_t<typename std::remove_cvref_t<decltype(field.field)>::
                                           non_indices_domain_t>> const point(field.fixed, element);
        visit_rule_basis(field.rules, field.field, point, component, weight, sample, constant);
    } else if constexpr (requires {
                             field.cochain;
                             field.prescribed_flux(element, component, std::declval<double&>());
                         }) {
        double prescribed = 0.0;
        if (field.prescribed_flux(element, component, prescribed)) {
            constant(weight * prescribed);
            return;
        }
        auto const coefficients = field.value(element, component);
        ddc::device_for_each(coefficients.domain(), [&](auto entry) {
            double const coefficient = weight * coefficients.mem(entry);
            if (coefficient != 0.0)
                visit_tensor_basis(
                        field.cochain,
                        stencil_sample(storage_element(element), entry),
                        typename std::remove_cvref_t<decltype(coefficients)>::indices_domain_t::
                                discrete_element_type(entry),
                        coefficient,
                        sample,
                        constant);
        });
    } else
        sample(field, storage_element(element), component, weight);
}
} // namespace sil::tensor::detail
