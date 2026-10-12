// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#pragma once

#include <similie/tensor/extrapolation_basis.hpp>

namespace sil::multidomains::detail {
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
                * sil::tensor::detail::extrapolation_source_value(rule, field, element, component));
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
            std::size_t const source
                    = entry.template uid<sil::tensor::detail::ExtrapolationSource>();
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
    if constexpr (requires { *field.field; })
        visit_tensor_basis(*field.field, element, component, weight, sample, constant);
    else if constexpr (requires {
                           field.rules;
                           field.field;
                           field.fixed;
                       }) {
        sil::misc::convert_type_seq_to_t<
                sil::misc::SignedDiscreteElement,
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
                        sil::tensor::detail::stencil_sample(
                                sil::tensor::detail::storage_element(element),
                                entry),
                        typename std::remove_cvref_t<decltype(coefficients)>::indices_domain_t::
                                discrete_element_type(entry),
                        coefficient,
                        sample,
                        constant);
        });
    } else
        sample(field, sil::tensor::detail::storage_element(element), component, weight);
}
} // namespace sil::multidomains::detail
