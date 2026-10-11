// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#pragma once

#include <similie/misc/signed_discrete_element.hpp>
#include <similie/misc/type_seq_conversion.hpp>

#include "extrapolation_rules.hpp"

namespace sil::tensor {

/** Lazy tensor sampling on its stored domain and its extrapolated exterior.
 * \important This operator and documentation is fully AI-generated.
 * Coordinates remain signed until the face rules resolve them to a stored
 * sample. value() substitutes a caller's sampler, retaining the source/donor
 * field identity for matrix assembly. This view performs no differentiation.
 * The view borrows tensor allocations through its field and rule handles;
 * their owners must outlive it and synchronize updates before sampling.
 */
template <class Field, class Rules>
struct TensorWithExtrapolationRules
{
    using non_indices_domain_t = typename Field::non_indices_domain_t;
    using memory_space = typename Field::memory_space;
    using indices_domain_t = typename Field::indices_domain_t;
    using discrete_domain_type = typename Field::discrete_domain_type;
    Field field;
    Rules rules;

    KOKKOS_FUNCTION non_indices_domain_t non_indices_domain() const
    {
        return field.non_indices_domain();
    }

    template <class Sampler, class Element, class Component>
    KOKKOS_FUNCTION double value(Sampler sampler, Element elem, Component component) const
    {
        misc::convert_type_seq_to_t<
                misc::SignedDiscreteElement,
                ddc::type_seq_remove_t<
                        ddc::to_type_seq_t<discrete_domain_type>,
                        ddc::to_type_seq_t<Component>>> const signed_elem(elem);
        auto const sample = [&](auto const& sampled_field, auto resolved, auto index) {
            if constexpr (misc::Specialization<decltype(resolved), misc::SignedDiscreteElement>)
                return sampler(sampled_field, resolved.storage_element(), index);
            else
                return sampler(sampled_field, resolved, index);
        };
        return detail::EvaluateRuleValue<decltype(sample)> {
                sample}(rules, field, signed_elem, component);
    }

    template <class Sampler, class Element>
    KOKKOS_FUNCTION double value(Sampler sampler, Element elem) const
    {
        misc::convert_type_seq_to_t<
                misc::SignedDiscreteElement,
                ddc::to_type_seq_t<indices_domain_t>> const component(elem);
        return value(sampler, elem, component.storage_element());
    }

    template <class Element>
    KOKKOS_FUNCTION double mem(Element elem) const
    {
        return value(StoredTensorSampler {}, elem);
    }

    template <class Element>
    KOKKOS_FUNCTION double operator()(Element elem) const
    {
        return mem(elem);
    }

    template <class Element, class Component>
    KOKKOS_FUNCTION double mem(Element elem, Component component) const
    {
        return value(StoredTensorSampler {}, elem, component);
    }

    template <class Element, class Component>
    KOKKOS_FUNCTION double operator()(Element elem, Component component) const
    {
        return mem(elem, component);
    }
};

template <class Field, class Rule>
KOKKOS_FUNCTION auto make_tensor_with_extrapolation_rules(Field field, Rule rule)
{
    if constexpr (misc::Specialization<Rule, ExtrapolationRules>)
        return TensorWithExtrapolationRules<Field, Rule> {field, rule};
    else
        return make_tensor_with_extrapolation_rules(field, make_extrapolation_rules(field, rule));
}

} // namespace sil::tensor
