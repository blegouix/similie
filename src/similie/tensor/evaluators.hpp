// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

#include <similie/misc/clamp_to_domain.hpp>
#include <similie/misc/domain_contains.hpp>
#include <similie/misc/specialization.hpp>

#include <Kokkos_Core.hpp>

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

/** Substitute a sampler for stored tensor values, retaining domain identity.
 * \important This operator and documentation is fully AI-generated.
 * A sampler receives (field, element, component), so it can distinguish identical
 * local coordinates in different domains when evaluating a global basis vector.
 * The field is borrowed and must outlive this sampling view.
 */
template <class Field, class Sampler>
struct SampledTensor
{
    using non_indices_domain_t = typename Field::non_indices_domain_t;
    using discrete_domain_type = typename Field::discrete_domain_type;
    Field const& field;
    Sampler sampler;

    KOKKOS_FUNCTION non_indices_domain_t non_indices_domain() const
    {
        return field.non_indices_domain();
    }

    template <class Element, class Component>
    KOKKOS_FUNCTION double mem(Element element, Component component) const
    {
        return sampler(field, element, component);
    }
};

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
struct EvaluateRule
{
    template <class Rule, class Field, class Element, class Component>
    KOKKOS_FUNCTION double operator()(
            Rule const& rule,
            Field const& field,
            Element element,
            Component component) const
    {
        return rule(field, element, component);
    }
};

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
        if constexpr (requires { rule.value(field, sampler, element, component); })
            return rule.value(field, sampler, element, component);
        else
            return rule(SampledTensor<Field, Sampler> {field, sampler}, element, component);
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
