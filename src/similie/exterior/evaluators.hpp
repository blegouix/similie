// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

#include <similie/misc/clamp_to_domain.hpp>
#include <similie/misc/domain_contains.hpp>

#include <Kokkos_Core.hpp>

namespace sil {

namespace exterior {

namespace detail {

struct IdentityStencilEvaluator
{
    KOKKOS_FUNCTION double operator()(auto, auto) const
    {
        return 0.0;
    }

    KOKKOS_FUNCTION double value(auto sampler, auto sampled_elem, auto cochain_elem) const
    {
        return sampler(sampled_elem, cochain_elem);
    }
};

template <class TensorType>
struct ClampedTensorEvaluator
{
    TensorType tensor;

    KOKKOS_FUNCTION double operator()(auto sampled_elem, auto cochain_elem) const
    {
        auto const clamped_elem = misc::clamp_to_domain(tensor.non_indices_domain(), sampled_elem);
        return tensor.mem(clamped_elem, cochain_elem);
    }

    KOKKOS_FUNCTION double value(auto sampler, auto sampled_elem, auto cochain_elem) const
    {
        auto const clamped_elem = misc::clamp_to_domain(tensor.non_indices_domain(), sampled_elem);
        return sampler(clamped_elem, cochain_elem);
    }
};

template <class TensorType>
struct ZeroOutsideTensorEvaluator
{
    TensorType tensor;

    KOKKOS_FUNCTION double operator()(auto sampled_elem, auto cochain_elem) const
    {
        if (!misc::domain_contains(tensor.non_indices_domain(), sampled_elem)) {
            return 0.0;
        }
        return tensor.mem(sampled_elem, cochain_elem);
    }

    KOKKOS_FUNCTION double value(auto sampler, auto sampled_elem, auto cochain_elem) const
    {
        if (!misc::domain_contains(tensor.non_indices_domain(), sampled_elem)) {
            return 0.0;
        }
        return sampler(sampled_elem, cochain_elem);
    }
};

} // namespace detail

} // namespace exterior

} // namespace sil

namespace sil::exterior {

/** Substitute a sampler for stored cochain values, retaining domain identity.
 * \important This operator and documentation is fully AI-generated.
 * A sampler receives (field, element, component), so it can distinguish identical
 * local coordinates in different domains when evaluating a global basis vector.
 * The field is borrowed and must outlive this sampling view.
 */
template <class Field, class Sampler>
struct SampledCochain
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

struct StoredCochainSampler
{
    template <class Field, class Element, class Component>
    KOKKOS_FUNCTION double operator()(Field const& field, Element element, Component component)
            const
    {
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
            return rule(SampledCochain<Field, Sampler> {field, sampler}, element, component);
    }
};
} // namespace detail
} // namespace sil::exterior
