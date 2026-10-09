// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#pragma once

#include <cstddef>
#include <type_traits>
#include <utility>

#include <ddc/ddc.hpp>

#include <similie/misc/domain_contains.hpp>
#include <similie/misc/specialization.hpp>
#include <similie/misc/type_seq_conversion.hpp>

#include "evaluators.hpp"

namespace sil::exterior {

namespace detail {

template <std::size_t Dimension, class... RulePairs>
class ExtrapolationRulesStorage;

template <std::size_t Dimension>
class ExtrapolationRulesStorage<Dimension>
{
public:
    KOKKOS_DEFAULTED_FUNCTION ExtrapolationRulesStorage() = default;

    template <class TensorType, class Element, class Component, class Evaluate>
    KOKKOS_FUNCTION void accumulate(
            TensorType const&,
            Element,
            Component,
            double&,
            double&,
            Evaluate) const
    {
    }
};

template <std::size_t Dimension, class RulePair, class... OtherRulePairs>
class ExtrapolationRulesStorage<Dimension, RulePair, OtherRulePairs...>
{
    RulePair m_rules;
    ExtrapolationRulesStorage<Dimension + 1, OtherRulePairs...> m_other_rules;

public:
    KOKKOS_DEFAULTED_FUNCTION ExtrapolationRulesStorage() = default;

    KOKKOS_FUNCTION explicit ExtrapolationRulesStorage(
            RulePair rules,
            OtherRulePairs... other_rules)
        : m_rules(rules)
        , m_other_rules(other_rules...)
    {
    }

    template <std::size_t Axis, bool Upper>
    KOKKOS_FUNCTION auto const& boundary_rule() const
    {
        if constexpr (Axis == Dimension) {
            if constexpr (Upper)
                return m_rules.second;
            else
                return m_rules.first;
        } else {
            return m_other_rules.template boundary_rule<Axis, Upper>();
        }
    }

    template <class TensorType, class Element, class Component, class Evaluate>
    KOKKOS_FUNCTION void accumulate(
            TensorType const& tensor,
            Element sampled_element,
            Component component,
            double& weighted_value,
            double& total_distance,
            Evaluate evaluate) const
    {
        using DDim = ddc::type_seq_element_t<
                Dimension,
                ddc::to_type_seq_t<typename TensorType::non_indices_domain_t>>;
        ddc::DiscreteElement<DDim> const sampled(sampled_element);
        ddc::DiscreteElement<DDim> const front(tensor.non_indices_domain().front());
        ddc::DiscreteElement<DDim> const back(tensor.non_indices_domain().back());
        // A negative offset also detects unsigned coordinates wrapping below zero.
        ddc::DiscreteVector<DDim> const offset = sampled - front;
        if (offset.template get<DDim>() < 0) {
            double const distance = -static_cast<double>(offset.template get<DDim>());
            weighted_value
                    += distance * evaluate(m_rules.first, tensor, sampled_element, component);
            total_distance += distance;
        } else if (sampled.template uid<DDim>() > back.template uid<DDim>()) {
            double const distance
                    = static_cast<double>(sampled.template uid<DDim>() - back.template uid<DDim>());
            weighted_value
                    += distance * evaluate(m_rules.second, tensor, sampled_element, component);
            total_distance += distance;
        }
        m_other_rules.accumulate(
                tensor,
                sampled_element,
                component,
                weighted_value,
                total_distance,
                evaluate);
    }
};

} // namespace detail

/**
 * Select an extrapolation rule independently on each side of each dimension.
 * \important This operator and documentation is fully AI-generated.
 *
 * Supply one std::pair of callable rules per non-index dimension of the tensor,
 * in its non_indices_domain_t order. The first rule samples beyond the lower
 * (left) boundary, and the second beyond the upper (right) boundary. Boundary
 * nodes themselves are stored samples and are returned directly, as are all
 * interior samples. The pair count is checked against the domain rank at
 * compile time, independently of the ordering of the sampled element's tags.
 *
 * Outside the domain, blend the selected boundary rules using normalized
 * outward distances in logical grid-index units: the result is
 * sum(distance_i * rule_i(tensor, element, component)) / sum(distance_i).
 * Each selected rule receives the original sample. Dimensions whose bounds
 * are not crossed contribute zero weight and their rules are not evaluated.
 * A sample outside only one dimension therefore uses that boundary rule alone.
 *
 * For example, beyond the upper-left corner, the left rule has weight
 * (x_min - x) / ((x_min - x) + (y - y_max)) and the upper rule has the
 * complementary weight. Approaching either adjacent single-boundary region
 * recovers its rule continuously, provided the individual rules are continuous.
 * Continuity at the stored corner itself also requires compatible rule limits.
 * Interior samples bypass normalization, including the corner where all
 * outward distances vanish. Each rule and its state must be copyable and
 * accessible in the differential operator's execution space. This class
 * implements the same (tensor, element, component) sampling
 * interface as individual rules and can be passed directly to deriv(),
 * transposed_coboundary(), codifferential(), or either stage of laplacian().
 */
template <misc::Specialization<std::pair>... RulePairs>
class ExtrapolationRules
{
    detail::ExtrapolationRulesStorage<0, RulePairs...> m_rules;

public:
    KOKKOS_DEFAULTED_FUNCTION ExtrapolationRules() = default;

    KOKKOS_FUNCTION explicit ExtrapolationRules(RulePairs... rules)
        requires(sizeof...(RulePairs) > 0)
        : m_rules(rules...)
    {
    }

    template <class ExtrapolationRule>
        requires(
                sizeof...(RulePairs) > 0
                && ((std::is_same_v<ExtrapolationRule, typename RulePairs::first_type>
                     && std::is_same_v<ExtrapolationRule, typename RulePairs::second_type>)
                    && ...))
    KOKKOS_FUNCTION ExtrapolationRules(ExtrapolationRule rule) : m_rules(RulePairs {rule, rule}...)
    {
    }

    /** Access one face policy for boundary integrals and nodal constraints. */
    template <std::size_t Axis, bool Upper>
    KOKKOS_FUNCTION auto const& boundary_rule() const
    {
        static_assert(Axis < sizeof...(RulePairs));
        return m_rules.template boundary_rule<Axis, Upper>();
    }

    template <class TensorType, class Element, class Component>
    KOKKOS_FUNCTION double operator()(
            TensorType const& tensor,
            Element sampled_element,
            Component component) const
    {
        static_assert(
                sizeof...(RulePairs)
                        == ddc::type_seq_size_v<
                                ddc::to_type_seq_t<typename TensorType::non_indices_domain_t>>,
                "ExtrapolationRules requires one left/right pair per non-index dimension");
        misc::convert_type_seq_to_t<
                ddc::DiscreteElement,
                ddc::type_seq_remove_t<
                        ddc::to_type_seq_t<typename TensorType::discrete_domain_type>,
                        ddc::to_type_seq_t<Component>>> const element(sampled_element);
        if (misc::domain_contains(tensor.non_indices_domain(), element))
            return tensor.mem(element, component);
        double weighted_value = 0.0;
        double total_distance = 0.0;
        m_rules.accumulate(
                tensor,
                element,
                component,
                weighted_value,
                total_distance,
                detail::EvaluateRule {});
        return weighted_value / total_distance;
    }

    /** Evaluate a basis/stencil sampler through the same face and corner rules.
     * The sampler receives the mapped donor field, including its domain tag.
     */
    template <class TensorType, class Sampler, class Element, class Component>
    KOKKOS_FUNCTION double value(
            TensorType const& tensor,
            Sampler sampler,
            Element sampled_element,
            Component component) const
    {
        static_assert(
                sizeof...(RulePairs)
                == ddc::type_seq_size_v<
                        ddc::to_type_seq_t<typename TensorType::non_indices_domain_t>>);
        misc::convert_type_seq_to_t<
                ddc::DiscreteElement,
                ddc::type_seq_remove_t<
                        ddc::to_type_seq_t<typename TensorType::discrete_domain_type>,
                        ddc::to_type_seq_t<Component>>> const element(sampled_element);
        if (misc::domain_contains(tensor.non_indices_domain(), element))
            return sampler(tensor, element, component);
        double weighted_value = 0.0, total_distance = 0.0;
        m_rules.accumulate(
                tensor,
                element,
                component,
                weighted_value,
                total_distance,
                detail::EvaluateRuleValue<Sampler> {sampler});
        return weighted_value / total_distance;
    }
};

template <class LeftRule, class RightRule, misc::Specialization<std::pair>... OtherRulePairs>
ExtrapolationRules(std::pair<LeftRule, RightRule>, OtherRulePairs...)
        -> ExtrapolationRules<std::pair<LeftRule, RightRule>, OtherRulePairs...>;

namespace detail {

template <class ExtrapolationRule, std::size_t... Dimension>
KOKKOS_FUNCTION auto repeat_extrapolation_rule(
        ExtrapolationRule rule,
        std::index_sequence<Dimension...>)
{
    if constexpr (sizeof...(Dimension) == 0)
        return ExtrapolationRules<> {};
    else
        return ExtrapolationRules((static_cast<void>(Dimension), std::pair {rule, rule})...);
}

} // namespace detail

/**
 * Broadcast a rule to both sides of all N non-index dimensions.
 * \important This operator and documentation is fully AI-generated.
 *
 * Each of the 2*N copies retains the supplied rule's state. This factory only
 * accepts single rules. Differential operators distinguish single rules from
 * ExtrapolationRules with if constexpr and only convert the former, with N
 * inferred from the cochain tensor domain. An explicitly typed uniform
 * ExtrapolationRules can also be implicitly constructed from its single rule.
 * The overload make_extrapolation_rules(tensor, rule) infers N directly from
 * tensor.non_indices_domain_t.
 */
template <std::size_t N, misc::NotSpecialization<ExtrapolationRules> ExtrapolationRule>
KOKKOS_FUNCTION auto make_extrapolation_rules(ExtrapolationRule rule)
{
    return detail::repeat_extrapolation_rule(rule, std::make_index_sequence<N>());
}

template <class TensorType, misc::NotSpecialization<ExtrapolationRules> ExtrapolationRule>
KOKKOS_FUNCTION auto make_extrapolation_rules(TensorType const&, ExtrapolationRule rule)
{
    return make_extrapolation_rules<
            ddc::type_seq_size_v<ddc::to_type_seq_t<typename TensorType::non_indices_domain_t>>>(
            rule);
}

} // namespace sil::exterior
