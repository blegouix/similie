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

#include "extrapolation_basis.hpp"
#include "scalar_extrapolation_rules.hpp"

namespace sil::tensor {

namespace detail {

template <class Result, class Contribution>
KOKKOS_FUNCTION void accumulate_extrapolation(
        Result& result,
        Contribution const& contribution,
        double weight,
        std::size_t source)
{
    if constexpr (ddc::type_seq_contains_v<
                          ddc::TypeSeq<ExtrapolationSource>,
                          ddc::to_type_seq_t<typename Result::discrete_domain_type>>) {
        using Spatial = misc::convert_type_seq_to_t<
                ddc::DiscreteDomain,
                ddc::type_seq_remove_t<
                        ddc::to_type_seq_t<typename Result::discrete_domain_type>,
                        ddc::TypeSeq<ExtrapolationSource>>>;
        ddc::device_for_each(contribution.domain(), [&](auto entry) {
            result
                    .mem(typename Spatial::discrete_element_type(entry),
                         ddc::DiscreteElement<ExtrapolationSource>(
                                 entry.template uid<ExtrapolationSource>() == 0 ? 0 : source))
                    += weight * contribution.mem(entry);
        });
    } else
        ddc::device_for_each(result.domain(), [&](auto component) {
            result.mem(component) += weight * contribution.mem(component);
        });
}

template <std::size_t Dimension, class... RulePairs>
class ExtrapolationRulesStorage;

template <std::size_t Dimension>
class ExtrapolationRulesStorage<Dimension>
{
public:
    KOKKOS_DEFAULTED_FUNCTION ExtrapolationRulesStorage() = default;

    template <class Domain, class Element, class Result, class Evaluate>
    KOKKOS_FUNCTION void accumulate(Domain, Element, Result&, double&, Evaluate) const
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

    template <class Domain, class Element, class Result, class Evaluate>
    KOKKOS_FUNCTION void accumulate(
            Domain domain,
            Element element,
            Result& result,
            double& total,
            Evaluate evaluate) const
    {
        using DDim = ddc::type_seq_element_t<Dimension, ddc::to_type_seq_t<Domain>>;
        std::ptrdiff_t const sampled = static_cast<std::ptrdiff_t>(element.template uid<DDim>());
        std::ptrdiff_t const front
                = static_cast<std::ptrdiff_t>(domain.front().template uid<DDim>());
        std::ptrdiff_t const back = static_cast<std::ptrdiff_t>(domain.back().template uid<DDim>());
        double const distance = sampled < front  ? double(front - sampled)
                                : sampled > back ? double(sampled - back)
                                                 : 0.0;
        if (distance != 0.0) {
            if (sampled < front)
                accumulate_extrapolation(
                        result,
                        evaluate(m_rules.first, element),
                        distance,
                        1 + 2 * Dimension);
            else
                accumulate_extrapolation(
                        result,
                        evaluate(m_rules.second, element),
                        distance,
                        2 + 2 * Dimension);
            total += distance;
        }
        m_other_rules.accumulate(domain, element, result, total, evaluate);
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
 * sum(distance_i * rule_i(tensor, element)) / sum(distance_i).
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
 * accessible in the caller's execution space. This class implements the same
 * (tensor, element) pointwise interface as individual rules.
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

    template <class Domain, class Element>
    KOKKOS_FUNCTION auto value(Domain domain, Element sampled_element) const
    {
        static_assert(sizeof...(RulePairs) == ddc::type_seq_size_v<ddc::to_type_seq_t<Domain>>);
        misc::convert_type_seq_to_t<misc::SignedDiscreteElement, ddc::to_type_seq_t<Domain>> const
                element(sampled_element);
        auto result
                = detail::make_extrapolation_basis<1 + 2 * sizeof...(RulePairs)>(domain, element);
        if (misc::domain_contains(domain, element)) {
            result
                    .mem(element.storage_element(),
                         ddc::DiscreteElement<detail::ExtrapolationSource>(0))
                    = 1.0;
            return result;
        }
        double total = 0.0;
        m_rules.accumulate(domain, element, result, total, [&](auto const& rule, auto sample) {
            return rule.value(domain, sample);
        });
        result *= 1.0 / total;
        return result;
    }

    template <class Field, class Element>
    KOKKOS_FUNCTION auto operator()(Field const& field, Element sampled_element) const
    {
        return evaluate(
                field,
                sampled_element,
                [&](auto const& rule, auto element) { return rule(field, element); },
                [&](auto element) {
                    return detail::pointwise_tensor<Field>([&](auto component) {
                        return field.mem(element.storage_element(), component);
                    });
                });
    }

private:
    template <class Field, class Element, class Evaluate, class Interior>
    KOKKOS_FUNCTION auto evaluate(
            Field const& field,
            Element sample,
            Evaluate evaluate,
            Interior interior) const
    {
        static_assert(
                sizeof...(RulePairs)
                == ddc::type_seq_size_v<ddc::to_type_seq_t<typename Field::non_indices_domain_t>>);
        misc::convert_type_seq_to_t<
                misc::SignedDiscreteElement,
                ddc::to_type_seq_t<typename Field::non_indices_domain_t>> const element(sample);
        if (misc::domain_contains(field.non_indices_domain(), element))
            return interior(element);
        auto result = detail::pointwise_tensor<Field>([](auto) { return 0.0; });
        double total = 0.0;
        m_rules.accumulate(field.non_indices_domain(), element, result, total, evaluate);
        result *= 1.0 / total;
        return result;
    }
};

static_assert(ExtrapolationRulesConcept<
              ExtrapolationRules<std::pair<ClampExtrapolationRule, ClampExtrapolationRule>>,
              detail::ConceptField<ScalarIndex>,
              ddc::DiscreteElement<detail::ConceptGrid>>);

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
 * accepts single rules. Consumers distinguish single rules from
 * ExtrapolationRules with if constexpr and only convert the former, with N
 * inferred from the tensor domain. An explicitly typed uniform
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

} // namespace sil::tensor
