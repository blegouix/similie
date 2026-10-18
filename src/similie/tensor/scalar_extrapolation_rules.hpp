// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#pragma once

#include <cstddef>

#include <ddc/ddc.hpp>

#include <similie/misc/clamp_to_domain.hpp>
#include <similie/misc/domain_contains.hpp>
#include <similie/misc/type_seq_conversion.hpp>

#include "extrapolation_basis.hpp"

namespace sil::tensor {

/** Spatial closure evaluated independently of the stored field values.
 * \important This operator and documentation is fully AI-generated.
 * operator() returns all stored components as a pointwise Tensor. value()
 * returns its spatial coefficients in an OwningTensor, depending only on the
 * domain and sample. The source axis separates stored samples from the rule
 * evaluation; prescribed data and donor mappings remain in the rule.
 */
struct ClampExtrapolationRule
{
    static constexpr bool STORED_SAMPLES_ONLY = true;
    template <class Domain, class Element>
    KOKKOS_FUNCTION auto value(Domain const& domain, Element element) const
    {
        auto result = detail::make_extrapolation_basis(domain, element);
        result
                .mem(detail::extrapolation_sample(domain, element),
                     ddc::DiscreteElement<detail::ExtrapolationSource>(0))
                = 1.0;
        return result;
    }
    template <class Field, class Element>
    KOKKOS_FUNCTION auto operator()(Field const& field, Element element) const
    {
        return detail::pointwise_tensor<Field>([&](auto component) {
            return field
                    .mem(detail::extrapolation_sample(field.non_indices_domain(), element),
                         component);
        });
    }
};

static_assert(ExtrapolationRuleConcept<
              ClampExtrapolationRule,
              detail::FieldConcept<ScalarIndex>,
              ddc::DiscreteElement<detail::GridConcept>>);

struct ZeroExtrapolationRule
{
    static constexpr bool STORED_SAMPLES_ONLY = true;
    template <class Domain, class Element>
    KOKKOS_FUNCTION auto value(Domain const& domain, Element element) const
    {
        auto result = detail::make_extrapolation_basis(domain, element);
        if (misc::domain_contains(domain, element))
            result
                    .mem(detail::extrapolation_sample(domain, element),
                         ddc::DiscreteElement<detail::ExtrapolationSource>(0))
                    = 1.0;
        return result;
    }
    template <class Field, class Element>
    KOKKOS_FUNCTION auto operator()(Field const& field, Element element) const
    {
        return detail::pointwise_tensor<Field>([&](auto component) {
            return misc::domain_contains(field.non_indices_domain(), element)
                           ? field
                                     .mem(detail::extrapolation_sample(
                                                  field.non_indices_domain(),
                                                  element),
                                          component)
                           : 0.0;
        });
    }
};

static_assert(ExtrapolationRuleConcept<
              ZeroExtrapolationRule,
              detail::FieldConcept<ScalarIndex>,
              ddc::DiscreteElement<detail::GridConcept>>);

struct PrescribedScalarExtrapolationRule
{
    static constexpr bool IS_DIRICHLET = true;
    double prescribed_value;
    template <class Domain, class Element>
    KOKKOS_FUNCTION auto value(Domain const& domain, Element element) const
    {
        auto result = detail::make_extrapolation_basis(domain, element);
        if (misc::domain_contains(domain, element))
            result
                    .mem(detail::extrapolation_sample(domain, element),
                         ddc::DiscreteElement<detail::ExtrapolationSource>(0))
                    = 1.0;
        else
            result
                    .mem(detail::extrapolation_sample(domain, element),
                         ddc::DiscreteElement<detail::ExtrapolationSource>(1))
                    = 1.0;
        return result;
    }
    template <class Field, class Element>
    KOKKOS_FUNCTION auto operator()(Field const& field, Element element) const
    {
        return detail::pointwise_tensor<Field>([&](auto component) {
            return misc::domain_contains(field.non_indices_domain(), element)
                           ? field
                                     .mem(detail::extrapolation_sample(
                                                  field.non_indices_domain(),
                                                  element),
                                          component)
                           : prescribed_value;
        });
    }
};

static_assert(ExtrapolationRuleConcept<
              PrescribedScalarExtrapolationRule,
              detail::FieldConcept<ScalarIndex>,
              ddc::DiscreteElement<detail::GridConcept>>);

struct ConstantScalarExtrapolationRule
{
    double exterior_value = 0.0;
    template <class Domain, class Element>
    KOKKOS_FUNCTION auto value(Domain const& domain, Element element) const
    {
        auto result = detail::make_extrapolation_basis(domain, element);
        if (misc::domain_contains(domain, element))
            result
                    .mem(detail::extrapolation_sample(domain, element),
                         ddc::DiscreteElement<detail::ExtrapolationSource>(0))
                    = 1.0;
        else
            result
                    .mem(detail::extrapolation_sample(domain, element),
                         ddc::DiscreteElement<detail::ExtrapolationSource>(1))
                    = 1.0;
        return result;
    }
    template <class Field, class Element>
    KOKKOS_FUNCTION auto operator()(Field const& field, Element element) const
    {
        return detail::pointwise_tensor<Field>([&](auto component) {
            return misc::domain_contains(field.non_indices_domain(), element)
                           ? field
                                     .mem(detail::extrapolation_sample(
                                                  field.non_indices_domain(),
                                                  element),
                                          component)
                           : exterior_value;
        });
    }
};

static_assert(ExtrapolationRuleConcept<
              ConstantScalarExtrapolationRule,
              detail::FieldConcept<ScalarIndex>,
              ddc::DiscreteElement<detail::GridConcept>>);

struct NaturalScalarExtrapolationRule
{
    static constexpr bool STORED_SAMPLES_ONLY = true;
    template <class Domain, class Element>
    KOKKOS_FUNCTION auto value(Domain const& domain, Element element) const
    {
        auto result = detail::make_extrapolation_basis(domain, element);
        misc::convert_type_seq_to_t<misc::SignedDiscreteElement, ddc::to_type_seq_t<Domain>> const
                sample(element),
                nearest(misc::clamp_to_domain(domain, sample)), front(domain.front()),
                back(domain.back());
        result.mem(nearest.storage_element(), ddc::DiscreteElement<detail::ExtrapolationSource>(0))
                = 1.0;
        for (std::size_t dim = 0; dim < ddc::type_seq_size_v<ddc::to_type_seq_t<Domain>>; ++dim) {
            std::ptrdiff_t const offset
                    = static_cast<std::ptrdiff_t>(ddc::detail::array(sample)[dim])
                      - static_cast<std::ptrdiff_t>(ddc::detail::array(nearest)[dim]);
            if (offset == 0 || ddc::detail::array(front)[dim] == ddc::detail::array(back)[dim])
                continue;
            misc::convert_type_seq_to_t<misc::SignedDiscreteElement, ddc::to_type_seq_t<Domain>>
                    inner(nearest);
            ddc::detail::array(inner)[dim] += offset < 0 ? 1 : -1;
            double const distance = offset < 0 ? -double(offset) : double(offset);
            result
                    .mem(nearest.storage_element(),
                         ddc::DiscreteElement<detail::ExtrapolationSource>(0))
                    += distance;
            result
                    .mem(inner.storage_element(),
                         ddc::DiscreteElement<detail::ExtrapolationSource>(0))
                    -= distance;
        }
        return result;
    }
    template <class Field, class Element>
    KOKKOS_FUNCTION auto operator()(Field const& field, Element element) const
    {
        auto const basis = value(field.non_indices_domain(), element);
        return detail::pointwise_tensor<Field>([&](auto component) {
            return detail::contract_extrapolation_basis(
                    basis,
                    [&](auto sample) { return field.mem(sample, component); },
                    [](std::size_t) { return 0.0; });
        });
    }
};

static_assert(ExtrapolationRuleConcept<
              NaturalScalarExtrapolationRule,
              detail::FieldConcept<ScalarIndex>,
              ddc::DiscreteElement<detail::GridConcept>>);

} // namespace sil::tensor
