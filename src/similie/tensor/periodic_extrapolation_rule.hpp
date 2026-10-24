// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#pragma once

#include <cassert>
#include <cstddef>

#include "scalar_extrapolation_rules.hpp"

namespace sil::tensor {

/** Continue a whole tensor periodically across a half-open logical domain.
 * Each axis wraps independently; the domain stores one period without a
 * duplicated endpoint. Geometry fields need their own, unwrapped continuation.
 * \important This operator and documentation is fully AI-generated.
 */
class PeriodicExtrapolationRule
{
    template <class Domain, class Element>
    KOKKOS_FUNCTION static typename Domain::discrete_element_type wrap(
            Domain const& domain,
            Element element)
    {
        misc::convert_type_seq_to_t<misc::SignedDiscreteElement, ddc::to_type_seq_t<Domain>> const
                sampled(element);
        typename Domain::discrete_element_type wrapped;
        for (std::size_t axis = 0; axis < Domain::rank(); ++axis) {
            std::ptrdiff_t const front = ddc::detail::array(domain.front())[axis];
            std::ptrdiff_t const count = ddc::detail::array(domain.extents())[axis];
            assert(count > 0);
            std::ptrdiff_t const index = ddc::detail::array(sampled)[axis] - front;
            ddc::detail::array(wrapped)[axis] = front + (index % count + count) % count;
        }
        return wrapped;
    }

public:
    template <class Domain, class Element>
    KOKKOS_FUNCTION auto value(Domain const& domain, Element element) const
    {
        auto result = detail::make_extrapolation_basis(domain, element);
        result
                .mem(detail::extrapolation_sample(domain, element),
                     ddc::DiscreteElement<detail::ExtrapolationSource>(
                             misc::domain_contains(domain, element) ? 0 : 1))
                = 1.0;
        return result;
    }

    template <class Field, class Element>
    KOKKOS_FUNCTION auto operator()(Field const& field, Element element) const
    {
        typename Field::non_indices_domain_t::discrete_element_type const wrapped
                = wrap(field.non_indices_domain(), element);
        return detail::pointwise_tensor<Field>(
                [&](auto component) { return field.mem(wrapped, component); });
    }
};

static_assert(ExtrapolationRuleConcept<
              PeriodicExtrapolationRule,
              detail::FieldConcept<ScalarIndex>,
              ddc::DiscreteElement<detail::GridConcept>>);

} // namespace sil::tensor
