// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#pragma once

#include <cstddef>

#include <ddc/ddc.hpp>

#include <similie/misc/clamp_to_domain.hpp>
#include <similie/misc/domain_contains.hpp>
#include <similie/misc/type_seq_conversion.hpp>

#include "extrapolation_stencil.hpp"

namespace sil::tensor {

/** Spatial closure evaluated independently of the stored field values.
 * \important This operator and documentation is fully AI-generated.
 * operator() returns all stored components as a pointwise Tensor. value()
 * returns its affine spatial basis, depending only on the domain and sample.
 */
struct ClampExtrapolationRule
{
    template <class Domain, class Element>
    KOKKOS_FUNCTION ExtrapolationStencil<Domain> value(Domain domain, Element element) const
    {
        ExtrapolationStencil<Domain> result;
        result.add(domain, misc::clamp_to_domain(domain, element), 1.0);
        return result;
    }
    template <class Field, class Element>
    KOKKOS_FUNCTION auto operator()(Field const& field, Element element) const
    {
        ExtrapolationStencil<typename Field::non_indices_domain_t> const stencil
                = value(field.non_indices_domain(), element);
        return detail::pointwise_tensor<Field>([&](auto component) {
            return stencil.evaluate(
                    [&](auto const& term) { return field.mem(term.element, component); });
        });
    }
};

struct ZeroExtrapolationRule
{
    template <class Domain, class Element>
    KOKKOS_FUNCTION ExtrapolationStencil<Domain> value(Domain domain, Element element) const
    {
        ExtrapolationStencil<Domain> result;
        if (misc::domain_contains(domain, element))
            result.add(domain, element, 1.0);
        return result;
    }
    template <class Field, class Element>
    KOKKOS_FUNCTION auto operator()(Field const& field, Element element) const
    {
        ExtrapolationStencil<typename Field::non_indices_domain_t> const stencil
                = value(field.non_indices_domain(), element);
        return detail::pointwise_tensor<Field>([&](auto component) {
            return stencil.evaluate(
                    [&](auto const& term) { return field.mem(term.element, component); });
        });
    }
};

struct PrescribedScalarExtrapolationRule
{
    static constexpr bool IS_DIRICHLET = true;
    double prescribed_value;
    template <class Domain, class Element>
    KOKKOS_FUNCTION ExtrapolationStencil<Domain> value(Domain domain, Element element) const
    {
        ExtrapolationStencil<Domain> result;
        if (misc::domain_contains(domain, element))
            result.add(domain, element, 1.0);
        else
            result.constant = prescribed_value;
        return result;
    }
    template <class Field, class Element>
    KOKKOS_FUNCTION auto operator()(Field const& field, Element element) const
    {
        ExtrapolationStencil<typename Field::non_indices_domain_t> const stencil
                = value(field.non_indices_domain(), element);
        return detail::pointwise_tensor<Field>([&](auto component) {
            return stencil.evaluate(
                    [&](auto const& term) { return field.mem(term.element, component); });
        });
    }
};

struct ConstantScalarExtrapolationRule
{
    double exterior_value = 0.0;
    template <class Domain, class Element>
    KOKKOS_FUNCTION ExtrapolationStencil<Domain> value(Domain domain, Element element) const
    {
        ExtrapolationStencil<Domain> result;
        if (misc::domain_contains(domain, element))
            result.add(domain, element, 1.0);
        else
            result.constant = exterior_value;
        return result;
    }
    template <class Field, class Element>
    KOKKOS_FUNCTION auto operator()(Field const& field, Element element) const
    {
        ExtrapolationStencil<typename Field::non_indices_domain_t> const stencil
                = value(field.non_indices_domain(), element);
        return detail::pointwise_tensor<Field>([&](auto component) {
            return stencil.evaluate(
                    [&](auto const& term) { return field.mem(term.element, component); });
        });
    }
};

struct NaturalScalarExtrapolationRule
{
    template <class Domain, class Element>
    KOKKOS_FUNCTION ExtrapolationStencil<Domain> value(Domain domain, Element element) const
    {
        ExtrapolationStencil<Domain> result;
        misc::convert_type_seq_to_t<misc::SignedDiscreteElement, ddc::to_type_seq_t<Domain>> const
                sample(element),
                nearest(misc::clamp_to_domain(domain, sample)), front(domain.front()),
                back(domain.back());
        result.add(domain, nearest, 1.0);
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
            result.add(domain, nearest, distance);
            result.add(domain, inner, -distance);
        }
        return result;
    }
    template <class Field, class Element>
    KOKKOS_FUNCTION auto operator()(Field const& field, Element element) const
    {
        ExtrapolationStencil<typename Field::non_indices_domain_t> const stencil
                = value(field.non_indices_domain(), element);
        return detail::pointwise_tensor<Field>([&](auto component) {
            return stencil.evaluate(
                    [&](auto const& term) { return field.mem(term.element, component); });
        });
    }
};

} // namespace sil::tensor
