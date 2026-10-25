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
 * sample. value(domain, element) returns spatial basis weights without reading
 * field values. This view performs no differentiation. Slices retain fixed
 * coordinates and the original policies, so donor mappings and corner weights
 * remain valid even when a slice fixes an exterior coordinate. Computed samples
 * are read-only values; the caller updates the stored field through its Tensor.
 * The view borrows tensor allocations through its field and rule handles;
 * their owners must outlive it and synchronize updates before sampling.
 */
template <class Field, class Rules, class Fixed = misc::SignedDiscreteElement<>>
struct TensorWithExtrapolationRules
{
    using discrete_domain_type = misc::convert_type_seq_to_t<
            ddc::DiscreteDomain,
            ddc::type_seq_remove_t<
                    ddc::to_type_seq_t<typename Field::discrete_domain_type>,
                    misc::to_type_seq_t<Fixed>>>;
    using discrete_element_type = typename discrete_domain_type::discrete_element_type;
    using accessor_t = tensor_accessor_for_domain_t<discrete_domain_type>;
    using indices_domain_t = typename accessor_t::discrete_domain_type;
    using non_indices_domain_t = misc::convert_type_seq_to_t<
            ddc::DiscreteDomain,
            ddc::type_seq_remove_t<
                    ddc::to_type_seq_t<discrete_domain_type>,
                    ddc::to_type_seq_t<indices_domain_t>>>;
    using natural_domain_t
            = misc::cartesian_prod_t<non_indices_domain_t, typename accessor_t::natural_domain_t>;
    using memory_space = typename Field::memory_space;
    using element_type = typename Field::element_type;
    Field field;
    Rules rules;
    Fixed fixed = {};

    KOKKOS_FUNCTION static constexpr accessor_t accessor()
    {
        return accessor_t();
    }
    KOKKOS_FUNCTION discrete_domain_type domain() const
    {
        return discrete_domain_type(field.domain());
    }
    KOKKOS_FUNCTION non_indices_domain_t non_indices_domain() const
    {
        return non_indices_domain_t(domain());
    }
    KOKKOS_FUNCTION indices_domain_t indices_domain() const
    {
        return indices_domain_t(domain());
    }
    KOKKOS_FUNCTION discrete_domain_type access_domain() const
    {
        return discrete_domain_type(non_indices_domain(), accessor().access_domain());
    }
    KOKKOS_FUNCTION natural_domain_t natural_domain() const
    {
        return natural_domain_t(non_indices_domain(), accessor().natural_domain());
    }
    template <class... Dimensions>
    KOKKOS_FUNCTION discrete_element_type access_element() const
    {
        return discrete_element_type(accessor_t::template access_element<Dimensions...>());
    }
    template <class... Elements>
    KOKKOS_FUNCTION discrete_element_type access_element(Elements... elements) const
    {
        return discrete_element_type(
                accessor_t::access_element(
                        typename accessor_t::natural_domain_t::discrete_element_type(elements...)),
                typename non_indices_domain_t::discrete_element_type(elements...));
    }
    template <class... Elements>
    KOKKOS_FUNCTION typename natural_domain_t::discrete_element_type canonical_natural_element(
            Elements... elements) const
    {
        return typename natural_domain_t::discrete_element_type(
                accessor_t::canonical_natural_element(
                        typename indices_domain_t::discrete_element_type(elements...)),
                typename non_indices_domain_t::discrete_element_type(elements...));
    }

    template <class Element>
    KOKKOS_FUNCTION auto operator[](Element elem) const
    {
        static_assert(ddc::type_seq_contains_v<
                      misc::to_type_seq_t<Element>,
                      ddc::to_type_seq_t<discrete_domain_type>>);
        auto const mapped = [&]<class... Dimensions>(ddc::TypeSeq<Dimensions...>) {
            return misc::SignedDiscreteElement<Dimensions...>([&] {
                if constexpr (TensorIndex<Dimensions>)
                    return static_cast<std::ptrdiff_t>(
                            detail::LambdaMemElem<Dimensions>::operator()(
                                    ddc::DiscreteElement<Dimensions>(
                                            elem.template uid<Dimensions>()))
                                    .uid());
                else
                    return static_cast<std::ptrdiff_t>(elem.template uid<Dimensions>());
            }()...);
        }(misc::to_type_seq_t<Element> {});
        using Selected
                = ddc::type_seq_merge_t<misc::to_type_seq_t<Fixed>, misc::to_type_seq_t<Element>>;
        misc::convert_type_seq_to_t<misc::SignedDiscreteElement, Selected> const
                selected(fixed, mapped);
        TensorWithExtrapolationRules<Field, Rules, std::remove_cvref_t<decltype(selected)>> const
                slice {field, rules, selected};
        if constexpr (ddc::type_seq_contains_v<
                              ddc::to_type_seq_t<typename Field::non_indices_domain_t>,
                              Selected>)
            return slice.point_at(selected);
        else
            return slice;
    }
    template <class... Elements>
    KOKKOS_FUNCTION double mem(Elements const&... elements) const
    {
        misc::convert_type_seq_to_t<
                misc::SignedDiscreteElement,
                ddc::to_type_seq_t<typename Field::non_indices_domain_t>> const
        elem(fixed, elements...);
        misc::convert_type_seq_to_t<
                misc::SignedDiscreteElement,
                ddc::to_type_seq_t<typename Field::indices_domain_t>> const
        component(fixed, elements...);
        return rules(field, elem).mem(component.storage_element());
    }
    template <class... Elements>
    KOKKOS_FUNCTION double operator()(Elements const&... elements) const
    {
        misc::convert_type_seq_to_t<
                misc::SignedDiscreteElement,
                ddc::to_type_seq_t<typename Field::non_indices_domain_t>> const
        elem(fixed, elements...);
        misc::convert_type_seq_to_t<
                misc::SignedDiscreteElement,
                ddc::to_type_seq_t<indices_domain_t>> const
        component(elements...);
        return point_at(elem)(component.storage_element());
    }
    template <class... Elements>
    KOKKOS_FUNCTION double get(Elements const&... elements) const
    {
        misc::convert_type_seq_to_t<
                misc::SignedDiscreteElement,
                ddc::to_type_seq_t<typename Field::non_indices_domain_t>> const
        elem(fixed, elements...);
        misc::convert_type_seq_to_t<
                misc::SignedDiscreteElement,
                ddc::to_type_seq_t<indices_domain_t>> const
        component(elements...);
        return point_at(elem).get(component.storage_element());
    }
    template <class Element>
    KOKKOS_FUNCTION auto value(typename Field::non_indices_domain_t domain, Element elem) const
    {
        return rules.value(domain, elem);
    }

private:
    template <class F, class R, class S>
    friend struct TensorWithExtrapolationRules;

    template <class Element>
    KOKKOS_FUNCTION auto point_at(Element elem) const
    {
        auto const point = rules(field, elem);
        using FixedIndices = misc::type_seq_intersect_t<
                misc::to_type_seq_t<Fixed>,
                ddc::to_type_seq_t<typename Field::indices_domain_t>>;
        if constexpr (ddc::type_seq_size_v<FixedIndices> == 0)
            return point;
        else {
            misc::convert_type_seq_to_t<misc::SignedDiscreteElement, FixedIndices> const indices(
                    fixed);
            return detail::pointwise_tensor<indices_domain_t, memory_space>([&](auto component) {
                return point.mem(
                        typename Field::indices_domain_t::
                                discrete_element_type(indices.storage_element(), component));
            });
        }
    }
};

template <class Field, class Rule>
KOKKOS_FUNCTION auto make_tensor_with_extrapolation_rules(Field const& field, Rule rule)
{
    if constexpr (misc::Specialization<Rule, ExtrapolationRules>)
        return TensorWithExtrapolationRules<Field, Rule> {field, rule};
    else
        return make_tensor_with_extrapolation_rules(field, make_extrapolation_rules(field, rule));
}

} // namespace sil::tensor
