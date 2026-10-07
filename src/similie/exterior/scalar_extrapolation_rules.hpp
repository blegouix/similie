// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#pragma once

#include <cstddef>

#include <ddc/ddc.hpp>

#include <similie/misc/clamp_to_domain.hpp>
#include <similie/misc/domain_contains.hpp>

namespace sil::exterior {

/**
 * Use the nearest value of a cochain at an out-of-domain sample.
 * \important This operator and documentation are fully AI-generated.
 *
 * This is the historical closure of the primal coboundary in SimiLie. It is
 * explicit so callers can select another rule for a connected domain.
 */
struct ClampCochainExtrapolationRule
{
    template <class TensorType, class Element, class Component>
    KOKKOS_FUNCTION double operator()(
            TensorType tensor,
            Element sampled_element,
            Component component) const
    {
        return tensor
                .mem(sil::misc::clamp_to_domain(tensor.non_indices_domain(), sampled_element),
                     component);
    }
};

/**
 * Return zero for samples outside a cochain domain.
 * \important This operator and documentation are fully AI-generated.
 *
 * This is the historical closure used for the dual cochain in the scalar
 * codifferential. Interior samples retain their stored values.
 */
struct ZeroCochainExtrapolationRule
{
    template <class TensorType, class Element, class Component>
    KOKKOS_FUNCTION double operator()(
            TensorType tensor,
            Element sampled_element,
            Component component) const
    {
        if (!sil::misc::domain_contains(tensor.non_indices_domain(), sampled_element))
            return 0.0;
        return tensor.mem(sampled_element, component);
    }
};

/**
 * Extend a primal scalar cochain with a prescribed exterior potential.
 * Stored samples, including boundary nodes, retain their values. Strong nodal
 * Dirichlet constraints belong to the solver, independently of this sampler.
 * \important This operator and documentation is fully AI-generated.
 */
struct PrescribedScalarExtrapolationRule
{
    double value;

    template <class TensorType, class Element, class Component>
    KOKKOS_FUNCTION double operator()(
            TensorType tensor,
            Element sampled_element,
            Component component) const
    {
        if (!sil::misc::domain_contains(tensor.non_indices_domain(), sampled_element))
            return value;
        return tensor.mem(sampled_element, component);
    }
};

/**
 * Extend a dual flux cochain with a prescribed exterior flux sample.
 * The value is an oriented, integrated cochain value in the same units as the
 * stored dual field; physical normal flux densities must first be integrated
 * on the corresponding dual face. Use this rule at the dual derivative stage.
 * \important This operator and documentation is fully AI-generated.
 */
struct NormalScalarFluxExtrapolationRule
{
    double value = 0.0;

    template <class TensorType, class Element, class Component>
    KOKKOS_FUNCTION double operator()(
            TensorType tensor,
            Element sampled_element,
            Component component) const
    {
        return PrescribedScalarExtrapolationRule {value}(tensor, sampled_element, component);
    }
};

/**
 * Continue the nearest one-sided slope in each logical grid direction.
 * This primal closure preserves affine fields without prescribing a potential
 * or a normal flux. At corners the directional continuations are added; on
 * singleton dimensions the continuation is constant.
 * \important This operator and documentation is fully AI-generated.
 */
struct NaturalScalarExtrapolationRule
{
    template <class TensorType, class Element, class Component>
    KOKKOS_FUNCTION double operator()(
            TensorType tensor,
            Element sampled_element,
            Component component) const
    {
        if (sil::misc::domain_contains(tensor.non_indices_domain(), sampled_element))
            return tensor.mem(sampled_element, component);
        Element const front(tensor.non_indices_domain().front());
        Element const back(tensor.non_indices_domain().back());
        Element nearest = sampled_element;
        // Discrete elements store unsigned coordinates, but an exterior sample
        // below index zero represents a negative logical offset.
        auto const offset = sampled_element - front;
        auto const extent = back - front;
        for (std::size_t dim = 0; dim < ddc::type_seq_size_v<ddc::to_type_seq_t<Element>>; ++dim) {
            if (ddc::detail::array(offset)[dim] < 0)
                ddc::detail::array(nearest)[dim] = ddc::detail::array(front)[dim];
            else if (ddc::detail::array(offset)[dim] > ddc::detail::array(extent)[dim])
                ddc::detail::array(nearest)[dim] = ddc::detail::array(back)[dim];
        }
        double const boundary_value = tensor.mem(nearest, component);
        double result = boundary_value;
        for (std::size_t dim = 0; dim < ddc::type_seq_size_v<ddc::to_type_seq_t<Element>>; ++dim) {
            if (ddc::detail::array(front)[dim] == ddc::detail::array(back)[dim])
                continue;
            Element inner = nearest;
            double distance = 0.0;
            if (ddc::detail::array(offset)[dim] < 0) {
                ++ddc::detail::array(inner)[dim];
                distance = -static_cast<double>(ddc::detail::array(offset)[dim]);
            } else if (ddc::detail::array(offset)[dim] > ddc::detail::array(extent)[dim]) {
                --ddc::detail::array(inner)[dim];
                distance = static_cast<double>(
                        ddc::detail::array(offset)[dim] - ddc::detail::array(extent)[dim]);
            }
            if (distance != 0.0)
                result += distance * (boundary_value - tensor.mem(inner, component));
        }
        return result;
    }
};

/**
 * Sample a neighboring cochain through a caller-supplied trace map.
 * The map supplies both the neighboring grid element and the stored component,
 * so it can account for different origins, axis permutations and orientations.
 * The signed affine jump is added to exterior samples only. A primal potential
 * jump and a dual flux connection can therefore use different rule instances.
 * The neighbor and map must be accessible in the operator's execution space.
 * \important This operator and documentation is fully AI-generated.
 */
template <class NeighborTensor, class TraceMap>
struct ConnectedScalarExtrapolationRule
{
    NeighborTensor neighbor;
    TraceMap trace_map;
    double jump = 0.0;

    template <class TensorType, class Element, class Component>
    KOKKOS_FUNCTION double operator()(
            TensorType tensor,
            Element sampled_element,
            Component component) const
    {
        if (sil::misc::domain_contains(tensor.non_indices_domain(), sampled_element))
            return tensor.mem(sampled_element, component);
        return trace_map(neighbor, sampled_element, component) + jump;
    }
};

} // namespace sil::exterior
