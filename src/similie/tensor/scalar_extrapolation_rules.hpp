// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#pragma once

#include <cstddef>

#include <ddc/ddc.hpp>

#include <similie/misc/clamp_to_domain.hpp>
#include <similie/misc/domain_contains.hpp>
#include <similie/misc/type_seq_conversion.hpp>

#include "evaluators.hpp"

namespace sil::tensor {

/**
 * Use the nearest stored tensor component at an out-of-domain sample.
 * \important This operator and documentation are fully AI-generated.
 *
 * Clamp each non-index coordinate to the stored tensor domain.
 */
struct ClampExtrapolationRule
{
    template <class TensorType, class Element, class Component>
    KOKKOS_FUNCTION double operator()(
            TensorType const& tensor,
            Element sampled_element,
            Component component) const
    {
        return StoredTensorSampler {}(
                tensor,
                sil::misc::clamp_to_domain(tensor.non_indices_domain(), sampled_element),
                component);
    }

    template <class TensorType, class Sampler, class Element, class Component>
    KOKKOS_FUNCTION double value(
            TensorType const& tensor,
            Sampler sampler,
            Element sampled_element,
            Component component) const
    {
        return (*this)(
                SampledTensor<TensorType, Sampler> {tensor, sampler},
                sampled_element,
                component);
    }
};

/**
 * Return zero for samples outside a tensor domain.
 * \important This operator and documentation are fully AI-generated.
 *
 * Interior samples retain their stored values.
 */
struct ZeroExtrapolationRule
{
    template <class TensorType, class Element, class Component>
    KOKKOS_FUNCTION double operator()(
            TensorType const& tensor,
            Element sampled_element,
            Component component) const
    {
        if (!sil::misc::domain_contains(tensor.non_indices_domain(), sampled_element))
            return 0.0;
        return StoredTensorSampler {}(tensor, sampled_element, component);
    }

    template <class TensorType, class Sampler, class Element, class Component>
    KOKKOS_FUNCTION double value(
            TensorType const& tensor,
            Sampler sampler,
            Element sampled_element,
            Component component) const
    {
        return (*this)(
                SampledTensor<TensorType, Sampler> {tensor, sampler},
                sampled_element,
                component);
    }
};

/**
 * Extend scalar tensor values with a prescribed exterior value.
 * Stored samples, including boundary samples, retain their values. The
 * Dirichlet marker lets consumers distinguish prescribed boundary data;
 * imposing constraints on stored values is independent of this sampler.
 * \important This operator and documentation is fully AI-generated.
 */
struct PrescribedScalarExtrapolationRule
{
    static constexpr bool IS_DIRICHLET = true;
    double prescribed_value;

    template <class TensorType, class Element, class Component>
    KOKKOS_FUNCTION double operator()(
            TensorType const& tensor,
            Element sampled_element,
            Component component) const
    {
        if (!sil::misc::domain_contains(tensor.non_indices_domain(), sampled_element))
            return prescribed_value;
        return StoredTensorSampler {}(tensor, sampled_element, component);
    }

    template <class TensorType, class Sampler, class Element, class Component>
    KOKKOS_FUNCTION double value(
            TensorType const& tensor,
            Sampler sampler,
            Element sampled_element,
            Component component) const
    {
        return (*this)(
                SampledTensor<TensorType, Sampler> {tensor, sampler},
                sampled_element,
                component);
    }
};

/**
 * Extend scalar tensor values with a constant exterior value.
 * Interior samples retain their stored values. Unlike the prescribed rule,
 * this rule carries no Dirichlet constraint marker.
 * \important This operator and documentation is fully AI-generated.
 */
struct ConstantScalarExtrapolationRule
{
    double exterior_value = 0.0;

    template <class TensorType, class Element, class Component>
    KOKKOS_FUNCTION double operator()(
            TensorType const& tensor,
            Element sampled_element,
            Component component) const
    {
        return PrescribedScalarExtrapolationRule {
                exterior_value}(tensor, sampled_element, component);
    }

    template <class TensorType, class Sampler, class Element, class Component>
    KOKKOS_FUNCTION double value(
            TensorType const& tensor,
            Sampler sampler,
            Element sampled_element,
            Component component) const
    {
        return (*this)(
                SampledTensor<TensorType, Sampler> {tensor, sampler},
                sampled_element,
                component);
    }
};

/**
 * Continue the nearest one-sided slope in each logical grid direction.
 * This continuation preserves affine fields. At corners the directional
 * continuations are added; singleton dimensions use a constant continuation.
 * \important This operator and documentation is fully AI-generated.
 */
struct NaturalScalarExtrapolationRule
{
    template <class TensorType, class Element, class Component>
    KOKKOS_FUNCTION double operator()(
            TensorType const& tensor,
            Element sampled_element,
            Component component) const
    {
        if (sil::misc::domain_contains(tensor.non_indices_domain(), sampled_element))
            return StoredTensorSampler {}(tensor, sampled_element, component);
        Element const front(tensor.non_indices_domain().front());
        Element const back(tensor.non_indices_domain().back());
        Element nearest = sampled_element;
        // Signed coordinates and legacy wrapped DDC coordinates both produce
        // signed logical offsets from the stored origin.
        auto const offset = sampled_element - front;
        auto const extent = back - front;
        for (std::size_t dim = 0; dim < ddc::type_seq_size_v<misc::to_type_seq_t<Element>>; ++dim) {
            if (ddc::detail::array(offset)[dim] < 0)
                ddc::detail::array(nearest)[dim] = ddc::detail::array(front)[dim];
            else if (ddc::detail::array(offset)[dim] > ddc::detail::array(extent)[dim])
                ddc::detail::array(nearest)[dim] = ddc::detail::array(back)[dim];
        }
        double const boundary_value = StoredTensorSampler {}(tensor, nearest, component);
        double result = boundary_value;
        for (std::size_t dim = 0; dim < ddc::type_seq_size_v<misc::to_type_seq_t<Element>>; ++dim) {
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
                result += distance
                          * (boundary_value - StoredTensorSampler {}(tensor, inner, component));
        }
        return result;
    }

    template <class TensorType, class Sampler, class Element, class Component>
    KOKKOS_FUNCTION double value(
            TensorType const& tensor,
            Sampler sampler,
            Element sampled_element,
            Component component) const
    {
        return (*this)(
                SampledTensor<TensorType, Sampler> {tensor, sampler},
                sampled_element,
                component);
    }
};

} // namespace sil::tensor
