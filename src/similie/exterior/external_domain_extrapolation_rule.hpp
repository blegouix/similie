// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#pragma once

#include <cassert>
#include <cstddef>
#include <type_traits>

#include <ddc/ddc.hpp>

#include "evaluators.hpp"
#include "scalar_extrapolation_rules.hpp"

namespace sil::exterior {

enum class BoundarySide { Lower, Upper };

/**
 * Map samples through an aligned interface, in any number of dimensions.
 * \important This operator and documentation is fully AI-generated.
 *
 * Dimension is the interface normal. The source and donor use the same grid
 * ordered tags and matching tangential extents, but may have different logical origins
 * and normal extents. Tangential offsets from the source origin are preserved.
 * source_side selects the boundary being left; target_side selects the boundary
 * being entered. Either lower or upper boundary can connect to either side.
 *
 * With shared_boundary=true, both domains store the interface node: one step
 * outside maps one step inside the donor. With false, adjacent domains have
 * disjoint stored nodes: one step outside maps to the donor boundary node.
 * map_boundary() identifies the coincident trace nodes in the shared case.
 * Spectator tensor indices pass through unchanged. Axis permutations, reversed
 * tangential coordinates, interpolation, and component frame changes are not
 * supported by this aligned map.
 */
template <class Dimension>
struct ExternalDomainBoundaryMap
{
    BoundarySide source_side;
    BoundarySide target_side;
    bool shared_boundary = true;

private:
    template <class DDim, class SourceDomain, class TargetDomain, class Element>
    KOKKOS_FUNCTION void map_coordinate(
            SourceDomain source,
            TargetDomain target,
            Element sampled,
            Element& mapped) const
    {
        if constexpr (std::is_same_v<DDim, Dimension>) {
            ddc::DiscreteElement<Dimension> const source_boundary(
                    source_side == BoundarySide::Lower ? source.front() : source.back());
            ddc::DiscreteElement<Dimension> const target_boundary(
                    target_side == BoundarySide::Lower ? target.front() : target.back());
            ddc::DiscreteVector<Dimension> const offset
                    = ddc::DiscreteElement<Dimension>(sampled) - source_boundary;
            // Signed differences also handle unsigned elements wrapping below zero.
            std::ptrdiff_t const distance = source_side == BoundarySide::Lower
                                                    ? -offset.template get<Dimension>()
                                                    : offset.template get<Dimension>();
            assert(distance >= 0);
            std::ptrdiff_t const inward = distance - (shared_boundary ? 0 : 1);
            assert(inward >= 0);
            mapped.template uid<Dimension>()
                    = (target_boundary
                       + ddc::DiscreteVector<Dimension>(
                               target_side == BoundarySide::Lower ? inward : -inward))
                              .template uid<Dimension>();
        } else {
            assert(source.template extent<DDim>() == target.template extent<DDim>());
            ddc::DiscreteVector<DDim> const offset = ddc::DiscreteElement<DDim>(sampled)
                                                     - ddc::DiscreteElement<DDim>(source.front());
            mapped.template uid<DDim>()
                    = (ddc::DiscreteElement<DDim>(target.front()) + offset).template uid<DDim>();
        }
    }

public:
    template <class... DDim, class TargetDomain, class Element>
    KOKKOS_FUNCTION Element
    operator()(ddc::DiscreteDomain<DDim...> source, TargetDomain target, Element sampled) const
    {
        static_assert((std::is_same_v<Dimension, DDim> || ...));
        static_assert(std::is_same_v<ddc::TypeSeq<DDim...>, ddc::to_type_seq_t<TargetDomain>>);
        Element mapped = sampled;
        (map_coordinate<DDim>(source, target, sampled, mapped), ...);
        return mapped;
    }

    template <class SourceDomain, class TargetDomain, class Element>
    KOKKOS_FUNCTION Element
    map_boundary(SourceDomain source, TargetDomain target, Element sampled) const
    {
        assert(shared_boundary);
        assert(sampled.template uid<Dimension>()
               == (source_side == BoundarySide::Lower ? source.front() : source.back())
                          .template uid<Dimension>());
        return (*this)(source, target, sampled);
    }
};

/**
 * Continue a cochain into a specified boundary of another tensor domain.
 * \important This operator and documentation is fully AI-generated.
 *
 * Install this rule on the corresponding side of ExtrapolationRules. Stored
 * samples retain their values. Exterior samples are mapped with boundary_map
 * and evaluated by target_rule in the donor. The default clamps samples that
 * also cross a tangential donor boundary; another rule can supply that closure.
 * ExtrapolationRules handles corners using its normalized-distance blend.
 *
 * orientation * donor_value + jump expresses an oriented component and an
 * affine potential jump. The reverse connection must use reciprocal orientation
 * and jump -jump/orientation. A scalar circulation cut normally uses orientation
 * 1 and opposite jumps. The tensor view and target rule must be accessible in
 * the execution space; the caller owns the donor allocation and synchronizes
 * donor updates before sampling. This sampler supplies ghost values; a solver
 * must still identify shared trace unknowns and enforce conservative flux balance.
 */
template <class Dimension, class NeighborTensor, class TargetRule = ClampCochainExtrapolationRule>
struct ExternalDomainExtrapolationRule
{
    static constexpr bool IS_INTERFACE = true;
    NeighborTensor neighbor;
    ExternalDomainBoundaryMap<Dimension> boundary_map;
    double jump = 0.0;
    double orientation = 1.0;
    TargetRule target_rule = {};

    template <class TensorType, class Element, class Component>
    KOKKOS_FUNCTION double operator()(
            TensorType const& tensor,
            Element sampled_element,
            Component component) const
    {
        if (misc::domain_contains(tensor.non_indices_domain(), sampled_element))
            return tensor.mem(sampled_element, component);
        Element const mapped = boundary_map(
                tensor.non_indices_domain(),
                neighbor.non_indices_domain(),
                sampled_element);
        return orientation * target_rule(neighbor, mapped, component) + jump;
    }

    /** Evaluate a stencil sampler after resolving the donor side and element. */
    template <class TensorType, class Sampler, class Element, class Component>
    KOKKOS_FUNCTION double value(
            TensorType const& tensor,
            Sampler sampler,
            Element sampled_element,
            Component component) const
    {
        if (misc::domain_contains(tensor.non_indices_domain(), sampled_element))
            return sampler(tensor, sampled_element, component);
        Element const mapped = boundary_map(
                tensor.non_indices_domain(),
                neighbor.non_indices_domain(),
                sampled_element);
        return orientation
                       * detail::EvaluateRuleValue<
                               Sampler> {sampler}(target_rule, neighbor, mapped, component)
               + jump;
    }
};

template <class NeighborTensor, class Dimension>
ExternalDomainExtrapolationRule(
        NeighborTensor,
        ExternalDomainBoundaryMap<Dimension>,
        double = 0.0,
        double = 1.0) -> ExternalDomainExtrapolationRule<Dimension, NeighborTensor>;

template <class NeighborTensor, class Dimension, class TargetRule>
ExternalDomainExtrapolationRule(
        NeighborTensor,
        ExternalDomainBoundaryMap<Dimension>,
        double,
        double,
        TargetRule) -> ExternalDomainExtrapolationRule<Dimension, NeighborTensor, TargetRule>;

} // namespace sil::exterior
