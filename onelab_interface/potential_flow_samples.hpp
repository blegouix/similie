// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#pragma once

#include <array>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include <similie/exterior/external_domain_extrapolation_rule.hpp>
#include <similie/exterior/extrapolation_rules.hpp>

#include "potential_flow_system.hpp"
#include "potential_flow_tensor_laplacian.hpp"

namespace similie::onelab_interface::potential_flow_onelab {

struct PotentialFlowSamples
{
    std::vector<std::array<double, 2>> differences;
    std::vector<std::array<double, 4>> cell_potentials;
};

/**
 * Reconstruct cell potentials and dphi with the ordinary exterior derivative.
 * \important This operator and documentation is fully AI-generated.
 *
 * Each patch owns its left nodes and samples its right trace in its neighbor.
 * Disjoint ownership makes the last forward difference cross the interface
 * through ExternalDomainExtrapolationRule. The same rule carries the signed
 * circulation jump, so velocity and exported potential use the same branch.
 * The radial boundary nodes have already been constrained by the solver;
 * one-sided primal continuation supplies the unused outer forward difference.
 * Physical normal flux constraints belong to the assembled nodal balances.
 */
template <std::size_t PatchCount>
PotentialFlowSamples sample_potential_flow_field(
        std::vector<double> const& potential,
        double circulation,
        std::size_t ring_nodes,
        std::size_t radial_nodes,
        std::array<std::size_t, PatchCount> const& starts,
        std::array<std::size_t, PatchCount> const& cells,
        std::vector<TraceConnection> const& connections)
{
    if (potential.size() != ring_nodes * radial_nodes || radial_nodes < 2
        || connections.size() != PatchCount)
        throw std::runtime_error("invalid potential-flow sample domains");
    [[maybe_unused]] sil::tensor::TensorAccessor<PotentialIndex> scalar_accessor;
    [[maybe_unused]] sil::tensor::TensorAccessor<GradientIndex> gradient_accessor;
    std::vector<ddc::Chunk<double, PotentialDomain, ddc::HostAllocator<double>>> allocations;
    allocations.reserve(PatchCount);
    for (std::size_t side = 0; side < PatchCount; ++side) {
        if (cells[side] == 0)
            throw std::runtime_error("empty potential-flow sample patch");
        GridDomain const
                grid(ddc::DiscreteElement<GridX, GridY>(starts[side], 0),
                     ddc::DiscreteVector<GridX, GridY>(cells[side], radial_nodes));
        allocations.emplace_back(
                PotentialDomain(grid, scalar_accessor.domain()),
                ddc::HostAllocator<double>());
        sil::tensor::Tensor field(allocations.back());
        ddc::host_for_each(grid, [&](ddc::DiscreteElement<GridX, GridY> elem) {
            field.mem(elem, ddc::DiscreteElement<PotentialIndex>(0))
                    = potential[elem.uid<GridY>() * ring_nodes + elem.uid<GridX>() % ring_nodes];
        });
    }
    std::array<std::size_t, PatchCount> lower {}, upper {};
    std::array<double, PatchCount> lower_jump {}, upper_jump {};
    std::array<bool, PatchCount> lower_connected {}, upper_connected {};
    for (TraceConnection const& connection : connections) {
        if (connection.first_domain >= PatchCount || connection.second_domain >= PatchCount
            || connection.first_side != TraceSide::UpperX
            || connection.second_side != TraceSide::LowerX
            || upper_connected[connection.first_domain]
            || lower_connected[connection.second_domain])
            throw std::runtime_error("invalid potential-flow sample connection");
        upper[connection.first_domain] = connection.second_domain;
        lower[connection.second_domain] = connection.first_domain;
        upper_jump[connection.first_domain] = circulation * connection.jump_coefficient;
        lower_jump[connection.second_domain] = -circulation * connection.jump_coefficient;
        upper_connected[connection.first_domain] = true;
        lower_connected[connection.second_domain] = true;
    }
    PotentialFlowSamples samples;
    samples.differences.resize(potential.size());
    samples.cell_potentials.resize(potential.size());
    for (std::size_t side = 0; side < PatchCount; ++side) {
        if (!lower_connected[side] || !upper_connected[side])
            throw std::runtime_error("unconnected potential-flow sample trace");
        sil::tensor::Tensor field(allocations[side]);
        sil::exterior::ExtrapolationRules const
                rules(std::
                              pair {sil::exterior::ExternalDomainExtrapolationRule {
                                            sil::tensor::Tensor(allocations[lower[side]]),
                                            sil::exterior::ExternalDomainBoundaryMap<GridX> {
                                                    sil::exterior::BoundarySide::Lower,
                                                    sil::exterior::BoundarySide::Upper,
                                                    false},
                                            lower_jump[side]},
                                    sil::exterior::ExternalDomainExtrapolationRule {
                                            sil::tensor::Tensor(allocations[upper[side]]),
                                            sil::exterior::ExternalDomainBoundaryMap<GridX> {
                                                    sil::exterior::BoundarySide::Upper,
                                                    sil::exterior::BoundarySide::Lower,
                                                    false},
                                            upper_jump[side]}},
                      std::
                              pair {sil::exterior::NaturalScalarExtrapolationRule {},
                                    sil::exterior::NaturalScalarExtrapolationRule {}});
        ddc::Chunk gradient_allocation(
                ddc::DiscreteDomain<
                        GridX,
                        GridY,
                        GradientIndex>(field.non_indices_domain(), gradient_accessor.domain()),
                ddc::HostAllocator<double>());
        sil::tensor::Tensor gradient(gradient_allocation);
        sil::exterior::deriv<
                GradientIndex,
                PotentialIndex>(Kokkos::DefaultHostExecutionSpace(), gradient, field, rules);
        Kokkos::DefaultHostExecutionSpace().fence();
        ddc::host_for_each(
                field.non_indices_domain(),
                [&](ddc::DiscreteElement<GridX, GridY> elem) {
                    std::size_t const index
                            = elem.uid<GridY>() * ring_nodes + elem.uid<GridX>() % ring_nodes;
                    samples.differences[index]
                            = {gradient(elem, gradient_accessor.template access_element<X>()),
                               gradient(elem, gradient_accessor.template access_element<Y>())};
                    if (elem.uid<GridY>() + 1 < radial_nodes)
                        samples.cell_potentials[index]
                                = {rules(field, elem, ddc::DiscreteElement<PotentialIndex>(0)),
                                   rules(field,
                                         elem + ddc::DiscreteVector<GridX, GridY>(1, 0),
                                         ddc::DiscreteElement<PotentialIndex>(0)),
                                   rules(field,
                                         elem + ddc::DiscreteVector<GridX, GridY>(1, 1),
                                         ddc::DiscreteElement<PotentialIndex>(0)),
                                   rules(field,
                                         elem + ddc::DiscreteVector<GridX, GridY>(0, 1),
                                         ddc::DiscreteElement<PotentialIndex>(0))};
                });
    }
    return samples;
}

} // namespace similie::onelab_interface::potential_flow_onelab
