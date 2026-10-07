// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#pragma once

#include <array>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include <similie/exterior/coboundary.hpp>
#include <similie/exterior/external_domain_extrapolation_rule.hpp>
#include <similie/exterior/extrapolation_rules.hpp>
#include <similie/multidomains/multidomains.hpp>

#include "potential_flow_grid.hpp"

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
template <std::size_t PatchCount, class BindDomains>
PotentialFlowSamples sample_potential_flow_field(
        std::vector<double> const& potential,
        double circulation,
        std::size_t ring_nodes,
        std::size_t radial_nodes,
        std::array<std::size_t, PatchCount> const& starts,
        std::array<std::size_t, PatchCount> const& cells,
        BindDomains bind_domains)
{
    if (potential.size() != ring_nodes * radial_nodes || radial_nodes < 2)
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
    std::vector<
            sil::tensor::Tensor<double, PotentialDomain, Kokkos::layout_right, Kokkos::HostSpace>>
            fields;
    for (std::size_t side = 0; side < PatchCount; ++side)
        fields.emplace_back(allocations[side]);
    auto const domains = bind_domains(fields);
    PotentialFlowSamples samples;
    samples.differences.resize(potential.size());
    samples.cell_potentials.resize(potential.size());
    domains.for_each_domain([&]<class Node>() {
        constexpr std::size_t side = Node::id::INDEX;
        static_assert(side < PatchCount);
        sil::tensor::Tensor field(allocations[side]);
        auto const rules = domains.template extrapolation_rules<typename Node::id>(circulation);
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
    });
    return samples;
}

} // namespace similie::onelab_interface::potential_flow_onelab
