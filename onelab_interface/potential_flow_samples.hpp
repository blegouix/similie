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

template <class ExecSpace, class Field>
void fill_potential_patch(
        ExecSpace const& exec,
        Field field,
        Kokkos::View<double*> potential,
        std::size_t ring_nodes)
{
    ddc::parallel_for_each(
            "fill_potential_patch",
            exec,
            field.non_indices_domain(),
            KOKKOS_LAMBDA(ddc::DiscreteElement<GridX, GridY> elem) {
                field.mem(elem, ddc::DiscreteElement<PotentialIndex>(0)) = potential(
                        elem.uid<GridY>() * ring_nodes + elem.uid<GridX>() % ring_nodes);
            });
}

template <class ExecSpace, class Field, class Gradient, class Rules>
void reconstruct_potential_patch(
        ExecSpace const& exec,
        Field field,
        Gradient gradient,
        Rules rules,
        std::size_t ring_nodes,
        std::size_t radial_nodes,
        Kokkos::View<double* [2]> differences,
        Kokkos::View<double* [4]> cells)
{
    sil::exterior::deriv<GradientIndex, PotentialIndex>(exec, gradient, field, rules);
    ddc::parallel_for_each(
            "reconstruct_potential_patch",
            exec,
            field.non_indices_domain(),
            KOKKOS_LAMBDA(ddc::DiscreteElement<GridX, GridY> elem) {
                std::size_t const index
                        = elem.uid<GridY>() * ring_nodes + elem.uid<GridX>() % ring_nodes;
                differences(index, 0)
                        = gradient(elem, gradient.accessor().template access_element<X>());
                differences(index, 1)
                        = gradient(elem, gradient.accessor().template access_element<Y>());
                if (elem.uid<GridY>() + 1 < radial_nodes) {
                    cells(index, 0) = rules(field, elem, ddc::DiscreteElement<PotentialIndex>(0));
                    cells(index, 1)
                            = rules(field,
                                    elem + ddc::DiscreteVector<GridX, GridY>(1, 0),
                                    ddc::DiscreteElement<PotentialIndex>(0));
                    cells(index, 2)
                            = rules(field,
                                    elem + ddc::DiscreteVector<GridX, GridY>(1, 1),
                                    ddc::DiscreteElement<PotentialIndex>(0));
                    cells(index, 3)
                            = rules(field,
                                    elem + ddc::DiscreteVector<GridX, GridY>(0, 1),
                                    ddc::DiscreteElement<PotentialIndex>(0));
                }
            });
}

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
    std::vector<ddc::Chunk<double, PotentialDomain, ddc::DeviceAllocator<double>>> allocations;
    std::vector<ddc::Chunk<
            double,
            ddc::DiscreteDomain<GridX, GridY, GradientIndex>,
            ddc::DeviceAllocator<double>>>
            gradients;
    allocations.reserve(PatchCount);
    gradients.reserve(PatchCount);
    for (std::size_t side = 0; side < PatchCount; ++side) {
        if (cells[side] == 0)
            throw std::runtime_error("empty potential-flow sample patch");
        GridDomain const
                grid(ddc::DiscreteElement<GridX, GridY>(starts[side], 0),
                     ddc::DiscreteVector<GridX, GridY>(cells[side], radial_nodes));
        allocations.emplace_back(
                PotentialDomain(grid, scalar_accessor.domain()),
                ddc::DeviceAllocator<double>());
        gradients.emplace_back(
                ddc::DiscreteDomain<GridX, GridY, GradientIndex>(grid, gradient_accessor.domain()),
                ddc::DeviceAllocator<double>());
    }
    std::vector<sil::tensor::Tensor<
            double,
            PotentialDomain,
            Kokkos::layout_right,
            Kokkos::DefaultExecutionSpace::memory_space>>
            fields;
    for (std::size_t side = 0; side < PatchCount; ++side)
        fields.emplace_back(allocations[side]);
    Kokkos::View<double*> values("sample_potential", potential.size());
    auto host_values = Kokkos::create_mirror_view(values);
    for (std::size_t i = 0; i < potential.size(); ++i)
        host_values(i) = potential[i];
    Kokkos::deep_copy(values, host_values);
    Kokkos::View<double* [2]> differences("potential_differences", potential.size());
    Kokkos::View<double* [4]> cell_values("potential_cell_values", potential.size());
    auto const domains = bind_domains(fields);
    sil::multidomains::DomainExecution<
            typename decltype(domains)::topology_type,
            Kokkos::DefaultExecutionSpace>
            partitions(Kokkos::DefaultExecutionSpace {});
    partitions.for_each([&]<class Node>(Kokkos::DefaultExecutionSpace const& stream) {
        static_assert(Node::id::INDEX < PatchCount);
        fill_potential_patch(stream, fields[Node::id::INDEX], values, ring_nodes);
    });
    partitions.fence(); // Every donor cochain must be ready before any derivative reads it.
    partitions.for_each([&]<class Node>(Kokkos::DefaultExecutionSpace const& stream) {
        reconstruct_potential_patch(
                stream,
                fields[Node::id::INDEX],
                sil::tensor::Tensor(gradients[Node::id::INDEX]),
                domains.template extrapolation_rules<typename Node::id>(circulation),
                ring_nodes,
                radial_nodes,
                differences,
                cell_values);
    });
    partitions.fence();
    auto const host_differences
            = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), differences);
    auto const host_cells = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), cell_values);
    PotentialFlowSamples samples;
    samples.differences.resize(potential.size());
    samples.cell_potentials.resize(potential.size());
    for (std::size_t i = 0; i < potential.size(); ++i) {
        samples.differences[i] = {host_differences(i, 0), host_differences(i, 1)};
        samples.cell_potentials[i]
                = {host_cells(i, 0), host_cells(i, 1), host_cells(i, 2), host_cells(i, 3)};
    }
    return samples;
}

} // namespace similie::onelab_interface::potential_flow_onelab
