// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include <ddc/ddc.hpp>

#include <similie/physics/hamilton_equations.hpp>
#include <similie/physics/scalar_field/scalar_field_with_power_coupling.hpp>
#include <similie/solvers/affine_scalar_system.hpp>
#include <similie/solvers/sparse_linear_operator.hpp>

#include "gmsh_structured_grid.hpp"
#include "potential_flow_connections.hpp"
#include "potential_flow_grid.hpp"
#include "potential_flow_samples.hpp"

namespace similie::onelab_interface::potential_flow_onelab {

struct Inputs
{
    double velocity = 100.0 / 3.6;
    double box_size = 7.0;
    double incidence = -10.0 * 3.14159265358979323846 / 180.0;
    double circulation = -10.0;
    double mass_flow_rate = -100.0;
    double density = 1.225;
    bool impose_circulation = false;
    bool airfoil = true;
};

struct Result
{
    double circulation = 0.0;
    double mass_flow_rate = 0.0;
    double lift_kutta_joukowski = 0.0;
    double max_speed = 0.0;
    std::size_t node_count = 0;
    std::size_t cell_count = 0;
    solvers::StrongFormulationSolverDiagnostics solver_diagnostics;
};

struct X
{
};
struct Y
{
};
struct T
{
};
using FreeScalarFieldHamiltonian
        = physics::scalar_field::ScalarFieldWithPowerCouplingHamiltonian<T, X, Y>;

inline Result run(
        std::filesystem::path const& mesh_file,
        std::filesystem::path const& output_file,
        Inputs const& inputs,
        solvers::StrongFormulationSolverSettings const& settings)
{
    auto const parsed = sil::onelab_interface::gmsh::parse_supported_msh2_mesh(mesh_file);
    if (!std::holds_alternative<sil::onelab_interface::gmsh::QuadrilateralMesh>(parsed))
        throw std::runtime_error("potential flow requires structured quadrilaterals");
    auto const& mesh = std::get<sil::onelab_interface::gmsh::QuadrilateralMesh>(parsed);
    constexpr std::size_t patch_count = 4;
    std::array<std::vector<sil::onelab_interface::gmsh::QuadrilateralCell const*>, patch_count>
            patches;
    for (auto const& cell : mesh.cells) {
        if ((cell.physical_tag != 2 && cell.physical_tag != 3) || cell.elementary_tag < 24
            || cell.elementary_tag > 27)
            throw std::runtime_error("potential-flow mesh contains a non-fluid quad");
        patches[cell.elementary_tag - 24].push_back(&cell);
    }
    std::size_t cells_along_outer_boundary = 0;
    for (auto const& edge : mesh.boundary_edges)
        if (edge.physical_tag == 10)
            ++cells_along_outer_boundary;
    if (cells_along_outer_boundary == 0 || patches[0].size() % cells_along_outer_boundary != 0)
        throw std::runtime_error("potential-flow mesh does not match the mapped quad topology");
    std::size_t const cells_per_patch_i = cells_along_outer_boundary;
    std::size_t const cells_per_patch_j = patches[0].size() / cells_per_patch_i;
    for (auto const& patch : patches)
        if (patch.size() != cells_per_patch_i * cells_per_patch_j)
            throw std::runtime_error("potential-flow quad patch has unexpected dimensions");
    std::size_t const ni = patch_count * cells_per_patch_i;
    std::size_t const nodes_j = cells_per_patch_j + 1;
    std::size_t const node_count = ni * nodes_j;
    if (mesh.cells.size() != ni * cells_per_patch_j || mesh.nodes.size() != node_count)
        throw std::runtime_error("potential-flow mesh does not match the mapped quad topology");
    std::map<std::size_t, std::size_t> logical_index;
    auto bind_node = [&](std::size_t tag, std::size_t index) {
        auto const [it, inserted] = logical_index.emplace(tag, index);
        if (!inserted && it->second != index)
            throw std::runtime_error("quad patches do not share a consistent structured grid");
    };
    for (std::size_t patch = 0; patch < patch_count; ++patch) {
        for (std::size_t a = 0; a < cells_per_patch_i; ++a)
            for (std::size_t j = 0; j < cells_per_patch_j; ++j) {
                auto const& cell = *patches[patch][a * cells_per_patch_j + j];
                std::size_t const i = patch * cells_per_patch_i + a;
                bind_node(cell.node_tags[0], j * ni + i);
                bind_node(cell.node_tags[1], j * ni + (i + 1) % ni);
                bind_node(cell.node_tags[2], (j + 1) * ni + (i + 1) % ni);
                bind_node(cell.node_tags[3], (j + 1) * ni + i);
            }
    }
    if (logical_index.size() != node_count)
        throw std::runtime_error("potential-flow mapped grid has missing nodes");
    std::vector<sil::onelab_interface::gmsh::MeshNode> nodes(node_count);
    for (auto const& node : mesh.nodes)
        nodes[logical_index.at(node.tag)] = node;
    auto index = [&](std::size_t i, std::size_t j) { return j * ni + i % ni; };
    auto point = [&](std::size_t i, std::size_t j) {
        auto const& n = nodes[index(i, j)];
        return std::array<double, 2> {n.x, n.y};
    };
    std::vector<int> boundary(node_count, 0);
    std::vector<double> prescribed(node_count, 0.0);
    std::vector<int> outer_boundary(ni, 0);
    bool has_cut = false;
    for (auto const& edge : mesh.boundary_edges) {
        has_cut |= edge.physical_tag == 13;
        if (edge.physical_tag != 10 && edge.physical_tag != 11)
            continue;
        std::size_t const first = logical_index.at(edge.node_tags[0]);
        std::size_t const second = logical_index.at(edge.node_tags[1]);
        if (first / ni == nodes_j - 1 && second / ni == nodes_j - 1) {
            std::size_t const begin = (first % ni + 1) % ni == second % ni ? first : second;
            outer_boundary[begin % ni] = edge.physical_tag;
        }
        for (std::size_t const tag : edge.node_tags) {
            std::size_t const id = logical_index.at(tag);
            if (boundary[id] && boundary[id] != edge.physical_tag)
                throw std::runtime_error("upstream and downstream boundaries overlap");
            boundary[id] = edge.physical_tag;
            prescribed[id] = edge.physical_tag == 11 ? inputs.velocity * inputs.box_size : 0.0;
        }
    }
    if (!has_cut || std::find(boundary.begin(), boundary.end(), 10) == boundary.end()
        || std::find(boundary.begin(), boundary.end(), 11) == boundary.end())
        throw std::runtime_error("potential-flow physical boundary is incomplete");
    // Each side has three tensor domains with a uniform outer boundary rule:
    // zero flux, prescribed potential, and zero flux. The domains at the
    // circulation cut share trace nodes with an affine potential jump.
    if (cells_per_patch_i % 2 != 0)
        throw std::runtime_error("potential-flow mesh has an odd number of cells along a patch");
    std::size_t const half_patch_cells = cells_per_patch_i / 2;
    std::size_t const cut_cell_i = 3 * cells_per_patch_i - 1;
    std::size_t const cut_node_i = cut_cell_i + 1;
    std::array<std::size_t, 6> const side_start {
            cells_per_patch_i,
            cells_per_patch_i + half_patch_cells,
            2 * cells_per_patch_i + half_patch_cells,
            3 * cells_per_patch_i,
            3 * cells_per_patch_i + half_patch_cells,
            half_patch_cells};
    std::array<std::size_t, 6> const side_cells {
            half_patch_cells,
            cells_per_patch_i,
            half_patch_cells,
            half_patch_cells,
            cells_per_patch_i,
            half_patch_cells};
    std::vector<TraceConnection> const connections {
            {0, TraceSide::UpperX, 1, TraceSide::LowerX, 0.0},
            {1, TraceSide::UpperX, 2, TraceSide::LowerX, 0.0},
            {2, TraceSide::UpperX, 3, TraceSide::LowerX, -1.0},
            {3, TraceSide::UpperX, 4, TraceSide::LowerX, 0.0},
            {4, TraceSide::UpperX, 5, TraceSide::LowerX, 0.0},
            {5, TraceSide::UpperX, 0, TraceSide::LowerX, 0.0},
    };
    [[maybe_unused]] sil::tensor::TensorAccessor<PositionIndex> position_accessor;
    std::vector<ddc::Chunk<double, PositionDomain, ddc::HostAllocator<double>>>
            position_allocations;
    std::vector<std::vector<std::size_t>> global_indices(side_start.size());
    position_allocations.reserve(side_start.size());
    for (std::size_t side = 0; side < side_start.size(); ++side) {
        GridDomain const
                grid(ddc::DiscreteElement<GridX, GridY>(side_start[side], 0),
                     ddc::DiscreteVector<GridX, GridY>(side_cells[side], nodes_j));
        position_allocations.emplace_back(
                PositionDomain(grid, position_accessor.domain()),
                ddc::HostAllocator<double>());
        sil::tensor::Tensor position(position_allocations.back());
        global_indices[side].reserve(grid.size());
        ddc::host_for_each(grid, [&](ddc::DiscreteElement<GridX, GridY> elem) {
            std::array<double, 2> const coordinates = point(elem.uid<GridX>(), elem.uid<GridY>());
            position(elem, position_accessor.access_element<X>()) = coordinates[0];
            position(elem, position_accessor.access_element<Y>()) = coordinates[1];
            global_indices[side].push_back(index(elem.uid<GridX>(), elem.uid<GridY>()));
        });
    }
    // The example supplies geometry, global numbering, and boundary policies.
    // All cell equations, affine interface contributions and nodal constraints
    // are assembled by the library through those policies' operator() calls.
    auto assemble = [&](double jump_scale, double prescribed_scale) {
        solvers::AffineScalarSystem system(node_count);
        std::vector<solvers::IndexedScalarField<GridX, GridY>> fields;
        for (std::size_t side = 0; side < side_start.size(); ++side)
            fields.push_back(
                    {sil::tensor::Tensor(position_allocations[side]).non_indices_domain(),
                     global_indices[side].data(),
                     system.probe_state()});
        for (std::size_t side = 0; side < fields.size(); ++side) {
            std::size_t const lower = (side + fields.size() - 1) % fields.size();
            std::size_t const upper = (side + 1) % fields.size();
            sil::exterior::ExternalDomainBoundaryMap<GridX> const lower_map {
                    sil::exterior::BoundarySide::Lower,
                    sil::exterior::BoundarySide::Upper,
                    false};
            sil::exterior::ExternalDomainBoundaryMap<GridX> const upper_map {
                    sil::exterior::BoundarySide::Upper,
                    sil::exterior::BoundarySide::Lower,
                    false};
            sil::tensor::Tensor position(position_allocations[side]);
            sil::exterior::ExtrapolationRules const geometry_rules(
                    std::
                            pair {sil::exterior::ExternalDomainExtrapolationRule {
                                          sil::tensor::Tensor(position_allocations[lower]),
                                          lower_map},
                                  sil::exterior::ExternalDomainExtrapolationRule {
                                          sil::tensor::Tensor(position_allocations[upper]),
                                          upper_map}},
                    std::
                            pair {sil::exterior::ClampCochainExtrapolationRule {},
                                  sil::exterior::ClampCochainExtrapolationRule {}});
            auto append_patch = [&](auto upper_rule) {
                sil::exterior::ExtrapolationRules const primal_rules(
                        std::
                                pair {sil::exterior::ExternalDomainExtrapolationRule {
                                              fields[lower],
                                              lower_map,
                                              -jump_scale * connections[lower].jump_coefficient},
                                      sil::exterior::ExternalDomainExtrapolationRule {
                                              fields[upper],
                                              upper_map,
                                              jump_scale * connections[side].jump_coefficient}},
                        std::pair {sil::exterior::NaturalScalarExtrapolationRule {}, upper_rule});
                solvers::assemble_integrated_laplacian(
                        system,
                        fields[side].grid.remove_last(ddc::DiscreteVector<GridX, GridY>(0, 1)),
                        fields[side],
                        position,
                        primal_rules,
                        geometry_rules,
                        sil::exterior::NormalScalarFluxExtrapolationRule {},
                        inputs.density);
            };
            int const outer_tag = outer_boundary[side_start[side] % ni];
            for (std::size_t i = 0; i < side_cells[side]; ++i)
                if (outer_boundary[(side_start[side] + i) % ni] != outer_tag)
                    throw std::runtime_error("outer boundary rule is not uniform on a patch");
            if (outer_tag == 10 || outer_tag == 11)
                append_patch(
                        sil::exterior::PrescribedScalarExtrapolationRule {
                                prescribed_scale
                                * prescribed[index(side_start[side] + 1, nodes_j - 1)]});
            else
                append_patch(sil::exterior::NaturalScalarExtrapolationRule {});
        }
        system.finalize();
        return system;
    };
    solvers::AffineScalarSystem const system = assemble(0.0, 1.0);
    std::vector<double> const jump_rhs = [&]() {
        solvers::AffineScalarSystem response = assemble(1.0, 0.0);
        return std::move(response.rhs);
    }();
    solvers::SparseLinearOperator const matrix(system.rows);
    Result result;
    result.node_count = node_count;
    result.cell_count = mesh.cells.size();
    std::vector<double> const base_ring
            = solvers::solve_linear_system(matrix, system.rhs, settings, result.solver_diagnostics);
    solvers::StrongFormulationSolverDiagnostics cut_diagnostics;
    std::vector<double> const response_ring
            = solvers::solve_linear_system(matrix, jump_rhs, settings, cut_diagnostics);
    FreeScalarFieldHamiltonian const hamiltonian(0.0, 0.0, 2.0);
    physics::HamiltonEquations const equations(hamiltonian);
    PotentialFlowSamples const base_samples = sample_potential_flow_field(
            base_ring,
            0.0,
            ni,
            nodes_j,
            side_start,
            side_cells,
            connections);
    PotentialFlowSamples const response_samples = sample_potential_flow_field(
            response_ring,
            1.0,
            ni,
            nodes_j,
            side_start,
            side_cells,
            connections);
    auto velocity = [&](PotentialFlowSamples const& samples,
                        std::size_t i,
                        std::size_t j,
                        double xi,
                        double eta) {
        auto const a = point(i, j), b = point(i + 1, j);
        auto const c = point(i + 1, j + 1), d = point(i, j + 1);
        double const tx = (1.0 - eta) * (b[0] - a[0]) + eta * (c[0] - d[0]);
        double const ty = (1.0 - eta) * (b[1] - a[1]) + eta * (c[1] - d[1]);
        double const rx = (1.0 - xi) * (d[0] - a[0]) + xi * (c[0] - b[0]);
        double const ry = (1.0 - xi) * (d[1] - a[1]) + xi * (c[1] - b[1]);
        double const dt = (1.0 - eta) * samples.differences[index(i, j)][0]
                          + eta * samples.differences[index(i, j + 1)][0];
        double const dr = (1.0 - xi) * samples.differences[index(i, j)][1]
                          + xi * samples.differences[index(i + 1, j)][1];
        double const determinant = tx * ry - ty * rx;
        if (std::abs(determinant) < 1.0e-16)
            throw std::runtime_error("degenerate quad metric");
        double const gradient_x = (dt * ry - ty * dr) / determinant;
        double const gradient_y = (tx * dr - dt * rx) / determinant;
        return std::array<
                double,
                2> {equations.dpotential_dt<X>(gradient_x), equations.dpotential_dt<Y>(gradient_y)};
    };
    auto mass_flow = [&](PotentialFlowSamples const& samples) {
        double flux = 0.0;
        for (std::size_t j = 0; j < cells_per_patch_j; ++j) {
            auto const v = velocity(samples, cut_cell_i, j, 1.0, 0.5);
            auto const p0 = point(cut_node_i, j), p1 = point(cut_node_i, j + 1);
            flux -= inputs.density * (v[0] * (p1[1] - p0[1]) - v[1] * (p1[0] - p0[0]));
        }
        return flux;
    };
    double const flow0 = mass_flow(base_samples);
    double const flow1 = mass_flow(response_samples);
    if (inputs.impose_circulation)
        result.circulation = inputs.circulation;
    else if (!inputs.airfoil) {
        if (std::abs(flow1) < 1.0e-14)
            throw std::runtime_error("zero mass-flow response");
        result.circulation = (inputs.mass_flow_rate - flow0) / flow1;
    } else {
        // Locate the GetDP Kutta probe in Cartesian space on the mapped quad mesh.
        std::size_t probe_i = ni;
        std::size_t probe_j = cells_per_patch_j;
        double probe_xi = 0.0;
        double probe_eta = 0.0;
        double best_clearance = -1.0;
        for (std::size_t i = 0; i < ni; ++i)
            for (std::size_t j = 0; j < cells_per_patch_j; ++j) {
                std::array<double, 2> const a = point(i, j);
                std::array<double, 2> const b = point(i + 1, j);
                std::array<double, 2> const c = point(i + 1, j + 1);
                std::array<double, 2> const d = point(i, j + 1);
                double const min_x = std::min({a[0], b[0], c[0], d[0]});
                double const max_x = std::max({a[0], b[0], c[0], d[0]});
                double const min_y = std::min({a[1], b[1], c[1], d[1]});
                double const max_y = std::max({a[1], b[1], c[1], d[1]});
                if (1.0001 < min_x - 1.0e-10 || 1.0001 > max_x + 1.0e-10 || 0.0 < min_y - 1.0e-10
                    || 0.0 > max_y + 1.0e-10)
                    continue;
                double xi = 0.5;
                double eta = 0.5;
                for (int iteration = 0; iteration < 12; ++iteration) {
                    double const x = (1.0 - xi) * ((1.0 - eta) * a[0] + eta * d[0])
                                     + xi * ((1.0 - eta) * b[0] + eta * c[0]);
                    double const y = (1.0 - xi) * ((1.0 - eta) * a[1] + eta * d[1])
                                     + xi * ((1.0 - eta) * b[1] + eta * c[1]);
                    double const tx = (1.0 - eta) * (b[0] - a[0]) + eta * (c[0] - d[0]);
                    double const ty = (1.0 - eta) * (b[1] - a[1]) + eta * (c[1] - d[1]);
                    double const rx = (1.0 - xi) * (d[0] - a[0]) + xi * (c[0] - b[0]);
                    double const ry = (1.0 - xi) * (d[1] - a[1]) + xi * (c[1] - b[1]);
                    double const determinant = tx * ry - ty * rx;
                    if (std::abs(determinant) < 1.0e-16)
                        break;
                    xi -= ((x - 1.0001) * ry - rx * y) / determinant;
                    eta -= (tx * y - ty * (x - 1.0001)) / determinant;
                }
                if (xi < -1.0e-9 || xi > 1.0 + 1.0e-9 || eta < -1.0e-9 || eta > 1.0 + 1.0e-9)
                    continue;
                double const x = (1.0 - xi) * ((1.0 - eta) * a[0] + eta * d[0])
                                 + xi * ((1.0 - eta) * b[0] + eta * c[0]);
                double const y = (1.0 - xi) * ((1.0 - eta) * a[1] + eta * d[1])
                                 + xi * ((1.0 - eta) * b[1] + eta * c[1]);
                if (std::hypot(x - 1.0001, y) > 1.0e-8)
                    continue;
                double const clearance = std::min({xi, 1.0 - xi, eta, 1.0 - eta});
                if (clearance > best_clearance) {
                    probe_i = i;
                    probe_j = j;
                    probe_xi = xi;
                    probe_eta = eta;
                    best_clearance = clearance;
                }
            }
        if (probe_i == ni)
            throw std::runtime_error("Kutta probe is outside the fluid mesh");
        auto kutta = [&](PotentialFlowSamples const& samples) {
            auto const v = velocity(samples, probe_i, probe_j, probe_xi, probe_eta);
            return v[1] - std::tan(inputs.incidence) * v[0];
        };
        double const v0 = kutta(base_samples), v1 = kutta(response_samples);
        if (std::abs(v1) < 1.0e-14)
            throw std::runtime_error("zero Kutta response");
        result.circulation = -v0 / v1;
    }
    result.mass_flow_rate = flow0 + result.circulation * flow1;
    result.lift_kutta_joukowski = -inputs.density * inputs.velocity * result.circulation;
    std::vector<double> potential(node_count);
    for (std::size_t i = 0; i < node_count; ++i)
        potential[i] = base_ring[i] + result.circulation * response_ring[i];
    PotentialFlowSamples const samples = sample_potential_flow_field(
            potential,
            result.circulation,
            ni,
            nodes_j,
            side_start,
            side_cells,
            connections);
    std::ofstream output(output_file);
    if (!output)
        throw std::runtime_error("cannot write potential-flow Gmsh view");
    output << std::setprecision(16);
    auto write_quad = [&](std::size_t i, std::size_t j, char const* code) {
        output << code << '(';
        for (int a = 0; a < 4; ++a) {
            std::size_t const ai = i + ((a == 1 || a == 2) ? 1 : 0);
            std::size_t const ar = j + ((a == 2 || a == 3) ? 1 : 0);
            auto const& p = nodes[index(ai, ar)];
            output << p.x << ',' << p.y << ',' << p.z;
            if (a != 3)
                output << ',';
        }
        output << "){ ";
    };
    output << "View \"SimiLie potential\" {\n";
    for (std::size_t i = 0; i < ni; ++i)
        for (std::size_t j = 0; j < cells_per_patch_j; ++j) {
            write_quad(i, j, "SQ");
            for (int a = 0; a < 4; ++a) {
                output << samples.cell_potentials[index(i, j)][a];
                if (a != 3)
                    output << ',';
            }
            output << " };\n";
        }
    output << "};\nView \"SimiLie velocity\" {\n";
    for (std::size_t i = 0; i < ni; ++i)
        for (std::size_t j = 0; j < cells_per_patch_j; ++j) {
            auto const v = velocity(samples, i, j, 0.5, 0.5);
            result.max_speed = std::max(result.max_speed, std::hypot(v[0], v[1]));
            write_quad(i, j, "VQ");
            for (int a = 0; a < 4; ++a) {
                output << v[0] << ',' << v[1] << ",0";
                if (a != 3)
                    output << ',';
            }
            output << " };\n";
        }
    output << "};\nView \"SimiLie pressure difference\" {\n";
    for (std::size_t i = 0; i < ni; ++i)
        for (std::size_t j = 0; j < cells_per_patch_j; ++j) {
            auto const v = velocity(samples, i, j, 0.5, 0.5);
            double const pressure = -0.5 * inputs.density * (v[0] * v[0] + v[1] * v[1]);
            write_quad(i, j, "SQ");
            output << pressure << ',' << pressure << ',' << pressure << ',' << pressure << " };\n";
        }
    output << "};\n";
    return result;
}

} // namespace similie::onelab_interface::potential_flow_onelab
