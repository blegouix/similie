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

#include <ginkgo/core/base/matrix_data.hpp>
#include <similie/exterior/boundary.hpp>
#include <similie/exterior/cochain.hpp>
#include <similie/solvers/minimize_strong_formulation_residual.hpp>

#include <Kokkos_Core.hpp>

#include "gmsh_structured_grid.hpp"

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
using Vertex = sil::exterior::Simplex<0, X, Y>;
using PrimalEdge = sil::exterior::Simplex<1, X, Y>;

struct Edge
{
    std::array<std::size_t, 2> nodes;
    std::array<double, 2> signs;
    double cut_value;
};

struct CellHodge
{
    std::array<Edge, 4> edges;
    std::array<std::array<double, 4>, 4> matrix;
};

// The primal coboundary is evaluated from exterior::boundary.
inline Edge make_edge(std::size_t i, std::size_t j, bool along_i, std::size_t ni, double cut_value)
{
    ddc::DiscreteElement<X, Y> const
            origin(static_cast<std::ptrdiff_t>(i), static_cast<std::ptrdiff_t>(j));
    PrimalEdge const primal(origin, ddc::DiscreteVector<X, Y>(along_i ? 1 : 0, along_i ? 0 : 1));
    auto incidence = sil::exterior::boundary(primal);
    Edge edge {{}, {}, cut_value};
    int k = 0;
    for (Vertex const& vertex : incidence) {
        auto const elem = vertex.discrete_element();
        edge.nodes[k] = static_cast<std::size_t>(elem.uid<Y>()) * ni
                        + static_cast<std::size_t>(elem.uid<X>()) % ni;
        edge.signs[k] = vertex.negative() ? -1.0 : 1.0;
        ++k;
    }
    if (k != 2 || edge.nodes[0] == edge.nodes[1])
        throw std::runtime_error("invalid DEC edge incidence");
    return edge;
}

inline double coboundary_value(Edge const& edge, std::vector<double> const& potential)
{
    ddc::DiscreteElement<X, Y> const origin(0, 0);
    PrimalEdge const primal(origin, ddc::DiscreteVector<X>(1));
    auto incidence = sil::exterior::boundary(primal);
    Kokkos::View<double*, Kokkos::LayoutRight, Kokkos::HostSpace>
            values("potential_flow_vertex_cochain", 2);
    values(0) = potential[edge.nodes[0]];
    values(1) = potential[edge.nodes[1]];
    sil::exterior::Cochain<decltype(incidence)> cochain(incidence, values);
    return cochain.integrate();
}

class SparseDecLaplacian
{
    Kokkos::View<int*> m_offsets;
    Kokkos::View<int*> m_columns;
    Kokkos::View<double*> m_values;
    std::size_t m_size;

public:
    static constexpr bool IS_LINEAR = true;
    static constexpr bool IS_SYMMETRIC = true;

    explicit SparseDecLaplacian(std::vector<std::map<std::size_t, double>> const& rows)
        : m_offsets("potential_flow_dec_offsets", rows.size() + 1)
        , m_columns(
                  "potential_flow_dec_columns",
                  [&]() {
                      std::size_t count = 0;
                      for (auto const& row : rows)
                          count += row.size();
                      return count;
                  }())
        , m_values("potential_flow_dec_values", m_columns.extent(0))
        , m_size(rows.size())
    {
        auto offsets = Kokkos::create_mirror_view(m_offsets);
        auto columns = Kokkos::create_mirror_view(m_columns);
        auto values = Kokkos::create_mirror_view(m_values);
        std::size_t slot = 0;
        for (std::size_t row = 0; row < rows.size(); ++row) {
            offsets(row) = static_cast<int>(slot);
            for (auto const& [column, value] : rows[row]) {
                columns(slot) = static_cast<int>(column);
                values(slot) = value;
                ++slot;
            }
        }
        offsets(rows.size()) = static_cast<int>(slot);
        Kokkos::deep_copy(m_offsets, offsets);
        Kokkos::deep_copy(m_columns, columns);
        Kokkos::deep_copy(m_values, values);
    }

    [[nodiscard]] std::size_t size() const
    {
        return m_size;
    }

    template <class ExecSpace, class InputView, class OutputView>
    void apply(ExecSpace exec_space, InputView input, OutputView output) const
    {
        auto const offsets = m_offsets;
        auto const columns = m_columns;
        auto const values = m_values;
        Kokkos::parallel_for(
                "potential_flow_dec_laplacian",
                Kokkos::RangePolicy<ExecSpace>(exec_space, 0, m_size),
                KOKKOS_LAMBDA(std::size_t row) {
                    double sum = 0.0;
                    for (int slot = offsets(row); slot < offsets(row + 1); ++slot)
                        sum += values(slot) * input(columns(slot), 0);
                    output(row, 0) = sum;
                });
        exec_space.fence();
    }

    friend gko::matrix_data<double, gko::int32> assemble_matrix_data(
            SparseDecLaplacian const& matrix)
    {
        gko::matrix_data<double, gko::int32> data(gko::dim<2>(matrix.m_size, matrix.m_size));
        auto const offsets
                = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), matrix.m_offsets);
        auto const columns
                = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), matrix.m_columns);
        auto const values
                = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), matrix.m_values);
        for (std::size_t row = 0; row < matrix.m_size; ++row)
            for (int slot = offsets(row); slot < offsets(row + 1); ++slot)
                data.nonzeros
                        .emplace_back(static_cast<gko::int32>(row), columns(slot), values(slot));
        return data;
    }
};

inline std::vector<double> solve_system(
        SparseDecLaplacian const& matrix,
        std::vector<double> const& values,
        solvers::StrongFormulationSolverSettings const& settings,
        solvers::StrongFormulationSolverDiagnostics& diagnostics)
{
    Kokkos::View<double**> rhs("potential_flow_rhs", values.size(), 1);
    Kokkos::View<double**> solution("potential_flow_solution", values.size(), 1);
    auto host = Kokkos::create_mirror_view(rhs);
    for (std::size_t i = 0; i < values.size(); ++i)
        host(i, 0) = values[i];
    Kokkos::deep_copy(rhs, host);
    diagnostics = solvers::minimize_strong_formulation_residual(
            Kokkos::DefaultExecutionSpace(),
            matrix,
            rhs,
            solution,
            settings);
    if (!std::isfinite(diagnostics.final_relative_residual)
        || diagnostics.final_relative_residual > settings.relative_tolerance * 10.0)
        throw std::runtime_error("potential-flow DEC solve did not converge");
    auto const solved = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), solution);
    std::vector<double> result(values.size());
    for (std::size_t i = 0; i < values.size(); ++i)
        result[i] = solved(i, 0);
    return result;
}

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
    constexpr std::size_t cells_per_patch_i = 40;
    constexpr std::size_t cells_per_patch_j = 24;
    constexpr std::size_t ni = patch_count * cells_per_patch_i;
    constexpr std::size_t nodes_j = cells_per_patch_j + 1;
    constexpr std::size_t node_count = ni * nodes_j;
    if (mesh.cells.size() != ni * cells_per_patch_j || mesh.nodes.size() != node_count)
        throw std::runtime_error("potential-flow mesh does not match the mapped quad topology");
    std::array<std::vector<sil::onelab_interface::gmsh::QuadrilateralCell const*>, patch_count>
            patches;
    for (auto const& cell : mesh.cells) {
        if ((cell.physical_tag != 2 && cell.physical_tag != 3) || cell.elementary_tag < 24
            || cell.elementary_tag > 27)
            throw std::runtime_error("potential-flow mesh contains a non-fluid quad");
        patches[cell.elementary_tag - 24].push_back(&cell);
    }
    std::map<std::size_t, std::size_t> logical_index;
    auto bind_node = [&](std::size_t tag, std::size_t index) {
        auto const [it, inserted] = logical_index.emplace(tag, index);
        if (!inserted && it->second != index)
            throw std::runtime_error("quad patches do not share a consistent structured grid");
    };
    for (std::size_t patch = 0; patch < patch_count; ++patch) {
        if (patches[patch].size() != cells_per_patch_i * cells_per_patch_j)
            throw std::runtime_error("potential-flow quad patch has unexpected dimensions");
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
    auto index = [](std::size_t i, std::size_t j) { return j * ni + i % ni; };
    auto point = [&](std::size_t i, std::size_t j) {
        auto const& n = nodes[index(i, j)];
        return std::array<double, 2> {n.x, n.y};
    };
    std::vector<int> boundary(node_count, 0);
    std::vector<double> prescribed(node_count, 0.0);
    bool has_cut = false;
    for (auto const& edge : mesh.boundary_edges) {
        has_cut |= edge.physical_tag == 13;
        if (edge.physical_tag != 10 && edge.physical_tag != 11)
            continue;
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
    std::vector<CellHodge> cells;
    cells.reserve(ni * cells_per_patch_j);
    std::array<double, 7> constexpr gauss_points {
            0.5,
            0.5 - 0.2029225756886985,
            0.5 + 0.2029225756886985,
            0.5 - 0.3707655927996970,
            0.5 + 0.3707655927996970,
            0.5 - 0.4745539561713792,
            0.5 + 0.4745539561713792,
    };
    std::array<double, 7> constexpr gauss_weights {
            0.2089795918367345,
            0.1909150252525595,
            0.1909150252525595,
            0.1398526957446385,
            0.1398526957446385,
            0.0647424830844350,
            0.0647424830844350,
    };
    for (std::size_t i = 0; i < ni; ++i)
        for (std::size_t j = 0; j < cells_per_patch_j; ++j) {
            CellHodge cell {
                    .edges
                    = {make_edge(i, j, true, ni, i == 119 ? -1.0 : 0.0),
                       make_edge(i, j + 1, true, ni, i == 119 ? -1.0 : 0.0),
                       make_edge(i, j, false, ni, 0.0),
                       make_edge(i + 1, j, false, ni, 0.0)},
                    .matrix = {},
            };
            auto const a = point(i, j), b = point(i + 1, j);
            auto const c = point(i + 1, j + 1), d = point(i, j + 1);
            // The quadrilateral Hodge star couples all four primal edge
            // cochains. The reconstruction and metric are integrated across
            // skew mapped quads, retaining their off-diagonal terms.
            for (std::size_t gi = 0; gi < gauss_points.size(); ++gi)
                for (std::size_t gj = 0; gj < gauss_points.size(); ++gj) {
                    double const xi = gauss_points[gi];
                    double const eta = gauss_points[gj];
                    double const tx = (1.0 - eta) * (b[0] - a[0]) + eta * (c[0] - d[0]);
                    double const ty = (1.0 - eta) * (b[1] - a[1]) + eta * (c[1] - d[1]);
                    double const rx = (1.0 - xi) * (d[0] - a[0]) + xi * (c[0] - b[0]);
                    double const ry = (1.0 - xi) * (d[1] - a[1]) + xi * (c[1] - b[1]);
                    double const determinant = tx * ry - ty * rx;
                    if (std::abs(determinant) <= 1.0e-16)
                        throw std::runtime_error("degenerate mapped quad");
                    std::array<double, 4> const dxi {1.0 - eta, eta, 0.0, 0.0};
                    std::array<double, 4> const deta {0.0, 0.0, 1.0 - xi, xi};
                    for (int p = 0; p < 4; ++p)
                        for (int q = 0; q < 4; ++q) {
                            double const px = (dxi[p] * ry - ty * deta[p]) / determinant;
                            double const py = (tx * deta[p] - rx * dxi[p]) / determinant;
                            double const qx = (dxi[q] * ry - ty * deta[q]) / determinant;
                            double const qy = (tx * deta[q] - rx * dxi[q]) / determinant;
                            cell.matrix[p][q] += gauss_weights[gi] * gauss_weights[gj]
                                                 * inputs.density * std::abs(determinant)
                                                 * (px * qx + py * qy);
                        }
                }
            cells.push_back(cell);
        }
    std::vector<std::map<std::size_t, double>> full_rows(node_count);
    std::vector<double> cut_rhs(node_count, 0.0);
    for (CellHodge const& cell : cells)
        for (int p = 0; p < 4; ++p)
            for (int q = 0; q < 4; ++q)
                for (int a = 0; a < 2; ++a) {
                    Edge const& row_edge = cell.edges[p];
                    Edge const& column_edge = cell.edges[q];
                    std::size_t const row = row_edge.nodes[a];
                    double const coefficient = cell.matrix[p][q] * row_edge.signs[a];
                    cut_rhs[row] -= coefficient * column_edge.cut_value;
                    for (int b = 0; b < 2; ++b)
                        full_rows[row][column_edge.nodes[b]] += coefficient * column_edge.signs[b];
                }
    std::vector<std::map<std::size_t, double>> rows(node_count);
    std::vector<double> base_rhs(node_count, 0.0);
    for (std::size_t i = 0; i < node_count; ++i) {
        if (boundary[i]) {
            rows[i][i] = 1.0;
            base_rhs[i] = prescribed[i];
            cut_rhs[i] = 0.0;
        } else {
            for (auto const& [column, value] : full_rows[i]) {
                if (boundary[column])
                    base_rhs[i] -= value * prescribed[column];
                else
                    rows[i][column] = value;
            }
        }
    }
    SparseDecLaplacian const matrix(rows);
    Result result;
    result.node_count = node_count;
    result.cell_count = mesh.cells.size();
    std::vector<double> const base
            = solve_system(matrix, base_rhs, settings, result.solver_diagnostics);
    solvers::StrongFormulationSolverDiagnostics cut_diagnostics;
    std::vector<double> const response = solve_system(matrix, cut_rhs, settings, cut_diagnostics);
    auto mass_flow = [&](std::vector<double> const& potential, double circulation) {
        double conjugate = 0.0;
        for (CellHodge const& cell : cells)
            for (int p = 0; p < 4; ++p)
                if (cell.edges[p].cut_value != 0.0)
                    for (int q = 0; q < 4; ++q)
                        conjugate += cell.edges[p].cut_value * cell.matrix[p][q]
                                     * (coboundary_value(cell.edges[q], potential)
                                        + circulation * cell.edges[q].cut_value);
        return conjugate;
    };
    auto velocity = [&](std::vector<double> const& potential,
                        double circulation,
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
        double const cut = i == 119 ? -circulation : 0.0;
        double const dt = (1.0 - eta) * (potential[index(i + 1, j)] - potential[index(i, j)])
                          + eta * (potential[index(i + 1, j + 1)] - potential[index(i, j + 1)])
                          + cut;
        double const dr = (1.0 - xi) * (potential[index(i, j + 1)] - potential[index(i, j)])
                          + xi * (potential[index(i + 1, j + 1)] - potential[index(i + 1, j)]);
        double const determinant = tx * ry - ty * rx;
        if (std::abs(determinant) < 1.0e-16)
            throw std::runtime_error("degenerate quad metric");
        return std::array<
                double,
                2> {(dt * ry - ty * dr) / determinant, (tx * dr - dt * rx) / determinant};
    };
    double const flow0 = mass_flow(base, 0.0);
    double const flow1 = mass_flow(response, 1.0);
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
        auto kutta = [&](std::vector<double> const& potential, double circulation) {
            auto const v = velocity(potential, circulation, probe_i, probe_j, probe_xi, probe_eta);
            return v[1] - std::tan(inputs.incidence) * v[0];
        };
        double const v0 = kutta(base, 0.0), v1 = kutta(response, 1.0);
        if (std::abs(v1) < 1.0e-14)
            throw std::runtime_error("zero Kutta response");
        result.circulation = -v0 / v1;
    }
    result.mass_flow_rate = flow0 + result.circulation * flow1;
    result.lift_kutta_joukowski = -inputs.density * inputs.velocity * result.circulation;
    std::vector<double> potential(node_count);
    for (std::size_t i = 0; i < node_count; ++i)
        potential[i] = base[i] + result.circulation * response[i];
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
                std::size_t const ai = i + ((a == 1 || a == 2) ? 1 : 0);
                std::size_t const ar = j + ((a == 2 || a == 3) ? 1 : 0);
                double const jump = i == 119 && (a == 1 || a == 2) ? -result.circulation : 0.0;
                output << potential[index(ai, ar)] + jump;
                if (a != 3)
                    output << ',';
            }
            output << " };\n";
        }
    output << "};\nView \"SimiLie velocity\" {\n";
    for (std::size_t i = 0; i < ni; ++i)
        for (std::size_t j = 0; j < cells_per_patch_j; ++j) {
            auto const v = velocity(potential, result.circulation, i, j, 0.5, 0.5);
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
            auto const v = velocity(potential, result.circulation, i, j, 0.5, 0.5);
            double const pressure = -0.5 * inputs.density * (v[0] * v[0] + v[1] * v[1]);
            write_quad(i, j, "SQ");
            output << pressure << ',' << pressure << ',' << pressure << ',' << pressure << " };\n";
        }
    output << "};\n";
    return result;
}

} // namespace similie::onelab_interface::potential_flow_onelab
