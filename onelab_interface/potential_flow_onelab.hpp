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

#include <similie/physics/scalar_field/triangular_laplace.hpp>
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

struct Cell
{
    std::array<std::size_t, 3> nodes;
    std::array<double, 3> cut_basis;
    physics::scalar_field::TriangleStiffness stiffness;
};

inline std::vector<double> solve_system(
        physics::scalar_field::TriangularLaplaceOperator const& operator_model,
        std::vector<double> const& values,
        solvers::StrongFormulationSolverSettings const& settings,
        solvers::StrongFormulationSolverDiagnostics& diagnostics)
{
    Kokkos::View<double**> rhs("potential_flow_rhs", values.size(), 1);
    Kokkos::View<double**> solution("potential_flow_solution", values.size(), 1);
    auto rhs_host = Kokkos::create_mirror_view(rhs);
    for (std::size_t i = 0; i < values.size(); ++i)
        rhs_host(i, 0) = values[i];
    Kokkos::deep_copy(rhs, rhs_host);
    diagnostics = solvers::minimize_strong_formulation_residual(
            Kokkos::DefaultExecutionSpace(),
            operator_model,
            rhs,
            solution,
            settings);
    if (!std::isfinite(diagnostics.final_relative_residual)
        || diagnostics.final_relative_residual > settings.relative_tolerance * 10.0) {
        throw std::runtime_error("potential-flow Laplace solve did not converge");
    }
    auto const solution_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), solution);
    std::vector<double> result(values.size());
    for (std::size_t i = 0; i < values.size(); ++i)
        result[i] = solution_host(i, 0);
    return result;
}

inline Result run(
        std::filesystem::path const& mesh_file,
        std::filesystem::path const& output_file,
        Inputs const& inputs,
        solvers::StrongFormulationSolverSettings const& settings)
{
    auto const supported_mesh = sil::onelab_interface::gmsh::parse_supported_msh2_mesh(mesh_file);
    if (!std::holds_alternative<sil::onelab_interface::gmsh::TriangularMesh>(supported_mesh)) {
        throw std::runtime_error("potential flow requires a two-dimensional triangular mesh");
    }
    auto const& mesh = std::get<sil::onelab_interface::gmsh::TriangularMesh>(supported_mesh);
    Result result;
    result.node_count = mesh.nodes.size();
    result.cell_count = mesh.cells.size();
    std::map<std::size_t, std::size_t> indices;
    for (std::size_t i = 0; i < mesh.nodes.size(); ++i)
        indices.emplace(mesh.nodes[i].tag, i);
    std::vector<int> cut_nodes(mesh.nodes.size(), 0);
    std::vector<int> boundary(mesh.nodes.size(), 0);
    std::vector<double> prescribed(mesh.nodes.size(), 0.0);
    for (auto const& edge : mesh.boundary_edges) {
        for (std::size_t tag : edge.node_tags) {
            std::size_t const i = indices.at(tag);
            if (edge.physical_tag == 13)
                cut_nodes[i] = 1;
            if (edge.physical_tag == 10 || edge.physical_tag == 11) {
                int const side = edge.physical_tag;
                if (boundary[i] && boundary[i] != side) {
                    throw std::runtime_error("upstream and downstream boundaries share a node");
                }
                boundary[i] = side;
                prescribed[i] = side == 11 ? inputs.velocity * inputs.box_size : 0.0;
            }
        }
    }
    bool has_upstream = false;
    bool has_downstream = false;
    bool has_cut = false;
    for (std::size_t i = 0; i < mesh.nodes.size(); ++i) {
        has_upstream |= boundary[i] == 10;
        has_downstream |= boundary[i] == 11;
        has_cut |= cut_nodes[i] != 0;
    }
    if (!has_upstream || !has_downstream || !has_cut) {
        throw std::runtime_error(
                "potential flow is missing upstream, downstream, or wake physical curves");
    }

    std::vector<Cell> cells;
    cells.reserve(mesh.cells.size());
    std::vector<std::map<std::size_t, double>> rows(mesh.nodes.size());
    std::vector<double> base_rhs(mesh.nodes.size(), 0.0);
    std::vector<double> cut_rhs(mesh.nodes.size(), 0.0);
    for (auto const& triangle : mesh.cells) {
        if (triangle.physical_tag != 2) {
            throw std::runtime_error("potential-flow mesh contains a non-fluid triangle");
        }
        Cell cell;
        std::array<std::array<double, 2>, 3> points;
        for (int a = 0; a < 3; ++a) {
            cell.nodes[a] = indices.at(triangle.node_tags[a]);
            auto const& node = mesh.nodes[cell.nodes[a]];
            points[a] = {node.x, node.y};
            // The wake basis has support only on surface 25 at wake nodes.
            // phi = u + circulation * q is discontinuous across the cut,
            // while its gradient remains well-defined in every triangle.
            cell.cut_basis[a]
                    = (triangle.elementary_tag == 25 && cut_nodes[cell.nodes[a]]) ? 1.0 : 0.0;
        }
        cell.stiffness
                = physics::scalar_field::triangular_laplace_stiffness(points, inputs.density);
        for (int a = 0; a < 3; ++a) {
            std::size_t const row = cell.nodes[a];
            if (boundary[row])
                continue;
            for (int b = 0; b < 3; ++b) {
                std::size_t const column = cell.nodes[b];
                double const value = cell.stiffness.matrix[a][b];
                if (boundary[column])
                    base_rhs[row] -= value * prescribed[column];
                else
                    rows[row][column] += value;
                cut_rhs[row] -= value * cell.cut_basis[b];
            }
        }
        cells.push_back(cell);
    }
    for (std::size_t i = 0; i < mesh.nodes.size(); ++i) {
        if (boundary[i]) {
            rows[i][i] = 1.0;
            base_rhs[i] = prescribed[i];
            cut_rhs[i] = 0.0;
        }
    }
    physics::scalar_field::TriangularLaplaceOperator const operator_model(rows);
    std::vector<double> const base_solution
            = solve_system(operator_model, base_rhs, settings, result.solver_diagnostics);
    solvers::StrongFormulationSolverDiagnostics cut_diagnostics;
    std::vector<double> const cut_solution
            = solve_system(operator_model, cut_rhs, settings, cut_diagnostics);

    auto mass_flow = [&](std::vector<double> const& field, double circulation) {
        double conjugate = 0.0;
        for (Cell const& cell : cells) {
            for (int a = 0; a < 3; ++a) {
                for (int b = 0; b < 3; ++b) {
                    conjugate += cell.cut_basis[a] * cell.stiffness.matrix[a][b]
                                 * (field[cell.nodes[b]] + circulation * cell.cut_basis[b]);
                }
            }
        }
        return conjugate;
    };
    double const flow0 = mass_flow(base_solution, 0.0);
    double const flow1 = mass_flow(cut_solution, 1.0);
    if (inputs.impose_circulation) {
        result.circulation = inputs.circulation;
    } else if (!inputs.airfoil) {
        if (std::abs(flow1) < 1.0e-14)
            throw std::runtime_error("zero circulation response");
        result.circulation = (inputs.mass_flow_rate - flow0) / flow1;
    } else {
        std::size_t trailing_cell = cells.size();
        for (std::size_t i = 0; i < cells.size(); ++i) {
            Cell const& cell = cells[i];
            auto const& a = mesh.nodes[cell.nodes[0]];
            auto const& b = mesh.nodes[cell.nodes[1]];
            auto const& c = mesh.nodes[cell.nodes[2]];
            double const determinant = (b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y);
            double const weight_b
                    = ((1.0001 - a.x) * (c.y - a.y) + (c.x - a.x) * a.y) / determinant;
            double const weight_c
                    = (-(b.x - a.x) * a.y - (1.0001 - a.x) * (b.y - a.y)) / determinant;
            double const weight_a = 1.0 - weight_b - weight_c;
            if (weight_a >= -1.0e-10 && weight_b >= -1.0e-10 && weight_c >= -1.0e-10) {
                trailing_cell = i;
                break;
            }
        }
        if (trailing_cell == cells.size())
            throw std::runtime_error("Kutta probe is outside the airfoil fluid mesh");
        auto transverse_speed = [&](std::vector<double> const& field, double circulation) {
            Cell const& cell = cells[trailing_cell];
            double vx = 0.0;
            double vy = 0.0;
            for (int a = 0; a < 3; ++a) {
                double const value = field[cell.nodes[a]] + circulation * cell.cut_basis[a];
                vx += value * cell.stiffness.gradients[a][0];
                vy += value * cell.stiffness.gradients[a][1];
            }
            return vy - std::tan(inputs.incidence) * vx;
        };
        double const v0 = transverse_speed(base_solution, 0.0);
        double const v1 = transverse_speed(cut_solution, 1.0);
        if (std::abs(v1) < 1.0e-14)
            throw std::runtime_error("zero Kutta circulation response");
        result.circulation = -v0 / v1;
    }
    result.mass_flow_rate = flow0 + result.circulation * flow1;
    result.lift_kutta_joukowski = -inputs.density * inputs.velocity * result.circulation;
    std::vector<double> field(mesh.nodes.size());
    for (std::size_t i = 0; i < field.size(); ++i) {
        field[i] = base_solution[i] + result.circulation * cut_solution[i];
    }
    std::ofstream output(output_file);
    if (!output)
        throw std::runtime_error("cannot write potential-flow Gmsh view");
    output << std::setprecision(16);
    output << "View \"SimiLie potential\" {\n";
    std::vector<std::array<double, 2>> velocities(cells.size());
    for (std::size_t cell_index = 0; cell_index < cells.size(); ++cell_index) {
        Cell const& cell = cells[cell_index];
        output << "ST(";
        for (int a = 0; a < 3; ++a) {
            auto const& node = mesh.nodes[cell.nodes[a]];
            output << node.x << ',' << node.y << ',' << node.z;
            if (a != 2)
                output << ',';
        }
        output << "){\n";
        for (int a = 0; a < 3; ++a) {
            output << field[cell.nodes[a]] + result.circulation * cell.cut_basis[a];
            if (a != 2)
                output << ',';
        }
        output << "};\n";
        double vx = 0.0;
        double vy = 0.0;
        for (int a = 0; a < 3; ++a) {
            double const value = field[cell.nodes[a]] + result.circulation * cell.cut_basis[a];
            vx += value * cell.stiffness.gradients[a][0];
            vy += value * cell.stiffness.gradients[a][1];
        }
        velocities[cell_index] = {vx, vy};
        result.max_speed = std::max(result.max_speed, std::hypot(vx, vy));
    }
    output << "};\n";
    output << "View \"SimiLie velocity\" {\n";
    for (std::size_t cell_index = 0; cell_index < cells.size(); ++cell_index) {
        Cell const& cell = cells[cell_index];
        output << "VT(";
        for (int a = 0; a < 3; ++a) {
            auto const& node = mesh.nodes[cell.nodes[a]];
            output << node.x << ',' << node.y << ',' << node.z;
            if (a != 2)
                output << ',';
        }
        output << "){";
        for (int a = 0; a < 3; ++a) {
            output << velocities[cell_index][0] << ',' << velocities[cell_index][1] << ",0";
            if (a != 2)
                output << ',';
        }
        output << "};\n";
    }
    output << "};\n";
    output << "View \"SimiLie pressure difference\" {\n";
    for (std::size_t cell_index = 0; cell_index < cells.size(); ++cell_index) {
        Cell const& cell = cells[cell_index];
        output << "ST(";
        for (int a = 0; a < 3; ++a) {
            auto const& node = mesh.nodes[cell.nodes[a]];
            output << node.x << ',' << node.y << ',' << node.z;
            if (a != 2)
                output << ',';
        }
        double const pressure = -0.5 * inputs.density
                                * (velocities[cell_index][0] * velocities[cell_index][0]
                                   + velocities[cell_index][1] * velocities[cell_index][1]);
        output << "){" << pressure << ',' << pressure << ',' << pressure << "};\n";
    }
    output << "};\n";
    return result;
}

} // namespace similie::onelab_interface::potential_flow_onelab
