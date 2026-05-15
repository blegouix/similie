// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include <Kokkos_Core.hpp>

#include <similie/mesher/gmsh_structured_msh2.hpp>
#include <similie/physics/hamilton_equations.hpp>
#include <similie/physics/magnetostatics/linear_magnetostatics.hpp>
#include <similie/physics/magnetostatics/linear_magnetic_induction_to_magnetic_field.hpp>
#include <similie/physics/magnetostatics/magnetostatics_quantities.hpp>
#include <similie/physics/magnetostatics/structured_linear_magnetostatics.hpp>
#include <similie/solvers/minimize_strong_formulation_residual.hpp>

namespace similie::physics::magnetostatics {

struct StructuredLinearMagnetostaticsInputs
{
    double current_density_magnitude;
    double core_mu;
    double mu0;
};

struct StructuredLinearMagnetostaticsRegionTags
{
    int e_core_tag;
    int i_core_tag;
    int coil_left_tag;
    int coil_right_tag;
    int air_gap_tag;
};

struct StructuredLinearMagnetostaticsResult
{
    std::string topology;
    std::size_t node_count = 0;
    std::array<std::size_t, 3> mesh_dimensions {0, 0, 0};
    std::size_t num_cells = 0;
    std::size_t num_air_cells = 0;
    std::size_t num_core_cells = 0;
    std::size_t num_coil_cells = 0;
    std::size_t num_air_gap_cells = 0;
    double max_abs_potential = 0.0;
    double max_abs_induction = 0.0;
    double max_abs_field = 0.0;
    double air_gap_induction_magnitude_sum = 0.0;
    double force_density_magnitude_sum = 0.0;
    solvers::StrongFormulationSolverDiagnostics solver_diagnostics;
};

namespace detail {

struct CellInputFields
{
    double mu;
    std::array<double, 3> current_density;
};

struct CellPostProcessFields
{
    std::array<double, 3> magnetic_induction;
    std::array<double, 3> magnetic_field;
    std::array<double, 6> maxwell_stress;
    std::array<double, 3> force_density;
};

template <class Logger>
void log_info(Logger&& logger, std::string const& message)
{
    if constexpr (std::is_invocable_v<Logger, std::string const&>) {
        logger(message);
    }
}

inline void write_results_view(
        std::filesystem::path const& output_file,
        sil::mesher::gmsh::StructuredGrid2D const& grid,
        std::vector<CellInputFields> const& cell_inputs,
        std::vector<double> const& magnetic_vector_potential)
{
    if (!output_file.parent_path().empty()) {
        std::filesystem::create_directories(output_file.parent_path());
    }

    std::ofstream stream(output_file);
    if (!stream.is_open()) {
        throw std::runtime_error("failed to open output view file: " + output_file.string());
    }

    stream << "View \"SimiLie linear magnetostatics permeability\" {\n";
    for (std::size_t j = 0; j < grid.ncell_y(); ++j) {
        for (std::size_t i = 0; i < grid.ncell_x(); ++i) {
            std::size_t const index = grid.cell_index(i, j);
            stream << "  SP(" << grid.cell_center_x(i) << ", " << grid.cell_center_y(j) << ", "
                   << grid.z_value << "){" << cell_inputs[index].mu << "};\n";
        }
    }
    stream << "};\n";

    stream << "View \"SimiLie linear magnetostatics current density z\" {\n";
    for (std::size_t j = 0; j < grid.ncell_y(); ++j) {
        for (std::size_t i = 0; i < grid.ncell_x(); ++i) {
            std::size_t const index = grid.cell_index(i, j);
            stream << "  SP(" << grid.cell_center_x(i) << ", " << grid.cell_center_y(j) << ", "
                   << grid.z_value << "){" << cell_inputs[index].current_density[2] << "};\n";
        }
    }
    stream << "};\n";

    stream << "View \"SimiLie linear magnetostatics magnetic vector potential z\" {\n";
    for (std::size_t j = 0; j < grid.ny(); ++j) {
        for (std::size_t i = 0; i < grid.nx(); ++i) {
            std::size_t const index = grid.node_index(i, j);
            stream << "  SP(" << grid.x_coords[i] << ", " << grid.y_coords[j] << ", " << grid.z_value
                   << "){" << magnetic_vector_potential[3 * index + 2] << "};\n";
        }
    }
    stream << "};\n";
}

inline void write_results_view(
        std::filesystem::path const& output_file,
        sil::mesher::gmsh::StructuredGrid3D const& grid,
        std::vector<CellInputFields> const& cell_inputs,
        std::vector<double> const& magnetic_vector_potential)
{
    if (!output_file.parent_path().empty()) {
        std::filesystem::create_directories(output_file.parent_path());
    }

    std::ofstream stream(output_file);
    if (!stream.is_open()) {
        throw std::runtime_error("failed to open output view file: " + output_file.string());
    }

    stream << "View \"SimiLie linear magnetostatics permeability\" {\n";
    for (std::size_t k = 0; k < grid.ncell_z(); ++k) {
        for (std::size_t j = 0; j < grid.ncell_y(); ++j) {
            for (std::size_t i = 0; i < grid.ncell_x(); ++i) {
                std::size_t const index = grid.cell_index(i, j, k);
                stream << "  SP(" << grid.cell_center_x(i) << ", " << grid.cell_center_y(j) << ", "
                       << grid.cell_center_z(k) << "){" << cell_inputs[index].mu << "};\n";
            }
        }
    }
    stream << "};\n";

    stream << "View \"SimiLie linear magnetostatics current density z\" {\n";
    for (std::size_t k = 0; k < grid.ncell_z(); ++k) {
        for (std::size_t j = 0; j < grid.ncell_y(); ++j) {
            for (std::size_t i = 0; i < grid.ncell_x(); ++i) {
                std::size_t const index = grid.cell_index(i, j, k);
                stream << "  SP(" << grid.cell_center_x(i) << ", " << grid.cell_center_y(j) << ", "
                       << grid.cell_center_z(k) << "){" << cell_inputs[index].current_density[2]
                       << "};\n";
            }
        }
    }
    stream << "};\n";

    stream << "View \"SimiLie linear magnetostatics magnetic vector potential z\" {\n";
    for (std::size_t k = 0; k < grid.nz(); ++k) {
        for (std::size_t j = 0; j < grid.ny(); ++j) {
            for (std::size_t i = 0; i < grid.nx(); ++i) {
                std::size_t const index = grid.node_index(i, j, k);
                stream << "  SP(" << grid.x_coords[i] << ", " << grid.y_coords[j] << ", "
                       << grid.z_coords[k] << "){" << magnetic_vector_potential[3 * index + 2]
                       << "};\n";
            }
        }
    }
    stream << "};\n";
}

template <class Logger>
StructuredLinearMagnetostaticsResult run_on_quadrilateral_grid(
        std::filesystem::path const& output_view_file,
        StructuredLinearMagnetostaticsInputs const& inputs,
        StructuredLinearMagnetostaticsRegionTags const& region_tags,
        solvers::StrongFormulationSolverSettings const& solver_settings,
        sil::mesher::gmsh::QuadrilateralMesh const& mesh,
        Logger&& logger)
{
    sil::mesher::gmsh::StructuredGrid2D const grid = sil::mesher::gmsh::build_structured_grid(mesh);
    log_info(
            logger,
            "SimiLie structured rectilinear quadrilateral mesh validated ("
                    + std::to_string(grid.ordered_nodes.size()) + " nodes, dimensions="
                    + std::to_string(grid.nx()) + "x" + std::to_string(grid.ny()) + ")");

    StructuredLinearMagnetostaticsResult result;
    result.topology = "quadrilateral";
    result.node_count = grid.ordered_nodes.size();
    result.mesh_dimensions = {grid.nx(), grid.ny(), 1};
    result.num_cells = grid.ncell_x() * grid.ncell_y();

    std::vector<CellInputFields> cell_inputs(result.num_cells);
    for (std::size_t cell_index = 0; cell_index < result.num_cells; ++cell_index) {
        CellInputFields field {
                .mu = inputs.mu0,
                .current_density = {0.0, 0.0, 0.0},
        };
        int const physical_tag = grid.ordered_cells[cell_index].physical_tag;
        if (physical_tag == region_tags.e_core_tag || physical_tag == region_tags.i_core_tag) {
            field.mu = inputs.core_mu;
            ++result.num_core_cells;
        } else if (physical_tag == region_tags.coil_left_tag) {
            field.current_density[2] = inputs.current_density_magnitude;
            ++result.num_coil_cells;
        } else if (physical_tag == region_tags.coil_right_tag) {
            field.current_density[2] = -inputs.current_density_magnitude;
            ++result.num_coil_cells;
        } else {
            ++result.num_air_cells;
        }
        cell_inputs[cell_index] = field;
    }

    Kokkos::View<double*> x_coords("similie_x_coords", grid.nx());
    Kokkos::View<double*> y_coords("similie_y_coords", grid.ny());
    auto x_coords_host = Kokkos::create_mirror_view(x_coords);
    auto y_coords_host = Kokkos::create_mirror_view(y_coords);
    for (std::size_t i = 0; i < grid.nx(); ++i) {
        x_coords_host(i) = grid.x_coords[i];
    }
    for (std::size_t j = 0; j < grid.ny(); ++j) {
        y_coords_host(j) = grid.y_coords[j];
    }
    Kokkos::deep_copy(x_coords, x_coords_host);
    Kokkos::deep_copy(y_coords, y_coords_host);

    std::size_t const num_nodes = grid.nx() * grid.ny();
    Kokkos::View<double**> rhs("similie_rhs", num_nodes, 1);
    Kokkos::View<double**> magnetic_vector_potential_z_view("similie_Az", num_nodes, 1);
    auto rhs_host = Kokkos::create_mirror_view(rhs);
    for (std::size_t j = 0; j < grid.ny(); ++j) {
        for (std::size_t i = 0; i < grid.nx(); ++i) {
            std::size_t const node_index = grid.node_index(i, j);
            bool const boundary = (i == 0 || j == 0 || i + 1 == grid.nx() || j + 1 == grid.ny());
            if (boundary) {
                rhs_host(node_index, 0) = 0.0;
                continue;
            }
            double accumulated_current_density_z = 0.0;
            std::size_t count = 0;
            for (int dj = -1; dj <= 0; ++dj) {
                for (int di = -1; di <= 0; ++di) {
                    std::ptrdiff_t const ci = static_cast<std::ptrdiff_t>(i) + di;
                    std::ptrdiff_t const cj = static_cast<std::ptrdiff_t>(j) + dj;
                    if (ci < 0 || cj < 0
                        || ci >= static_cast<std::ptrdiff_t>(grid.ncell_x())
                        || cj >= static_cast<std::ptrdiff_t>(grid.ncell_y())) {
                        continue;
                    }
                    accumulated_current_density_z += cell_inputs[grid.cell_index(
                                                                         static_cast<std::size_t>(ci),
                                                                         static_cast<std::size_t>(cj))]
                                                             .current_density[2];
                    ++count;
                }
            }
            rhs_host(node_index, 0)
                    = count == 0 ? 0.0
                                 : inputs.mu0 * accumulated_current_density_z / static_cast<double>(count);
        }
    }
    Kokkos::deep_copy(rhs, rhs_host);
    log_info(logger, "SimiLie right-hand side assembled on rectilinear nodes");

    [[maybe_unused]] LinearMagnetostaticsHamiltonian const hamiltonian(inputs.core_mu);
    physics::HamiltonEquations equations {hamiltonian};
    auto const operator_model = physics::make_stationary_equations_operator(
            equations,
            StructuredScalarPoissonStrongFormOperator2D<typename Kokkos::DefaultExecutionSpace::memory_space>(
                    x_coords,
                    y_coords));
    log_info(
            logger,
            solver_settings.use_matrix_free
                    ? "SimiLie starting matrix-free preconditioned conjugate-gradient solve"
                    : "SimiLie starting assembled-matrix Ginkgo preconditioned conjugate-gradient solve");
    result.solver_diagnostics = solvers::minimize_strong_formulation_residual(
            Kokkos::DefaultExecutionSpace(),
            operator_model,
            rhs,
            magnetic_vector_potential_z_view,
            solver_settings);
    log_info(
            logger,
            solver_settings.use_matrix_free
                    ? "SimiLie matrix-free preconditioned conjugate-gradient solve finished"
                    : "SimiLie assembled-matrix Ginkgo preconditioned conjugate-gradient solve finished");

    auto magnetic_vector_potential_z_host = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(),
            magnetic_vector_potential_z_view);
    std::vector<double> magnetic_vector_potential(3 * num_nodes, 0.0);
    for (std::size_t j = 0; j < grid.ny(); ++j) {
        for (std::size_t i = 0; i < grid.nx(); ++i) {
            std::size_t const node_index = grid.node_index(i, j);
            magnetic_vector_potential[3 * node_index + 2] = magnetic_vector_potential_z_host(node_index, 0);
        }
    }

    MagneticVectorPotentialToMagneticInduction curl_operator;
    std::vector<CellPostProcessFields> cell_outputs(result.num_cells);
    for (double value : magnetic_vector_potential) {
        result.max_abs_potential = std::max(result.max_abs_potential, std::abs(value));
    }
    log_info(logger, "SimiLie starting magnetostatics post-processing");

    auto node_value_z = [&](std::size_t i, std::size_t j) {
        return magnetic_vector_potential[3 * grid.node_index(i, j) + 2];
    };
    auto derivative_az_at_cell = [&](std::size_t i, std::size_t j, char axis) {
        if (axis == 'x') {
            double const dx = grid.x_coords[i + 1] - grid.x_coords[i];
            return ((node_value_z(i + 1, j) - node_value_z(i, j))
                    + (node_value_z(i + 1, j + 1) - node_value_z(i, j + 1)))
                   / (2.0 * dx);
        }
        double const dy = grid.y_coords[j + 1] - grid.y_coords[j];
        return ((node_value_z(i, j + 1) - node_value_z(i, j))
                + (node_value_z(i + 1, j + 1) - node_value_z(i + 1, j)))
               / (2.0 * dy);
    };

    for (std::size_t j = 0; j < grid.ncell_y(); ++j) {
        for (std::size_t i = 0; i < grid.ncell_x(); ++i) {
            std::size_t const cell_index = grid.cell_index(i, j);

            std::array<double, MagneticInductionIndex::access_size()> magnetic_induction_storage {};
            std::array<double, MagneticFieldIndex::access_size()> magnetic_field_storage {};
            auto magnetic_induction = detail::make_local_tensor<MagneticInductionIndex>(magnetic_induction_storage);
            auto magnetic_field = detail::make_local_tensor<MagneticFieldIndex>(magnetic_field_storage);

            curl_operator.forward(
                    magnetic_induction,
                    magnetic_field,
                    0.0,
                    derivative_az_at_cell(i, j, 'y'),
                    derivative_az_at_cell(i, j, 'x'),
                    0.0,
                    0.0,
                    0.0);

            LinearMagneticInductionToMagneticField constitutive_law(cell_inputs[cell_index].mu);
            constitutive_law.forward(magnetic_field, magnetic_induction);

            CellPostProcessFields cell_output {};
            cell_output.magnetic_induction = {
                    magnetic_induction(magnetic_induction.template access_element<Y, Z>()),
                    -magnetic_induction(magnetic_induction.template access_element<X, Z>()),
                    magnetic_induction(magnetic_induction.template access_element<X, Y>()),
            };
            cell_output.magnetic_field = {
                    magnetic_field(magnetic_field.template access_element<X>()),
                    magnetic_field(magnetic_field.template access_element<Y>()),
                    magnetic_field(magnetic_field.template access_element<Z>()),
            };
            double const half_trace = 0.5
                                      * (cell_output.magnetic_induction[0] * cell_output.magnetic_field[0]
                                         + cell_output.magnetic_induction[1] * cell_output.magnetic_field[1]
                                         + cell_output.magnetic_induction[2] * cell_output.magnetic_field[2]);
            cell_output.maxwell_stress = {
                    cell_output.magnetic_induction[0] * cell_output.magnetic_field[0] - half_trace,
                    cell_output.magnetic_induction[1] * cell_output.magnetic_field[1] - half_trace,
                    cell_output.magnetic_induction[2] * cell_output.magnetic_field[2] - half_trace,
                    cell_output.magnetic_induction[0] * cell_output.magnetic_field[1],
                    cell_output.magnetic_induction[0] * cell_output.magnetic_field[2],
                    cell_output.magnetic_induction[1] * cell_output.magnetic_field[2],
            };
            cell_outputs[cell_index] = cell_output;

            for (double value : cell_output.magnetic_induction) {
                result.max_abs_induction = std::max(result.max_abs_induction, std::abs(value));
            }
            for (double value : cell_output.magnetic_field) {
                result.max_abs_field = std::max(result.max_abs_field, std::abs(value));
            }
            if (grid.ordered_cells[cell_index].physical_tag == region_tags.air_gap_tag) {
                double const induction_magnitude = std::sqrt(
                        cell_output.magnetic_induction[0] * cell_output.magnetic_induction[0]
                        + cell_output.magnetic_induction[1] * cell_output.magnetic_induction[1]
                        + cell_output.magnetic_induction[2] * cell_output.magnetic_induction[2]);
                result.air_gap_induction_magnitude_sum += induction_magnitude;
                ++result.num_air_gap_cells;
            }
        }
    }

    std::vector<double> cell_x_coords(grid.ncell_x());
    std::vector<double> cell_y_coords(grid.ncell_y());
    for (std::size_t i = 0; i < grid.ncell_x(); ++i) {
        cell_x_coords[i] = grid.cell_center_x(i);
    }
    for (std::size_t j = 0; j < grid.ncell_y(); ++j) {
        cell_y_coords[j] = grid.cell_center_y(j);
    }

    auto stress_component = [&](std::size_t i, std::size_t j, std::size_t component) {
        return cell_outputs[grid.cell_index(i, j)].maxwell_stress[component];
    };
    for (std::size_t j = 0; j < grid.ncell_y(); ++j) {
        for (std::size_t i = 0; i < grid.ncell_x(); ++i) {
            auto derivative = [&](std::size_t component, char axis) {
                if (axis == 'x') {
                    return sil::mesher::gmsh::centered_first_derivative(
                            cell_x_coords,
                            [&](std::size_t index) {
                                std::size_t const clamped = std::min(index, grid.ncell_x() - 1);
                                return stress_component(clamped, j, component);
                            },
                            i);
                }
                return sil::mesher::gmsh::centered_first_derivative(
                        cell_y_coords,
                        [&](std::size_t index) {
                            std::size_t const clamped = std::min(index, grid.ncell_y() - 1);
                            return stress_component(i, clamped, component);
                        },
                        j);
            };
            std::array<double, 3>& force_density = cell_outputs[grid.cell_index(i, j)].force_density;
            force_density[0] = derivative(0, 'x') + derivative(3, 'y');
            force_density[1] = derivative(3, 'x') + derivative(1, 'y');
            force_density[2] = derivative(4, 'x') + derivative(5, 'y');
            result.force_density_magnitude_sum += std::sqrt(
                    force_density[0] * force_density[0] + force_density[1] * force_density[1]
                    + force_density[2] * force_density[2]);
        }
    }

    write_results_view(output_view_file, grid, cell_inputs, magnetic_vector_potential);
    log_info(logger, "SimiLie magnetostatics post-processing exported");
    return result;
}

template <class Logger>
StructuredLinearMagnetostaticsResult run_on_hexahedral_grid(
        std::filesystem::path const& output_view_file,
        StructuredLinearMagnetostaticsInputs const& inputs,
        StructuredLinearMagnetostaticsRegionTags const& region_tags,
        solvers::StrongFormulationSolverSettings const& solver_settings,
        sil::mesher::gmsh::HexahedralMesh const& mesh,
        Logger&& logger)
{
    sil::mesher::gmsh::StructuredGrid3D const grid = sil::mesher::gmsh::build_structured_grid(mesh);
    log_info(
            logger,
            "SimiLie structured rectilinear hexahedral mesh validated ("
                    + std::to_string(grid.ordered_nodes.size()) + " nodes, dimensions="
                    + std::to_string(grid.nx()) + "x" + std::to_string(grid.ny()) + "x"
                    + std::to_string(grid.nz()) + ")");

    StructuredLinearMagnetostaticsResult result;
    result.topology = "hexahedral";
    result.node_count = grid.ordered_nodes.size();
    result.mesh_dimensions = {grid.nx(), grid.ny(), grid.nz()};
    result.num_cells = grid.ncell_x() * grid.ncell_y() * grid.ncell_z();

    std::vector<CellInputFields> cell_inputs(result.num_cells);
    for (std::size_t cell_index = 0; cell_index < result.num_cells; ++cell_index) {
        CellInputFields field {
                .mu = inputs.mu0,
                .current_density = {0.0, 0.0, 0.0},
        };
        int const physical_tag = grid.ordered_cells[cell_index].physical_tag;
        if (physical_tag == region_tags.e_core_tag || physical_tag == region_tags.i_core_tag) {
            field.mu = inputs.core_mu;
            ++result.num_core_cells;
        } else if (physical_tag == region_tags.coil_left_tag) {
            field.current_density[2] = inputs.current_density_magnitude;
            ++result.num_coil_cells;
        } else if (physical_tag == region_tags.coil_right_tag) {
            field.current_density[2] = -inputs.current_density_magnitude;
            ++result.num_coil_cells;
        } else {
            ++result.num_air_cells;
        }
        cell_inputs[cell_index] = field;
    }

    Kokkos::View<double*> x_coords("similie_x_coords", grid.nx());
    Kokkos::View<double*> y_coords("similie_y_coords", grid.ny());
    auto x_coords_host = Kokkos::create_mirror_view(x_coords);
    auto y_coords_host = Kokkos::create_mirror_view(y_coords);
    for (std::size_t i = 0; i < grid.nx(); ++i) {
        x_coords_host(i) = grid.x_coords[i];
    }
    for (std::size_t j = 0; j < grid.ny(); ++j) {
        y_coords_host(j) = grid.y_coords[j];
    }
    Kokkos::deep_copy(x_coords, x_coords_host);
    Kokkos::deep_copy(y_coords, y_coords_host);

    std::size_t const num_xy_nodes = grid.nx() * grid.ny();
    Kokkos::View<double**> rhs("similie_rhs", num_xy_nodes, 1);
    Kokkos::View<double**> magnetic_vector_potential_z_xy_view("similie_Az", num_xy_nodes, 1);
    auto rhs_host = Kokkos::create_mirror_view(rhs);
    for (std::size_t j = 0; j < grid.ny(); ++j) {
        for (std::size_t i = 0; i < grid.nx(); ++i) {
            std::size_t const node_index_xy = i + grid.nx() * j;
            bool const boundary = (i == 0 || j == 0 || i + 1 == grid.nx() || j + 1 == grid.ny());
            if (boundary) {
                rhs_host(node_index_xy, 0) = 0.0;
                continue;
            }
            double accumulated_current_density_z = 0.0;
            std::size_t count = 0;
            for (int dj = -1; dj <= 0; ++dj) {
                for (int di = -1; di <= 0; ++di) {
                    std::ptrdiff_t const ci = static_cast<std::ptrdiff_t>(i) + di;
                    std::ptrdiff_t const cj = static_cast<std::ptrdiff_t>(j) + dj;
                    if (ci < 0 || cj < 0
                        || ci >= static_cast<std::ptrdiff_t>(grid.ncell_x())
                        || cj >= static_cast<std::ptrdiff_t>(grid.ncell_y())) {
                        continue;
                    }
                    double slice_sum = 0.0;
                    for (std::size_t ck = 0; ck < grid.ncell_z(); ++ck) {
                        slice_sum += cell_inputs[grid.cell_index(
                                                          static_cast<std::size_t>(ci),
                                                          static_cast<std::size_t>(cj),
                                                          ck)]
                                             .current_density[2];
                    }
                    accumulated_current_density_z += slice_sum / static_cast<double>(grid.ncell_z());
                    ++count;
                }
            }
            rhs_host(node_index_xy, 0)
                    = count == 0 ? 0.0
                                 : inputs.mu0 * accumulated_current_density_z / static_cast<double>(count);
        }
    }
    Kokkos::deep_copy(rhs, rhs_host);
    log_info(logger, "SimiLie right-hand side assembled on rectilinear nodes");

    [[maybe_unused]] LinearMagnetostaticsHamiltonian const hamiltonian(inputs.core_mu);
    physics::HamiltonEquations equations {hamiltonian};
    auto const operator_model = physics::make_stationary_equations_operator(
            equations,
            StructuredScalarPoissonStrongFormOperator2D<typename Kokkos::DefaultExecutionSpace::memory_space>(
                    x_coords,
                    y_coords));
    log_info(
            logger,
            solver_settings.use_matrix_free
                    ? "SimiLie starting matrix-free preconditioned conjugate-gradient solve"
                    : "SimiLie starting assembled-matrix Ginkgo preconditioned conjugate-gradient solve");
    result.solver_diagnostics = solvers::minimize_strong_formulation_residual(
            Kokkos::DefaultExecutionSpace(),
            operator_model,
            rhs,
            magnetic_vector_potential_z_xy_view,
            solver_settings);
    log_info(
            logger,
            solver_settings.use_matrix_free
                    ? "SimiLie matrix-free preconditioned conjugate-gradient solve finished"
                    : "SimiLie assembled-matrix Ginkgo preconditioned conjugate-gradient solve finished");

    std::size_t const num_nodes = grid.nx() * grid.ny() * grid.nz();
    auto magnetic_vector_potential_z_xy_host = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(),
            magnetic_vector_potential_z_xy_view);
    std::vector<double> magnetic_vector_potential(3 * num_nodes, 0.0);
    for (std::size_t k = 0; k < grid.nz(); ++k) {
        for (std::size_t j = 0; j < grid.ny(); ++j) {
            for (std::size_t i = 0; i < grid.nx(); ++i) {
                std::size_t const node_index = grid.node_index(i, j, k);
                std::size_t const node_index_xy = i + grid.nx() * j;
                magnetic_vector_potential[3 * node_index + 2]
                        = magnetic_vector_potential_z_xy_host(node_index_xy, 0);
            }
        }
    }

    MagneticVectorPotentialToMagneticInduction curl_operator;
    std::vector<CellPostProcessFields> cell_outputs(result.num_cells);
    for (double value : magnetic_vector_potential) {
        result.max_abs_potential = std::max(result.max_abs_potential, std::abs(value));
    }
    log_info(logger, "SimiLie starting magnetostatics post-processing");

    auto node_value_z = [&](std::size_t i, std::size_t j, std::size_t k) {
        return magnetic_vector_potential[3 * grid.node_index(i, j, k) + 2];
    };
    auto derivative_az_at_cell = [&](std::size_t i, std::size_t j, std::size_t k, char axis) {
        if (axis == 'x') {
            double const dx = grid.x_coords[i + 1] - grid.x_coords[i];
            return ((node_value_z(i + 1, j, k) - node_value_z(i, j, k))
                    + (node_value_z(i + 1, j + 1, k) - node_value_z(i, j + 1, k))
                    + (node_value_z(i + 1, j, k + 1) - node_value_z(i, j, k + 1))
                    + (node_value_z(i + 1, j + 1, k + 1) - node_value_z(i, j + 1, k + 1)))
                   / (4.0 * dx);
        }
        double const dy = grid.y_coords[j + 1] - grid.y_coords[j];
        return ((node_value_z(i, j + 1, k) - node_value_z(i, j, k))
                + (node_value_z(i + 1, j + 1, k) - node_value_z(i + 1, j, k))
                + (node_value_z(i, j + 1, k + 1) - node_value_z(i, j, k + 1))
                + (node_value_z(i + 1, j + 1, k + 1) - node_value_z(i + 1, j, k + 1)))
               / (4.0 * dy);
    };

    for (std::size_t k = 0; k < grid.ncell_z(); ++k) {
        for (std::size_t j = 0; j < grid.ncell_y(); ++j) {
            for (std::size_t i = 0; i < grid.ncell_x(); ++i) {
                std::size_t const cell_index = grid.cell_index(i, j, k);

                std::array<double, MagneticInductionIndex::access_size()> magnetic_induction_storage {};
                std::array<double, MagneticFieldIndex::access_size()> magnetic_field_storage {};
                auto magnetic_induction = detail::make_local_tensor<MagneticInductionIndex>(magnetic_induction_storage);
                auto magnetic_field = detail::make_local_tensor<MagneticFieldIndex>(magnetic_field_storage);

                curl_operator.forward(
                        magnetic_induction,
                        magnetic_field,
                        0.0,
                        derivative_az_at_cell(i, j, k, 'y'),
                        derivative_az_at_cell(i, j, k, 'x'),
                        0.0,
                        0.0,
                        0.0);

                LinearMagneticInductionToMagneticField constitutive_law(cell_inputs[cell_index].mu);
                constitutive_law.forward(magnetic_field, magnetic_induction);

                CellPostProcessFields cell_output {};
                cell_output.magnetic_induction = {
                        magnetic_induction(magnetic_induction.template access_element<Y, Z>()),
                        -magnetic_induction(magnetic_induction.template access_element<X, Z>()),
                        magnetic_induction(magnetic_induction.template access_element<X, Y>()),
                };
                cell_output.magnetic_field = {
                        magnetic_field(magnetic_field.template access_element<X>()),
                        magnetic_field(magnetic_field.template access_element<Y>()),
                        magnetic_field(magnetic_field.template access_element<Z>()),
                };
                double const half_trace = 0.5
                                          * (cell_output.magnetic_induction[0] * cell_output.magnetic_field[0]
                                             + cell_output.magnetic_induction[1] * cell_output.magnetic_field[1]
                                             + cell_output.magnetic_induction[2] * cell_output.magnetic_field[2]);
                cell_output.maxwell_stress = {
                        cell_output.magnetic_induction[0] * cell_output.magnetic_field[0] - half_trace,
                        cell_output.magnetic_induction[1] * cell_output.magnetic_field[1] - half_trace,
                        cell_output.magnetic_induction[2] * cell_output.magnetic_field[2] - half_trace,
                        cell_output.magnetic_induction[0] * cell_output.magnetic_field[1],
                        cell_output.magnetic_induction[0] * cell_output.magnetic_field[2],
                        cell_output.magnetic_induction[1] * cell_output.magnetic_field[2],
                };
                cell_outputs[cell_index] = cell_output;

                for (double value : cell_output.magnetic_induction) {
                    result.max_abs_induction = std::max(result.max_abs_induction, std::abs(value));
                }
                for (double value : cell_output.magnetic_field) {
                    result.max_abs_field = std::max(result.max_abs_field, std::abs(value));
                }
                if (grid.ordered_cells[cell_index].physical_tag == region_tags.air_gap_tag) {
                    double const induction_magnitude = std::sqrt(
                            cell_output.magnetic_induction[0] * cell_output.magnetic_induction[0]
                            + cell_output.magnetic_induction[1] * cell_output.magnetic_induction[1]
                            + cell_output.magnetic_induction[2] * cell_output.magnetic_induction[2]);
                    result.air_gap_induction_magnitude_sum += induction_magnitude;
                    ++result.num_air_gap_cells;
                }
            }
        }
    }

    std::vector<double> cell_x_coords(grid.ncell_x());
    std::vector<double> cell_y_coords(grid.ncell_y());
    std::vector<double> cell_z_coords(grid.ncell_z());
    for (std::size_t i = 0; i < grid.ncell_x(); ++i) {
        cell_x_coords[i] = grid.cell_center_x(i);
    }
    for (std::size_t j = 0; j < grid.ncell_y(); ++j) {
        cell_y_coords[j] = grid.cell_center_y(j);
    }
    for (std::size_t k = 0; k < grid.ncell_z(); ++k) {
        cell_z_coords[k] = grid.cell_center_z(k);
    }

    auto stress_component = [&](std::size_t i, std::size_t j, std::size_t k, std::size_t component) {
        return cell_outputs[grid.cell_index(i, j, k)].maxwell_stress[component];
    };
    for (std::size_t k = 0; k < grid.ncell_z(); ++k) {
        for (std::size_t j = 0; j < grid.ncell_y(); ++j) {
            for (std::size_t i = 0; i < grid.ncell_x(); ++i) {
                auto derivative = [&](std::size_t component, char axis) {
                    if (axis == 'x') {
                        return sil::mesher::gmsh::centered_first_derivative(
                                cell_x_coords,
                                [&](std::size_t index) {
                                    std::size_t const clamped = std::min(index, grid.ncell_x() - 1);
                                    return stress_component(clamped, j, k, component);
                                },
                                i);
                    }
                    if (axis == 'y') {
                        return sil::mesher::gmsh::centered_first_derivative(
                                cell_y_coords,
                                [&](std::size_t index) {
                                    std::size_t const clamped = std::min(index, grid.ncell_y() - 1);
                                    return stress_component(i, clamped, k, component);
                                },
                                j);
                    }
                    return sil::mesher::gmsh::centered_first_derivative(
                            cell_z_coords,
                            [&](std::size_t index) {
                                std::size_t const clamped = std::min(index, grid.ncell_z() - 1);
                                return stress_component(i, j, clamped, component);
                            },
                            k);
                };
                std::array<double, 3>& force_density = cell_outputs[grid.cell_index(i, j, k)].force_density;
                force_density[0] = derivative(0, 'x') + derivative(3, 'y') + derivative(4, 'z');
                force_density[1] = derivative(3, 'x') + derivative(1, 'y') + derivative(5, 'z');
                force_density[2] = derivative(4, 'x') + derivative(5, 'y') + derivative(2, 'z');
                result.force_density_magnitude_sum += std::sqrt(
                        force_density[0] * force_density[0] + force_density[1] * force_density[1]
                        + force_density[2] * force_density[2]);
            }
        }
    }

    write_results_view(output_view_file, grid, cell_inputs, magnetic_vector_potential);
    log_info(logger, "SimiLie magnetostatics post-processing exported");
    return result;
}

template <class Logger>
StructuredLinearMagnetostaticsResult run_structured_linear_magnetostatics_problem(
        std::filesystem::path const& mesh_file,
        std::filesystem::path const& output_view_file,
        StructuredLinearMagnetostaticsInputs const& inputs,
        StructuredLinearMagnetostaticsRegionTags const& region_tags,
        solvers::StrongFormulationSolverSettings const& solver_settings,
        Logger&& logger)
{
    sil::mesher::gmsh::SupportedMesh const mesh = sil::mesher::gmsh::parse_supported_msh2_mesh(mesh_file);
    if (std::holds_alternative<sil::mesher::gmsh::QuadrilateralMesh>(mesh)) {
        return run_on_quadrilateral_grid(
                output_view_file,
                inputs,
                region_tags,
                solver_settings,
                std::get<sil::mesher::gmsh::QuadrilateralMesh>(mesh),
                std::forward<Logger>(logger));
    }
    return run_on_hexahedral_grid(
            output_view_file,
            inputs,
            region_tags,
            solver_settings,
            std::get<sil::mesher::gmsh::HexahedralMesh>(mesh),
            std::forward<Logger>(logger));
}

} // namespace detail

} // namespace similie::physics::magnetostatics
