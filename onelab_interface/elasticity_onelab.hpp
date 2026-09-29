// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <ddc/ddc.hpp>

#include <ginkgo/core/base/matrix_data.hpp>
#include <similie/physics/elasticity/elastic_material_hodge_2d.hpp>
#include <similie/physics/elasticity/linear_elasticity.hpp>
#include <similie/physics/hamilton_equations.hpp>
#include <similie/solvers/minimize_strong_formulation_residual.hpp>
#include <similie/tensor/tensor.hpp>

#include <Kokkos_Core.hpp>

#include "gmsh_structured_grid.hpp"

namespace similie::onelab_interface::elasticity_onelab {

struct Inputs
{
    double young_modulus = 200.0e9;
    double poisson_ratio = 0.3;
    double thickness = 0.01;
    double applied_force = 100.0;
    std::vector<int> material_tags;
};

struct Result
{
    std::size_t node_count = 0;
    std::array<std::size_t, 3> mesh_dimensions {0, 0, 0};
    std::size_t num_cells = 0;
    std::size_t num_material_cells = 0;
    std::size_t num_clamped_nodes = 0;
    std::size_t num_loaded_nodes = 0;
    double max_displacement = 0.0;
    double probe_displacement_y = 0.0;
    double max_von_mises = 0.0;
    solvers::StrongFormulationSolverDiagnostics solver_diagnostics;
};

template <class Problem, class ProblemParameterName, class PublishString, class PublishNumber>
void synchronize_controls(
        Problem const&,
        ProblemParameterName&& problem_parameter_name,
        PublishString&& publish_or_sync_string,
        PublishNumber&& publish_or_sync_number)
{
    publish_or_sync_string(
            problem_parameter_name("2LinearElasticity", "0Preprocess"),
            "Preprocess",
            "Linear elasticity preprocessing strategy selected in the .silpro file.",
            "TwoDomainTransfiniteWrenchInterior",
            true);
    (void)publish_or_sync_number;
}

template <class Problem, class ReadNumberParameter, class ReadRequiredIntegerParameter>
Inputs read_inputs(
        Problem const& problem,
        ReadNumberParameter&& read_number_parameter,
        ReadRequiredIntegerParameter&& read_required_integer_parameter)
{
    Inputs inputs;
    // The .silpro controls use GPa and mm, independently of their magnitudes.
    inputs.young_modulus = 1.0e9
                           * read_number_parameter(
                                   problem.linear_elasticity.young_modulus_parameter,
                                   std::nullopt,
                                   inputs.young_modulus / 1.0e9);
    inputs.poisson_ratio = read_number_parameter(
            problem.linear_elasticity.poisson_ratio_parameter,
            std::nullopt,
            inputs.poisson_ratio);
    inputs.thickness = 1.0e-3
                       * read_number_parameter(
                               problem.linear_elasticity.thickness_parameter,
                               std::nullopt,
                               inputs.thickness / 1.0e-3);
    inputs.applied_force = read_number_parameter(
            problem.linear_elasticity.applied_force_parameter,
            std::nullopt,
            inputs.applied_force);

    for (std::string const& parameter_name : problem.linear_elasticity.material_tags) {
        inputs.material_tags.push_back(read_required_integer_parameter(parameter_name));
    }
    if (!(inputs.young_modulus > 0.0) || !std::isfinite(inputs.young_modulus)) {
        throw std::runtime_error("missing or invalid Young modulus ONELAB parameter");
    }
    if (!(inputs.poisson_ratio > -1.0 && inputs.poisson_ratio < 0.5)) {
        throw std::runtime_error("invalid Poisson coefficient for linear elasticity");
    }
    if (!(inputs.thickness > 0.0) || !std::isfinite(inputs.thickness)) {
        throw std::runtime_error("missing or invalid wrench thickness ONELAB parameter");
    }
    if (!std::isfinite(inputs.applied_force)) {
        throw std::runtime_error("invalid applied force");
    }
    if (inputs.material_tags.empty()) {
        inputs.material_tags.push_back(1);
    }
    return inputs;
}

template <class PublishOutputString, class PublishOutputNumber, class PublishStatus>
void publish_outputs(
        std::filesystem::path const& mesh_file,
        Inputs const& inputs,
        solvers::StrongFormulationSolverSettings const& solver_settings,
        Result const& result,
        PublishOutputString&& publish_output_string,
        PublishOutputNumber&& publish_output_number,
        PublishStatus&& publish_status)
{
    publish_output_string(
            "Mesh file",
            mesh_file.string(),
            "Mesh file",
            "Mesh file exported by Gmsh for the linear elasticity interface.",
            "file");
    publish_output_number(
            "Young modulus [Pa]",
            inputs.young_modulus,
            "Young modulus [Pa]",
            "Young modulus used by the intrinsic linear elasticity law.");
    publish_output_number(
            "Poisson coefficient",
            inputs.poisson_ratio,
            "Poisson coefficient",
            "Poisson coefficient used by the intrinsic linear elasticity law.");
    publish_output_number(
            "Applied force [N]",
            inputs.applied_force,
            "Applied force [N]",
            "Total downward force applied to the handle end.");
    publish_output_number(
            "Material cells",
            static_cast<double>(result.num_material_cells),
            "Material cells",
            "Number of structured cells in the meshed wrench interior.");
    publish_output_number(
            "Loaded nodes",
            static_cast<double>(result.num_loaded_nodes),
            "Loaded nodes",
            "Number of active nodes receiving the end load.");
    publish_output_number(
            "Solver iterations",
            static_cast<double>(result.solver_diagnostics.iterations),
            "Solver iterations",
            "Number of iterations performed by the strong-formulation solver.");
    publish_output_string(
            "Solver backend",
            solver_settings.use_matrix_free ? "matrix-free" : "assembled-matrix",
            "Solver backend",
            "Backend used by the stationary strong-formulation solver.",
            "generic");
    publish_output_number(
            "Final relative residual",
            result.solver_diagnostics.final_relative_residual,
            "Final relative residual",
            "Final residual divided by the initial residual.");
    publish_output_number(
            "Probe displacement y [mm]",
            1.0e3 * result.probe_displacement_y,
            "Probe displacement y [mm]",
            "Vertical displacement at the active node closest to the original wrench probe.");
    publish_output_number(
            "Maximum displacement [mm]",
            1.0e3 * result.max_displacement,
            "Maximum displacement [mm]",
            "Maximum displacement magnitude on active wrench nodes.");
    publish_output_number(
            "Maximum von Mises stress [Pa]",
            result.max_von_mises,
            "Maximum von Mises stress [Pa]",
            "Maximum von Mises value on active wrench cells.");
    publish_status("Linear elasticity solve completed");
}

namespace detail {

struct X
{
    static constexpr bool PERIODIC = false;
};

struct Y
{
    static constexpr bool PERIODIC = false;
};

struct DDimX
{
    using continuous_dimension_type = X;
    static constexpr bool PERIODIC = false;
};

struct DDimY
{
    using continuous_dimension_type = Y;
    static constexpr bool PERIODIC = false;
};

using PositionIndex2D = sil::tensor::Contravariant<sil::tensor::TensorNaturalIndex<X, Y>>;

inline bool has_tag(std::vector<int> const& tags, int physical_tag)
{
    return std::find(tags.begin(), tags.end(), physical_tag) != tags.end();
}

template <class Logger>
void log_info(Logger&& logger, std::string const& message)
{
    if constexpr (std::is_invocable_v<Logger, std::string const&>) {
        logger(message);
    }
}

struct CellFields
{
    double density = 0.0;
    physics::elasticity::Strain2D strain;
    physics::elasticity::CauchyStress2D stress;
};

template <class Equations>
[[nodiscard]] inline physics::elasticity::CauchyStress2D linear_elasticity_stress(
        Equations equations,
        physics::elasticity::Strain2D strain)
{
    int const elem = 0;
    return {
            .xx = equations.template dpotential_dt<physics::elasticity::StrainXX>(strain, elem),
            .yy = equations.template dpotential_dt<physics::elasticity::StrainYY>(strain, elem),
            .xy
            = 0.5 * equations.template dpotential_dt<physics::elasticity::StrainXY>(strain, elem),
    };
}

struct CurvilinearStructuredGrid2D
{
    std::size_t ncell_x = 0;
    std::size_t ncell_y = 0;
    std::vector<sil::onelab_interface::gmsh::MeshNode> ordered_nodes;
    std::vector<sil::onelab_interface::gmsh::QuadrilateralCell> ordered_cells;
    std::vector<int> active_nodes;
    std::vector<int> active_cells;

    [[nodiscard]] std::size_t nx() const
    {
        return ncell_x + 1;
    }

    [[nodiscard]] std::size_t ny() const
    {
        return ncell_y + 1;
    }

    [[nodiscard]] std::size_t node_index(std::size_t i, std::size_t j) const
    {
        return i + nx() * j;
    }

    [[nodiscard]] std::size_t cell_index(std::size_t i, std::size_t j) const
    {
        return i + ncell_x * j;
    }

    [[nodiscard]] double node_x(std::size_t i, std::size_t j) const
    {
        return ordered_nodes[node_index(i, j)].x;
    }

    [[nodiscard]] double node_y(std::size_t i, std::size_t j) const
    {
        return ordered_nodes[node_index(i, j)].y;
    }

    [[nodiscard]] double cell_center_x(std::size_t i, std::size_t j) const
    {
        return 0.25 * (node_x(i, j) + node_x(i + 1, j) + node_x(i, j + 1) + node_x(i + 1, j + 1));
    }

    [[nodiscard]] double cell_center_y(std::size_t i, std::size_t j) const
    {
        return 0.25 * (node_y(i, j) + node_y(i + 1, j) + node_y(i, j + 1) + node_y(i + 1, j + 1));
    }

    [[nodiscard]] bool has_node(std::size_t index) const
    {
        return active_nodes.empty() || active_nodes[index] != 0;
    }

    [[nodiscard]] bool has_cell(std::size_t index) const
    {
        return active_cells.empty() || active_cells[index] != 0;
    }
};

inline std::vector<std::pair<std::size_t, std::size_t>> structured_cell_dimension_candidates(
        std::size_t node_count,
        std::size_t cell_count)
{
    std::vector<std::pair<std::size_t, std::size_t>> candidates;
    for (std::size_t ncell_x = 1; ncell_x <= cell_count; ++ncell_x) {
        if (cell_count % ncell_x != 0) {
            continue;
        }
        std::size_t const ncell_y = cell_count / ncell_x;
        if ((ncell_x + 1) * (ncell_y + 1) == node_count) {
            candidates.emplace_back(ncell_x, ncell_y);
        }
    }
    if (candidates.empty()) {
        throw std::runtime_error(
                "failed to infer structured quadrilateral dimensions from the mesh");
    }
    return candidates;
}

inline CurvilinearStructuredGrid2D build_curvilinear_structured_grid(
        sil::onelab_interface::gmsh::QuadrilateralMesh const& mesh)
{
    std::map<std::size_t, sil::onelab_interface::gmsh::MeshNode> nodes_by_tag;
    for (auto const& node : mesh.nodes) {
        nodes_by_tag.emplace(node.tag, node);
    }
    std::map<std::size_t, bool> referenced_node_tags;
    for (auto const& cell : mesh.cells) {
        for (std::size_t node_tag : cell.node_tags) {
            referenced_node_tags[node_tag] = true;
        }
    }

    auto const candidates
            = structured_cell_dimension_candidates(referenced_node_tags.size(), mesh.cells.size());
    for (auto const& [ncell_x, ncell_y] : candidates) {
        CurvilinearStructuredGrid2D grid;
        grid.ncell_x = ncell_x;
        grid.ncell_y = ncell_y;
        grid.ordered_nodes.resize((ncell_x + 1) * (ncell_y + 1));
        grid.ordered_cells = mesh.cells;
        grid.active_nodes.assign(grid.ordered_nodes.size(), 1);
        grid.active_cells.assign(grid.ordered_cells.size(), 1);
        std::vector<std::size_t> assigned_node_tags(grid.ordered_nodes.size(), 0);

        bool consistent = true;
        auto assign_node = [&](std::size_t i, std::size_t j, std::size_t node_tag) {
            std::size_t const index = grid.node_index(i, j);
            if (assigned_node_tags[index] != 0 && assigned_node_tags[index] != node_tag) {
                consistent = false;
                return;
            }
            auto const node_it = nodes_by_tag.find(node_tag);
            if (node_it == nodes_by_tag.end()) {
                consistent = false;
                return;
            }
            assigned_node_tags[index] = node_tag;
            grid.ordered_nodes[index] = node_it->second;
        };

        for (std::size_t j = 0; consistent && j < ncell_y; ++j) {
            for (std::size_t i = 0; consistent && i < ncell_x; ++i) {
                auto const& cell = mesh.cells[grid.cell_index(i, j)];
                assign_node(i, j, cell.node_tags[0]);
                assign_node(i, j + 1, cell.node_tags[1]);
                assign_node(i + 1, j + 1, cell.node_tags[2]);
                assign_node(i + 1, j, cell.node_tags[3]);
            }
        }
        if (!consistent) {
            continue;
        }
        for (std::size_t tag : assigned_node_tags) {
            if (tag == 0) {
                consistent = false;
                break;
            }
        }
        if (consistent) {
            return grid;
        }
    }
    throw std::runtime_error("the quadrilateral mesh does not form a full transfinite grid");
}

template <class MemorySpace, class Equations>
class ElasticityOperator2D
{
    using view_type = Kokkos::View<double*, MemorySpace>;
    using int_view_type = Kokkos::View<int*, MemorySpace>;
    using columns_view_type = Kokkos::View<int**, Kokkos::LayoutRight, MemorySpace>;
    using coefficients_view_type = Kokkos::View<double**, Kokkos::LayoutRight, MemorySpace>;

    static constexpr int MAX_ROW_ENTRIES = 18;

    std::size_t m_nx = 0;
    std::size_t m_ny = 0;
    columns_view_type m_columns;
    coefficients_view_type m_coefficients;
    int_view_type m_counts;

public:
    static constexpr bool IS_LINEAR = true;
    static constexpr bool IS_SYMMETRIC = true;

    ElasticityOperator2D(
            std::size_t nx,
            std::size_t ny,
            view_type node_x,
            view_type node_y,
            Equations equations,
            int_view_type active,
            int_view_type dirichlet)
        : m_nx(nx)
        , m_ny(ny)
        , m_columns("similie_elasticity_flux_columns", 2 * nx * ny, MAX_ROW_ENTRIES)
        , m_coefficients("similie_elasticity_flux_coefficients", 2 * nx * ny, MAX_ROW_ENTRIES)
        , m_counts("similie_elasticity_flux_counts", 2 * nx * ny)
    {
        auto const x = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), node_x);
        auto const y = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), node_y);
        auto const active_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), active);
        auto const dirichlet_host
                = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), dirichlet);
        std::vector<std::map<std::size_t, double>> rows(2 * nx * ny);
        std::array<std::array<double, 8>, 8> incidence {};
        for (int input = 0; input < 8; ++input) {
            std::array<std::array<double, 2>, 4> basis {};
            basis[input / 2][input % 2] = 1.0;
            std::array<double, 8> const differences
                    = physics::elasticity::ElasticMaterialHodge2D::edge_differences<X, Y>(basis);
            for (int edge_component = 0; edge_component < 8; ++edge_component)
                incidence[edge_component][input] = differences[edge_component];
        }

        // Each mapped quadrilateral is split into four vertex control volumes.
        // The four internal half-faces carry one traction with opposite signs
        // in the two neighboring balances. Natural boundary tractions are RHS
        // data; zero traction contributes no stiffness on exterior half-faces.
        for (std::size_t j = 0; j + 1 < ny; ++j) {
            for (std::size_t i = 0; i + 1 < nx; ++i) {
                std::array<std::size_t, 4> const
                        nodes {i + nx * j, i + 1 + nx * j, i + nx * (j + 1), i + 1 + nx * (j + 1)};
                std::array<std::array<double, 2>, 4> positions {};
                for (std::size_t a = 0; a < 4; ++a)
                    positions[a] = {x(nodes[a]), y(nodes[a])};
                physics::elasticity::ElasticMaterialHodge2D const
                        material(positions, [&](physics::elasticity::Strain2D strain) {
                            return linear_elasticity_stress(equations, strain);
                        });
                std::array<std::array<double, 8>, 8> force_from_nodal {};
                for (int output_edge = 0; output_edge < 8; ++output_edge)
                    for (int input = 0; input < 8; ++input)
                        for (int input_edge = 0; input_edge < 8; ++input_edge)
                            force_from_nodal[output_edge][input]
                                    += material.matrix()[output_edge][input_edge]
                                       * incidence[input_edge][input];
                for (int output = 0; output < 8; ++output)
                    for (int input = 0; input < 8; ++input) {
                        double coefficient = 0.0;
                        for (int output_edge = 0; output_edge < 8; ++output_edge)
                            coefficient += incidence[output_edge][output]
                                           * force_from_nodal[output_edge][input];
                        rows[2 * nodes[output / 2] + output % 2][2 * nodes[input / 2] + input % 2]
                                += coefficient;
                    }
            }
        }

        auto columns_host = Kokkos::create_mirror_view(m_columns);
        auto coefficients_host = Kokkos::create_mirror_view(m_coefficients);
        auto counts_host = Kokkos::create_mirror_view(m_counts);
        for (std::size_t row = 0; row < rows.size(); ++row) {
            int count = 0;
            if (active_host(row / 2) == 0 || dirichlet_host(row / 2) != 0) {
                columns_host(row, count) = static_cast<int>(row);
                coefficients_host(row, count++) = 1.0;
            } else {
                for (auto const& [column, coefficient] : rows[row]) {
                    if (coefficient == 0.0 || active_host(column / 2) == 0
                        || dirichlet_host(column / 2) != 0)
                        continue;
                    if (count == MAX_ROW_ENTRIES)
                        throw std::runtime_error("elasticity flux stencil capacity exceeded");
                    columns_host(row, count) = static_cast<int>(column);
                    coefficients_host(row, count++) = coefficient;
                }
            }
            counts_host(row) = count;
        }
        Kokkos::deep_copy(m_columns, columns_host);
        Kokkos::deep_copy(m_coefficients, coefficients_host);
        Kokkos::deep_copy(m_counts, counts_host);
    }

    [[nodiscard]] std::size_t size() const
    {
        return 2 * m_nx * m_ny;
    }

    template <class ExecSpace, class InputView, class OutputView>
    void apply(ExecSpace exec_space, InputView input, OutputView output) const
    {
        auto const columns = m_columns;
        auto const coefficients = m_coefficients;
        auto const counts = m_counts;
        Kokkos::parallel_for(
                "similie_elasticity_flux_balance",
                Kokkos::RangePolicy<ExecSpace>(exec_space, 0, size()),
                KOKKOS_LAMBDA(std::size_t row) {
                    double balance = 0.0;
                    for (int slot = 0; slot < counts(row); ++slot)
                        balance += coefficients(row, slot) * input(columns(row, slot), 0);
                    output(row, 0) = balance;
                });
        exec_space.fence();
    }

    [[nodiscard]] auto columns() const
    {
        return m_columns;
    }
    [[nodiscard]] auto coefficients() const
    {
        return m_coefficients;
    }
    [[nodiscard]] auto counts() const
    {
        return m_counts;
    }
};

template <class MemorySpace, class Equations>
gko::matrix_data<double, gko::int32> assemble_matrix_data(
        ElasticityOperator2D<MemorySpace, Equations> const& operator_model)
{
    gko::matrix_data<double, gko::int32> matrix_data(
            gko::dim<2>(operator_model.size(), operator_model.size()));
    auto const columns
            = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), operator_model.columns());
    auto const coefficients = Kokkos::
            create_mirror_view_and_copy(Kokkos::HostSpace(), operator_model.coefficients());
    auto const counts
            = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), operator_model.counts());
    for (std::size_t row = 0; row < operator_model.size(); ++row) {
        for (int slot = 0; slot < counts(row); ++slot)
            matrix_data.nonzeros.emplace_back(
                    static_cast<gko::int32>(row),
                    static_cast<gko::int32>(columns(row, slot)),
                    coefficients(row, slot));
    }
    return matrix_data;
}

inline void write_results_view(
        std::filesystem::path const& output_view_file,
        CurvilinearStructuredGrid2D const& grid,
        std::vector<CellFields> const& cell_fields,
        std::vector<double> const& displacement)
{
    std::ofstream stream(output_view_file);
    if (!stream.is_open()) {
        throw std::runtime_error("failed to open output view file: " + output_view_file.string());
    }
    stream << std::setprecision(17);
    stream << "View \"SimiLie linear elasticity displacement\" {\n";
    for (std::size_t j = 0; j < grid.ny(); ++j) {
        for (std::size_t i = 0; i < grid.nx(); ++i) {
            std::size_t const node = grid.node_index(i, j);
            stream << "VP(" << grid.node_x(i, j) << "," << grid.node_y(i, j) << ",0" << "){"
                   << displacement[2 * node] << "," << displacement[2 * node + 1] << ",0};\n";
        }
    }
    stream << "};\n";

    auto write_scalar_cell_view = [&](std::string const& name, auto value) {
        stream << "View \"" << name << "\" {\n";
        for (std::size_t j = 0; j < grid.ncell_y; ++j) {
            for (std::size_t i = 0; i < grid.ncell_x; ++i) {
                CellFields const& fields = cell_fields[grid.cell_index(i, j)];
                stream << "SP(" << grid.cell_center_x(i, j) << "," << grid.cell_center_y(i, j)
                       << ",0){" << value(fields) << "};\n";
            }
        }
        stream << "};\n";
    };
    write_scalar_cell_view("SimiLie linear elasticity material density", [](CellFields const& f) {
        return f.density;
    });
    write_scalar_cell_view("SimiLie linear elasticity stress xx", [](CellFields const& f) {
        return f.stress.xx;
    });
    write_scalar_cell_view("SimiLie linear elasticity stress yy", [](CellFields const& f) {
        return f.stress.yy;
    });
    write_scalar_cell_view("SimiLie linear elasticity stress xy", [](CellFields const& f) {
        return f.stress.xy;
    });
    write_scalar_cell_view("SimiLie linear elasticity von Mises", [](CellFields const& f) {
        return f.stress.von_mises();
    });
}

} // namespace detail

template <class Logger>
Result run_on_quadrilateral_grid(
        std::filesystem::path const& output_view_file,
        Inputs const& inputs,
        solvers::StrongFormulationSolverSettings const& solver_settings,
        sil::onelab_interface::gmsh::QuadrilateralMesh const& mesh,
        Logger&& logger)
{
    auto const grid = detail::build_curvilinear_structured_grid(mesh);
    detail::log_info(
            logger,
            "SimiLie transfinite quadrilateral wrench mesh validated for elasticity ("
                    + std::to_string(grid.ordered_nodes.size()) + " nodes, dimensions="
                    + std::to_string(grid.nx()) + "x" + std::to_string(grid.ny()) + ")");

    Result result;
    result.node_count = grid.ordered_nodes.size();
    result.mesh_dimensions = {grid.nx(), grid.ny(), 1};
    result.num_cells = grid.ncell_x * grid.ncell_y;

    std::vector<detail::CellFields> cell_fields(result.num_cells);
    for (std::size_t cell_index = 0; cell_index < result.num_cells; ++cell_index) {
        int const tag = grid.ordered_cells[cell_index].physical_tag;
        bool const material = detail::has_tag(inputs.material_tags, tag);
        if (!material) {
            throw std::runtime_error("the wrench interior mesh contains a non-material cell");
        }
        cell_fields[cell_index].density = 1.0;
        ++result.num_material_cells;
    }

    using memory_space = typename Kokkos::DefaultExecutionSpace::memory_space;
    Kokkos::View<int*, memory_space> active("similie_elasticity_active", grid.nx() * grid.ny());
    Kokkos::View<int*, memory_space>
            dirichlet("similie_elasticity_dirichlet", grid.nx() * grid.ny());
    Kokkos::View<double*, memory_space> node_x("similie_elasticity_node_x", grid.nx() * grid.ny());
    Kokkos::View<double*, memory_space> node_y("similie_elasticity_node_y", grid.nx() * grid.ny());
    auto active_host = Kokkos::create_mirror_view(active);
    auto dirichlet_host = Kokkos::create_mirror_view(dirichlet);
    auto node_x_host = Kokkos::create_mirror_view(node_x);
    auto node_y_host = Kokkos::create_mirror_view(node_y);
    std::map<std::size_t, std::size_t> node_by_tag;
    for (std::size_t node = 0; node < grid.ordered_nodes.size(); ++node) {
        node_by_tag.emplace(grid.ordered_nodes[node].tag, node);
    }
    std::vector<double> load_weights(grid.ordered_nodes.size(), 0.0);
    double loaded_length = 0.0;
    for (auto const& edge : mesh.boundary_edges) {
        if (edge.physical_tag != 2 && edge.physical_tag != 3) {
            continue;
        }
        auto const a = node_by_tag.at(edge.node_tags[0]);
        auto const b = node_by_tag.at(edge.node_tags[1]);
        if (edge.physical_tag == 2) {
            dirichlet_host(a) = dirichlet_host(b) = 1;
        } else {
            double const length = std::
                    hypot(grid.ordered_nodes[a].x - grid.ordered_nodes[b].x,
                          grid.ordered_nodes[a].y - grid.ordered_nodes[b].y);
            loaded_length += length;
            load_weights[a] += 0.5 * length;
            load_weights[b] += 0.5 * length;
        }
    }
    for (std::size_t j = 0; j < grid.ny(); ++j) {
        for (std::size_t i = 0; i < grid.nx(); ++i) {
            std::size_t const node = grid.node_index(i, j);
            active_host(node) = 1;
            double const x = grid.node_x(i, j);
            double const y = grid.node_y(i, j);
            node_x_host(node) = x;
            node_y_host(node) = y;
            result.num_clamped_nodes += dirichlet_host(node) != 0;
            result.num_loaded_nodes += load_weights[node] > 0.0;
        }
    }
    if (result.num_clamped_nodes == 0 || !(loaded_length > 0.0)) {
        throw std::runtime_error("missing physical clamp (2) or load (3) boundary");
    }
    Kokkos::deep_copy(active, active_host);
    Kokkos::deep_copy(dirichlet, dirichlet_host);
    Kokkos::deep_copy(node_x, node_x_host);
    Kokkos::deep_copy(node_y, node_y_host);

    physics::elasticity::LinearElasticityHamiltonian<> const
            hamiltonian(inputs.young_modulus, inputs.poisson_ratio);
    auto const equations = physics::HamiltonEquations {hamiltonian};

    Kokkos::View<double**> rhs("similie_elasticity_rhs", 2 * grid.nx() * grid.ny(), 1);
    Kokkos::View<double**>
            displacement_view("similie_elasticity_displacement", 2 * grid.nx() * grid.ny(), 1);
    auto rhs_host = Kokkos::create_mirror_view(rhs);
    // The dual-face tractions are forces per unit thickness in this 2D model.
    // Distribute the prescribed boundary force over its length on the same scale.
    for (std::size_t node = 0; node < load_weights.size(); ++node) {
        rhs_host(2 * node + 1, 0)
                = -inputs.applied_force * load_weights[node] / (inputs.thickness * loaded_length);
    }
    Kokkos::deep_copy(rhs, rhs_host);

    detail::ElasticityOperator2D<memory_space, decltype(equations)> const
            operator_model(grid.nx(), grid.ny(), node_x, node_y, equations, active, dirichlet);

    detail::log_info(logger, "SimiLie starting linear elasticity solve");
    result.solver_diagnostics = solvers::minimize_strong_formulation_residual(
            Kokkos::DefaultExecutionSpace(),
            operator_model,
            rhs,
            displacement_view,
            solver_settings);
    if (!std::isfinite(result.solver_diagnostics.final_relative_residual)
        || result.solver_diagnostics.final_relative_residual > solver_settings.relative_tolerance) {
        throw std::runtime_error(
                "linear elasticity solver did not reach the requested residual tolerance");
    }
    detail::log_info(logger, "SimiLie linear elasticity solve finished");

    auto displacement_host
            = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), displacement_view);
    std::vector<double> displacement(2 * grid.nx() * grid.ny(), 0.0);
    for (std::size_t node = 0; node < grid.nx() * grid.ny(); ++node) {
        displacement[2 * node] = displacement_host(2 * node, 0);
        displacement[2 * node + 1] = displacement_host(2 * node + 1, 0);
        if (active_host(node) != 0) {
            result.max_displacement = std::
                    max(result.max_displacement,
                        std::hypot(displacement[2 * node], displacement[2 * node + 1]));
        }
    }

    std::size_t const probe_node = grid.node_index(grid.ncell_x / 2, grid.ncell_y);
    result.probe_displacement_y = displacement[2 * probe_node + 1];

    double constexpr material_threshold = 0.1;
    for (std::size_t j = 0; j < grid.ncell_y; ++j) {
        for (std::size_t i = 0; i < grid.ncell_x; ++i) {
            detail::CellFields& fields = cell_fields[grid.cell_index(i, j)];
            std::array<std::size_t, 4> const
                    nodes {grid.node_index(i, j),
                           grid.node_index(i + 1, j),
                           grid.node_index(i, j + 1),
                           grid.node_index(i + 1, j + 1)};
            std::array<std::array<double, 2>, 4> positions;
            for (int a = 0; a < 4; ++a) {
                positions[a] = {grid.ordered_nodes[nodes[a]].x, grid.ordered_nodes[nodes[a]].y};
            }
            physics::elasticity::ElasticMaterialHodge2D const
                    material(positions, [&](physics::elasticity::Strain2D strain) {
                        return detail::linear_elasticity_stress(equations, strain);
                    });
            std::array<std::array<double, 2>, 4> nodal_displacement {};
            for (int a = 0; a < 4; ++a)
                nodal_displacement[a]
                        = {displacement[2 * nodes[a]], displacement[2 * nodes[a] + 1]};
            std::array<double, 4> const gradient = material.recover_gradient(
                    physics::elasticity::ElasticMaterialHodge2D::
                            edge_differences<detail::X, detail::Y>(nodal_displacement));
            fields.strain = physics::elasticity::DisplacementToStrain::
                    from_gradient(gradient[0], gradient[3], gradient[1], gradient[2]);
            fields.stress = detail::linear_elasticity_stress(equations, fields.strain);
            fields.stress.xx *= fields.density;
            fields.stress.yy *= fields.density;
            fields.stress.xy *= fields.density;
            if (fields.density > material_threshold) {
                result.max_von_mises = std::max(result.max_von_mises, fields.stress.von_mises());
            }
        }
    }

    detail::write_results_view(output_view_file, grid, cell_fields, displacement);
    detail::log_info(logger, "SimiLie linear elasticity post-processing exported");
    return result;
}

template <class Logger>
Result run(
        std::filesystem::path const& mesh_file,
        std::filesystem::path const& output_view_file,
        Inputs const& inputs,
        solvers::StrongFormulationSolverSettings const& solver_settings,
        Logger&& logger)
{
    auto const mesh = sil::onelab_interface::gmsh::parse_supported_msh2_mesh(mesh_file);
    if (!std::holds_alternative<sil::onelab_interface::gmsh::QuadrilateralMesh>(mesh)) {
        throw std::runtime_error(
                "the current linear elasticity example expects a 2D quadrilateral grid");
    }
    return run_on_quadrilateral_grid(
            output_view_file,
            inputs,
            solver_settings,
            std::get<sil::onelab_interface::gmsh::QuadrilateralMesh>(mesh),
            std::forward<Logger>(logger));
}

} // namespace similie::onelab_interface::elasticity_onelab
