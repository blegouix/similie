// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <ginkgo/core/base/matrix_data.hpp>
#include <similie/physics/elasticity/linear_elasticity.hpp>
#include <similie/physics/elasticity/linear_elasticity_constitutive_law.hpp>
#include <similie/solvers/minimize_strong_formulation_residual.hpp>

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
    inputs.young_modulus = read_number_parameter(
            problem.linear_elasticity.young_modulus_parameter,
            std::nullopt,
            inputs.young_modulus);
    if (inputs.young_modulus < 1.0e7) {
        inputs.young_modulus *= 1.0e9;
    }
    inputs.poisson_ratio = read_number_parameter(
            problem.linear_elasticity.poisson_ratio_parameter,
            std::nullopt,
            inputs.poisson_ratio);
    inputs.thickness = read_number_parameter(
            problem.linear_elasticity.thickness_parameter,
            std::nullopt,
            inputs.thickness);
    if (inputs.thickness > 1.0) {
        inputs.thickness *= 1.0e-3;
    }
    inputs.applied_force = read_number_parameter(
            problem.linear_elasticity.applied_force_parameter,
            std::nullopt,
            inputs.applied_force);

    for (std::string const& parameter_name : problem.linear_elasticity.material_tags) {
        inputs.material_tags.push_back(read_required_integer_parameter(parameter_name));
    }
    if (!(inputs.young_modulus > 0.0)) {
        throw std::runtime_error("missing or invalid Young modulus ONELAB parameter");
    }
    if (!(inputs.poisson_ratio > -1.0 && inputs.poisson_ratio < 0.5)) {
        throw std::runtime_error("invalid Poisson coefficient for linear elasticity");
    }
    if (!(inputs.thickness > 0.0)) {
        throw std::runtime_error("missing or invalid wrench thickness ONELAB parameter");
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
            "Young modulus used by the plane-stress Hooke law.");
    publish_output_number(
            "Poisson coefficient",
            inputs.poisson_ratio,
            "Poisson coefficient",
            "Poisson coefficient used by the plane-stress Hooke law.");
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
            "Maximum plane-stress von Mises value on active wrench cells.");
    publish_status("Linear elasticity solve completed");
}

namespace detail {

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
    physics::elasticity::SmallStrain2D strain;
    physics::elasticity::CauchyStress2D stress;
};

struct ElasticityMaterialCoefficients
{
    double c11 = 1.0;
    double c12 = 0.0;
    double c66 = 1.0;
};

[[nodiscard]] inline ElasticityMaterialCoefficients material_coefficients(
        double young_modulus,
        double poisson_ratio)
{
    physics::elasticity::LinearElasticityHamiltonian<> const
            hamiltonian(young_modulus, poisson_ratio);
    physics::elasticity::SmallStrain2D const zero_strain {};
    int const elem = 0;
    return {
            .c11 = hamiltonian.template jacobian<
                    physics::elasticity::StrainXX,
                    physics::elasticity::StrainXX>(zero_strain, elem),
            .c12 = hamiltonian.template jacobian<
                    physics::elasticity::StrainXX,
                    physics::elasticity::StrainYY>(zero_strain, elem),
            .c66 = 0.25
                   * hamiltonian.template jacobian<
                           physics::elasticity::StrainXY,
                           physics::elasticity::StrainXY>(zero_strain, elem),
    };
}

[[nodiscard]] inline physics::elasticity::CauchyStress2D hooke_plane_stress(
        ElasticityMaterialCoefficients coefficients,
        physics::elasticity::SmallStrain2D strain)
{
    physics::elasticity::LinearElasticStrainToStress const
            normal_x(2.0 * coefficients.c66, coefficients.c12);
    physics::elasticity::LinearElasticStrainToStress const
            normal_y(2.0 * coefficients.c66, coefficients.c12);
    physics::elasticity::LinearElasticStrainToStress const shear(2.0 * coefficients.c66, 0.0);
    double const trace_strain = strain.xx + strain.yy;
    return {
            .xx = normal_x(trace_strain, strain.xx),
            .yy = normal_y(trace_strain, strain.yy),
            .xy = shear(trace_strain, strain.xy),
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

inline double distance(
        sil::onelab_interface::gmsh::MeshNode const& lhs,
        sil::onelab_interface::gmsh::MeshNode const& rhs)
{
    double const dx = lhs.x - rhs.x;
    double const dy = lhs.y - rhs.y;
    double const dz = lhs.z - rhs.z;
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

inline std::pair<double, double> average_logical_spacings(CurvilinearStructuredGrid2D const& grid)
{
    double sum_x = 0.0;
    double sum_y = 0.0;
    for (std::size_t j = 0; j < grid.ny(); ++j) {
        for (std::size_t i = 0; i < grid.ncell_x; ++i) {
            sum_x += distance(
                    grid.ordered_nodes[grid.node_index(i, j)],
                    grid.ordered_nodes[grid.node_index(i + 1, j)]);
        }
    }
    for (std::size_t j = 0; j < grid.ncell_y; ++j) {
        for (std::size_t i = 0; i < grid.nx(); ++i) {
            sum_y += distance(
                    grid.ordered_nodes[grid.node_index(i, j)],
                    grid.ordered_nodes[grid.node_index(i, j + 1)]);
        }
    }
    return {
            sum_x / static_cast<double>(grid.ncell_x * grid.ny()),
            sum_y / static_cast<double>(grid.nx() * grid.ncell_y),
    };
}

template <class MemorySpace>
class ElasticityOperator2D
{
    using view_type = Kokkos::View<double*, MemorySpace>;
    using int_view_type = Kokkos::View<int*, MemorySpace>;

    std::size_t m_nx = 0;
    std::size_t m_ny = 0;
    double m_hx = 1.0;
    double m_hy = 1.0;
    double m_c11 = 1.0;
    double m_c12_plus_c66 = 1.0;
    double m_c66 = 1.0;
    int_view_type m_active;
    int_view_type m_dirichlet;
    view_type m_density;

public:
    static constexpr bool IS_LINEAR = true;

    ElasticityOperator2D(
            std::size_t nx,
            std::size_t ny,
            double hx,
            double hy,
            ElasticityMaterialCoefficients material,
            int_view_type active,
            int_view_type dirichlet,
            view_type density)
        : m_nx(nx)
        , m_ny(ny)
        , m_hx(hx)
        , m_hy(hy)
        , m_c11(material.c11)
        , m_c12_plus_c66(material.c12 + material.c66)
        , m_c66(material.c66)
        , m_active(active)
        , m_dirichlet(dirichlet)
        , m_density(density)
    {
    }

    [[nodiscard]] KOKKOS_INLINE_FUNCTION std::size_t size() const
    {
        return 2 * m_nx * m_ny;
    }

    template <class ExecSpace, class InputView, class OutputView>
    void apply(ExecSpace exec_space, InputView input, OutputView output) const
    {
        std::size_t const nx = m_nx;
        std::size_t const ny = m_ny;
        double const inv_hx2 = 1.0 / (m_hx * m_hx);
        double const inv_hy2 = 1.0 / (m_hy * m_hy);
        double const inv_4hxhy = 1.0 / (4.0 * m_hx * m_hy);
        double const c11 = m_c11;
        double const c66 = m_c66;
        double const c12_plus_c66 = m_c12_plus_c66;
        auto const active = m_active;
        auto const dirichlet = m_dirichlet;
        auto const density = m_density;

        Kokkos::parallel_for(
                "similie_elasticity_operator_apply",
                Kokkos::RangePolicy<ExecSpace>(exec_space, 0, nx * ny),
                KOKKOS_LAMBDA(std::size_t node) {
                    std::size_t const i = node % nx;
                    std::size_t const j = node / nx;
                    std::size_t const row_x = 2 * node;
                    std::size_t const row_y = row_x + 1;
                    if (active(node) == 0 || dirichlet(node) != 0) {
                        output(row_x, 0) = input(row_x, 0);
                        output(row_y, 0) = input(row_y, 0);
                        return;
                    }

                    double ux = 0.0;
                    double uy = 0.0;
                    auto add_same = [&](std::ptrdiff_t ii,
                                        std::ptrdiff_t jj,
                                        double coeff_x,
                                        double coeff_y) {
                        if (ii < 0 || jj < 0 || ii >= static_cast<std::ptrdiff_t>(nx)
                            || jj >= static_cast<std::ptrdiff_t>(ny)) {
                            ux += coeff_x * input(row_x, 0);
                            uy += coeff_y * input(row_y, 0);
                            return;
                        }
                        std::size_t const neighbor
                                = static_cast<std::size_t>(ii) + nx * static_cast<std::size_t>(jj);
                        if (active(neighbor) == 0) {
                            return;
                        }
                        ux += coeff_x * input(2 * neighbor, 0);
                        uy += coeff_y * input(2 * neighbor + 1, 0);
                    };
                    auto add_cross = [&](std::ptrdiff_t ii,
                                         std::ptrdiff_t jj,
                                         double coeff_for_x_row,
                                         double coeff_for_y_row) {
                        if (ii < 0 || jj < 0 || ii >= static_cast<std::ptrdiff_t>(nx)
                            || jj >= static_cast<std::ptrdiff_t>(ny)) {
                            return;
                        }
                        std::size_t const neighbor
                                = static_cast<std::size_t>(ii) + nx * static_cast<std::size_t>(jj);
                        if (active(neighbor) == 0) {
                            return;
                        }
                        ux += coeff_for_x_row * input(2 * neighbor + 1, 0);
                        uy += coeff_for_y_row * input(2 * neighbor, 0);
                    };

                    double const w = density(node);
                    double const ax = w * c11 * inv_hx2;
                    double const ay = w * c66 * inv_hy2;
                    double const bx = w * c66 * inv_hx2;
                    double const by = w * c11 * inv_hy2;
                    ux += 2.0 * (ax + ay) * input(row_x, 0);
                    uy += 2.0 * (bx + by) * input(row_y, 0);
                    add_same(static_cast<std::ptrdiff_t>(i) - 1, j, -ax, -bx);
                    add_same(static_cast<std::ptrdiff_t>(i) + 1, j, -ax, -bx);
                    add_same(i, static_cast<std::ptrdiff_t>(j) - 1, -ay, -by);
                    add_same(i, static_cast<std::ptrdiff_t>(j) + 1, -ay, -by);

                    double const cross = -w * c12_plus_c66 * inv_4hxhy;
                    add_cross(
                            static_cast<std::ptrdiff_t>(i) + 1,
                            static_cast<std::ptrdiff_t>(j) + 1,
                            cross,
                            cross);
                    add_cross(
                            static_cast<std::ptrdiff_t>(i) - 1,
                            static_cast<std::ptrdiff_t>(j) - 1,
                            cross,
                            cross);
                    add_cross(
                            static_cast<std::ptrdiff_t>(i) - 1,
                            static_cast<std::ptrdiff_t>(j) + 1,
                            -cross,
                            -cross);
                    add_cross(
                            static_cast<std::ptrdiff_t>(i) + 1,
                            static_cast<std::ptrdiff_t>(j) - 1,
                            -cross,
                            -cross);

                    output(row_x, 0) = ux;
                    output(row_y, 0) = uy;
                });
        exec_space.fence();
    }

    [[nodiscard]] auto active() const
    {
        return m_active;
    }

    [[nodiscard]] auto dirichlet() const
    {
        return m_dirichlet;
    }

    [[nodiscard]] auto density() const
    {
        return m_density;
    }

    [[nodiscard]] double hx() const
    {
        return m_hx;
    }

    [[nodiscard]] double hy() const
    {
        return m_hy;
    }

    [[nodiscard]] double c11() const
    {
        return m_c11;
    }

    [[nodiscard]] double c66() const
    {
        return m_c66;
    }

    [[nodiscard]] double c12_plus_c66() const
    {
        return m_c12_plus_c66;
    }

    [[nodiscard]] std::size_t nx() const
    {
        return m_nx;
    }

    [[nodiscard]] std::size_t ny() const
    {
        return m_ny;
    }
};

template <class MemorySpace>
gko::matrix_data<double, gko::int32> assemble_matrix_data(
        ElasticityOperator2D<MemorySpace> const& operator_model)
{
    gko::matrix_data<double, gko::int32> matrix_data(
            gko::dim<2>(operator_model.size(), operator_model.size()));
    auto const active_host
            = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), operator_model.active());
    auto const dirichlet_host
            = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), operator_model.dirichlet());
    auto const density_host
            = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), operator_model.density());
    double const inv_hx2 = 1.0 / (operator_model.hx() * operator_model.hx());
    double const inv_hy2 = 1.0 / (operator_model.hy() * operator_model.hy());
    double const inv_4hxhy = 1.0 / (4.0 * operator_model.hx() * operator_model.hy());
    for (std::size_t j = 0; j < operator_model.ny(); ++j) {
        for (std::size_t i = 0; i < operator_model.nx(); ++i) {
            std::size_t const node = i + operator_model.nx() * j;
            std::size_t const row_x = 2 * node;
            std::size_t const row_y = row_x + 1;
            if (active_host(node) == 0 || dirichlet_host(node) != 0) {
                matrix_data.nonzeros.emplace_back(row_x, row_x, 1.0);
                matrix_data.nonzeros.emplace_back(row_y, row_y, 1.0);
                continue;
            }
            double const w = density_host(node);
            double const ax = w * operator_model.c11() * inv_hx2;
            double const ay = w * operator_model.c66() * inv_hy2;
            double const bx = w * operator_model.c66() * inv_hx2;
            double const by = w * operator_model.c11() * inv_hy2;
            matrix_data.nonzeros.emplace_back(row_x, row_x, 2.0 * (ax + ay));
            matrix_data.nonzeros.emplace_back(row_y, row_y, 2.0 * (bx + by));

            auto add_same = [&](std::ptrdiff_t ii, std::ptrdiff_t jj, double cx, double cy) {
                if (ii < 0 || jj < 0 || ii >= static_cast<std::ptrdiff_t>(operator_model.nx())
                    || jj >= static_cast<std::ptrdiff_t>(operator_model.ny())) {
                    return;
                }
                std::size_t const neighbor = static_cast<std::size_t>(ii)
                                             + operator_model.nx() * static_cast<std::size_t>(jj);
                if (active_host(neighbor) == 0) {
                    return;
                }
                matrix_data.nonzeros.emplace_back(row_x, 2 * neighbor, cx);
                matrix_data.nonzeros.emplace_back(row_y, 2 * neighbor + 1, cy);
            };
            auto add_cross = [&](std::ptrdiff_t ii, std::ptrdiff_t jj, double coeff) {
                if (ii < 0 || jj < 0 || ii >= static_cast<std::ptrdiff_t>(operator_model.nx())
                    || jj >= static_cast<std::ptrdiff_t>(operator_model.ny())) {
                    return;
                }
                std::size_t const neighbor = static_cast<std::size_t>(ii)
                                             + operator_model.nx() * static_cast<std::size_t>(jj);
                if (active_host(neighbor) == 0) {
                    return;
                }
                matrix_data.nonzeros.emplace_back(row_x, 2 * neighbor + 1, coeff);
                matrix_data.nonzeros.emplace_back(row_y, 2 * neighbor, coeff);
            };

            add_same(static_cast<std::ptrdiff_t>(i) - 1, j, -ax, -bx);
            add_same(static_cast<std::ptrdiff_t>(i) + 1, j, -ax, -bx);
            add_same(i, static_cast<std::ptrdiff_t>(j) - 1, -ay, -by);
            add_same(i, static_cast<std::ptrdiff_t>(j) + 1, -ay, -by);
            double const cross = -w * operator_model.c12_plus_c66() * inv_4hxhy;
            add_cross(
                    static_cast<std::ptrdiff_t>(i) + 1,
                    static_cast<std::ptrdiff_t>(j) + 1,
                    cross);
            add_cross(
                    static_cast<std::ptrdiff_t>(i) - 1,
                    static_cast<std::ptrdiff_t>(j) - 1,
                    cross);
            add_cross(
                    static_cast<std::ptrdiff_t>(i) - 1,
                    static_cast<std::ptrdiff_t>(j) + 1,
                    -cross);
            add_cross(
                    static_cast<std::ptrdiff_t>(i) + 1,
                    static_cast<std::ptrdiff_t>(j) - 1,
                    -cross);
        }
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

    double active_min_x = std::numeric_limits<double>::infinity();
    double active_max_x = -std::numeric_limits<double>::infinity();
    double active_min_y = std::numeric_limits<double>::infinity();
    double active_max_y = -std::numeric_limits<double>::infinity();
    for (std::size_t j = 0; j < grid.ny(); ++j) {
        for (std::size_t i = 0; i < grid.nx(); ++i) {
            active_min_x = std::min(active_min_x, grid.node_x(i, j));
            active_max_x = std::max(active_max_x, grid.node_x(i, j));
            active_min_y = std::min(active_min_y, grid.node_y(i, j));
            active_max_y = std::max(active_max_y, grid.node_y(i, j));
        }
    }
    if (!(active_min_x < active_max_x && active_min_y < active_max_y)) {
        throw std::runtime_error("failed to detect the wrench bounds in the structured grid");
    }
    double const active_width_x = active_max_x - active_min_x;
    double const active_width_y = active_max_y - active_min_y;
    double const clamp_limit_x = active_min_x + 0.08 * active_width_x;
    double const force_start_x = active_max_x - 0.05 * active_width_x;
    double const force_band_half_height = 0.55 * active_width_y;

    using memory_space = typename Kokkos::DefaultExecutionSpace::memory_space;
    Kokkos::View<int*, memory_space> active("similie_elasticity_active", grid.nx() * grid.ny());
    Kokkos::View<int*, memory_space>
            dirichlet("similie_elasticity_dirichlet", grid.nx() * grid.ny());
    Kokkos::View<double*, memory_space>
            density("similie_elasticity_density", grid.nx() * grid.ny());
    auto active_host = Kokkos::create_mirror_view(active);
    auto dirichlet_host = Kokkos::create_mirror_view(dirichlet);
    auto density_host = Kokkos::create_mirror_view(density);
    std::vector<std::size_t> loaded_nodes;
    loaded_nodes.reserve(grid.ny());
    for (std::size_t j = 0; j < grid.ny(); ++j) {
        for (std::size_t i = 0; i < grid.nx(); ++i) {
            std::size_t const node = grid.node_index(i, j);
            active_host(node) = 1;
            density_host(node) = 1.0;
            double const x = grid.node_x(i, j);
            double const y = grid.node_y(i, j);
            bool const clamped = x <= clamp_limit_x && std::abs(y) <= 0.36 * active_width_y;
            dirichlet_host(node) = clamped ? 1 : 0;
            if (clamped) {
                ++result.num_clamped_nodes;
            }
            bool const loaded = x >= force_start_x && std::abs(y) <= force_band_half_height;
            if (loaded) {
                loaded_nodes.push_back(node);
            }
        }
    }
    result.num_loaded_nodes = loaded_nodes.size();
    if (result.num_clamped_nodes == 0 || result.num_loaded_nodes == 0) {
        throw std::runtime_error(
                "failed to detect clamped or loaded active nodes in the structured wrench grid");
    }
    Kokkos::deep_copy(active, active_host);
    Kokkos::deep_copy(dirichlet, dirichlet_host);
    Kokkos::deep_copy(density, density_host);

    detail::ElasticityMaterialCoefficients const material
            = detail::material_coefficients(inputs.young_modulus, inputs.poisson_ratio);
    auto const [hx, hy] = detail::average_logical_spacings(grid);

    Kokkos::View<double**> rhs("similie_elasticity_rhs", 2 * grid.nx() * grid.ny(), 1);
    Kokkos::View<double**>
            displacement_view("similie_elasticity_displacement", 2 * grid.nx() * grid.ny(), 1);
    auto rhs_host = Kokkos::create_mirror_view(rhs);
    double const nodal_force_density
            = -inputs.applied_force
              / (inputs.thickness * hx * hy * static_cast<double>(loaded_nodes.size()));
    for (std::size_t node : loaded_nodes) {
        rhs_host(2 * node + 1, 0) = nodal_force_density;
    }
    Kokkos::deep_copy(rhs, rhs_host);

    detail::ElasticityOperator2D<memory_space> const
            operator_model(grid.nx(), grid.ny(), hx, hy, material, active, dirichlet, density);

    detail::log_info(logger, "SimiLie starting linear elasticity solve");
    result.solver_diagnostics = solvers::minimize_strong_formulation_residual(
            Kokkos::DefaultExecutionSpace(),
            operator_model,
            rhs,
            displacement_view,
            solver_settings);
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

    std::size_t probe_node = loaded_nodes.front();
    for (std::size_t node : loaded_nodes) {
        if (grid.ordered_nodes[node].x > grid.ordered_nodes[probe_node].x) {
            probe_node = node;
        }
    }
    result.probe_displacement_y = displacement[2 * probe_node + 1];

    double constexpr material_threshold = 0.1;
    for (std::size_t j = 0; j < grid.ncell_y; ++j) {
        for (std::size_t i = 0; i < grid.ncell_x; ++i) {
            std::size_t const n00 = grid.node_index(i, j);
            std::size_t const n10 = grid.node_index(i + 1, j);
            std::size_t const n01 = grid.node_index(i, j + 1);
            std::size_t const n11 = grid.node_index(i + 1, j + 1);
            double const dux_dx = 0.5
                                  * ((displacement[2 * n10] - displacement[2 * n00])
                                     + (displacement[2 * n11] - displacement[2 * n01]))
                                  / hx;
            double const duy_dy = 0.5
                                  * ((displacement[2 * n01 + 1] - displacement[2 * n00 + 1])
                                     + (displacement[2 * n11 + 1] - displacement[2 * n10 + 1]))
                                  / hy;
            double const dux_dy = 0.5
                                  * ((displacement[2 * n01] - displacement[2 * n00])
                                     + (displacement[2 * n11] - displacement[2 * n10]))
                                  / hy;
            double const duy_dx = 0.5
                                  * ((displacement[2 * n10 + 1] - displacement[2 * n00 + 1])
                                     + (displacement[2 * n11 + 1] - displacement[2 * n01 + 1]))
                                  / hx;
            detail::CellFields& fields = cell_fields[grid.cell_index(i, j)];
            fields.strain = {
                    .xx = dux_dx,
                    .yy = duy_dy,
                    .xy = 0.5 * (dux_dy + duy_dx),
            };
            fields.stress = detail::hooke_plane_stress(material, fields.strain);
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
