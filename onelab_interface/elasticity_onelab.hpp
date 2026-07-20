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

#include <ddc/ddc.hpp>
#include <ginkgo/core/base/matrix_data.hpp>
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
[[nodiscard]] KOKKOS_FUNCTION double elasticity_c11(Equations equations)
{
    physics::elasticity::Strain2D const unit_strain {.xx = 1.0};
    int const elem = 0;
    return equations.template dpotential_dt<physics::elasticity::StrainXX>(unit_strain, elem);
}

template <class Equations>
[[nodiscard]] KOKKOS_FUNCTION double elasticity_c12(Equations equations)
{
    physics::elasticity::Strain2D const unit_strain {.yy = 1.0};
    int const elem = 0;
    return equations.template dpotential_dt<physics::elasticity::StrainXX>(unit_strain, elem);
}

template <class Equations>
[[nodiscard]] KOKKOS_FUNCTION double elasticity_c66(Equations equations)
{
    physics::elasticity::Strain2D const unit_strain {.xy = 1.0};
    int const elem = 0;
    return 0.25
           * equations.template dpotential_dt<physics::elasticity::StrainXY>(unit_strain, elem);
}

template <class StressIndex, class StrainIndex, class Equations>
[[nodiscard]] KOKKOS_FUNCTION double stress_derivative_from_unit_strain(Equations equations)
{
    physics::elasticity::Strain2D unit_strain;
    if constexpr (std::is_same_v<StrainIndex, physics::elasticity::StrainXX>) {
        unit_strain.xx = 1.0;
    } else if constexpr (std::is_same_v<StrainIndex, physics::elasticity::StrainYY>) {
        unit_strain.yy = 1.0;
    } else {
        unit_strain.xy = 1.0;
    }
    int const elem = 0;
    return equations.template dpotential_dt<StressIndex>(unit_strain, elem);
}

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

template <class MemorySpace, class Equations>
class ElasticityOperator2D
{
    using view_type = Kokkos::View<double*, MemorySpace>;
    using int_view_type = Kokkos::View<int*, MemorySpace>;
    using stencil_columns_view_type = Kokkos::View<int****, Kokkos::LayoutRight, MemorySpace>;
    using stencil_coefficients_view_type =
            Kokkos::View<double****, Kokkos::LayoutRight, MemorySpace>;
    using stencil_counts_view_type = Kokkos::View<int***, Kokkos::LayoutRight, MemorySpace>;
    using transposed_columns_view_type = Kokkos::View<int***, Kokkos::LayoutRight, MemorySpace>;
    using transposed_coefficients_view_type =
            Kokkos::View<double***, Kokkos::LayoutRight, MemorySpace>;
    using transposed_counts_view_type = Kokkos::View<int**, Kokkos::LayoutRight, MemorySpace>;

    static constexpr int NUM_STRAIN_COMPONENTS = 3;
    static constexpr int NUM_DISPLACEMENT_COMPONENTS = 2;
    static constexpr int STRAIN_STENCIL_MAX_SIZE = 4;
    static constexpr int TRANSPOSED_STRAIN_STENCIL_MAX_SIZE = 16;

    std::size_t m_nx = 0;
    std::size_t m_ny = 0;
    Equations m_equations;
    int_view_type m_active;
    int_view_type m_dirichlet;
    view_type m_density;
    view_type m_cell_measure;
    stencil_columns_view_type m_strain_columns;
    stencil_coefficients_view_type m_strain_coefficients;
    stencil_counts_view_type m_strain_counts;
    transposed_columns_view_type m_transposed_strain_columns;
    transposed_coefficients_view_type m_transposed_strain_coefficients;
    transposed_counts_view_type m_transposed_strain_counts;

public:
    static constexpr bool IS_LINEAR = true;

    ElasticityOperator2D(
            std::size_t nx,
            std::size_t ny,
            view_type node_x,
            view_type node_y,
            Equations equations,
            int_view_type active,
            int_view_type dirichlet,
            view_type density)
        : m_nx(nx)
        , m_ny(ny)
        , m_equations(std::move(equations))
        , m_active(active)
        , m_dirichlet(dirichlet)
        , m_density(density)
        , m_cell_measure("similie_elasticity_cell_measure", nx * ny)
        , m_strain_columns("similie_elasticity_strain_columns",
                           NUM_STRAIN_COMPONENTS,
                           NUM_DISPLACEMENT_COMPONENTS,
                           nx * ny,
                           STRAIN_STENCIL_MAX_SIZE)
        , m_strain_coefficients("similie_elasticity_strain_coefficients",
                                NUM_STRAIN_COMPONENTS,
                                NUM_DISPLACEMENT_COMPONENTS,
                                nx * ny,
                                STRAIN_STENCIL_MAX_SIZE)
        , m_strain_counts("similie_elasticity_strain_counts",
                          NUM_STRAIN_COMPONENTS,
                          NUM_DISPLACEMENT_COMPONENTS,
                          nx * ny)
        , m_transposed_strain_columns("similie_elasticity_transposed_strain_columns",
                                      NUM_STRAIN_COMPONENTS,
                                      2 * nx * ny,
                                      TRANSPOSED_STRAIN_STENCIL_MAX_SIZE)
        , m_transposed_strain_coefficients("similie_elasticity_transposed_strain_coefficients",
                                           NUM_STRAIN_COMPONENTS,
                                           2 * nx * ny,
                                           TRANSPOSED_STRAIN_STENCIL_MAX_SIZE)
        , m_transposed_strain_counts("similie_elasticity_transposed_strain_counts",
                                     NUM_STRAIN_COMPONENTS,
                                     2 * nx * ny)
    {
        auto const node_x_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), node_x);
        auto const node_y_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), node_y);
        auto cell_measure_host = Kokkos::create_mirror_view(m_cell_measure);
        auto const node_domain = ddc::DiscreteDomain<DDimX, DDimY>(
                ddc::DiscreteElement<DDimX, DDimY>(0, 0),
                ddc::DiscreteVector<DDimX, DDimY>(nx, ny));
        [[maybe_unused]] sil::tensor::TensorAccessor<PositionIndex2D> position_accessor;
        ddc::DiscreteDomain<DDimX, DDimY, PositionIndex2D> const position_domain(
                node_domain,
                position_accessor.domain());
        ddc::Chunk position_alloc(position_domain, ddc::HostAllocator<double>());
        sil::tensor::Tensor position(position_alloc);
        ddc::host_for_each(node_domain, [&](auto elem) {
            std::size_t const i
                    = static_cast<std::size_t>(ddc::DiscreteElement<DDimX>(elem).uid());
            std::size_t const j
                    = static_cast<std::size_t>(ddc::DiscreteElement<DDimY>(elem).uid());
            std::size_t const node = i + nx * j;
            position(elem, position_accessor.template access_element<X>()) = node_x_host(node);
            position(elem, position_accessor.template access_element<Y>()) = node_y_host(node);
        });
        for (std::size_t j = 0; j < ny; ++j) {
            for (std::size_t i = 0; i < nx; ++i) {
                std::size_t const row = i + nx * j;
                cell_measure_host(row) = 0.0;
                if (i + 1 < nx && j + 1 < ny) {
                    std::size_t const n00 = i + nx * j;
                    std::size_t const n10 = i + 1 + nx * j;
                    std::size_t const n01 = i + nx * (j + 1);
                    double const dx_i = node_x_host(n10) - node_x_host(n00);
                    double const dy_i = node_y_host(n10) - node_y_host(n00);
                    double const dx_j = node_x_host(n01) - node_x_host(n00);
                    double const dy_j = node_y_host(n01) - node_y_host(n00);
                    cell_measure_host(row) = std::abs(dx_i * dy_j - dx_j * dy_i);
                }
            }
        }

        auto strain_columns_host = Kokkos::create_mirror_view(m_strain_columns);
        auto strain_coefficients_host = Kokkos::create_mirror_view(m_strain_coefficients);
        auto strain_counts_host = Kokkos::create_mirror_view(m_strain_counts);
        auto transposed_columns_host = Kokkos::create_mirror_view(m_transposed_strain_columns);
        auto transposed_coefficients_host
                = Kokkos::create_mirror_view(m_transposed_strain_coefficients);
        auto transposed_counts_host = Kokkos::create_mirror_view(m_transposed_strain_counts);

        for (int strain_id = 0; strain_id < NUM_STRAIN_COMPONENTS; ++strain_id) {
            for (int component_id = 0; component_id < NUM_DISPLACEMENT_COMPONENTS;
                 ++component_id) {
                for (std::size_t row = 0; row < nx * ny; ++row) {
                    strain_counts_host(strain_id, component_id, row) = 0;
                    for (int slot = 0; slot < STRAIN_STENCIL_MAX_SIZE; ++slot) {
                        strain_columns_host(strain_id, component_id, row, slot) = 0;
                        strain_coefficients_host(strain_id, component_id, row, slot) = 0.0;
                    }
                }
            }
            for (std::size_t row = 0; row < 2 * nx * ny; ++row) {
                transposed_counts_host(strain_id, row) = 0;
                for (int slot = 0; slot < TRANSPOSED_STRAIN_STENCIL_MAX_SIZE; ++slot) {
                    transposed_columns_host(strain_id, row, slot) = 0;
                    transposed_coefficients_host(strain_id, row, slot) = 0.0;
                }
            }
        }

        auto fill_forward_stencil = [&](auto strain_tag,
                                        auto displacement_tag,
                                        int strain_id,
                                        int displacement_component_id,
                                        std::size_t sample_row,
                                        auto elem) {
            using StrainIndex = decltype(strain_tag);
            using DisplacementComponent = decltype(displacement_tag);
            int count = 0;
            auto stencil = physics::elasticity::DisplacementToStrain::
                    template forward_value<StrainIndex, DisplacementComponent, X, Y>(
                            elem, position);
            ddc::device_for_each(stencil.domain(), [&](auto stencil_elem) {
                double const coefficient = stencil.mem(stencil_elem);
                if (coefficient == 0.0) {
                    return;
                }
                auto const displacement_elem = ddc::DiscreteElement<DDimX, DDimY>(stencil_elem);
                std::size_t const i = static_cast<std::size_t>(
                        ddc::DiscreteElement<DDimX>(displacement_elem).uid());
                std::size_t const j = static_cast<std::size_t>(
                        ddc::DiscreteElement<DDimY>(displacement_elem).uid());
                if (i >= nx || j >= ny) {
                    return;
                }
                if (count >= STRAIN_STENCIL_MAX_SIZE) {
                    throw std::runtime_error("strain stencil capacity exceeded");
                }
                strain_columns_host(strain_id, displacement_component_id, sample_row, count)
                        = static_cast<int>(2 * (i + nx * j) + displacement_component_id);
                strain_coefficients_host(strain_id, displacement_component_id, sample_row, count)
                        = coefficient;
                ++count;
            });
            strain_counts_host(strain_id, displacement_component_id, sample_row) = count;
        };

        for (std::size_t j = 0; j + 1 < ny; ++j) {
            for (std::size_t i = 0; i + 1 < nx; ++i) {
                std::size_t const sample_row = i + nx * j;
                auto const elem = ddc::DiscreteElement<DDimX, DDimY>(i, j);
                fill_forward_stencil(
                        physics::elasticity::StrainXX {}, X {}, 0, 0, sample_row, elem);
                fill_forward_stencil(
                        physics::elasticity::StrainXX {}, Y {}, 0, 1, sample_row, elem);
                fill_forward_stencil(
                        physics::elasticity::StrainYY {}, X {}, 1, 0, sample_row, elem);
                fill_forward_stencil(
                        physics::elasticity::StrainYY {}, Y {}, 1, 1, sample_row, elem);
                fill_forward_stencil(
                        physics::elasticity::StrainXY {}, X {}, 2, 0, sample_row, elem);
                fill_forward_stencil(
                        physics::elasticity::StrainXY {}, Y {}, 2, 1, sample_row, elem);
            }
        }

        for (std::size_t sample_row = 0; sample_row < nx * ny; ++sample_row) {
            for (int strain_id = 0; strain_id < NUM_STRAIN_COMPONENTS; ++strain_id) {
                for (int component_id = 0; component_id < NUM_DISPLACEMENT_COMPONENTS;
                     ++component_id) {
                    for (int slot = 0;
                         slot < strain_counts_host(strain_id, component_id, sample_row);
                         ++slot) {
                        std::size_t const row = static_cast<std::size_t>(
                                strain_columns_host(strain_id, component_id, sample_row, slot));
                        int const count = transposed_counts_host(strain_id, row);
                        if (count >= TRANSPOSED_STRAIN_STENCIL_MAX_SIZE) {
                            throw std::runtime_error(
                                    "transposed strain stencil capacity exceeded");
                        }
                        transposed_columns_host(strain_id, row, count)
                                = static_cast<int>(sample_row);
                        transposed_coefficients_host(strain_id, row, count)
                                = strain_coefficients_host(
                                        strain_id, component_id, sample_row, slot);
                        transposed_counts_host(strain_id, row) = count + 1;
                    }
                }
            }
        }

        Kokkos::deep_copy(m_strain_columns, strain_columns_host);
        Kokkos::deep_copy(m_strain_coefficients, strain_coefficients_host);
        Kokkos::deep_copy(m_strain_counts, strain_counts_host);
        Kokkos::deep_copy(m_transposed_strain_columns, transposed_columns_host);
        Kokkos::deep_copy(m_transposed_strain_coefficients, transposed_coefficients_host);
        Kokkos::deep_copy(m_transposed_strain_counts, transposed_counts_host);
        Kokkos::deep_copy(m_cell_measure, cell_measure_host);
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
        auto const equations = m_equations;
        auto const active = m_active;
        auto const dirichlet = m_dirichlet;
        auto const density = m_density;
        auto const cell_measure = m_cell_measure;
        auto const strain_columns = m_strain_columns;
        auto const strain_coefficients = m_strain_coefficients;
        auto const strain_counts = m_strain_counts;
        auto const transposed_strain_columns = m_transposed_strain_columns;
        auto const transposed_strain_coefficients = m_transposed_strain_coefficients;
        auto const transposed_strain_counts = m_transposed_strain_counts;

        Kokkos::parallel_for(
                "similie_elasticity_operator_apply",
                Kokkos::RangePolicy<ExecSpace>(exec_space, 0, 2 * nx * ny),
                KOKKOS_LAMBDA(std::size_t row) {
                    std::size_t const node = row / 2;
                    if (active(node) == 0 || dirichlet(node) != 0) {
                        output(row, 0) = input(row, 0);
                        return;
                    }

                    double residual = 0.0;
                    for (int stress_id = 0; stress_id < NUM_STRAIN_COMPONENTS; ++stress_id) {
                        for (int slot = 0; slot < transposed_strain_counts(stress_id, row);
                             ++slot) {
                            std::size_t const sample_row = static_cast<std::size_t>(
                                    transposed_strain_columns(stress_id, row, slot));
                            physics::elasticity::Strain2D strain;
                            for (int component_id = 0; component_id < NUM_DISPLACEMENT_COMPONENTS;
                                 ++component_id) {
                                for (int k = 0;
                                     k < strain_counts(0, component_id, sample_row);
                                     ++k) {
                                    strain.xx += strain_coefficients(
                                                         0, component_id, sample_row, k)
                                                 * input(static_cast<std::size_t>(strain_columns(
                                                                 0,
                                                                 component_id,
                                                                 sample_row,
                                                                 k)),
                                                         0);
                                }
                                for (int k = 0;
                                     k < strain_counts(1, component_id, sample_row);
                                     ++k) {
                                    strain.yy += strain_coefficients(
                                                         1, component_id, sample_row, k)
                                                 * input(static_cast<std::size_t>(strain_columns(
                                                                 1,
                                                                 component_id,
                                                                 sample_row,
                                                                 k)),
                                                         0);
                                }
                                for (int k = 0;
                                     k < strain_counts(2, component_id, sample_row);
                                     ++k) {
                                    strain.xy += strain_coefficients(
                                                         2, component_id, sample_row, k)
                                                 * input(static_cast<std::size_t>(strain_columns(
                                                                 2,
                                                                 component_id,
                                                                 sample_row,
                                                                 k)),
                                                         0);
                                }
                            }
                            double stress = 0.0;
                            auto const elem = ddc::DiscreteElement<DDimX, DDimY>(
                                    sample_row % nx, sample_row / nx);
                            if (stress_id == 0) {
                                stress = equations.template dpotential_dt<
                                        physics::elasticity::StrainXX>(strain, elem);
                            } else if (stress_id == 1) {
                                stress = equations.template dpotential_dt<
                                        physics::elasticity::StrainYY>(strain, elem);
                            } else {
                                stress = equations.template dpotential_dt<
                                        physics::elasticity::StrainXY>(strain, elem);
                            }
                            residual += transposed_strain_coefficients(stress_id, row, slot)
                                        * density(sample_row) * cell_measure(sample_row) * stress;
                        }
                    }
                    output(row, 0) = residual;
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

    [[nodiscard]] auto cell_measure() const
    {
        return m_cell_measure;
    }

    [[nodiscard]] auto equations() const
    {
        return m_equations;
    }

    [[nodiscard]] std::size_t nx() const
    {
        return m_nx;
    }

    [[nodiscard]] std::size_t ny() const
    {
        return m_ny;
    }

    [[nodiscard]] auto strain_columns() const
    {
        return m_strain_columns;
    }

    [[nodiscard]] auto strain_coefficients() const
    {
        return m_strain_coefficients;
    }

    [[nodiscard]] auto strain_counts() const
    {
        return m_strain_counts;
    }

    [[nodiscard]] auto transposed_strain_columns() const
    {
        return m_transposed_strain_columns;
    }

    [[nodiscard]] auto transposed_strain_coefficients() const
    {
        return m_transposed_strain_coefficients;
    }

    [[nodiscard]] auto transposed_strain_counts() const
    {
        return m_transposed_strain_counts;
    }
};

template <class MemorySpace, class Equations>
gko::matrix_data<double, gko::int32> assemble_matrix_data(
        ElasticityOperator2D<MemorySpace, Equations> const& operator_model)
{
    gko::matrix_data<double, gko::int32> matrix_data(
            gko::dim<2>(operator_model.size(), operator_model.size()));
    auto const active_host
            = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), operator_model.active());
    auto const dirichlet_host
            = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), operator_model.dirichlet());
    auto const density_host
            = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), operator_model.density());
    auto const cell_measure_host = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(),
            operator_model.cell_measure());
    auto const strain_columns_host = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), operator_model.strain_columns());
    auto const strain_coefficients_host = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), operator_model.strain_coefficients());
    auto const strain_counts_host = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), operator_model.strain_counts());
    auto const transposed_columns_host = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), operator_model.transposed_strain_columns());
    auto const transposed_coefficients_host = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), operator_model.transposed_strain_coefficients());
    auto const transposed_counts_host = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), operator_model.transposed_strain_counts());

    auto coefficient = [&](int stress_id, int strain_id) {
        auto const equations = operator_model.equations();
        if (stress_id == 0 && strain_id == 0) {
            return stress_derivative_from_unit_strain<
                    physics::elasticity::StrainXX,
                    physics::elasticity::StrainXX>(equations);
        }
        if (stress_id == 0 && strain_id == 1) {
            return stress_derivative_from_unit_strain<
                    physics::elasticity::StrainXX,
                    physics::elasticity::StrainYY>(equations);
        }
        if (stress_id == 0 && strain_id == 2) {
            return stress_derivative_from_unit_strain<
                    physics::elasticity::StrainXX,
                    physics::elasticity::StrainXY>(equations);
        }
        if (stress_id == 1 && strain_id == 0) {
            return stress_derivative_from_unit_strain<
                    physics::elasticity::StrainYY,
                    physics::elasticity::StrainXX>(equations);
        }
        if (stress_id == 1 && strain_id == 1) {
            return stress_derivative_from_unit_strain<
                    physics::elasticity::StrainYY,
                    physics::elasticity::StrainYY>(equations);
        }
        if (stress_id == 1 && strain_id == 2) {
            return stress_derivative_from_unit_strain<
                    physics::elasticity::StrainYY,
                    physics::elasticity::StrainXY>(equations);
        }
        if (stress_id == 2 && strain_id == 0) {
            return stress_derivative_from_unit_strain<
                    physics::elasticity::StrainXY,
                    physics::elasticity::StrainXX>(equations);
        }
        if (stress_id == 2 && strain_id == 1) {
            return stress_derivative_from_unit_strain<
                    physics::elasticity::StrainXY,
                    physics::elasticity::StrainYY>(equations);
        }
        return stress_derivative_from_unit_strain<
                physics::elasticity::StrainXY,
                physics::elasticity::StrainXY>(equations);
    };

    for (std::size_t row = 0; row < operator_model.size(); ++row) {
        std::size_t const node = row / 2;
        if (active_host(node) == 0 || dirichlet_host(node) != 0) {
            matrix_data.nonzeros.emplace_back(row, row, 1.0);
            continue;
        }
        for (int stress_id = 0; stress_id < 3; ++stress_id) {
            for (int transpose_slot = 0; transpose_slot < transposed_counts_host(stress_id, row);
                 ++transpose_slot) {
                std::size_t const sample_row = static_cast<std::size_t>(
                        transposed_columns_host(stress_id, row, transpose_slot));
                double const transpose_coefficient
                        = transposed_coefficients_host(stress_id, row, transpose_slot);
                double const weight = density_host(sample_row) * cell_measure_host(sample_row);
                for (int strain_id = 0; strain_id < 3; ++strain_id) {
                    double const material_coefficient = coefficient(stress_id, strain_id);
                    if (material_coefficient == 0.0) {
                        continue;
                    }
                    for (int component_id = 0; component_id < 2; ++component_id) {
                        for (int slot = 0;
                             slot < strain_counts_host(strain_id, component_id, sample_row);
                             ++slot) {
                            std::size_t const column = static_cast<std::size_t>(
                                    strain_columns_host(strain_id, component_id, sample_row, slot));
                            std::size_t const column_node = column / 2;
                            if (active_host(column_node) == 0) {
                                continue;
                            }
                            matrix_data.nonzeros.emplace_back(
                                    static_cast<gko::int32>(row),
                                    static_cast<gko::int32>(column),
                                    weight * transpose_coefficient * material_coefficient
                                            * strain_coefficients_host(
                                                    strain_id, component_id, sample_row, slot));
                        }
                    }
                }
            }
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
    Kokkos::View<double*, memory_space>
            node_x("similie_elasticity_node_x", grid.nx() * grid.ny());
    Kokkos::View<double*, memory_space>
            node_y("similie_elasticity_node_y", grid.nx() * grid.ny());
    auto active_host = Kokkos::create_mirror_view(active);
    auto dirichlet_host = Kokkos::create_mirror_view(dirichlet);
    auto density_host = Kokkos::create_mirror_view(density);
    auto node_x_host = Kokkos::create_mirror_view(node_x);
    auto node_y_host = Kokkos::create_mirror_view(node_y);
    std::vector<std::size_t> loaded_nodes;
    loaded_nodes.reserve(grid.ny());
    for (std::size_t j = 0; j < grid.ny(); ++j) {
        for (std::size_t i = 0; i < grid.nx(); ++i) {
            std::size_t const node = grid.node_index(i, j);
            active_host(node) = 1;
            density_host(node) = 1.0;
            double const x = grid.node_x(i, j);
            double const y = grid.node_y(i, j);
            node_x_host(node) = x;
            node_y_host(node) = y;
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
    Kokkos::deep_copy(node_x, node_x_host);
    Kokkos::deep_copy(node_y, node_y_host);

    physics::elasticity::LinearElasticityHamiltonian<> const
            hamiltonian(inputs.young_modulus, inputs.poisson_ratio);
    auto const equations = physics::HamiltonEquations {hamiltonian};
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

    detail::ElasticityOperator2D<memory_space, decltype(equations)> const
            operator_model(grid.nx(), grid.ny(), node_x, node_y, equations, active, dirichlet, density);

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

    auto const node_domain = ddc::DiscreteDomain<detail::DDimX, detail::DDimY>(
            ddc::DiscreteElement<detail::DDimX, detail::DDimY>(0, 0),
            ddc::DiscreteVector<detail::DDimX, detail::DDimY>(grid.nx(), grid.ny()));
    [[maybe_unused]] sil::tensor::TensorAccessor<detail::PositionIndex2D> position_accessor;
    ddc::DiscreteDomain<detail::DDimX, detail::DDimY, detail::PositionIndex2D> const
            position_domain(node_domain, position_accessor.domain());
    ddc::Chunk position_alloc(position_domain, ddc::HostAllocator<double>());
    sil::tensor::Tensor position(position_alloc);
    ddc::host_for_each(node_domain, [&](auto elem) {
        std::size_t const i
                = static_cast<std::size_t>(ddc::DiscreteElement<detail::DDimX>(elem).uid());
        std::size_t const j
                = static_cast<std::size_t>(ddc::DiscreteElement<detail::DDimY>(elem).uid());
        position(elem, position_accessor.template access_element<detail::X>())
                = grid.node_x(i, j);
        position(elem, position_accessor.template access_element<detail::Y>())
                = grid.node_y(i, j);
    });

    auto strain_component_from_displacement = [&](auto strain_tag,
                                                 auto displacement_component_tag,
                                                 auto elem) {
        using StrainIndex = decltype(strain_tag);
        using DisplacementComponent = decltype(displacement_component_tag);
        auto stencil = physics::elasticity::DisplacementToStrain::
                template forward_value<
                        StrainIndex,
                        DisplacementComponent,
                        detail::X,
                        detail::Y>(elem, position);
        double value = 0.0;
        ddc::host_for_each(stencil.domain(), [&](auto stencil_elem) {
            auto const displacement_elem
                    = ddc::DiscreteElement<detail::DDimX, detail::DDimY>(stencil_elem);
            std::size_t const i = static_cast<std::size_t>(
                    ddc::DiscreteElement<detail::DDimX>(displacement_elem).uid());
            std::size_t const j = static_cast<std::size_t>(
                    ddc::DiscreteElement<detail::DDimY>(displacement_elem).uid());
            if (i >= grid.nx() || j >= grid.ny()) {
                return;
            }
            std::size_t const node = grid.node_index(i, j);
            int const component = std::is_same_v<DisplacementComponent, detail::X> ? 0 : 1;
            value += stencil.mem(stencil_elem) * displacement[2 * node + component];
        });
        return value;
    };

    double constexpr material_threshold = 0.1;
    for (std::size_t j = 0; j < grid.ncell_y; ++j) {
        for (std::size_t i = 0; i < grid.ncell_x; ++i) {
            auto const elem = ddc::DiscreteElement<detail::DDimX, detail::DDimY>(i, j);
            detail::CellFields& fields = cell_fields[grid.cell_index(i, j)];
            fields.strain = {
                    .xx = strain_component_from_displacement(
                            physics::elasticity::StrainXX {}, detail::X {}, elem),
                    .yy = strain_component_from_displacement(
                            physics::elasticity::StrainYY {}, detail::Y {}, elem),
                    .xy = strain_component_from_displacement(
                                  physics::elasticity::StrainXY {}, detail::X {}, elem)
                          + strain_component_from_displacement(
                                  physics::elasticity::StrainXY {}, detail::Y {}, elem),
            };
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
