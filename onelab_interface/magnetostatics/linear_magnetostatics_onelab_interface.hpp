// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <numbers>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <similie/physics/magnetostatics/magnetostatics_quantities.hpp>
#include <similie/physics/magnetostatics/structured_linear_magnetostatics.hpp>
#include <similie/solvers/minimize_strong_formulation_residual.hpp>

#include "base_onelab_interface.hpp"

namespace similie::onelab_interface {

namespace detail {

constexpr int ECORE_TAG = 1000;
constexpr int ICORE_TAG = 1100;
constexpr int COIL_LEFT_TAG = 2000;
constexpr int COIL_RIGHT_TAG = 2001;
constexpr int AIR_TAG = 3000;
constexpr int AIRGAP_TAG = 3200;
constexpr double COORD_TOL = 1.0e-12;

struct MeshNode
{
    std::size_t tag;
    double x;
    double y;
    double z;
};

struct HexahedralCell
{
    int physical_tag;
    std::array<std::size_t, 8> node_tags;
};

struct HexahedralMesh
{
    std::vector<MeshNode> nodes;
    std::vector<HexahedralCell> cells;
};

struct StructuredGrid
{
    std::vector<double> x_coords;
    std::vector<double> y_coords;
    std::vector<double> z_coords;
    std::map<std::size_t, std::array<std::size_t, 3>> node_indices_by_tag;
    std::vector<MeshNode> ordered_nodes;
    std::vector<HexahedralCell> ordered_cells;

    [[nodiscard]] std::size_t nx() const { return x_coords.size(); }
    [[nodiscard]] std::size_t ny() const { return y_coords.size(); }
    [[nodiscard]] std::size_t nz() const { return z_coords.size(); }
    [[nodiscard]] std::size_t ncell_x() const { return x_coords.size() - 1; }
    [[nodiscard]] std::size_t ncell_y() const { return y_coords.size() - 1; }
    [[nodiscard]] std::size_t ncell_z() const { return z_coords.size() - 1; }

    [[nodiscard]] std::size_t node_index(std::size_t i, std::size_t j, std::size_t k) const
    {
        return i + nx() * (j + ny() * k);
    }

    [[nodiscard]] std::size_t cell_index(std::size_t i, std::size_t j, std::size_t k) const
    {
        return i + ncell_x() * (j + ncell_y() * k);
    }

    [[nodiscard]] double cell_center_x(std::size_t i) const { return 0.5 * (x_coords[i] + x_coords[i + 1]); }
    [[nodiscard]] double cell_center_y(std::size_t j) const { return 0.5 * (y_coords[j] + y_coords[j + 1]); }
    [[nodiscard]] double cell_center_z(std::size_t k) const { return 0.5 * (z_coords[k] + z_coords[k + 1]); }
};

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

inline bool nearly_equal(double lhs, double rhs)
{
    return std::abs(lhs - rhs) < COORD_TOL;
}

inline std::vector<double> unique_sorted(std::vector<double> values)
{
    std::sort(values.begin(), values.end());
    std::vector<double> result;
    result.reserve(values.size());
    for (double value : values) {
        if (result.empty() || !nearly_equal(result.back(), value)) {
            result.push_back(value);
        }
    }
    return result;
}

inline std::size_t nearest_index(std::vector<double> const& coordinates, double value)
{
    for (std::size_t index = 0; index < coordinates.size(); ++index) {
        if (nearly_equal(coordinates[index], value)) {
            return index;
        }
    }
    throw std::runtime_error("coordinate does not belong to the detected structured grid");
}

inline int element_dimension(int element_type)
{
    switch (element_type) {
    case 15:
        return 0;
    case 1:
        return 1;
    case 2:
    case 3:
        return 2;
    case 4:
    case 5:
    case 6:
    case 7:
        return 3;
    default:
        return -1;
    }
}

inline bool is_supported_boundary_element(int element_type)
{
    return element_type == 1 || element_type == 2 || element_type == 3 || element_type == 15;
}

inline HexahedralMesh parse_hexahedral_msh2_mesh(std::filesystem::path const& mesh_file)
{
    std::ifstream stream(mesh_file);
    if (!stream.is_open()) {
        throw std::runtime_error("failed to open mesh file: " + mesh_file.string());
    }

    HexahedralMesh mesh;
    std::string token;
    bool saw_elements_section = false;

    while (stream >> token) {
        if (token == "$MeshFormat") {
            double version = 0.0;
            int file_type = 0;
            int data_size = 0;
            stream >> version >> file_type >> data_size;
            if (version < 2.0 || version >= 3.0 || file_type != 0) {
                throw std::runtime_error("only Gmsh msh2 ASCII meshes are supported");
            }
        } else if (token == "$Nodes") {
            std::size_t node_count = 0;
            stream >> node_count;
            mesh.nodes.reserve(node_count);
            for (std::size_t i = 0; i < node_count; ++i) {
                MeshNode node;
                stream >> node.tag >> node.x >> node.y >> node.z;
                mesh.nodes.push_back(node);
            }
        } else if (token == "$Elements") {
            saw_elements_section = true;
            std::size_t element_count = 0;
            stream >> element_count;
            for (std::size_t i = 0; i < element_count; ++i) {
                std::size_t element_tag = 0;
                int element_type = 0;
                std::size_t num_tags = 0;
                stream >> element_tag >> element_type >> num_tags;

                std::vector<int> tags(num_tags, 0);
                for (std::size_t tag_id = 0; tag_id < num_tags; ++tag_id) {
                    stream >> tags[tag_id];
                }

                if (element_dimension(element_type) < 0) {
                    std::ostringstream error_stream;
                    error_stream << "unsupported Gmsh element type " << element_type
                                 << " found in mesh";
                    throw std::runtime_error(error_stream.str());
                }

                if (element_type == 5) {
                    HexahedralCell cell;
                    cell.physical_tag = tags.empty() ? 0 : tags[0];
                    for (std::size_t k = 0; k < cell.node_tags.size(); ++k) {
                        stream >> cell.node_tags[k];
                    }
                    mesh.cells.push_back(cell);
                } else if (is_supported_boundary_element(element_type)) {
                    int nodes_to_skip = 0;
                    if (element_type == 1) {
                        nodes_to_skip = 2;
                    } else if (element_type == 2) {
                        nodes_to_skip = 3;
                    } else if (element_type == 3) {
                        nodes_to_skip = 4;
                    } else if (element_type == 15) {
                        nodes_to_skip = 1;
                    }
                    for (int k = 0; k < nodes_to_skip; ++k) {
                        std::size_t ignored_node = 0;
                        stream >> ignored_node;
                    }
                } else {
                    int nodes_to_skip = 0;
                    if (element_type == 4) {
                        nodes_to_skip = 4;
                    } else if (element_type == 6) {
                        nodes_to_skip = 6;
                    } else if (element_type == 7) {
                        nodes_to_skip = 5;
                    }
                    for (int k = 0; k < nodes_to_skip; ++k) {
                        std::size_t ignored_node = 0;
                        stream >> ignored_node;
                    }
                    throw std::runtime_error(
                            "unsupported mesh topology: SimiLie currently requires the whole "
                            "mesh to be made of quadrilaterals or hexahedra");
                }
            }
        }
    }

    if (mesh.nodes.empty()) {
        throw std::runtime_error("the provided mesh does not contain any node");
    }
    if (!saw_elements_section || mesh.cells.empty()) {
        throw std::runtime_error(
                "unsupported mesh topology: SimiLie currently requires the whole mesh to be made "
                "of quadrilaterals or hexahedra");
    }

    return mesh;
}

inline StructuredGrid build_structured_grid(HexahedralMesh const& mesh)
{
    StructuredGrid grid;
    grid.x_coords.reserve(mesh.nodes.size());
    grid.y_coords.reserve(mesh.nodes.size());
    grid.z_coords.reserve(mesh.nodes.size());

    for (MeshNode const& node : mesh.nodes) {
        grid.x_coords.push_back(node.x);
        grid.y_coords.push_back(node.y);
        grid.z_coords.push_back(node.z);
    }
    grid.x_coords = unique_sorted(std::move(grid.x_coords));
    grid.y_coords = unique_sorted(std::move(grid.y_coords));
    grid.z_coords = unique_sorted(std::move(grid.z_coords));

    if (grid.nx() * grid.ny() * grid.nz() != mesh.nodes.size()) {
        throw std::runtime_error("the mesh nodes do not form a full rectilinear grid");
    }

    std::vector<bool> occupied_nodes(grid.nx() * grid.ny() * grid.nz(), false);
    grid.ordered_nodes.resize(mesh.nodes.size());
    for (MeshNode const& node : mesh.nodes) {
        std::array<std::size_t, 3> const index {
                nearest_index(grid.x_coords, node.x),
                nearest_index(grid.y_coords, node.y),
                nearest_index(grid.z_coords, node.z),
        };
        std::size_t const linear_index = grid.node_index(index[0], index[1], index[2]);
        if (occupied_nodes[linear_index]) {
            throw std::runtime_error("duplicated nodes on the detected rectilinear grid");
        }
        occupied_nodes[linear_index] = true;
        grid.node_indices_by_tag.emplace(node.tag, index);
        grid.ordered_nodes[linear_index] = node;
    }

    std::size_t const num_cells = grid.ncell_x() * grid.ncell_y() * grid.ncell_z();
    if (mesh.cells.size() != num_cells) {
        throw std::runtime_error("the hexahedral cells do not cover a full rectilinear grid");
    }

    std::vector<bool> occupied_cells(num_cells, false);
    grid.ordered_cells.resize(num_cells);
    for (HexahedralCell const& cell : mesh.cells) {
        std::size_t min_x = std::numeric_limits<std::size_t>::max();
        std::size_t min_y = std::numeric_limits<std::size_t>::max();
        std::size_t min_z = std::numeric_limits<std::size_t>::max();
        std::size_t max_x = 0;
        std::size_t max_y = 0;
        std::size_t max_z = 0;

        for (std::size_t node_tag : cell.node_tags) {
            auto const iterator = grid.node_indices_by_tag.find(node_tag);
            if (iterator == grid.node_indices_by_tag.end()) {
                throw std::runtime_error("a hexahedron references an unknown node");
            }
            min_x = std::min(min_x, iterator->second[0]);
            min_y = std::min(min_y, iterator->second[1]);
            min_z = std::min(min_z, iterator->second[2]);
            max_x = std::max(max_x, iterator->second[0]);
            max_y = std::max(max_y, iterator->second[1]);
            max_z = std::max(max_z, iterator->second[2]);
        }

        if (max_x != min_x + 1 || max_y != min_y + 1 || max_z != min_z + 1) {
            throw std::runtime_error("a hexahedron does not match a single rectilinear grid cell");
        }

        std::size_t const linear_index = grid.cell_index(min_x, min_y, min_z);
        if (occupied_cells[linear_index]) {
            throw std::runtime_error("duplicated hexahedron on the detected rectilinear grid");
        }
        occupied_cells[linear_index] = true;
        grid.ordered_cells[linear_index] = cell;
    }

    for (bool occupied_cell : occupied_cells) {
        if (!occupied_cell) {
            throw std::runtime_error("the hexahedral cells do not cover a full rectilinear grid");
        }
    }

    return grid;
}

inline void write_results_view(
        std::filesystem::path const& output_file,
        StructuredGrid const& grid,
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
                std::array<double, 3> const& current_density = cell_inputs[index].current_density;
                stream << "  SP(" << grid.cell_center_x(i) << ", " << grid.cell_center_y(j) << ", "
                       << grid.cell_center_z(k) << "){" << current_density[2] << "};\n";
            }
        }
    }
    stream << "};\n";

    stream << "View \"SimiLie linear magnetostatics magnetic vector potential z\" {\n";
    for (std::size_t k = 0; k < grid.nz(); ++k) {
        for (std::size_t j = 0; j < grid.ny(); ++j) {
            for (std::size_t i = 0; i < grid.nx(); ++i) {
                std::size_t const index = grid.node_index(i, j, k);
                stream << "  SP(" << grid.x_coords[i] << ", " << grid.y_coords[j] << ", " << grid.z_coords[k]
                       << "){" << magnetic_vector_potential[3 * index + 2] << "};\n";
            }
        }
    }
    stream << "};\n";

}

inline double centered_first_derivative(
        std::vector<double> const& coordinates,
        auto const& value_at,
        std::size_t index)
{
    if (coordinates.size() <= 1) {
        return 0.0;
    }
    if (index == 0) {
        return (value_at(1) - value_at(0)) / (coordinates[1] - coordinates[0]);
    }
    if (index + 1 == coordinates.size()) {
        return (value_at(index) - value_at(index - 1))
               / (coordinates[index] - coordinates[index - 1]);
    }
    return (value_at(index + 1) - value_at(index - 1))
           / (coordinates[index + 1] - coordinates[index - 1]);
}

} // namespace detail

class LinearMagnetostaticsOnelabInterface : public BaseOnelabInterface
{
public:
    LinearMagnetostaticsOnelabInterface() : BaseOnelabInterface("SimiLie") {}
    ~LinearMagnetostaticsOnelabInterface() override = default;

protected:
    void publish_module_parameters() override
    {
        onelab::string formulation = get_or_create_string(
                control_parameter_name("Formulation"),
                "Linear magnetostatics",
                "Formulation",
                "Structured-grid stationary magnetostatics solved by SimiLie.");
        formulation.setReadOnly(true);
        client().set(formulation);

        onelab::number coil_current = get_or_create_number(
                control_parameter_name("Coil current (rms) [A]"),
                10.0,
                "Coil current (rms) [A]",
                "Fallback current value used when the driving ONELAB model does not provide one.",
                0.0,
                1.e9,
                1.0);
        client().set(coil_current);

        onelab::number number_of_turns = get_or_create_number(
                control_parameter_name("Number of turns"),
                288.0,
                "Number of turns",
                "Fallback number of turns used when the driving ONELAB model does not provide one.",
                1.0,
                1.e9,
                1.0);
        client().set(number_of_turns);

        onelab::number core_relative_permeability = get_or_create_number(
                control_parameter_name("Core relative permeability"),
                2000.0,
                "Core relative permeability",
                "Fallback core relative permeability used when the driving ONELAB model does not provide one.",
                1.0,
                1.e9,
                1.0);
        client().set(core_relative_permeability);

        onelab::number coil_width = get_or_create_number(
                control_parameter_name("Coil width [m]"),
                0.03,
                "Coil width [m]",
                "Fallback coil width used to derive the current density from the current and the number of turns.",
                0.0,
                1.e9,
                1.e-3);
        client().set(coil_width);

        onelab::number coil_height = get_or_create_number(
                control_parameter_name("Coil height [m]"),
                0.09,
                "Coil height [m]",
                "Fallback coil height used to derive the current density from the current and the number of turns.",
                0.0,
                1.e9,
                1.e-3);
        client().set(coil_height);

        onelab::number export_result_view = get_or_create_number(
                control_parameter_name("Export input fields view"),
                1.0,
                "Export result view",
                "When enabled, the interface writes a .pos file containing the inputs and the computed magnetostatics fields.",
                0.0,
                1.0,
                1.0);
        export_result_view.setChoices({0.0, 1.0});
        export_result_view.setValueLabels({{0.0, "No"}, {1.0, "Yes"}});
        client().set(export_result_view);

        onelab::string output_view_file = get_or_create_string(
                control_parameter_name("Input fields view file"),
                "",
                "Result view file",
                "Optional output .pos file used to visualize the permeability, source current density and computed magnetostatics fields.");
        output_view_file.setKind("file");
        client().set(output_view_file);

        onelab::number merge_result_view = get_or_create_number(
                control_parameter_name("Merge result view in Gmsh"),
                0.0,
                "Merge result view in Gmsh",
                "When enabled, Gmsh immediately merges the exported .pos result view after the SimiLie run.",
                0.0,
                1.0,
                1.0);
        merge_result_view.setChoices({0.0, 1.0});
        merge_result_view.setValueLabels({{0.0, "No"}, {1.0, "Yes"}});
        client().set(merge_result_view);
    }

    void run_module() override
    {
        client().sendProgress(module_name() + " ONELAB interface: exporting mesh for linear magnetostatics");
        std::filesystem::path const mesh_file = export_input_mesh_from_gmsh();

        double const current_rms
                = read_number_parameter("Input/4Coil Parameters/0Current (rms) [A]",
                                        control_parameter_name("Coil current (rms) [A]"),
                                        10.0);
        double const number_of_turns
                = read_number_parameter("Input/4Coil Parameters/1Number of turns",
                                        control_parameter_name("Number of turns"),
                                        288.0);
        double const core_relative_permeability
                = read_number_parameter("Input/42Core relative permeability",
                                        control_parameter_name("Core relative permeability"),
                                        2000.0);
        double const coil_width
                = read_number_parameter("Input/10Geometric dimensions/03Coil width [m]",
                                        control_parameter_name("Coil width [m]"),
                                        0.03);
        double const coil_height
                = read_number_parameter("Input/10Geometric dimensions/04Coil height [m]",
                                        control_parameter_name("Coil height [m]"),
                                        0.09);
        bool const export_result_view
                = (get_first_number_value(control_parameter_name("Export input fields view"), 1.0)
                   != 0.0);
        bool const merge_result_view
                = (get_first_number_value(control_parameter_name("Merge result view in Gmsh"), 0.0)
                   != 0.0);
        std::filesystem::path output_view_file
                = get_first_string_value(control_parameter_name("Input fields view file"));
        if (output_view_file.empty()) {
            output_view_file = mesh_file.parent_path() / "similie_linear_magnetostatics_inputs.pos";
        }

        double const mu0 = 4.e-7 * std::numbers::pi_v<double>;
        double const core_mu = core_relative_permeability * mu0;
        double const coil_section = coil_width * coil_height;
        if (coil_section <= 0.0) {
            throw std::runtime_error("the coil width and height must define a strictly positive section");
        }
        double const current_density_magnitude
                = std::sqrt(2.0) * current_rms * number_of_turns / coil_section;

        detail::HexahedralMesh const mesh = detail::parse_hexahedral_msh2_mesh(mesh_file);
        detail::StructuredGrid const grid = detail::build_structured_grid(mesh);
        client().sendInfo("SimiLie structured rectilinear mesh validated");

        std::size_t const num_cells = grid.ncell_x() * grid.ncell_y() * grid.ncell_z();
        std::vector<detail::CellInputFields> cell_inputs(num_cells);
        std::size_t num_air_cells = 0;
        std::size_t num_core_cells = 0;
        std::size_t num_coil_cells = 0;

        for (std::size_t cell_index = 0; cell_index < num_cells; ++cell_index) {
            detail::CellInputFields field {
                    .mu = mu0,
                    .current_density = {0.0, 0.0, 0.0},
            };
            int const physical_tag = grid.ordered_cells[cell_index].physical_tag;
            if (physical_tag == detail::ECORE_TAG || physical_tag == detail::ICORE_TAG) {
                field.mu = core_mu;
                ++num_core_cells;
            } else if (physical_tag == detail::COIL_LEFT_TAG) {
                field.current_density[2] = current_density_magnitude;
                ++num_coil_cells;
            } else if (physical_tag == detail::COIL_RIGHT_TAG) {
                field.current_density[2] = -current_density_magnitude;
                ++num_coil_cells;
            } else {
                ++num_air_cells;
            }
            cell_inputs[cell_index] = field;
        }

        std::size_t const num_nodes = grid.nx() * grid.ny() * grid.nz();
        Kokkos::View<double*> x_coords("similie_x_coords", grid.nx());
        Kokkos::View<double*> y_coords("similie_y_coords", grid.ny());
        auto x_coords_host = Kokkos::create_mirror_view(x_coords);
        auto y_coords_host = Kokkos::create_mirror_view(y_coords);
        for (std::size_t i = 0; i < grid.nx(); ++i) {
            x_coords_host(i) = grid.x_coords[i];
        }
        for (std::size_t i = 0; i < grid.ny(); ++i) {
            y_coords_host(i) = grid.y_coords[i];
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
                                     : mu0 * accumulated_current_density_z / static_cast<double>(count);
            }
        }
        Kokkos::deep_copy(rhs, rhs_host);
        client().sendInfo("SimiLie right-hand side assembled on rectilinear nodes");

        physics::magnetostatics::StructuredScalarPoissonStrongFormOperator2D<
                typename Kokkos::DefaultExecutionSpace::memory_space>
                operator_model(x_coords, y_coords);
        physics::dedonder_weyl::StationaryStrongFormulation formulation {operator_model};
        client().sendInfo("SimiLie starting matrix-free conjugate-gradient solve");
        solvers::StrongFormulationSolverSettings solver_settings;
        solver_settings.max_iterations = 2000U;
        solver_settings.relative_tolerance = 1.0e-10;
        solvers::StrongFormulationSolverDiagnostics const solver_diagnostics
                = solvers::minimize_strong_formulation_residual(
                Kokkos::DefaultExecutionSpace(),
                formulation,
                rhs,
                magnetic_vector_potential_z_xy_view,
                solver_settings);
        client().sendInfo("SimiLie matrix-free conjugate-gradient solve finished");

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

        physics::magnetostatics::MagneticVectorPotentialToMagneticInduction curl_operator;
        std::vector<detail::CellPostProcessFields> cell_outputs(num_cells);
        double max_abs_potential = 0.0;
        double max_abs_induction = 0.0;
        double max_abs_field = 0.0;
        double air_gap_induction_magnitude_sum = 0.0;
        std::size_t num_air_gap_cells = 0;
        double force_density_magnitude_sum = 0.0;
        for (double value : magnetic_vector_potential) {
            max_abs_potential = std::max(max_abs_potential, std::abs(value));
        }
        client().sendInfo("SimiLie starting magnetostatics post-processing");

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
            if (axis == 'y') {
                return ((node_value_z(i, j + 1, k) - node_value_z(i, j, k))
                        + (node_value_z(i + 1, j + 1, k) - node_value_z(i + 1, j, k))
                        + (node_value_z(i, j + 1, k + 1) - node_value_z(i, j, k + 1))
                        + (node_value_z(i + 1, j + 1, k + 1) - node_value_z(i + 1, j, k + 1)))
                       / (4.0 * dy);
            }
            return 0.0;
        };

        for (std::size_t k = 0; k < grid.ncell_z(); ++k) {
            for (std::size_t j = 0; j < grid.ncell_y(); ++j) {
                for (std::size_t i = 0; i < grid.ncell_x(); ++i) {
                    std::size_t const cell_index = grid.cell_index(i, j, k);

                    std::array<double, physics::magnetostatics::MagneticInductionIndex::access_size()>
                            magnetic_induction_storage {};
                    std::array<double, physics::magnetostatics::MagneticFieldIndex::access_size()>
                            magnetic_field_storage {};
                    auto magnetic_induction = physics::magnetostatics::detail::
                            make_local_tensor<physics::magnetostatics::MagneticInductionIndex>(
                                    magnetic_induction_storage);
                    auto magnetic_field = physics::magnetostatics::detail::
                            make_local_tensor<physics::magnetostatics::MagneticFieldIndex>(
                                    magnetic_field_storage);

                    curl_operator.forward(
                            magnetic_induction,
                            magnetic_field,
                            0.0,
                            derivative_az_at_cell(i, j, k, 'y'),
                            derivative_az_at_cell(i, j, k, 'x'),
                            0.0,
                            0.0,
                            0.0);

                    physics::magnetostatics::LinearMagneticInductionToMagneticField constitutive_law(
                            cell_inputs[cell_index].mu);
                    constitutive_law.forward(magnetic_field, magnetic_induction);

                    detail::CellPostProcessFields cell_output {};
                    cell_output.magnetic_induction = std::array<double, 3> {
                            magnetic_induction(magnetic_induction.template access_element<
                                               physics::magnetostatics::Y,
                                               physics::magnetostatics::Z>()),
                            -magnetic_induction(magnetic_induction.template access_element<
                                                physics::magnetostatics::X,
                                                physics::magnetostatics::Z>()),
                            magnetic_induction(magnetic_induction.template access_element<
                                               physics::magnetostatics::X,
                                               physics::magnetostatics::Y>()),
                    };
                    cell_output.magnetic_field = std::array<double, 3> {
                            magnetic_field(magnetic_field.template access_element<
                                           physics::magnetostatics::X>()),
                            magnetic_field(magnetic_field.template access_element<
                                           physics::magnetostatics::Y>()),
                            magnetic_field(magnetic_field.template access_element<
                                           physics::magnetostatics::Z>()),
                    };
                    double const half_trace = 0.5
                                              * (cell_output.magnetic_induction[0]
                                                         * cell_output.magnetic_field[0]
                                                 + cell_output.magnetic_induction[1]
                                                           * cell_output.magnetic_field[1]
                                                 + cell_output.magnetic_induction[2]
                                                           * cell_output.magnetic_field[2]);
                    cell_output.maxwell_stress = std::array<double, 6> {
                            cell_output.magnetic_induction[0] * cell_output.magnetic_field[0]
                                    - half_trace,
                            cell_output.magnetic_induction[1] * cell_output.magnetic_field[1]
                                    - half_trace,
                            cell_output.magnetic_induction[2] * cell_output.magnetic_field[2]
                                    - half_trace,
                            cell_output.magnetic_induction[0] * cell_output.magnetic_field[1],
                            cell_output.magnetic_induction[0] * cell_output.magnetic_field[2],
                            cell_output.magnetic_induction[1] * cell_output.magnetic_field[2],
                    };
                    cell_outputs[cell_index] = cell_output;

                    for (double value : cell_output.magnetic_induction) {
                        max_abs_induction = std::max(max_abs_induction, std::abs(value));
                    }
                    for (double value : cell_output.magnetic_field) {
                        max_abs_field = std::max(max_abs_field, std::abs(value));
                    }
                    if (grid.ordered_cells[cell_index].physical_tag == detail::AIRGAP_TAG) {
                        double const induction_magnitude = std::sqrt(
                                cell_output.magnetic_induction[0] * cell_output.magnetic_induction[0]
                                + cell_output.magnetic_induction[1] * cell_output.magnetic_induction[1]
                                + cell_output.magnetic_induction[2] * cell_output.magnetic_induction[2]);
                        air_gap_induction_magnitude_sum += induction_magnitude;
                        ++num_air_gap_cells;
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
                            return detail::centered_first_derivative(
                                    cell_x_coords,
                                    [&](std::size_t index) {
                                        std::size_t const clamped = std::min(index, grid.ncell_x() - 1);
                                        return stress_component(clamped, j, k, component);
                                    },
                                    i);
                        }
                        if (axis == 'y') {
                            return detail::centered_first_derivative(
                                    cell_y_coords,
                                    [&](std::size_t index) {
                                        std::size_t const clamped = std::min(index, grid.ncell_y() - 1);
                                        return stress_component(i, clamped, k, component);
                                    },
                                    j);
                        }
                        return detail::centered_first_derivative(
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
                    force_density_magnitude_sum += std::sqrt(
                            force_density[0] * force_density[0]
                            + force_density[1] * force_density[1]
                            + force_density[2] * force_density[2]);
                }
            }
        }

        if (export_result_view) {
            detail::write_results_view(
                    output_view_file,
                    grid,
                    cell_inputs,
                    magnetic_vector_potential);
            if (merge_result_view) {
                client().sendMergeFileRequest(output_view_file.string());
            }
        }
        client().sendInfo("SimiLie magnetostatics post-processing exported");

        publish_output_string(
                "Input mesh file",
                mesh_file.string(),
                "Input mesh file",
                "Mesh file exported by Gmsh for the linear magnetostatics interface.",
                "file");
        publish_output_string(
                "Input fields view file",
                output_view_file.string(),
                "Result view file",
                "View file containing the input fields and computed stationary magnetostatics fields.",
                "file");
        publish_output_number(
                "Current density magnitude [A/m^2]",
                current_density_magnitude,
                "Current density magnitude [A/m^2]",
                "Current density magnitude derived from the current, the number of turns and the coil section.");
        publish_output_number(
                "Air permeability [H/m]",
                mu0,
                "Air permeability [H/m]",
                "Magnetic permeability used in air and coil cells.");
        publish_output_number(
                "Core permeability [H/m]",
                core_mu,
                "Core permeability [H/m]",
                "Magnetic permeability used in core cells.");
        publish_output_number(
                "Number of air cells",
                static_cast<double>(num_air_cells),
                "Number of air cells",
                "Number of hexahedral cells tagged as air or air gap.");
        publish_output_number(
                "Number of core cells",
                static_cast<double>(num_core_cells),
                "Number of core cells",
                "Number of hexahedral cells tagged as core.");
        publish_output_number(
                "Number of coil cells",
                static_cast<double>(num_coil_cells),
                "Number of coil cells",
                "Number of hexahedral cells tagged as coil.");
        publish_output_number(
                "Solver iterations",
                static_cast<double>(solver_diagnostics.iterations),
                "Solver iterations",
                "Number of conjugate-gradient iterations performed by the stationary strong-formulation solver.");
        publish_output_number(
                "Solver converged",
                solver_diagnostics.converged ? 1.0 : 0.0,
                "Solver converged",
                "Equals 1 when the stationary strong-formulation solver met its relative-residual target, 0 otherwise.");
        publish_output_number(
                "Final residual L2",
                solver_diagnostics.final_residual_l2,
                "Final residual L2",
                "Final L2 norm of the strong-formulation residual returned by the stationary solver.");
        publish_output_number(
                "Final relative residual",
                solver_diagnostics.final_relative_residual,
                "Final relative residual",
                "Final residual divided by the initial residual, as returned by the stationary solver.");
        publish_output_number(
                "Maximum magnetic vector potential [SI]",
                max_abs_potential,
                "Maximum magnetic vector potential [SI]",
                "Maximum absolute nodal magnetic vector potential over the structured mesh.");
        publish_output_number(
                "Maximum magnetic induction [T]",
                max_abs_induction,
                "Maximum magnetic induction [T]",
                "Maximum absolute cell-centered magnetic induction over the structured mesh.");
        publish_output_number(
                "Maximum magnetic field [A/m]",
                max_abs_field,
                "Maximum magnetic field [A/m]",
                "Maximum absolute cell-centered magnetic field over the structured mesh.");
        publish_output_number(
                "Air-gap mean magnetic induction [T]",
                num_air_gap_cells == 0
                        ? 0.0
                        : air_gap_induction_magnitude_sum / static_cast<double>(num_air_gap_cells),
                "Air-gap mean magnetic induction [T]",
                "Mean magnetic-induction magnitude over the hexahedral cells tagged as air gap.");
        publish_output_number(
                "Mean force density magnitude [N/m^3]",
                num_cells == 0 ? 0.0 : force_density_magnitude_sum / static_cast<double>(num_cells),
                "Mean force density magnitude [N/m^3]",
                "Mean force-density magnitude over all hexahedral cells of the structured mesh.");
        {
            std::ostringstream summary;
            summary << "SimiLie solver diagnostics: iterations=" << solver_diagnostics.iterations
                    << ", final residual L2=" << solver_diagnostics.final_residual_l2
                    << ", final relative residual=" << solver_diagnostics.final_relative_residual
                    << ", air-gap mean |B|="
                    << (num_air_gap_cells == 0
                                ? 0.0
                                : air_gap_induction_magnitude_sum
                                          / static_cast<double>(num_air_gap_cells))
                    << " T, mean |f|="
                    << (num_cells == 0 ? 0.0 : force_density_magnitude_sum / static_cast<double>(num_cells))
                    << " N/m^3";
            client().sendInfo(summary.str());
        }
        publish_status("linear magnetostatics solved on the rectilinear grid");
    }

private:
    double read_number_parameter(
            std::string const& external_name,
            std::string const& fallback_name,
            double default_value)
    {
        double const external_value
                = get_first_number_value(external_name, std::numeric_limits<double>::quiet_NaN());
        if (!std::isnan(external_value)) {
            return external_value;
        }
        return get_first_number_value(fallback_name, default_value);
    }
};

} // namespace similie::onelab_interface
