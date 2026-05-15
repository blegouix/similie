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
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "base_onelab_interface.hpp"

namespace similie::onelab_interface {

namespace detail {

constexpr int ECORE_TAG = 1000;
constexpr int ICORE_TAG = 1100;
constexpr int COIL_LEFT_TAG = 2000;
constexpr int COIL_RIGHT_TAG = 2001;
constexpr int AIR_TAG = 3000;
constexpr int AIRGAP_TAG = 3200;

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

struct CellInputFields
{
    double center_x;
    double center_y;
    double center_z;
    double mu;
    double jx;
    double jy;
    double jz;
};

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

inline std::map<std::size_t, MeshNode> make_node_map(HexahedralMesh const& mesh)
{
    std::map<std::size_t, MeshNode> node_map;
    for (MeshNode const& node : mesh.nodes) {
        node_map.emplace(node.tag, node);
    }
    return node_map;
}

inline void write_input_fields_view(
        std::filesystem::path const& output_file,
        std::vector<CellInputFields> const& cell_fields)
{
    if (!output_file.parent_path().empty()) {
        std::filesystem::create_directories(output_file.parent_path());
    }

    std::ofstream stream(output_file);
    if (!stream.is_open()) {
        throw std::runtime_error("failed to open output view file: " + output_file.string());
    }

    stream << "View \"SimiLie linear magnetostatics permeability\" {\n";
    for (CellInputFields const& field : cell_fields) {
        stream << "  SP(" << field.center_x << ", " << field.center_y << ", " << field.center_z
               << "){" << field.mu << "};\n";
    }
    stream << "};\n";

    stream << "View \"SimiLie linear magnetostatics current density\" {\n";
    for (CellInputFields const& field : cell_fields) {
        stream << "  VP(" << field.center_x << ", " << field.center_y << ", " << field.center_z
               << "){" << field.jx << ", " << field.jy << ", " << field.jz << "};\n";
    }
    stream << "};\n";
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
                "The magnetostatics ONELAB entry point currently prepared by SimiLie.");
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

        onelab::number export_input_fields_view = get_or_create_number(
                control_parameter_name("Export input fields view"),
                1.0,
                "Export input fields view",
                "When enabled, the interface writes a .pos file containing the permeability and current density inputs.",
                0.0,
                1.0,
                1.0);
        export_input_fields_view.setChoices({0.0, 1.0});
        export_input_fields_view.setValueLabels({{0.0, "No"}, {1.0, "Yes"}});
        client().set(export_input_fields_view);

        onelab::string output_view_file = get_or_create_string(
                control_parameter_name("Input fields view file"),
                "",
                "Input fields view file",
                "Optional output .pos file used to visualize the permeability and current density prepared by the linear magnetostatics interface.");
        output_view_file.setKind("file");
        client().set(output_view_file);
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
        bool const export_input_fields_view
                = (get_first_number_value(control_parameter_name("Export input fields view"), 1.0)
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
        std::map<std::size_t, detail::MeshNode> const node_map = detail::make_node_map(mesh);

        std::vector<detail::CellInputFields> cell_fields;
        cell_fields.reserve(mesh.cells.size());

        std::size_t num_air_cells = 0;
        std::size_t num_core_cells = 0;
        std::size_t num_coil_cells = 0;

        for (detail::HexahedralCell const& cell : mesh.cells) {
            double center_x = 0.0;
            double center_y = 0.0;
            double center_z = 0.0;
            for (std::size_t const node_tag : cell.node_tags) {
                auto const iterator = node_map.find(node_tag);
                if (iterator == node_map.end()) {
                    throw std::runtime_error("a cell references an unknown node");
                }
                center_x += iterator->second.x;
                center_y += iterator->second.y;
                center_z += iterator->second.z;
            }
            center_x /= static_cast<double>(cell.node_tags.size());
            center_y /= static_cast<double>(cell.node_tags.size());
            center_z /= static_cast<double>(cell.node_tags.size());

            detail::CellInputFields field {
                    .center_x = center_x,
                    .center_y = center_y,
                    .center_z = center_z,
                    .mu = mu0,
                    .jx = 0.0,
                    .jy = 0.0,
                    .jz = 0.0,
            };

            if (cell.physical_tag == detail::ECORE_TAG || cell.physical_tag == detail::ICORE_TAG) {
                field.mu = core_mu;
                ++num_core_cells;
            } else if (cell.physical_tag == detail::COIL_LEFT_TAG) {
                field.jz = current_density_magnitude;
                ++num_coil_cells;
            } else if (cell.physical_tag == detail::COIL_RIGHT_TAG) {
                field.jz = -current_density_magnitude;
                ++num_coil_cells;
            } else if (cell.physical_tag == detail::AIR_TAG || cell.physical_tag == detail::AIRGAP_TAG) {
                ++num_air_cells;
            } else {
                ++num_air_cells;
            }

            cell_fields.push_back(field);
        }

        if (export_input_fields_view) {
            detail::write_input_fields_view(output_view_file, cell_fields);
            client().sendMergeFileRequest(output_view_file.string());
        }

        publish_output_string(
                "Input mesh file",
                mesh_file.string(),
                "Input mesh file",
                "Mesh file exported by Gmsh for the linear magnetostatics interface.",
                "file");
        publish_output_string(
                "Input fields view file",
                output_view_file.string(),
                "Input fields view file",
                "View file containing the permeability and current density prepared by the linear magnetostatics interface.",
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
        publish_status("linear magnetostatics inputs prepared; solver implementation is pending");
        client().sendWarning("Linear magnetostatics solver implementation is pending.");
    }

private:
    double read_number_parameter(
            std::string const& external_name,
            std::string const& fallback_name,
            double default_value)
    {
        double const external_value = get_first_number_value(external_name, std::numeric_limits<double>::quiet_NaN());
        if (!std::isnan(external_value)) {
            return external_value;
        }
        return get_first_number_value(fallback_name, default_value);
    }
};

} // namespace similie::onelab_interface
