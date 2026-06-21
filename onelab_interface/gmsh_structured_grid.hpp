// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace sil::onelab_interface::gmsh {

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

struct QuadrilateralCell
{
    int physical_tag;
    std::array<std::size_t, 4> node_tags;
};

struct QuadrilateralMesh
{
    std::vector<MeshNode> nodes;
    std::vector<QuadrilateralCell> cells;
};

struct HexahedralMesh
{
    std::vector<MeshNode> nodes;
    std::vector<HexahedralCell> cells;
};

using SupportedMesh = std::variant<QuadrilateralMesh, HexahedralMesh>;

struct StructuredGrid2D
{
    std::vector<double> x_coords;
    std::vector<double> y_coords;
    double z_value = 0.0;
    std::map<std::size_t, std::array<std::size_t, 2>> node_indices_by_tag;
    std::vector<MeshNode> ordered_nodes;
    std::vector<QuadrilateralCell> ordered_cells;

    [[nodiscard]] std::size_t nx() const
    {
        return x_coords.size();
    }
    [[nodiscard]] std::size_t ny() const
    {
        return y_coords.size();
    }
    [[nodiscard]] std::size_t ncell_x() const
    {
        return x_coords.size() - 1;
    }
    [[nodiscard]] std::size_t ncell_y() const
    {
        return y_coords.size() - 1;
    }

    [[nodiscard]] std::size_t node_index(std::size_t i, std::size_t j) const
    {
        return i + nx() * j;
    }

    [[nodiscard]] std::size_t cell_index(std::size_t i, std::size_t j) const
    {
        return i + ncell_x() * j;
    }

    [[nodiscard]] double cell_center_x(std::size_t i) const
    {
        return 0.5 * (x_coords[i] + x_coords[i + 1]);
    }
    [[nodiscard]] double cell_center_y(std::size_t j) const
    {
        return 0.5 * (y_coords[j] + y_coords[j + 1]);
    }
};

struct StructuredGrid3D
{
    std::vector<double> x_coords;
    std::vector<double> y_coords;
    std::vector<double> z_coords;
    std::map<std::size_t, std::array<std::size_t, 3>> node_indices_by_tag;
    std::vector<MeshNode> ordered_nodes;
    std::vector<HexahedralCell> ordered_cells;

    [[nodiscard]] std::size_t nx() const
    {
        return x_coords.size();
    }
    [[nodiscard]] std::size_t ny() const
    {
        return y_coords.size();
    }
    [[nodiscard]] std::size_t nz() const
    {
        return z_coords.size();
    }
    [[nodiscard]] std::size_t ncell_x() const
    {
        return x_coords.size() - 1;
    }
    [[nodiscard]] std::size_t ncell_y() const
    {
        return y_coords.size() - 1;
    }
    [[nodiscard]] std::size_t ncell_z() const
    {
        return z_coords.size() - 1;
    }

    [[nodiscard]] std::size_t node_index(std::size_t i, std::size_t j, std::size_t k) const
    {
        return i + nx() * (j + ny() * k);
    }

    [[nodiscard]] std::size_t cell_index(std::size_t i, std::size_t j, std::size_t k) const
    {
        return i + ncell_x() * (j + ncell_y() * k);
    }

    [[nodiscard]] double cell_center_x(std::size_t i) const
    {
        return 0.5 * (x_coords[i] + x_coords[i + 1]);
    }
    [[nodiscard]] double cell_center_y(std::size_t j) const
    {
        return 0.5 * (y_coords[j] + y_coords[j + 1]);
    }
    [[nodiscard]] double cell_center_z(std::size_t k) const
    {
        return 0.5 * (z_coords[k] + z_coords[k + 1]);
    }
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

inline int nodes_per_element_type(int element_type)
{
    switch (element_type) {
    case 1:
        return 2;
    case 2:
        return 3;
    case 3:
        return 4;
    case 4:
        return 4;
    case 5:
        return 8;
    case 6:
        return 6;
    case 7:
        return 5;
    case 15:
        return 1;
    default:
        return 0;
    }
}

inline SupportedMesh parse_supported_msh2_mesh(std::filesystem::path const& mesh_file)
{
    std::ifstream stream(mesh_file);
    if (!stream.is_open()) {
        throw std::runtime_error("failed to open mesh file: " + mesh_file.string());
    }

    std::vector<MeshNode> nodes;
    std::vector<QuadrilateralCell> quadrilateral_cells;
    std::vector<HexahedralCell> hexahedral_cells;
    std::string token;
    bool saw_elements_section = false;
    int supported_topology_dimension = 0;

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
            nodes.reserve(node_count);
            for (std::size_t i = 0; i < node_count; ++i) {
                MeshNode node;
                stream >> node.tag >> node.x >> node.y >> node.z;
                nodes.push_back(node);
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

                if (element_type == 3) {
                    if (supported_topology_dimension == 0) {
                        supported_topology_dimension = 2;
                    } else if (supported_topology_dimension != 2) {
                        throw std::runtime_error(
                                "unsupported mesh topology: SimiLie currently requires the whole "
                                "mesh to be made of quadrilaterals or hexahedra");
                    }
                    QuadrilateralCell cell;
                    cell.physical_tag = tags.empty() ? 0 : tags[0];
                    for (std::size_t k = 0; k < cell.node_tags.size(); ++k) {
                        stream >> cell.node_tags[k];
                    }
                    quadrilateral_cells.push_back(cell);
                } else if (element_type == 5) {
                    if (supported_topology_dimension == 0) {
                        supported_topology_dimension = 3;
                    } else if (supported_topology_dimension != 3) {
                        throw std::runtime_error(
                                "unsupported mesh topology: SimiLie currently requires the whole "
                                "mesh to be made of quadrilaterals or hexahedra");
                    }
                    HexahedralCell cell;
                    cell.physical_tag = tags.empty() ? 0 : tags[0];
                    for (std::size_t k = 0; k < cell.node_tags.size(); ++k) {
                        stream >> cell.node_tags[k];
                    }
                    hexahedral_cells.push_back(cell);
                } else if (is_supported_boundary_element(element_type)) {
                    int const nodes_to_skip = nodes_per_element_type(element_type);
                    for (int k = 0; k < nodes_to_skip; ++k) {
                        std::size_t ignored_node = 0;
                        stream >> ignored_node;
                    }
                } else {
                    int const nodes_to_skip = nodes_per_element_type(element_type);
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

    if (nodes.empty()) {
        throw std::runtime_error("the provided mesh does not contain any node");
    }
    if (!saw_elements_section || (quadrilateral_cells.empty() && hexahedral_cells.empty())) {
        throw std::runtime_error(
                "unsupported mesh topology: SimiLie currently requires the whole mesh to be made "
                "of quadrilaterals or hexahedra");
    }

    if (!quadrilateral_cells.empty()) {
        return QuadrilateralMesh {
                .nodes = std::move(nodes),
                .cells = std::move(quadrilateral_cells),
        };
    }
    return HexahedralMesh {
            .nodes = std::move(nodes),
            .cells = std::move(hexahedral_cells),
    };
}

inline StructuredGrid2D build_structured_grid(QuadrilateralMesh const& mesh)
{
    StructuredGrid2D grid;
    grid.x_coords.reserve(mesh.nodes.size());
    grid.y_coords.reserve(mesh.nodes.size());
    std::vector<double> z_coords;
    z_coords.reserve(mesh.nodes.size());

    for (MeshNode const& node : mesh.nodes) {
        grid.x_coords.push_back(node.x);
        grid.y_coords.push_back(node.y);
        z_coords.push_back(node.z);
    }
    grid.x_coords = unique_sorted(std::move(grid.x_coords));
    grid.y_coords = unique_sorted(std::move(grid.y_coords));
    z_coords = unique_sorted(std::move(z_coords));

    if (z_coords.size() != 1) {
        throw std::runtime_error("the quadrilateral mesh nodes do not lie on a single plane");
    }
    grid.z_value = z_coords.front();

    std::vector<bool> occupied_nodes(grid.nx() * grid.ny(), false);
    grid.ordered_nodes.resize(grid.nx() * grid.ny());
    for (MeshNode const& node : mesh.nodes) {
        std::array<std::size_t, 2> const index {
                nearest_index(grid.x_coords, node.x),
                nearest_index(grid.y_coords, node.y),
        };
        std::size_t const linear_index = grid.node_index(index[0], index[1]);
        if (occupied_nodes[linear_index]) {
            MeshNode const& reference = grid.ordered_nodes[linear_index];
            if (!nearly_equal(reference.x, node.x) || !nearly_equal(reference.y, node.y)
                || !nearly_equal(reference.z, node.z)) {
                throw std::runtime_error("duplicated nodes on the detected rectilinear grid");
            }
            grid.node_indices_by_tag.emplace(node.tag, index);
            continue;
        }
        occupied_nodes[linear_index] = true;
        grid.node_indices_by_tag.emplace(node.tag, index);
        grid.ordered_nodes[linear_index] = node;
    }

    for (bool occupied_node : occupied_nodes) {
        if (!occupied_node) {
            throw std::runtime_error("the mesh nodes do not form a full rectilinear grid");
        }
    }

    std::size_t const num_cells = grid.ncell_x() * grid.ncell_y();
    if (mesh.cells.size() != num_cells) {
        throw std::runtime_error("the quadrilateral cells do not cover a full rectilinear grid");
    }

    std::vector<bool> occupied_cells(num_cells, false);
    grid.ordered_cells.resize(num_cells);
    for (QuadrilateralCell const& cell : mesh.cells) {
        std::size_t min_x = std::numeric_limits<std::size_t>::max();
        std::size_t min_y = std::numeric_limits<std::size_t>::max();
        std::size_t max_x = 0;
        std::size_t max_y = 0;

        for (std::size_t node_tag : cell.node_tags) {
            auto const iterator = grid.node_indices_by_tag.find(node_tag);
            if (iterator == grid.node_indices_by_tag.end()) {
                throw std::runtime_error("a quadrilateral references an unknown node");
            }
            min_x = std::min(min_x, iterator->second[0]);
            min_y = std::min(min_y, iterator->second[1]);
            max_x = std::max(max_x, iterator->second[0]);
            max_y = std::max(max_y, iterator->second[1]);
        }

        if (max_x != min_x + 1 || max_y != min_y + 1) {
            throw std::runtime_error(
                    "a quadrilateral does not match a single rectilinear grid cell");
        }

        std::size_t const linear_index = grid.cell_index(min_x, min_y);
        if (occupied_cells[linear_index]) {
            throw std::runtime_error("duplicated quadrilateral on the detected rectilinear grid");
        }
        occupied_cells[linear_index] = true;
        grid.ordered_cells[linear_index] = cell;
    }

    for (bool occupied_cell : occupied_cells) {
        if (!occupied_cell) {
            throw std::runtime_error(
                    "the quadrilateral cells do not cover a full rectilinear grid");
        }
    }

    return grid;
}

inline StructuredGrid3D build_structured_grid(HexahedralMesh const& mesh)
{
    StructuredGrid3D grid;
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

    std::vector<bool> occupied_nodes(grid.nx() * grid.ny() * grid.nz(), false);
    grid.ordered_nodes.resize(grid.nx() * grid.ny() * grid.nz());
    for (MeshNode const& node : mesh.nodes) {
        std::array<std::size_t, 3> const index {
                nearest_index(grid.x_coords, node.x),
                nearest_index(grid.y_coords, node.y),
                nearest_index(grid.z_coords, node.z),
        };
        std::size_t const linear_index = grid.node_index(index[0], index[1], index[2]);
        if (occupied_nodes[linear_index]) {
            MeshNode const& reference = grid.ordered_nodes[linear_index];
            if (!nearly_equal(reference.x, node.x) || !nearly_equal(reference.y, node.y)
                || !nearly_equal(reference.z, node.z)) {
                throw std::runtime_error("duplicated nodes on the detected rectilinear grid");
            }
            grid.node_indices_by_tag.emplace(node.tag, index);
            continue;
        }
        occupied_nodes[linear_index] = true;
        grid.node_indices_by_tag.emplace(node.tag, index);
        grid.ordered_nodes[linear_index] = node;
    }

    for (bool occupied_node : occupied_nodes) {
        if (!occupied_node) {
            throw std::runtime_error("the mesh nodes do not form a full rectilinear grid");
        }
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

} // namespace sil::onelab_interface::gmsh
