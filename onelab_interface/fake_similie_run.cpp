// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: MIT

#include "fake_similie_run.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include <ddc/ddc.hpp>

#include <similie/similie.hpp>

namespace similie::onelab_interface {
namespace {

struct GridX
{
};

struct GridY
{
};

struct GridZ
{
};

struct Coord
{
};

struct MeshNode
{
    std::size_t tag;
    double x;
    double y;
    double z;
};

struct Hexahedron
{
    std::array<std::size_t, 8> node_tags;
};

struct RectilinearGridData
{
    std::vector<MeshNode> nodes;
    std::vector<Hexahedron> cells;
    std::vector<double> x_coordinates;
    std::vector<double> y_coordinates;
    std::vector<double> z_coordinates;
};

int element_dimension(int element_type)
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

bool is_supported_boundary_element(int element_type)
{
    return element_type == 1 || element_type == 2 || element_type == 3 || element_type == 15;
}

bool is_supported_topological_cell(int element_type)
{
    return element_type == 3 || element_type == 5;
}

std::filesystem::path default_output_file()
{
    return std::filesystem::path(SIMILIE_ONELAB_DEFAULT_OUTPUT_DIR)
           / "similie_rectilinear_positions.pos";
}

RectilinearGridData parse_msh2_mesh(std::filesystem::path const& mesh_file)
{
    std::ifstream stream(mesh_file);
    if (!stream.is_open()) {
        throw std::runtime_error("failed to open mesh file: " + mesh_file.string());
    }

    RectilinearGridData grid;
    std::string token;
    bool saw_nodes_section = false;
    bool saw_elements_section = false;
    int mesh_dimension = -1;
    bool found_supported_cell_type = false;

    while (stream >> token) {
        if (token == "$MeshFormat") {
            double version = 0.0;
            int file_type = 0;
            int data_size = 0;
            stream >> version >> file_type >> data_size;
            if (version < 2.0 || version >= 3.0) {
                throw std::runtime_error("only Gmsh msh2 ASCII meshes are supported");
            }
            if (file_type != 0) {
                throw std::runtime_error("only ASCII meshes are supported");
            }
        } else if (token == "$Nodes") {
            saw_nodes_section = true;
            std::size_t node_count = 0;
            stream >> node_count;
            grid.nodes.reserve(node_count);
            for (std::size_t i = 0; i < node_count; ++i) {
                MeshNode node;
                stream >> node.tag >> node.x >> node.y >> node.z;
                grid.nodes.push_back(node);
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

                for (std::size_t tag_id = 0; tag_id < num_tags; ++tag_id) {
                    std::size_t ignored_tag = 0;
                    stream >> ignored_tag;
                }

                int const current_dimension = element_dimension(element_type);
                if (current_dimension < 0) {
                    std::ostringstream error_stream;
                    error_stream << "unsupported Gmsh element type " << element_type
                                 << " found in mesh";
                    throw std::runtime_error(error_stream.str());
                }
                mesh_dimension = std::max(mesh_dimension, current_dimension);

                if (element_type == 5) {
                    found_supported_cell_type = true;
                    Hexahedron cell;
                    for (std::size_t k = 0; k < cell.node_tags.size(); ++k) {
                        stream >> cell.node_tags[k];
                    }
                    grid.cells.push_back(cell);
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
                    if (is_supported_topological_cell(element_type)) {
                        found_supported_cell_type = true;
                    }
                    std::ostringstream error_stream;
                    error_stream
                            << "unsupported cell type in mesh: SimiLie currently requires the "
                               "whole mesh to be made of quadrilaterals or hexahedra";
                    throw std::runtime_error(error_stream.str());
                }
            }
        }
    }

    if (grid.nodes.empty()) {
        throw std::runtime_error("the provided mesh does not contain any node");
    }
    if (grid.cells.empty()) {
        if (saw_elements_section && mesh_dimension >= 0) {
            std::ostringstream error_stream;
            error_stream << "unsupported mesh topology: SimiLie currently requires the whole "
                            "mesh to be made of quadrilaterals or hexahedra";
            if (!found_supported_cell_type) {
                error_stream << " (no quadrilateral or hexahedral cell was found)";
            }
            throw std::runtime_error(error_stream.str());
        }
        throw std::runtime_error("the provided mesh does not contain any hexahedral cell");
    }

    return grid;
}

std::vector<double> unique_sorted_coordinates(std::vector<double> coordinates)
{
    std::sort(coordinates.begin(), coordinates.end());
    coordinates.erase(
            std::unique(
                    coordinates.begin(),
                    coordinates.end(),
                    [](double left, double right) { return std::abs(left - right) < 1.e-12; }),
            coordinates.end());
    return coordinates;
}

std::size_t nearest_coordinate_index(std::vector<double> const& coordinates, double value)
{
    auto const iterator = std::lower_bound(coordinates.begin(), coordinates.end(), value - 1.e-12);
    if (iterator != coordinates.end() && std::abs(*iterator - value) < 1.e-12) {
        return static_cast<std::size_t>(std::distance(coordinates.begin(), iterator));
    }
    if (iterator != coordinates.begin()) {
        auto const previous = std::prev(iterator);
        if (std::abs(*previous - value) < 1.e-12) {
            return static_cast<std::size_t>(std::distance(coordinates.begin(), previous));
        }
    }
    throw std::runtime_error("node coordinate does not match the detected rectilinear grid");
}

using grid_triplet_t = std::tuple<std::size_t, std::size_t, std::size_t>;

grid_triplet_t make_grid_triplet(std::size_t ix, std::size_t iy, std::size_t iz)
{
    return std::make_tuple(ix, iy, iz);
}

RectilinearGridData validate_rectilinear_grid(RectilinearGridData grid)
{
    std::vector<double> x_coordinates;
    std::vector<double> y_coordinates;
    std::vector<double> z_coordinates;
    x_coordinates.reserve(grid.nodes.size());
    y_coordinates.reserve(grid.nodes.size());
    z_coordinates.reserve(grid.nodes.size());

    for (MeshNode const& node : grid.nodes) {
        x_coordinates.push_back(node.x);
        y_coordinates.push_back(node.y);
        z_coordinates.push_back(node.z);
    }

    grid.x_coordinates = unique_sorted_coordinates(x_coordinates);
    grid.y_coordinates = unique_sorted_coordinates(y_coordinates);
    grid.z_coordinates = unique_sorted_coordinates(z_coordinates);

    std::size_t const nx = grid.x_coordinates.size();
    std::size_t const ny = grid.y_coordinates.size();
    std::size_t const nz = grid.z_coordinates.size();

    if (nx < 2 || ny < 2 || nz < 2) {
        throw std::runtime_error("the mesh is not a three-dimensional rectilinear grid");
    }
    if (nx * ny * nz != grid.nodes.size()) {
        throw std::runtime_error("the mesh nodes do not form a full rectilinear tensor-product grid");
    }

    std::map<std::size_t, grid_triplet_t> node_to_indices;
    std::set<grid_triplet_t> occupied_nodes;
    for (MeshNode const& node : grid.nodes) {
        std::size_t const ix = nearest_coordinate_index(grid.x_coordinates, node.x);
        std::size_t const iy = nearest_coordinate_index(grid.y_coordinates, node.y);
        std::size_t const iz = nearest_coordinate_index(grid.z_coordinates, node.z);
        grid_triplet_t const key = make_grid_triplet(ix, iy, iz);
        if (!occupied_nodes.insert(key).second) {
            throw std::runtime_error("the mesh contains duplicated nodes on the rectilinear grid");
        }
        node_to_indices.emplace(node.tag, key);
    }

    if (grid.cells.size() != (nx - 1) * (ny - 1) * (nz - 1)) {
        throw std::runtime_error("the hexahedra count does not match a full rectilinear grid");
    }

    std::set<grid_triplet_t> occupied_cells;
    for (Hexahedron const& cell : grid.cells) {
        std::set<grid_triplet_t> corners;
        std::size_t min_x = nx - 1;
        std::size_t min_y = ny - 1;
        std::size_t min_z = nz - 1;
        std::size_t max_x = 0;
        std::size_t max_y = 0;
        std::size_t max_z = 0;

        for (std::size_t node_tag : cell.node_tags) {
            auto const iterator = node_to_indices.find(node_tag);
            if (iterator == node_to_indices.end()) {
                throw std::runtime_error("a cell references an unknown node");
            }
            auto const& [ix, iy, iz] = iterator->second;
            corners.insert(iterator->second);
            min_x = std::min(min_x, ix);
            min_y = std::min(min_y, iy);
            min_z = std::min(min_z, iz);
            max_x = std::max(max_x, ix);
            max_y = std::max(max_y, iy);
            max_z = std::max(max_z, iz);
        }

        if (corners.size() != 8 || max_x != min_x + 1 || max_y != min_y + 1
            || max_z != min_z + 1) {
            throw std::runtime_error("a hexahedron does not match a rectilinear cell of the grid");
        }

        for (std::size_t dx : {0U, 1U}) {
            for (std::size_t dy : {0U, 1U}) {
                for (std::size_t dz : {0U, 1U}) {
                    if (!corners.contains(make_grid_triplet(min_x + dx, min_y + dy, min_z + dz))) {
                        throw std::runtime_error("a hexahedron does not contain the expected grid corners");
                    }
                }
            }
        }

        grid_triplet_t const cell_key = make_grid_triplet(min_x, min_y, min_z);
        if (!occupied_cells.insert(cell_key).second) {
            throw std::runtime_error("the mesh contains duplicated hexahedra");
        }
    }

    return grid;
}

double store_positions_in_tensor(RectilinearGridData const& grid)
{
    ddc::DiscreteDomain<GridX, GridY, GridZ, Coord> const tensor_domain(
            ddc::DiscreteElement<GridX, GridY, GridZ, Coord>(
                    ddc::DiscreteElement<GridX>(0),
                    ddc::DiscreteElement<GridY>(0),
                    ddc::DiscreteElement<GridZ>(0),
                    ddc::DiscreteElement<Coord>(0)),
            ddc::DiscreteVector<GridX, GridY, GridZ, Coord>(
                    ddc::DiscreteVector<GridX>(grid.x_coordinates.size()),
                    ddc::DiscreteVector<GridY>(grid.y_coordinates.size()),
                    ddc::DiscreteVector<GridZ>(grid.z_coordinates.size()),
                    ddc::DiscreteVector<Coord>(3)));

    ddc::Chunk tensor_alloc(tensor_domain, ddc::HostAllocator<double>());
    sil::tensor::Tensor positions_tensor(tensor_alloc);

    double checksum = 0.0;
    for (std::size_t ix = 0; ix < grid.x_coordinates.size(); ++ix) {
        for (std::size_t iy = 0; iy < grid.y_coordinates.size(); ++iy) {
            for (std::size_t iz = 0; iz < grid.z_coordinates.size(); ++iz) {
                auto const x_elem = ddc::DiscreteElement<GridX>(ix);
                auto const y_elem = ddc::DiscreteElement<GridY>(iy);
                auto const z_elem = ddc::DiscreteElement<GridZ>(iz);
                positions_tensor(x_elem, y_elem, z_elem, ddc::DiscreteElement<Coord>(0))
                        = grid.x_coordinates[ix];
                positions_tensor(x_elem, y_elem, z_elem, ddc::DiscreteElement<Coord>(1))
                        = grid.y_coordinates[iy];
                positions_tensor(x_elem, y_elem, z_elem, ddc::DiscreteElement<Coord>(2))
                        = grid.z_coordinates[iz];
                checksum += positions_tensor.get(
                                    x_elem,
                                    y_elem,
                                    z_elem,
                                    ddc::DiscreteElement<Coord>(0))
                            + 10.0 * positions_tensor.get(
                                               x_elem,
                                               y_elem,
                                               z_elem,
                                               ddc::DiscreteElement<Coord>(1))
                            + 100.0 * positions_tensor.get(
                                                x_elem,
                                                y_elem,
                                                z_elem,
                                                ddc::DiscreteElement<Coord>(2));
            }
        }
    }

    return checksum;
}

void write_positions_view(std::filesystem::path const& output_file, RectilinearGridData const& grid)
{
    if (!output_file.parent_path().empty()) {
        std::filesystem::create_directories(output_file.parent_path());
    }

    std::ofstream stream(output_file);
    if (!stream.is_open()) {
        throw std::runtime_error("failed to open output view file: " + output_file.string());
    }

    stream << "View \"SimiLie rectilinear node positions\" {\n";
    for (MeshNode const& node : grid.nodes) {
        stream << "  VP(" << node.x << ", " << node.y << ", " << node.z << "){"
               << node.x << ", " << node.y << ", " << node.z << "};\n";
    }
    stream << "};\n";
}

} // namespace

FakeRunResult run_fake_similie_job(FakeRunConfig const& config)
{
    if (config.input_mesh_file.empty()) {
        throw std::runtime_error("no input mesh file was provided to the ONELAB interface");
    }

    RectilinearGridData const grid = validate_rectilinear_grid(parse_msh2_mesh(config.input_mesh_file));
    double const checksum = store_positions_in_tensor(grid);

    FakeRunResult result;
    result.status = "rectilinear mesh accepted";
    result.nx = grid.x_coordinates.size();
    result.ny = grid.y_coordinates.size();
    result.nz = grid.z_coordinates.size();
    result.num_nodes = grid.nodes.size();
    result.num_cells = grid.cells.size();
    result.checksum = checksum;
    result.output_file = config.output_file.empty() ? default_output_file() : config.output_file;

    if (config.export_view) {
        write_positions_view(result.output_file, grid);
    }

    return result;
}

} // namespace similie::onelab_interface
