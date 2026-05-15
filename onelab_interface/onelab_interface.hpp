// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cctype>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <numbers>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <ddc/ddc.hpp>

#include <similie/physics/dedonder_weyl.hpp>
#include <similie/physics/magnetostatics/linear_magnetostatics.hpp>
#include <similie/physics/magnetostatics/magnetostatics_quantities.hpp>
#include <similie/physics/magnetostatics/structured_linear_magnetostatics.hpp>
#include <similie/physics/scalar_field/scalar_field_with_power_coupling.hpp>
#include <similie/solvers/minimize_strong_formulation_residual.hpp>

#include <onelab.h>

namespace similie::onelab_interface {

enum class SupportedPhysics
{
    ScalarFieldWithPowerCoupling,
    Magnetostatics,
};

enum class SupportedSolver
{
    MinimizeStrongFormulationResidual,
};

struct ScalarFieldWithPowerCouplingProblem
{
    double mass = 1.0;
    double coupling_constant = 0.0;
    double coupling_power = 4.0;
};

struct MagnetostaticsProblem
{
    std::string current_rms_parameter = "Input/4Coil Parameters/0Current (rms) [A]";
    std::string number_of_turns_parameter = "Input/4Coil Parameters/1Number of turns";
    std::string core_relative_permeability_parameter = "Input/42Core relative permeability";
    std::string coil_width_parameter = "Input/10Geometric dimensions/03Coil width [m]";
    std::string coil_height_parameter = "Input/10Geometric dimensions/04Coil height [m]";

    double current_rms = 10.0;
    double number_of_turns = 288.0;
    double core_relative_permeability = 2000.0;
    double coil_width = 0.03;
    double coil_height = 0.09;

    bool export_input_fields_view = true;
    bool merge_result_view_in_gmsh = false;
    std::string input_fields_view_file;
};

struct MinimizeStrongFormulationResidualProblem
{
    unsigned int max_iterations = 10000U;
    double relative_tolerance = 1.0e-10;
    unsigned int jacobi_max_block_size = 1U;
    bool use_matrix_free = true;
};

struct SilproProblem
{
    std::string name = "SimiLie problem";
    SupportedPhysics physics = SupportedPhysics::Magnetostatics;
    SupportedSolver solver = SupportedSolver::MinimizeStrongFormulationResidual;
    ScalarFieldWithPowerCouplingProblem scalar_field;
    MagnetostaticsProblem magnetostatics;
    MinimizeStrongFormulationResidualProblem solver_settings;
};

namespace detail {

inline std::string trim(std::string value)
{
    auto const is_space = [](unsigned char c) { return std::isspace(c) != 0; };
    auto const begin = std::find_if_not(value.begin(), value.end(), is_space);
    auto const end = std::find_if_not(value.rbegin(), value.rend(), is_space).base();
    if (begin >= end) {
        return "";
    }
    return std::string(begin, end);
}

inline std::string lowercase(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

inline bool parse_bool(std::string const& value)
{
    std::string const normalized = lowercase(trim(value));
    if (normalized == "1" || normalized == "true" || normalized == "yes" || normalized == "on") {
        return true;
    }
    if (normalized == "0" || normalized == "false" || normalized == "no" || normalized == "off") {
        return false;
    }
    throw std::runtime_error("invalid boolean value '" + value + "'");
}

template <class ValueType>
ValueType parse_number(std::string const& value)
{
    std::istringstream stream(value);
    ValueType parsed {};
    stream >> parsed;
    if (!stream || !stream.eof()) {
        throw std::runtime_error("invalid numeric value '" + value + "'");
    }
    return parsed;
}

struct SilproSection
{
    std::map<std::string, std::string> values;
    std::map<std::string, SilproSection> sections;
};

struct SilproToken
{
    enum class Kind
    {
        Word,
        String,
        LBrace,
        RBrace,
        Semicolon,
    };

    Kind kind;
    std::string text;
};

inline std::vector<SilproToken> lex_silpro(std::string const& content)
{
    std::vector<SilproToken> tokens;
    std::size_t i = 0;
    while (i < content.size()) {
        char const c = content[i];
        if (std::isspace(static_cast<unsigned char>(c)) != 0) {
            ++i;
            continue;
        }
        if (c == '#') {
            while (i < content.size() && content[i] != '\n') {
                ++i;
            }
            continue;
        }
        if (c == '/' && i + 1 < content.size() && content[i + 1] == '/') {
            i += 2;
            while (i < content.size() && content[i] != '\n') {
                ++i;
            }
            continue;
        }
        if (c == '{') {
            tokens.push_back({SilproToken::Kind::LBrace, "{"});
            ++i;
            continue;
        }
        if (c == '}') {
            tokens.push_back({SilproToken::Kind::RBrace, "}"});
            ++i;
            continue;
        }
        if (c == ';') {
            tokens.push_back({SilproToken::Kind::Semicolon, ";"});
            ++i;
            continue;
        }
        if (c == '"') {
            ++i;
            std::string text;
            while (i < content.size() && content[i] != '"') {
                if (content[i] == '\\' && i + 1 < content.size()) {
                    ++i;
                }
                text.push_back(content[i]);
                ++i;
            }
            if (i == content.size()) {
                throw std::runtime_error("unterminated string literal in .silpro file");
            }
            ++i;
            tokens.push_back({SilproToken::Kind::String, text});
            continue;
        }

        std::string text;
        while (i < content.size()) {
            char const inner = content[i];
            if (std::isspace(static_cast<unsigned char>(inner)) != 0 || inner == '{' || inner == '}'
                || inner == ';') {
                break;
            }
            if (inner == '/' && i + 1 < content.size() && content[i + 1] == '/') {
                break;
            }
            text.push_back(inner);
            ++i;
        }
        if (!text.empty()) {
            tokens.push_back({SilproToken::Kind::Word, text});
        }
    }
    return tokens;
}

class SilproParser
{
    std::vector<SilproToken> const& m_tokens;
    std::size_t m_index = 0;

public:
    explicit SilproParser(std::vector<SilproToken> const& tokens) : m_tokens(tokens) {}

    SilproSection parse_document()
    {
        SilproSection root;
        while (!at_end()) {
            std::string const section_name = parse_name();
            root.sections.emplace(section_name, parse_section_body());
        }
        return root;
    }

private:
    [[nodiscard]] bool at_end() const { return m_index >= m_tokens.size(); }

    [[nodiscard]] SilproToken const& peek() const
    {
        if (at_end()) {
            throw std::runtime_error("unexpected end of .silpro file");
        }
        return m_tokens[m_index];
    }

    SilproToken const& advance()
    {
        SilproToken const& token = peek();
        ++m_index;
        return token;
    }

    std::string parse_name()
    {
        SilproToken const& token = advance();
        if (token.kind != SilproToken::Kind::Word && token.kind != SilproToken::Kind::String) {
            throw std::runtime_error("expected identifier in .silpro file");
        }
        return token.text;
    }

    void expect(SilproToken::Kind kind, std::string const& message)
    {
        if (peek().kind != kind) {
            throw std::runtime_error(message);
        }
        advance();
    }

    SilproSection parse_section_body()
    {
        expect(SilproToken::Kind::LBrace, "expected '{' after section name in .silpro file");
        SilproSection section;
        while (peek().kind != SilproToken::Kind::RBrace) {
            std::string const name = parse_name();
            if (peek().kind == SilproToken::Kind::LBrace) {
                section.sections.emplace(name, parse_section_body());
                continue;
            }

            std::string const value = parse_name();
            expect(SilproToken::Kind::Semicolon, "expected ';' after assignment in .silpro file");
            section.values.emplace(name, value);
        }
        expect(SilproToken::Kind::RBrace, "expected '}' in .silpro file");
        return section;
    }
};

inline SilproSection parse_silpro_tree(std::filesystem::path const& file)
{
    std::ifstream stream(file);
    if (!stream.is_open()) {
        throw std::runtime_error("failed to open .silpro file: " + file.string());
    }

    std::string const content(
            (std::istreambuf_iterator<char>(stream)),
            std::istreambuf_iterator<char>());
    std::vector<SilproToken> const tokens = lex_silpro(content);
    SilproParser parser(tokens);
    return parser.parse_document();
}

inline SilproSection const& required_section(
        SilproSection const& root,
        std::string const& name,
        std::string const& context)
{
    auto const iterator = root.sections.find(name);
    if (iterator == root.sections.end()) {
        throw std::runtime_error("missing section '" + name + "' in " + context);
    }
    return iterator->second;
}

inline std::string get_value_or(
        SilproSection const& section,
        std::string const& key,
        std::string default_value)
{
    auto const iterator = section.values.find(key);
    return iterator == section.values.end() ? std::move(default_value) : iterator->second;
}

inline SupportedPhysics parse_physics_kind(std::string const& value)
{
    if (value == "Magnetostatics") {
        return SupportedPhysics::Magnetostatics;
    }
    if (value == "ScalarFieldWithPowerCoupling") {
        return SupportedPhysics::ScalarFieldWithPowerCoupling;
    }
    throw std::runtime_error("unsupported physics '" + value + "' in .silpro file");
}

inline SupportedSolver parse_solver_kind(std::string const& value)
{
    if (value == "MinimizeStrongFormulationResidual") {
        return SupportedSolver::MinimizeStrongFormulationResidual;
    }
    throw std::runtime_error("unsupported solver '" + value + "' in .silpro file");
}

inline SilproProblem parse_silpro_problem(std::filesystem::path const& file)
{
    SilproSection const root = parse_silpro_tree(file);
    SilproSection const& problem_section = required_section(root, "Problem", file.string());
    SilproSection const& solver_section = required_section(root, "Solver", file.string());

    SilproProblem problem;
    problem.name = get_value_or(problem_section, "Name", problem.name);
    problem.physics = parse_physics_kind(get_value_or(problem_section, "Physics", "Magnetostatics"));
    problem.solver = parse_solver_kind(
            get_value_or(problem_section, "Solver", "MinimizeStrongFormulationResidual"));

    problem.solver_settings.max_iterations = parse_number<unsigned int>(
            get_value_or(
                    solver_section,
                    "MaxIterations",
                    std::to_string(problem.solver_settings.max_iterations)));
    problem.solver_settings.relative_tolerance = parse_number<double>(
            get_value_or(
                    solver_section,
                    "RelativeTolerance",
                    std::to_string(problem.solver_settings.relative_tolerance)));
    problem.solver_settings.jacobi_max_block_size = parse_number<unsigned int>(
            get_value_or(
                    solver_section,
                    "JacobiMaxBlockSize",
                    std::to_string(problem.solver_settings.jacobi_max_block_size)));
    problem.solver_settings.use_matrix_free = parse_bool(
            get_value_or(
                    solver_section,
                    "UseMatrixFree",
                    problem.solver_settings.use_matrix_free ? "1" : "0"));

    if (problem.physics == SupportedPhysics::Magnetostatics) {
        SilproSection const& section = required_section(root, "Magnetostatics", file.string());
        problem.magnetostatics.current_rms_parameter = get_value_or(
                section,
                "CurrentRmsParameter",
                problem.magnetostatics.current_rms_parameter);
        problem.magnetostatics.number_of_turns_parameter = get_value_or(
                section,
                "NumberOfTurnsParameter",
                problem.magnetostatics.number_of_turns_parameter);
        problem.magnetostatics.core_relative_permeability_parameter = get_value_or(
                section,
                "CoreRelativePermeabilityParameter",
                problem.magnetostatics.core_relative_permeability_parameter);
        problem.magnetostatics.coil_width_parameter = get_value_or(
                section,
                "CoilWidthParameter",
                problem.magnetostatics.coil_width_parameter);
        problem.magnetostatics.coil_height_parameter = get_value_or(
                section,
                "CoilHeightParameter",
                problem.magnetostatics.coil_height_parameter);
        problem.magnetostatics.current_rms = parse_number<double>(
                get_value_or(
                        section,
                        "CurrentRms",
                        std::to_string(problem.magnetostatics.current_rms)));
        problem.magnetostatics.number_of_turns = parse_number<double>(
                get_value_or(
                        section,
                        "NumberOfTurns",
                        std::to_string(problem.magnetostatics.number_of_turns)));
        problem.magnetostatics.core_relative_permeability = parse_number<double>(
                get_value_or(
                        section,
                        "CoreRelativePermeability",
                        std::to_string(problem.magnetostatics.core_relative_permeability)));
        problem.magnetostatics.coil_width = parse_number<double>(
                get_value_or(
                        section,
                        "CoilWidth",
                        std::to_string(problem.magnetostatics.coil_width)));
        problem.magnetostatics.coil_height = parse_number<double>(
                get_value_or(
                        section,
                        "CoilHeight",
                        std::to_string(problem.magnetostatics.coil_height)));
        problem.magnetostatics.export_input_fields_view = parse_bool(
                get_value_or(
                        section,
                        "ExportInputFieldsView",
                        problem.magnetostatics.export_input_fields_view ? "1" : "0"));
        problem.magnetostatics.merge_result_view_in_gmsh = parse_bool(
                get_value_or(
                        section,
                        "MergeResultViewInGmsh",
                        problem.magnetostatics.merge_result_view_in_gmsh ? "1" : "0"));
        problem.magnetostatics.input_fields_view_file = get_value_or(
                section,
                "InputFieldsViewFile",
                problem.magnetostatics.input_fields_view_file);
    } else {
        SilproSection const& section = required_section(root, "ScalarFieldWithPowerCoupling", file.string());
        problem.scalar_field.mass = parse_number<double>(
                get_value_or(section, "Mass", std::to_string(problem.scalar_field.mass)));
        problem.scalar_field.coupling_constant = parse_number<double>(get_value_or(
                section,
                "CouplingConstant",
                std::to_string(problem.scalar_field.coupling_constant)));
        problem.scalar_field.coupling_power = parse_number<double>(get_value_or(
                section,
                "CouplingPower",
                std::to_string(problem.scalar_field.coupling_power)));
    }

    return problem;
}

inline physics::scalar_field::ScalarFieldWithPowerCoupling assemble_scalar_field_hamiltonian(
        ScalarFieldWithPowerCouplingProblem const& problem)
{
    return physics::scalar_field::ScalarFieldWithPowerCoupling(
            problem.mass,
            problem.coupling_constant,
            problem.coupling_power);
}

inline physics::magnetostatics::LinearMagnetostaticsHamiltonian assemble_linear_magnetostatics_hamiltonian(
        double mu)
{
    return physics::magnetostatics::LinearMagnetostaticsHamiltonian(mu);
}

inline solvers::StrongFormulationSolverSettings assemble_solver_settings(
        MinimizeStrongFormulationResidualProblem const& problem)
{
    solvers::StrongFormulationSolverSettings settings;
    settings.max_iterations = problem.max_iterations;
    settings.relative_tolerance = problem.relative_tolerance;
    settings.jacobi_max_block_size = problem.jacobi_max_block_size;
    settings.use_matrix_free = problem.use_matrix_free;
    return settings;
}

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
                stream << "  SP(" << grid.cell_center_x(i) << ", " << grid.cell_center_y(j) << ", "
                       << grid.cell_center_z(k) << "){"
                       << cell_inputs[index].current_density[2] << "};\n";
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

class OnelabInterface
{
    using Client = onelab::remoteNetworkClient;

public:
    explicit OnelabInterface(std::string module_name = "SimiLie")
        : m_module_name(std::move(module_name))
    {
    }

    int run(int argc, char** argv)
    {
        OnelabArguments parsed_arguments;
        if (!parse_onelab_arguments(argc, argv, parsed_arguments)) {
            print_usage(argv[0]);
            return 1;
        }

        Kokkos::ScopeGuard const kokkos_scope(argc, argv);
        ddc::ScopeGuard const ddc_scope(argc, argv);

        try {
            connect(parsed_arguments.client_name, parsed_arguments.socket_address);
        } catch (std::exception const& exception) {
            std::cerr << exception.what() << '\n';
            return 2;
        }

        client().sendInfo(module_name() + " ONELAB interface connected");
        publish_common_parameters();
        publish_interface_parameters();

        try {
            run_problem();
            client().sendInfo(module_name() + " ONELAB interface finished");
        } catch (std::exception const& exception) {
            client().sendError(exception.what());
            return 3;
        }

        return 0;
    }

    [[nodiscard]] static SilproProblem parse_silpro_file(std::filesystem::path const& file)
    {
        return detail::parse_silpro_problem(file);
    }

private:
    struct OnelabArguments
    {
        std::string client_name;
        std::string socket_address;
    };

    [[nodiscard]] std::string module_name() const
    {
        return m_module_name;
    }

    [[nodiscard]] std::string control_parameter_name(std::string const& parameter_name) const
    {
        return "0Modules/" + module_name() + "/0Control/" + parameter_name;
    }

    [[nodiscard]] std::string output_parameter_name(std::string const& parameter_name) const
    {
        return "0Modules/" + module_name() + "/1Output/" + parameter_name;
    }

    [[nodiscard]] Client& client()
    {
        if (m_client == nullptr) {
            throw std::logic_error("the ONELAB client is not connected");
        }
        return *m_client;
    }

    onelab::number get_or_create_number(
            std::string const& name,
            double default_value,
            std::string const& label,
            std::string const& help,
            double min_value,
            double max_value,
            double step)
    {
        std::vector<onelab::number> parameters;
        client().get(parameters, name);
        onelab::number parameter = parameters.empty()
                                           ? onelab::number(name, default_value, label, help)
                                           : parameters.front();
        if (parameters.empty()) {
            parameter.setMin(min_value);
            parameter.setMax(max_value);
            parameter.setStep(step);
            client().set(parameter);
        }
        return parameter;
    }

    onelab::string get_or_create_string(
            std::string const& name,
            std::string const& default_value,
            std::string const& label,
            std::string const& help)
    {
        std::vector<onelab::string> parameters;
        client().get(parameters, name);
        onelab::string parameter = parameters.empty()
                                           ? onelab::string(name, default_value, label, help)
                                           : parameters.front();
        if (parameters.empty()) {
            client().set(parameter);
        }
        return parameter;
    }

    [[nodiscard]] std::string get_first_string_value(std::string const& parameter_name)
    {
        std::vector<onelab::string> parameters;
        client().get(parameters, parameter_name);
        if (parameters.empty()) {
            return "";
        }
        return parameters.front().getValue();
    }

    [[nodiscard]] double get_first_number_value(std::string const& parameter_name, double default_value)
    {
        std::vector<onelab::number> parameters;
        client().get(parameters, parameter_name);
        if (parameters.empty()) {
            return default_value;
        }
        return parameters.front().getValue();
    }

    [[nodiscard]] double read_number_parameter(
            std::string const& preferred_parameter,
            std::optional<std::string> const& fallback_parameter,
            double fallback_value)
    {
        double const preferred_value = get_first_number_value(preferred_parameter, std::numeric_limits<double>::quiet_NaN());
        if (!std::isnan(preferred_value)) {
            return preferred_value;
        }
        if (fallback_parameter.has_value()) {
            return get_first_number_value(*fallback_parameter, fallback_value);
        }
        return fallback_value;
    }

    void publish_status(std::string const& status)
    {
        publish_output_string(
                "Last status",
                status,
                "Last status",
                "Status reported by the SimiLie ONELAB interface.");
    }

    void publish_output_string(
            std::string const& name,
            std::string const& value,
            std::string const& label,
            std::string const& help,
            std::string const& kind = "generic")
    {
        onelab::string parameter(output_parameter_name(name), value, label, help);
        parameter.setKind(kind);
        parameter.setReadOnly(true);
        client().set(parameter);
    }

    void publish_output_number(
            std::string const& name,
            double value,
            std::string const& label,
            std::string const& help)
    {
        onelab::number parameter(output_parameter_name(name), value, label, help);
        parameter.setReadOnly(true);
        client().set(parameter);
    }

    [[nodiscard]] std::filesystem::path resolve_input_mesh_file()
    {
        std::filesystem::path input_mesh_file
                = get_first_string_value(control_parameter_name("Input mesh file"));
        if (!input_mesh_file.empty()) {
            return input_mesh_file;
        }

        std::filesystem::path gmsh_mesh_file = get_first_string_value("Gmsh/MshFileName");
        if (gmsh_mesh_file.empty()) {
            throw std::runtime_error(
                    "No mesh file available: set 'Input mesh file' or let Gmsh manage the current mesh file.");
        }

        if (gmsh_mesh_file.is_absolute()) {
            return gmsh_mesh_file;
        }

        std::filesystem::path gmsh_model_path = get_first_string_value("Gmsh/Model absolute path");
        if (!gmsh_model_path.empty()) {
            return gmsh_model_path / gmsh_mesh_file;
        }

        return gmsh_mesh_file;
    }

    [[nodiscard]] std::filesystem::path export_input_mesh_from_gmsh()
    {
        std::filesystem::path const input_mesh_file = resolve_input_mesh_file();
        export_current_mesh_from_gmsh(input_mesh_file);
        return input_mesh_file;
    }

    [[nodiscard]] std::filesystem::path resolve_problem_file()
    {
        std::string const configured_file = get_first_string_value(control_parameter_name("Problem file"));
        if (configured_file.empty()) {
            throw std::runtime_error(
                    "no .silpro problem file configured: set '0Modules/SimiLie/0Control/Problem file'");
        }
        return configured_file;
    }

    void publish_common_parameters()
    {
        onelab::number is_metamodel("IsMetamodel", 1.0);
        is_metamodel.setNeverChanged(true);
        client().set(is_metamodel);

        onelab::number require_rectilinear_mesh = get_or_create_number(
                control_parameter_name("Require rectilinear mesh"),
                1.0,
                "Require rectilinear mesh",
                "Reject any incoming mesh that is not a full rectilinear grid supported by SimiLie.",
                0.0,
                1.0,
                1.0);
        require_rectilinear_mesh.setChoices({0.0, 1.0});
        require_rectilinear_mesh.setValueLabels({{0.0, "No"}, {1.0, "Yes"}});
        client().set(require_rectilinear_mesh);

        onelab::string input_mesh_file = get_or_create_string(
                control_parameter_name("Input mesh file"),
                "",
                "Input mesh file",
                "Optional path to the Gmsh .msh file that the SimiLie ONELAB interface should read. "
                "If empty, the interface uses Gmsh/MshFileName.");
        input_mesh_file.setKind("file");
        client().set(input_mesh_file);
    }

    void publish_interface_parameters()
    {
        onelab::string problem_file = get_or_create_string(
                control_parameter_name("Problem file"),
                "",
                "Problem file",
                "Path to the SimiLie .silpro problem description file.");
        problem_file.setKind("file");
        client().set(problem_file);

        onelab::string formulation = get_or_create_string(
                control_parameter_name("Supported physics"),
                "ScalarFieldWithPowerCoupling, Magnetostatics",
                "Supported physics",
                "Physics currently supported by the SimiLie .silpro interface.");
        formulation.setReadOnly(true);
        client().set(formulation);

        onelab::string solver = get_or_create_string(
                control_parameter_name("Supported solvers"),
                "MinimizeStrongFormulationResidual",
                "Supported solvers",
                "Solvers currently supported by the SimiLie .silpro interface.");
        solver.setReadOnly(true);
        client().set(solver);

        onelab::string result_view_file = get_or_create_string(
                control_parameter_name("Input fields view file"),
                "",
                "Input fields view file",
                "Optional override for the .pos file exported by magnetostatics problems.");
        result_view_file.setKind("file");
        client().set(result_view_file);
    }

    void run_problem()
    {
        std::filesystem::path const silpro_file = resolve_problem_file();
        SilproProblem const problem = parse_silpro_file(silpro_file);
        publish_output_string(
                "Problem file",
                silpro_file.string(),
                "Problem file",
                "SimiLie .silpro file parsed by the ONELAB interface.",
                "file");
        publish_output_string(
                "Problem name",
                problem.name,
                "Problem name",
                "Problem name declared in the .silpro file.");
        publish_output_string(
                "Physics",
                problem.physics == SupportedPhysics::Magnetostatics ? "Magnetostatics"
                                                                    : "ScalarFieldWithPowerCoupling",
                "Physics",
                "Physics selected by the .silpro file.");
        publish_output_string(
                "Solver",
                "MinimizeStrongFormulationResidual",
                "Solver",
                "Solver selected by the .silpro file.");

        if (problem.solver != SupportedSolver::MinimizeStrongFormulationResidual) {
            throw std::runtime_error("unsupported solver selected by the .silpro file");
        }

        if (problem.physics == SupportedPhysics::ScalarFieldWithPowerCoupling) {
            [[maybe_unused]] auto const physics
                    = detail::assemble_scalar_field_hamiltonian(problem.scalar_field);
            [[maybe_unused]] auto const solver_settings
                    = detail::assemble_solver_settings(problem.solver_settings);
            throw std::runtime_error(
                    "ScalarFieldWithPowerCoupling .silpro files are parsed successfully, but ONELAB "
                    "execution is not implemented yet in this interface");
        }

        run_magnetostatics_problem(problem);
    }

    void run_magnetostatics_problem(SilproProblem const& problem)
    {
        MagnetostaticsProblem const& cfg = problem.magnetostatics;
        client().sendProgress(module_name() + " ONELAB interface: exporting mesh for problem '" + problem.name + "'");
        std::filesystem::path const mesh_file = export_input_mesh_from_gmsh();

        double const current_rms = read_number_parameter(
                cfg.current_rms_parameter,
                std::nullopt,
                cfg.current_rms);
        double const number_of_turns = read_number_parameter(
                cfg.number_of_turns_parameter,
                std::nullopt,
                cfg.number_of_turns);
        double const core_relative_permeability = read_number_parameter(
                cfg.core_relative_permeability_parameter,
                std::nullopt,
                cfg.core_relative_permeability);
        double const coil_width = read_number_parameter(
                cfg.coil_width_parameter,
                std::nullopt,
                cfg.coil_width);
        double const coil_height = read_number_parameter(
                cfg.coil_height_parameter,
                std::nullopt,
                cfg.coil_height);

        std::filesystem::path output_view_file = get_first_string_value(control_parameter_name("Input fields view file"));
        if (output_view_file.empty()) {
            output_view_file = cfg.input_fields_view_file;
        }
        if (output_view_file.empty()) {
            output_view_file = mesh_file.parent_path() / "similie_linear_magnetostatics_inputs.pos";
        }

        double const mu0 = 4.e-7 * std::numbers::pi_v<double>;
        double const core_mu = core_relative_permeability * mu0;
        [[maybe_unused]] auto const magnetostatics_hamiltonian
                = detail::assemble_linear_magnetostatics_hamiltonian(core_mu);
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
        solvers::StrongFormulationSolverSettings const solver_settings
                = detail::assemble_solver_settings(problem.solver_settings);
        client().sendInfo(
                solver_settings.use_matrix_free
                        ? "SimiLie starting matrix-free preconditioned conjugate-gradient solve"
                        : "SimiLie starting assembled-matrix Ginkgo preconditioned conjugate-gradient solve");
        solvers::StrongFormulationSolverDiagnostics const solver_diagnostics
                = solvers::minimize_strong_formulation_residual(
                        Kokkos::DefaultExecutionSpace(),
                        formulation,
                        rhs,
                        magnetic_vector_potential_z_xy_view,
                        solver_settings);
        client().sendInfo(
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

        if (cfg.export_input_fields_view) {
            detail::write_results_view(output_view_file, grid, cell_inputs, magnetic_vector_potential);
            if (cfg.merge_result_view_in_gmsh) {
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
                "Input fields view file",
                "View file containing the permeability, current density and magnetic vector potential exported by SimiLie.",
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
        publish_output_string(
                "Solver backend",
                solver_settings.use_matrix_free ? "matrix-free" : "assembled-matrix",
                "Solver backend",
                "Backend used by the stationary strong-formulation solver.");
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
                "Solver optimization wall time [s]",
                solver_diagnostics.optimization_wall_seconds,
                "Solver optimization wall time [s]",
                "Wall-clock time spent in the effective iterative optimization, excluding matrix and preconditioner assembly.");
        publish_output_number(
                "Maximum magnetic vector potential [SI]",
                max_abs_potential,
                "Maximum magnetic vector potential [SI]",
                "Maximum absolute value of the computed magnetic vector potential.");
        publish_output_number(
                "Maximum magnetic induction [T]",
                max_abs_induction,
                "Maximum magnetic induction [T]",
                "Maximum absolute value of the computed magnetic induction.");
        publish_output_number(
                "Maximum magnetic field [A/m]",
                max_abs_field,
                "Maximum magnetic field [A/m]",
                "Maximum absolute value of the computed magnetic field.");
        publish_output_number(
                "Air-gap mean magnetic induction [T]",
                num_air_gap_cells == 0 ? 0.0
                                       : air_gap_induction_magnitude_sum
                                                 / static_cast<double>(num_air_gap_cells),
                "Air-gap mean magnetic induction [T]",
                "Mean magnitude of the magnetic induction over air-gap cells.");
        publish_output_number(
                "Mean force density magnitude [N/m^3]",
                num_cells == 0 ? 0.0 : force_density_magnitude_sum / static_cast<double>(num_cells),
                "Mean force density magnitude [N/m^3]",
                "Mean magnitude of the force density over all hexahedral cells.");

        std::ostringstream diagnostics_stream;
        diagnostics_stream << "SimiLie solver diagnostics: iterations=" << solver_diagnostics.iterations
                           << ", final residual L2=" << solver_diagnostics.final_residual_l2
                           << ", final relative residual=" << solver_diagnostics.final_relative_residual
                           << ", optimization wall time=" << solver_diagnostics.optimization_wall_seconds
                           << " s, air-gap mean |B|="
                           << (num_air_gap_cells == 0
                                       ? 0.0
                                       : air_gap_induction_magnitude_sum
                                                 / static_cast<double>(num_air_gap_cells))
                           << " T, mean |f|="
                           << (num_cells == 0 ? 0.0
                                              : force_density_magnitude_sum / static_cast<double>(num_cells))
                           << " N/m^3";
        client().sendInfo(diagnostics_stream.str());
        publish_status("Magnetostatics solve completed");
    }

    static void print_usage(char const* program_name)
    {
        std::cerr << "usage: " << program_name << " -onelab <client-name> <socket>\n";
    }

    static bool parse_onelab_arguments(int argc, char** argv, OnelabArguments& parsed_arguments)
    {
        for (int i = 0; i < argc; ++i) {
            if (std::string(argv[i]) == "-onelab" && i + 2 < argc) {
                parsed_arguments.client_name = argv[i + 1];
                parsed_arguments.socket_address = argv[i + 2];
                return true;
            }
        }
        return false;
    }

    void connect(std::string const& client_name, std::string const& socket_address)
    {
        static std::unique_ptr<Client> connected_client;
        connected_client = std::make_unique<Client>(client_name, socket_address);
        if (connected_client->getGmshClient() == nullptr) {
            throw std::runtime_error("failed to connect to the ONELAB server at " + socket_address);
        }
        m_client = connected_client.get();
    }

    void export_current_mesh_from_gmsh(std::filesystem::path const& mesh_file)
    {
        std::filesystem::path const absolute_mesh_file = std::filesystem::absolute(mesh_file);
        if (!absolute_mesh_file.parent_path().empty()) {
            std::filesystem::create_directories(absolute_mesh_file.parent_path());
        }
        if (std::filesystem::exists(absolute_mesh_file)) {
            std::filesystem::remove(absolute_mesh_file);
        }

        std::string const gmsh_command = "Mesh.Binary = 0;"
                                         "Mesh.MshFileVersion = 2.2;"
                                         "Save \"" + absolute_mesh_file.string() + "\";";

        client().sendInfo("Asking Gmsh to export the current mesh to " + absolute_mesh_file.string());
        client().sendParseStringRequest(gmsh_command);

        std::uintmax_t previous_size = 0;
        int stable_size_count = 0;

        for (int attempt = 0; attempt < 100; ++attempt) {
            if (std::filesystem::exists(absolute_mesh_file)) {
                std::uintmax_t const current_size = std::filesystem::file_size(absolute_mesh_file);
                if (current_size > 0 && current_size == previous_size) {
                    ++stable_size_count;
                } else {
                    stable_size_count = 0;
                }
                previous_size = current_size;

                if (stable_size_count >= 2) {
                    std::ifstream stream(absolute_mesh_file);
                    std::string const content(
                            (std::istreambuf_iterator<char>(stream)),
                            std::istreambuf_iterator<char>());
                    if (content.find("$EndElements") != std::string::npos) {
                        client().sendInfo(
                                "Detected completed exported mesh file at "
                                + absolute_mesh_file.string());
                        return;
                    }
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }

        throw std::runtime_error(
                "Gmsh did not export the mesh file '" + absolute_mesh_file.string()
                + "' before the ONELAB timeout.");
    }

    std::string m_module_name;
    Client* m_client = nullptr;
};

} // namespace similie::onelab_interface
