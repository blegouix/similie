// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
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
#include <variant>
#include <vector>

#include <ddc/ddc.hpp>

#include <similie/physics/hamilton_equations.hpp>
#include <similie/physics/magnetostatics/structured_linear_magnetostatics_problem.hpp>
#include <similie/physics/scalar_field/scalar_field_with_power_coupling.hpp>
#include <similie/solvers/minimize_strong_formulation_residual.hpp>

#include <onelab.h>

#include "gmsh_structured_msh2.hpp"

namespace similie::onelab_interface {

enum class SupportedPhysics {
    ScalarFieldWithPowerCoupling,
    Magnetostatics,
};

enum class SupportedSolver {
    MinimizeStrongFormulationResidual,
};

struct ScalarFieldWithPowerCouplingProblem
{
    double mass = 1.0;
    double coupling_constant = 0.0;
    double coupling_power = 4.0;
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
    enum class Kind {
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
    [[nodiscard]] bool at_end() const
    {
        return m_index >= m_tokens.size();
    }

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

    std::string const
            content((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());
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
    problem.physics
            = parse_physics_kind(get_value_or(problem_section, "Physics", "Magnetostatics"));
    problem.solver = parse_solver_kind(
            get_value_or(problem_section, "Solver", "MinimizeStrongFormulationResidual"));

    problem.solver_settings.max_iterations = parse_number<unsigned int>(get_value_or(
            solver_section,
            "MaxIterations",
            std::to_string(problem.solver_settings.max_iterations)));
    problem.solver_settings.relative_tolerance = parse_number<double>(get_value_or(
            solver_section,
            "RelativeTolerance",
            std::to_string(problem.solver_settings.relative_tolerance)));
    problem.solver_settings.jacobi_max_block_size = parse_number<unsigned int>(get_value_or(
            solver_section,
            "JacobiMaxBlockSize",
            std::to_string(problem.solver_settings.jacobi_max_block_size)));
    problem.solver_settings.use_matrix_free = parse_bool(get_value_or(
            solver_section,
            "UseMatrixFree",
            problem.solver_settings.use_matrix_free ? "1" : "0"));

    if (problem.physics == SupportedPhysics::ScalarFieldWithPowerCoupling) {
        SilproSection const& section
                = required_section(root, "ScalarFieldWithPowerCoupling", file.string());
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

using sil::mesher::gmsh::HexahedralCell;
using sil::mesher::gmsh::HexahedralMesh;
using sil::mesher::gmsh::MeshNode;
using sil::mesher::gmsh::QuadrilateralCell;
using sil::mesher::gmsh::QuadrilateralMesh;
using sil::mesher::gmsh::StructuredGrid2D;
using sil::mesher::gmsh::SupportedMesh;
using StructuredGrid = sil::mesher::gmsh::StructuredGrid3D;
using sil::mesher::gmsh::build_structured_grid;
using sil::mesher::gmsh::centered_first_derivative;
using sil::mesher::gmsh::parse_supported_msh2_mesh;

} // namespace detail

class OnelabInterface
{
    using Client = onelab::remoteNetworkClient;
    using MagnetostaticsInputs = physics::magnetostatics::StructuredLinearMagnetostaticsInputs;
    using MagnetostaticsRegionTags
            = physics::magnetostatics::StructuredLinearMagnetostaticsRegionTags;
    using MagnetostaticsResult = physics::magnetostatics::StructuredLinearMagnetostaticsResult;

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

    [[nodiscard]] double get_first_number_value(
            std::string const& parameter_name,
            double default_value)
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
        double const preferred_value = get_first_number_value(
                preferred_parameter,
                std::numeric_limits<double>::quiet_NaN());
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
        parameter.setVisible(false);
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
        parameter.setVisible(false);
        client().set(parameter);
    }

    [[nodiscard]] std::filesystem::path resolve_input_mesh_file()
    {
        std::filesystem::path input_mesh_file
                = get_first_string_value(control_parameter_name("Mesh file"));
        if (!input_mesh_file.empty()) {
            return input_mesh_file;
        }

        std::filesystem::path gmsh_mesh_file = get_first_string_value("Gmsh/MshFileName");
        if (gmsh_mesh_file.empty()) {
            std::filesystem::path gmsh_model_path
                    = get_first_string_value("Gmsh/Model absolute path");
            if (!gmsh_model_path.empty()) {
                return gmsh_model_path / "similie_onelab_current_mesh.msh";
            }
            return std::filesystem::temp_directory_path() / "similie_onelab_current_mesh.msh";
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

    [[nodiscard]] bool has_explicit_mesh_file_control()
    {
        return !get_first_string_value(control_parameter_name("Mesh file")).empty();
    }

    [[nodiscard]] std::filesystem::path export_input_mesh_from_gmsh()
    {
        std::filesystem::path const input_mesh_file = resolve_input_mesh_file();
        if (has_explicit_mesh_file_control() && std::filesystem::exists(input_mesh_file)
            && std::filesystem::file_size(input_mesh_file) > 0) {
            client().sendInfo(
                    "Using existing mesh file "
                    + std::filesystem::absolute(input_mesh_file).string());
            return input_mesh_file;
        }
        export_current_mesh_from_gmsh(input_mesh_file);
        return input_mesh_file;
    }

    [[nodiscard]] std::filesystem::path resolve_problem_file()
    {
        std::string const configured_file
                = get_first_string_value(control_parameter_name("Problem file"));
        if (configured_file.empty()) {
            throw std::runtime_error(
                    "no .silpro problem file configured: set '0Modules/SimiLie/0Control/Problem "
                    "file'");
        }
        return configured_file;
    }

    [[nodiscard]] std::string action()
    {
        return get_first_string_value(module_name() + "/Action");
    }

    [[nodiscard]] std::string problem_parameter_name(
            std::string const& section,
            std::string const& name) const
    {
        return "0Modules/" + module_name() + "/1Problem/" + section + "/" + name;
    }

    [[nodiscard]] std::string internal_parameter_name(std::string const& name) const
    {
        return "0Modules/" + module_name() + "/9Internal/" + name;
    }

    void publish_hidden_string(std::string const& name, std::string const& value)
    {
        onelab::string parameter(name, value, "", "");
        parameter.setVisible(false);
        parameter.setReadOnly(true);
        client().set(parameter);
    }

    void publish_common_parameters()
    {
        onelab::number is_metamodel("IsMetamodel", 1.0);
        is_metamodel.setNeverChanged(true);
        client().set(is_metamodel);

        onelab::string input_mesh_file = get_or_create_string(
                control_parameter_name("Mesh file"),
                "",
                "Mesh file",
                "Optional path to the Gmsh .msh file that the SimiLie ONELAB interface should "
                "read. "
                "If empty, the interface uses the current Gmsh mesh export.");
        input_mesh_file.setKind("file");
        client().set(input_mesh_file);
    }

    void publish_interface_parameters()
    {
        onelab::string problem_file = get_or_create_string(
                control_parameter_name("Problem file"),
                "",
                "Problem file (.silpro)",
                "Path to the SimiLie .silpro problem description file.");
        problem_file.setKind("file");
        client().set(problem_file);
    }

    bool synchronize_problem_controls(
            SilproProblem const& problem,
            std::filesystem::path const& silpro_file)
    {
        std::string const current_file = std::filesystem::absolute(silpro_file).string();
        std::string const last_file
                = get_first_string_value(internal_parameter_name("Last synchronized problem file"));
        bool const synchronize = (current_file != last_file);

        auto publish_or_sync_string = [&](std::string const& name,
                                          std::string const& label,
                                          std::string const& help,
                                          std::string const& value,
                                          bool read_only = false) {
            onelab::string parameter = get_or_create_string(name, value, label, help);
            if (synchronize) {
                parameter.setValue(value);
            }
            parameter.setReadOnly(read_only);
            client().set(parameter);
        };
        auto publish_or_sync_number = [&](std::string const& name,
                                          std::string const& label,
                                          std::string const& help,
                                          double value,
                                          double min_value,
                                          double max_value,
                                          double step,
                                          std::optional<std::vector<double>> choices = std::nullopt,
                                          std::optional<std::map<double, std::string>> value_labels
                                          = std::nullopt) {
            onelab::number parameter
                    = get_or_create_number(name, value, label, help, min_value, max_value, step);
            if (synchronize) {
                parameter.setValue(value);
            }
            if (choices.has_value()) {
                parameter.setChoices(*choices);
            }
            if (value_labels.has_value()) {
                parameter.setValueLabels(*value_labels);
            }
            client().set(parameter);
        };

        publish_or_sync_string(
                problem_parameter_name("0Problem", "0Name"),
                "Name",
                "Problem name declared in the .silpro file.",
                problem.name,
                true);
        publish_or_sync_string(
                problem_parameter_name("0Problem", "1Physics"),
                "Physics",
                "Physics selected in the .silpro file.",
                problem.physics == SupportedPhysics::Magnetostatics
                        ? "Magnetostatics"
                        : "ScalarFieldWithPowerCoupling",
                true);
        publish_or_sync_string(
                problem_parameter_name("0Problem", "2Solver"),
                "Solver",
                "Solver selected in the .silpro file.",
                "MinimizeStrongFormulationResidual",
                true);

        publish_or_sync_number(
                problem_parameter_name("1Solver", "0Max iterations"),
                "Max iterations",
                "Maximum number of iterations for the stationary strong-formulation solver.",
                static_cast<double>(problem.solver_settings.max_iterations),
                1.0,
                1.e9,
                1.0);
        publish_or_sync_number(
                problem_parameter_name("1Solver", "1Relative tolerance"),
                "Relative tolerance",
                "Relative convergence tolerance for the stationary strong-formulation solver.",
                problem.solver_settings.relative_tolerance,
                0.0,
                1.0,
                1.e-12);
        publish_or_sync_number(
                problem_parameter_name("1Solver", "2Jacobi max block size"),
                "Jacobi max block size",
                "Maximum Jacobi block size used by the auxiliary preconditioner.",
                static_cast<double>(problem.solver_settings.jacobi_max_block_size),
                1.0,
                1.e9,
                1.0);
        publish_or_sync_number(
                problem_parameter_name("1Solver", "3Use matrix-free"),
                "Use matrix-free",
                "When enabled, the strong-form operator is applied matrix-free and only the Jacobi "
                "preconditioner uses an auxiliary assembled matrix.",
                problem.solver_settings.use_matrix_free ? 1.0 : 0.0,
                0.0,
                1.0,
                1.0,
                std::vector<double> {0.0, 1.0},
                std::map<double, std::string> {{0.0, "No"}, {1.0, "Yes"}});

        if (problem.physics == SupportedPhysics::ScalarFieldWithPowerCoupling) {
            ScalarFieldWithPowerCouplingProblem const& cfg = problem.scalar_field;
            publish_or_sync_number(
                    problem_parameter_name("2ScalarFieldWithPowerCoupling", "0Mass"),
                    "Mass",
                    "Scalar-field mass parameter declared in the .silpro file.",
                    cfg.mass,
                    0.0,
                    1.e12,
                    1.e-3);
            publish_or_sync_number(
                    problem_parameter_name("2ScalarFieldWithPowerCoupling", "1Coupling constant"),
                    "Coupling constant",
                    "Scalar-field coupling constant declared in the .silpro file.",
                    cfg.coupling_constant,
                    -1.e12,
                    1.e12,
                    1.e-3);
            publish_or_sync_number(
                    problem_parameter_name("2ScalarFieldWithPowerCoupling", "2Coupling power"),
                    "Coupling power",
                    "Scalar-field coupling power declared in the .silpro file.",
                    cfg.coupling_power,
                    0.0,
                    1.e12,
                    1.0);
        }

        if (synchronize) {
            publish_hidden_string(
                    internal_parameter_name("Last synchronized problem file"),
                    current_file);
        }
        return synchronize;
    }

    [[nodiscard]] SilproProblem apply_problem_control_overrides(SilproProblem problem)
    {
        problem.solver_settings.max_iterations = static_cast<unsigned int>(get_first_number_value(
                problem_parameter_name("1Solver", "0Max iterations"),
                static_cast<double>(problem.solver_settings.max_iterations)));
        problem.solver_settings.relative_tolerance = get_first_number_value(
                problem_parameter_name("1Solver", "1Relative tolerance"),
                problem.solver_settings.relative_tolerance);
        problem.solver_settings.jacobi_max_block_size
                = static_cast<unsigned int>(get_first_number_value(
                        problem_parameter_name("1Solver", "2Jacobi max block size"),
                        static_cast<double>(problem.solver_settings.jacobi_max_block_size)));
        problem.solver_settings.use_matrix_free
                = (get_first_number_value(
                           problem_parameter_name("1Solver", "3Use matrix-free"),
                           problem.solver_settings.use_matrix_free ? 1.0 : 0.0)
                   != 0.0);

        if (problem.physics == SupportedPhysics::ScalarFieldWithPowerCoupling) {
            ScalarFieldWithPowerCouplingProblem& cfg = problem.scalar_field;
            cfg.mass = get_first_number_value(
                    problem_parameter_name("2ScalarFieldWithPowerCoupling", "0Mass"),
                    cfg.mass);
            cfg.coupling_constant = get_first_number_value(
                    problem_parameter_name("2ScalarFieldWithPowerCoupling", "1Coupling constant"),
                    cfg.coupling_constant);
            cfg.coupling_power = get_first_number_value(
                    problem_parameter_name("2ScalarFieldWithPowerCoupling", "2Coupling power"),
                    cfg.coupling_power);
        }

        return problem;
    }

    void run_problem()
    {
        std::filesystem::path const silpro_file = resolve_problem_file();
        SilproProblem problem = parse_silpro_file(silpro_file);
        bool const synchronized = synchronize_problem_controls(problem, silpro_file);
        problem = apply_problem_control_overrides(std::move(problem));
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
                problem.physics == SupportedPhysics::Magnetostatics
                        ? "Magnetostatics"
                        : "ScalarFieldWithPowerCoupling",
                "Physics",
                "Physics selected by the .silpro file.");
        publish_output_string(
                "Solver",
                "MinimizeStrongFormulationResidual",
                "Solver",
                "Solver selected by the .silpro file.");

        if (action() == "initialize") {
            publish_status(
                    synchronized ? "Problem configuration synchronized from .silpro file"
                                 : "Problem configuration initialized");
            client().sendInfo("SimiLie ONELAB interface initialized without solving");
            return;
        }

        if (problem.solver != SupportedSolver::MinimizeStrongFormulationResidual) {
            throw std::runtime_error("unsupported solver selected by the .silpro file");
        }

        if (problem.physics == SupportedPhysics::ScalarFieldWithPowerCoupling) {
            [[maybe_unused]] auto const physics
                    = detail::assemble_scalar_field_hamiltonian(problem.scalar_field);
            [[maybe_unused]] auto const solver_settings
                    = detail::assemble_solver_settings(problem.solver_settings);
            throw std::runtime_error(
                    "ScalarFieldWithPowerCoupling .silpro files are parsed successfully, but "
                    "ONELAB "
                    "execution is not implemented yet in this interface");
        }

        run_magnetostatics_problem(problem);
    }

    [[nodiscard]] MagnetostaticsInputs read_magnetostatics_inputs()
    {
        double const current_density_magnitude = read_number_parameter(
                "Input/90SimiLie/0Coil current density magnitude z [A/m^2]",
                std::nullopt,
                std::numeric_limits<double>::quiet_NaN());
        double const core_mu = read_number_parameter(
                "Input/90SimiLie/1Core magnetic permeability [H/m]",
                std::nullopt,
                std::numeric_limits<double>::quiet_NaN());
        double const mu0 = 4.e-7 * std::numbers::pi_v<double>;

        if (!(current_density_magnitude > 0.0)) {
            throw std::runtime_error(
                    "missing or invalid 'Input/90SimiLie/0Coil current density magnitude z "
                    "[A/m^2]' ONELAB parameter");
        }
        if (!(core_mu > 0.0)) {
            throw std::runtime_error(
                    "missing or invalid 'Input/90SimiLie/1Core magnetic permeability [H/m]' ONELAB "
                    "parameter");
        }
        return {
                .current_density_magnitude = current_density_magnitude,
                .core_mu = core_mu,
                .mu0 = mu0,
        };
    }

    [[nodiscard]] MagnetostaticsRegionTags read_magnetostatics_region_tags()
    {
        auto read_integer_tag = [&](std::string const& parameter_name) {
            double const value = read_number_parameter(
                    parameter_name,
                    std::nullopt,
                    std::numeric_limits<double>::quiet_NaN());
            if (std::isnan(value)) {
                throw std::runtime_error(
                        "missing ONELAB parameter '" + parameter_name
                        + "' needed to identify magnetostatics regions");
            }
            return static_cast<int>(std::lround(value));
        };

        return {
                .e_core_tag = read_integer_tag("Input/90SimiLie/2E-core physical tag"),
                .i_core_tag = read_integer_tag("Input/90SimiLie/3I-core physical tag"),
                .coil_left_tag = read_integer_tag("Input/90SimiLie/4Left coil physical tag"),
                .coil_right_tag = read_integer_tag("Input/90SimiLie/5Right coil physical tag"),
                .air_gap_tag = read_integer_tag("Input/90SimiLie/6Air-gap physical tag"),
        };
    }

    void publish_magnetostatics_outputs(
            std::filesystem::path const& mesh_file,
            MagnetostaticsInputs const& inputs,
            solvers::StrongFormulationSolverSettings const& solver_settings,
            MagnetostaticsResult const& result)
    {
        publish_output_string(
                "Mesh file",
                mesh_file.string(),
                "Mesh file",
                "Mesh file exported by Gmsh for the linear magnetostatics interface.",
                "file");
        publish_output_number(
                "Coil current density magnitude [A/m^2]",
                inputs.current_density_magnitude,
                "Coil current density magnitude [A/m^2]",
                "Current density magnitude read from the ONELAB model inputs.");
        publish_output_number(
                "Air permeability [H/m]",
                inputs.mu0,
                "Air permeability [H/m]",
                "Magnetic permeability used in air and coil cells.");
        publish_output_number(
                "Core permeability [H/m]",
                inputs.core_mu,
                "Core permeability [H/m]",
                "Magnetic permeability read from the ONELAB model inputs and used in core cells.");
        publish_output_number(
                "Number of air cells",
                static_cast<double>(result.num_air_cells),
                "Number of air cells",
                "Number of " + result.topology + " cells tagged as air or air gap.");
        publish_output_number(
                "Number of core cells",
                static_cast<double>(result.num_core_cells),
                "Number of core cells",
                "Number of " + result.topology + " cells tagged as core.");
        publish_output_number(
                "Number of coil cells",
                static_cast<double>(result.num_coil_cells),
                "Number of coil cells",
                "Number of " + result.topology + " cells tagged as coil.");
        publish_output_number(
                "Solver iterations",
                static_cast<double>(result.solver_diagnostics.iterations),
                "Solver iterations",
                "Number of conjugate-gradient iterations performed by the stationary "
                "strong-formulation solver.");
        publish_output_string(
                "Solver backend",
                solver_settings.use_matrix_free ? "matrix-free" : "assembled-matrix",
                "Solver backend",
                "Backend used by the stationary strong-formulation solver.");
        publish_output_number(
                "Solver converged",
                result.solver_diagnostics.converged ? 1.0 : 0.0,
                "Solver converged",
                "Equals 1 when the stationary strong-formulation solver met its relative-residual "
                "target, 0 otherwise.");
        publish_output_number(
                "Final residual L2",
                result.solver_diagnostics.final_residual_l2,
                "Final residual L2",
                "Final L2 norm of the strong-formulation residual returned by the stationary "
                "solver.");
        publish_output_number(
                "Final relative residual",
                result.solver_diagnostics.final_relative_residual,
                "Final relative residual",
                "Final residual divided by the initial residual, as returned by the stationary "
                "solver.");
        publish_output_number(
                "Solver optimization wall time [s]",
                result.solver_diagnostics.optimization_wall_seconds,
                "Solver optimization wall time [s]",
                "Wall-clock time spent in the effective iterative optimization, excluding matrix "
                "and preconditioner assembly.");
        publish_output_number(
                "Maximum magnetic vector potential [SI]",
                result.max_abs_potential,
                "Maximum magnetic vector potential [SI]",
                "Maximum absolute value of the computed magnetic vector potential.");
        publish_output_number(
                "Maximum magnetic induction [T]",
                result.max_abs_induction,
                "Maximum magnetic induction [T]",
                "Maximum absolute value of the computed magnetic induction.");
        publish_output_number(
                "Maximum magnetic field [A/m]",
                result.max_abs_field,
                "Maximum magnetic field [A/m]",
                "Maximum absolute value of the computed magnetic field.");
        publish_output_number(
                "Air-gap mean magnetic induction [T]",
                result.num_air_gap_cells == 0
                        ? 0.0
                        : result.air_gap_induction_magnitude_sum
                                  / static_cast<double>(result.num_air_gap_cells),
                "Air-gap mean magnetic induction [T]",
                "Mean magnitude of the magnetic induction over air-gap cells.");
        publish_output_number(
                "Mean force density magnitude [N/m^3]",
                result.num_cells == 0 ? 0.0
                                      : result.force_density_magnitude_sum
                                                / static_cast<double>(result.num_cells),
                "Mean force density magnitude [N/m^3]",
                "Mean magnitude of the force density over all supported cells.");

        std::ostringstream diagnostics_stream;
        diagnostics_stream << "SimiLie solver diagnostics: iterations="
                           << result.solver_diagnostics.iterations
                           << ", final residual L2=" << result.solver_diagnostics.final_residual_l2
                           << ", final relative residual="
                           << result.solver_diagnostics.final_relative_residual
                           << ", optimization wall time="
                           << result.solver_diagnostics.optimization_wall_seconds
                           << " s, air-gap mean |B|="
                           << (result.num_air_gap_cells == 0
                                       ? 0.0
                                       : result.air_gap_induction_magnitude_sum
                                                 / static_cast<double>(result.num_air_gap_cells))
                           << " T, mean |f|="
                           << (result.num_cells == 0
                                       ? 0.0
                                       : result.force_density_magnitude_sum
                                                 / static_cast<double>(result.num_cells))
                           << " N/m^3";
        client().sendInfo(diagnostics_stream.str());
        publish_status("Magnetostatics solve completed");
    }

    void run_magnetostatics_problem(SilproProblem const& problem)
    {
        client().sendProgress(
                module_name() + " ONELAB interface: exporting mesh for problem '" + problem.name
                + "'");
        std::filesystem::path const mesh_file = export_input_mesh_from_gmsh();
        MagnetostaticsInputs const inputs = read_magnetostatics_inputs();
        MagnetostaticsRegionTags const region_tags = read_magnetostatics_region_tags();
        std::filesystem::path const output_view_file
                = mesh_file.parent_path() / "similie_linear_magnetostatics_inputs.pos";
        solvers::StrongFormulationSolverSettings const solver_settings
                = detail::assemble_solver_settings(problem.solver_settings);
        MagnetostaticsResult const result
                = physics::magnetostatics::detail::run_structured_linear_magnetostatics_problem(
                        mesh_file,
                        output_view_file,
                        inputs,
                        region_tags,
                        solver_settings,
                        [&](std::string const& message) { client().sendInfo(message); });
        client().sendMergeFileRequest(std::filesystem::absolute(output_view_file).string());
        publish_magnetostatics_outputs(mesh_file, inputs, solver_settings, result);
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
                                         "Save \""
                                         + absolute_mesh_file.string() + "\";";

        client().sendInfo(
                "Asking Gmsh to export the current mesh to " + absolute_mesh_file.string());
        client().sendParseStringRequest(gmsh_command);

        std::uintmax_t previous_size = 0;
        int stable_size_count = 0;

        for (int attempt = 0; attempt < 1200; ++attempt) {
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
                    std::string const
                            content((std::istreambuf_iterator<char>(stream)),
                                    std::istreambuf_iterator<char>());
                    bool const has_completed_elements
                            = content.find("$EndElements") != std::string::npos;
                    bool const has_empty_nodes
                            = content.find("$Nodes\n0\n$EndNodes") != std::string::npos;
                    bool const has_empty_elements
                            = content.find("$Elements\n0\n$EndElements") != std::string::npos;
                    if (has_completed_elements && !has_empty_nodes && !has_empty_elements) {
                        client().sendInfo(
                                "Detected completed exported mesh file at "
                                + absolute_mesh_file.string());
                        return;
                    }
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }

        if (std::filesystem::exists(absolute_mesh_file)) {
            std::ifstream stream(absolute_mesh_file);
            std::string const
                    content((std::istreambuf_iterator<char>(stream)),
                            std::istreambuf_iterator<char>());
            bool const has_empty_nodes = content.find("$Nodes\n0\n$EndNodes") != std::string::npos;
            bool const has_empty_elements
                    = content.find("$Elements\n0\n$EndElements") != std::string::npos;
            if (has_empty_nodes || has_empty_elements) {
                throw std::runtime_error(
                        "no current mesh available in Gmsh: mesh the model first or set 'Mesh "
                        "file' explicitly");
            }
        }

        throw std::runtime_error(
                "Gmsh did not export the current mesh file '" + absolute_mesh_file.string()
                + "' before the ONELAB timeout.");
    }

    std::string m_module_name;
    Client* m_client = nullptr;
};

} // namespace similie::onelab_interface
