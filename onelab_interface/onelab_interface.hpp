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
#include <streambuf>
#include <string>
#include <thread>
#include <utility>
#include <variant>
#include <vector>

#include <ddc/ddc.hpp>

#include <similie/physics/hamilton_equations.hpp>
#include <similie/physics/scalar_field/scalar_field_with_power_coupling.hpp>
#include <similie/solvers/minimize_strong_formulation_residual.hpp>

#include <onelab.h>

#include "gmsh_structured_grid.hpp"
#include "linear_magnetostatics_onelab.hpp"
#include "minimize_strong_formulation_residual_onelab.hpp"
#include "scalar_field_with_power_coupling_onelab.hpp"

namespace similie::onelab_interface {

enum class SupportedPhysics {
    ScalarFieldWithPowerCoupling,
    LinearMagnetostatics,
    NonLinearMagnetostatics,
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
    solvers::Criterion criterion = solvers::Criterion::MomentsTemporalDerivative;
};

struct SingleElectricalConductorMaterialWithSingleLinearMagneticMaterialPreprocess
{
    std::string name;
    std::vector<std::string> positive_electrical_conductor_tags;
    std::vector<std::string> negative_electrical_conductor_tags;
    std::vector<std::string> magnetic_material_tags;
    std::string current_density_z_parameter;
    std::string magnetic_permeability_parameter;
    bool use_nonlinear_magnetic_material = false;
    std::string nonlinear_bh_curve = "EIcore";
};

struct ForceDensityDiagnosticsPostprocess
{
    std::string name;
    std::vector<std::string> diagnostic_region_tags;
};

struct SilproProblem
{
    std::string name;
    SupportedPhysics physics;
    SupportedSolver solver;
    ScalarFieldWithPowerCouplingProblem scalar_field;
    MinimizeStrongFormulationResidualProblem solver_settings;
    SingleElectricalConductorMaterialWithSingleLinearMagneticMaterialPreprocess
            single_electrical_conductor_material_with_single_linear_magnetic_material_preprocess;
    ForceDensityDiagnosticsPostprocess force_density_diagnostics_postprocess;
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

inline std::vector<std::string> collect_values(SilproSection const& section)
{
    std::vector<std::string> values;
    values.reserve(section.values.size());
    for (auto const& [_, value] : section.values) {
        values.push_back(value);
    }
    return values;
}

inline SupportedPhysics parse_physics_kind(std::string const& value)
{
    if (value == "LinearMagnetostatics") {
        return SupportedPhysics::LinearMagnetostatics;
    }
    if (value == "NonLinearMagnetostatics") {
        return SupportedPhysics::NonLinearMagnetostatics;
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

inline solvers::Criterion parse_solver_criterion(std::string const& value)
{
    if (value == "PotentialTemporalDerivative") {
        return solvers::Criterion::PotentialTemporalDerivative;
    }
    if (value == "MomentsTemporalDerivative") {
        return solvers::Criterion::MomentsTemporalDerivative;
    }
    if (value == "PotentialAndMomentsTemporalDerivative") {
        return solvers::Criterion::PotentialAndMomentsTemporalDerivative;
    }
    throw std::runtime_error("unsupported solver criterion '" + value + "' in .silpro file");
}

inline SilproProblem parse_silpro_problem(std::filesystem::path const& file)
{
    SilproSection const root = parse_silpro_tree(file);
    SilproSection const& problem_section = required_section(root, "Problem", file.string());
    SilproSection const& solver_section = required_section(root, "Solver", file.string());

    SilproProblem problem {
            .name = get_value_or(problem_section, "Name", "SimiLie problem"),
            .physics = parse_physics_kind(
                    get_value_or(problem_section, "Physics", "LinearMagnetostatics")),
            .solver = parse_solver_kind(
                    get_value_or(problem_section, "Solver", "MinimizeStrongFormulationResidual")),
            .scalar_field = {},
            .solver_settings = {},
            .single_electrical_conductor_material_with_single_linear_magnetic_material_preprocess
            = {
                    .name = "SingleElectricalConductorMaterialWithSingleLinearMagneticMaterial",
                    .positive_electrical_conductor_tags = {},
                    .negative_electrical_conductor_tags = {},
                    .magnetic_material_tags = {},
                    .current_density_z_parameter
                    = "Input/90SimiLie/0Coil current density magnitude z [A/m^2]",
                    .magnetic_permeability_parameter
                    = "Input/90SimiLie/1Core magnetic permeability [H/m]",
                    .use_nonlinear_magnetic_material = false,
                    .nonlinear_bh_curve = "EIcore",
            },
            .force_density_diagnostics_postprocess
            = {
                    .name = "ForceDensityDiagnostics",
                    .diagnostic_region_tags = {},
            },
    };
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
    problem.solver_settings.criterion = parse_solver_criterion(get_value_or(
            solver_section,
            "Criterion",
            "MomentsTemporalDerivative"));

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
    } else if (problem.physics == SupportedPhysics::LinearMagnetostatics
               || problem.physics == SupportedPhysics::NonLinearMagnetostatics) {
        if (auto preprocess_it = root.sections.find("Preprocess");
            preprocess_it != root.sections.end()) {
            SilproSection const& preprocess = preprocess_it->second;
            problem.single_electrical_conductor_material_with_single_linear_magnetic_material_preprocess
                    .name
                    = get_value_or(
                            preprocess,
                            "Name",
                            problem.single_electrical_conductor_material_with_single_linear_magnetic_material_preprocess
                                    .name);
            if (auto it = preprocess.sections.find("PositiveElectricalConductorTags");
                it != preprocess.sections.end()) {
                problem.single_electrical_conductor_material_with_single_linear_magnetic_material_preprocess
                        .positive_electrical_conductor_tags
                        = collect_values(it->second);
            }
            if (auto it = preprocess.sections.find("NegativeElectricalConductorTags");
                it != preprocess.sections.end()) {
                problem.single_electrical_conductor_material_with_single_linear_magnetic_material_preprocess
                        .negative_electrical_conductor_tags
                        = collect_values(it->second);
            }
            if (auto it = preprocess.sections.find("MagneticMaterialTags");
                it != preprocess.sections.end()) {
                problem.single_electrical_conductor_material_with_single_linear_magnetic_material_preprocess
                        .magnetic_material_tags
                        = collect_values(it->second);
            } else if (auto it = preprocess.sections.find("LinearMagneticMaterialTags");
                       it != preprocess.sections.end()) {
                problem.single_electrical_conductor_material_with_single_linear_magnetic_material_preprocess
                        .magnetic_material_tags
                        = collect_values(it->second);
            }
            problem.single_electrical_conductor_material_with_single_linear_magnetic_material_preprocess
                    .current_density_z_parameter
                    = get_value_or(
                            preprocess,
                            "CurrentDensityZ",
                            problem.single_electrical_conductor_material_with_single_linear_magnetic_material_preprocess
                                    .current_density_z_parameter);
            problem.single_electrical_conductor_material_with_single_linear_magnetic_material_preprocess
                    .magnetic_permeability_parameter
                    = get_value_or(
                            preprocess,
                            "MagneticPermeability",
                            problem.single_electrical_conductor_material_with_single_linear_magnetic_material_preprocess
                                    .magnetic_permeability_parameter);
            problem.single_electrical_conductor_material_with_single_linear_magnetic_material_preprocess
                    .use_nonlinear_magnetic_material
                    = parse_bool(get_value_or(
                            preprocess,
                            "UseNonlinearMagneticMaterial",
                            problem.single_electrical_conductor_material_with_single_linear_magnetic_material_preprocess
                                            .use_nonlinear_magnetic_material
                                    ? "1"
                                    : "0"));
            problem.single_electrical_conductor_material_with_single_linear_magnetic_material_preprocess
                    .nonlinear_bh_curve
                    = get_value_or(
                            preprocess,
                            "NonlinearBHCurve",
                            problem.single_electrical_conductor_material_with_single_linear_magnetic_material_preprocess
                                    .nonlinear_bh_curve);
        }
        if (auto postprocess_it = root.sections.find("PostProcess");
            postprocess_it != root.sections.end()) {
            SilproSection const& postprocess = postprocess_it->second;
            problem.force_density_diagnostics_postprocess.name = get_value_or(
                    postprocess,
                    "Name",
                    problem.force_density_diagnostics_postprocess.name);
            if (auto it = postprocess.sections.find("DiagnosticRegionTags");
                it != postprocess.sections.end()) {
                problem.force_density_diagnostics_postprocess.diagnostic_region_tags
                        = collect_values(it->second);
            }
        }
    }

    return problem;
}

using sil::onelab_interface::gmsh::HexahedralCell;
using sil::onelab_interface::gmsh::HexahedralMesh;
using sil::onelab_interface::gmsh::MeshNode;
using sil::onelab_interface::gmsh::QuadrilateralCell;
using sil::onelab_interface::gmsh::QuadrilateralMesh;
using sil::onelab_interface::gmsh::StructuredGrid2D;
using sil::onelab_interface::gmsh::SupportedMesh;
using StructuredGrid = sil::onelab_interface::gmsh::StructuredGrid3D;
using sil::onelab_interface::gmsh::build_structured_grid;
using sil::onelab_interface::gmsh::parse_supported_msh2_mesh;

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
        ScopedOnelabStreamForwarding const stream_forwarding(client());
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

    class OnelabStreambuf : public std::streambuf
    {
        Client* m_client;
        bool m_error_stream;
        std::string m_buffer;

    public:
        explicit OnelabStreambuf(Client* client, bool error_stream)
            : m_client(client)
            , m_error_stream(error_stream)
        {
        }

        ~OnelabStreambuf() override
        {
            flush_buffer();
        }

    protected:
        int overflow(int ch) override
        {
            if (ch == traits_type::eof()) {
                flush_buffer();
                return traits_type::not_eof(ch);
            }

            char const c = static_cast<char>(ch);
            if (c == '\n') {
                flush_buffer();
            } else {
                m_buffer.push_back(c);
            }
            return ch;
        }

        int sync() override
        {
            flush_buffer();
            return 0;
        }

    private:
        void flush_buffer()
        {
            if (m_client == nullptr || m_buffer.empty()) {
                m_buffer.clear();
                return;
            }
            if (m_error_stream) {
                m_client->sendError(m_buffer);
            } else {
                m_client->sendInfo(m_buffer);
            }
            m_buffer.clear();
        }
    };

    class ScopedOnelabStreamForwarding
    {
        OnelabStreambuf m_cout_buffer;
        OnelabStreambuf m_cerr_buffer;
        std::streambuf* m_old_cout_buffer;
        std::streambuf* m_old_cerr_buffer;

    public:
        explicit ScopedOnelabStreamForwarding(Client& client)
            : m_cout_buffer(&client, false)
            , m_cerr_buffer(&client, true)
            , m_old_cout_buffer(std::cout.rdbuf(&m_cout_buffer))
            , m_old_cerr_buffer(std::cerr.rdbuf(&m_cerr_buffer))
        {
        }

        ~ScopedOnelabStreamForwarding()
        {
            std::cout.flush();
            std::cerr.flush();
            std::cout.rdbuf(m_old_cout_buffer);
            std::cerr.rdbuf(m_old_cerr_buffer);
        }
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
                problem.physics == SupportedPhysics::LinearMagnetostatics
                        ? "LinearMagnetostatics"
                        : problem.physics == SupportedPhysics::NonLinearMagnetostatics
                                  ? "NonLinearMagnetostatics"
                                  : "ScalarFieldWithPowerCoupling",
                true);
        publish_or_sync_string(
                problem_parameter_name("0Problem", "2Solver"),
                "Solver",
                "Solver selected in the .silpro file.",
                "MinimizeStrongFormulationResidual",
                true);

        minimize_strong_formulation_residual_onelab::synchronize_controls(
                problem.solver_settings,
                [&](std::string const& section, std::string const& name) {
                    return problem_parameter_name(section, name);
                },
                publish_or_sync_number);

        if (problem.physics == SupportedPhysics::ScalarFieldWithPowerCoupling) {
            scalar_field_with_power_coupling_onelab::synchronize_controls(
                    problem,
                    [&](std::string const& section, std::string const& name) {
                        return problem_parameter_name(section, name);
                    },
                    publish_or_sync_number);
        }

        if (problem.physics == SupportedPhysics::LinearMagnetostatics
            || problem.physics == SupportedPhysics::NonLinearMagnetostatics) {
            linear_magnetostatics_onelab::synchronize_controls(
                    problem,
                    [&](std::string const& section, std::string const& name) {
                        return problem_parameter_name(section, name);
                    },
                    publish_or_sync_string,
                    publish_or_sync_number);
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
        problem.solver_settings
                = minimize_strong_formulation_residual_onelab::apply_control_overrides(
                        problem.solver_settings,
                        [&](std::string const& section, std::string const& name) {
                            return problem_parameter_name(section, name);
                        },
                        [&](std::string const& name, double default_value) {
                            return get_first_number_value(name, default_value);
                        });

        if (problem.physics == SupportedPhysics::ScalarFieldWithPowerCoupling) {
            problem = scalar_field_with_power_coupling_onelab::apply_control_overrides(
                    std::move(problem),
                    [&](std::string const& section, std::string const& name) {
                        return problem_parameter_name(section, name);
                    },
                    [&](std::string const& name, double default_value) {
                        return get_first_number_value(name, default_value);
                    });
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
                problem.physics == SupportedPhysics::LinearMagnetostatics
                        ? "LinearMagnetostatics"
                        : problem.physics == SupportedPhysics::NonLinearMagnetostatics
                                  ? "NonLinearMagnetostatics"
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
                    = scalar_field_with_power_coupling_onelab::assemble_hamiltonian(problem);
            scalar_field_with_power_coupling_onelab::run();
            return;
        }

        run_linear_magnetostatics_problem(problem, silpro_file);
    }

    [[nodiscard]] int read_required_integer_parameter(std::string const& parameter_name)
    {
        double const value = read_number_parameter(
                parameter_name,
                std::nullopt,
                std::numeric_limits<double>::quiet_NaN());
        if (std::isnan(value)) {
            throw std::runtime_error("missing ONELAB parameter '" + parameter_name + "'");
        }
        return static_cast<int>(std::lround(value));
    }

    void run_linear_magnetostatics_problem(
            SilproProblem const& problem,
            std::filesystem::path const& silpro_file)
    {
        client().sendProgress(
                module_name() + " ONELAB interface: exporting mesh for problem '" + problem.name
                + "'");
        std::filesystem::path const mesh_file = export_input_mesh_from_gmsh();
        SilproProblem problem_for_inputs = problem;
        problem_for_inputs
                .single_electrical_conductor_material_with_single_linear_magnetic_material_preprocess
                .use_nonlinear_magnetic_material
                = problem.physics == SupportedPhysics::NonLinearMagnetostatics;
        auto const inputs = linear_magnetostatics_onelab::read_inputs(
                problem_for_inputs,
                [&](std::string const& preferred_parameter,
                    std::optional<std::string> const& fallback_parameter,
                    double fallback_value) {
                    return read_number_parameter(
                            preferred_parameter,
                            fallback_parameter,
                            fallback_value);
                },
                [&](std::string const& parameter_name) {
                    return read_required_integer_parameter(parameter_name);
                });
        auto mutable_inputs = inputs;
        if (mutable_inputs.use_nonlinear_magnetic_material) {
            linear_magnetostatics_onelab::detail::validate_nonlinear_bh_curve(
                    mutable_inputs.nonlinear_bh_curve);
            linear_magnetostatics_onelab::detail::load_bh_curve_from_bh_pro(
                    mutable_inputs,
                    silpro_file.parent_path() / "BH.pro");
        }
        std::filesystem::path const output_view_file
                = mesh_file.parent_path() / "similie_linear_magnetostatics_inputs.pos";
        solvers::StrongFormulationSolverSettings const solver_settings {
                .max_iterations = problem.solver_settings.max_iterations,
                .relative_tolerance = problem.solver_settings.relative_tolerance,
                .jacobi_max_block_size = problem.solver_settings.jacobi_max_block_size,
                .use_matrix_free = problem.solver_settings.use_matrix_free,
                .criterion = problem.solver_settings.criterion,
        };
        auto const result = linear_magnetostatics_onelab::
                run(mesh_file,
                    output_view_file,
                    mutable_inputs,
                    solver_settings,
                    [&](std::string const& message) { client().sendInfo(message); });
        client().sendMergeFileRequest(std::filesystem::absolute(output_view_file).string());
        linear_magnetostatics_onelab::publish_outputs(
                mesh_file,
                mutable_inputs,
                solver_settings,
                result,
                [&](std::string const& name,
                    std::string const& value,
                    std::string const& label,
                    std::string const& help,
                    std::string const& kind) {
                    publish_output_string(name, value, label, help, kind);
                },
                [&](std::string const& name,
                    double value,
                    std::string const& label,
                    std::string const& help) { publish_output_number(name, value, label, help); },
                [&](std::string const& status) { publish_status(status); });
        std::ostringstream diagnostics_stream;
        diagnostics_stream << "SimiLie solver diagnostics: iterations="
                           << result.solver_diagnostics.iterations
                           << ", final residual L2=" << result.solver_diagnostics.final_residual_l2
                           << ", final relative residual="
                           << result.solver_diagnostics.final_relative_residual
                           << ", duration="
                           << result.solver_diagnostics.duration << " s, mean |f|="
                           << (result.num_diagnostic_cells == 0
                                       ? 0.0
                                       : result.diagnostic_force_density_magnitude_sum
                                                 / static_cast<double>(result.num_diagnostic_cells))
                           << " N/m^3";
        client().sendInfo(diagnostics_stream.str());
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
