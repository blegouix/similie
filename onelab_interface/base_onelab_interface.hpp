// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: MIT

#pragma once

#include <chrono>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <ddc/ddc.hpp>

#include <onelab.h>

namespace similie::onelab_interface {

class BaseOnelabInterface
{
public:
    explicit BaseOnelabInterface(std::string module_name = "SimiLie")
        : m_module_name(std::move(module_name))
    {
    }
    virtual ~BaseOnelabInterface() = default;

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
        publish_module_parameters();

        try {
            run_module();
            client().sendInfo(module_name() + " ONELAB interface finished");
        } catch (std::exception const& exception) {
            client().sendError(exception.what());
            return 3;
        }

        return 0;
    }

protected:
    using Client = onelab::remoteNetworkClient;

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
    [[nodiscard]] Client const& client() const
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
        std::vector<onelab::string> string_parameters;
        client().get(string_parameters, parameter_name);
        if (string_parameters.empty()) {
            return "";
        }
        return string_parameters.front().getValue();
    }
    [[nodiscard]] double get_first_number_value(
            std::string const& parameter_name,
            double default_value)
    {
        std::vector<onelab::number> number_parameters;
        client().get(number_parameters, parameter_name);
        if (number_parameters.empty()) {
            return default_value;
        }
        return number_parameters.front().getValue();
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

    virtual void publish_common_parameters()
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
    virtual void publish_module_parameters() = 0;
    virtual void run_module() = 0;

private:
    struct OnelabArguments
    {
        std::string client_name;
        std::string socket_address;
    };

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

private:
    std::string m_module_name;
    Client* m_client = nullptr;
};

} // namespace similie::onelab_interface
