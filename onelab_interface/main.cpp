// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: MIT

#include <filesystem>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

#include <ddc/ddc.hpp>

#include <onelab.h>

#include "fake_similie_run.hpp"

namespace {

struct OnelabArguments
{
    std::string client_name;
    std::string socket_address;
};

void print_usage(char const* program_name)
{
    std::cerr << "usage: " << program_name << " -onelab <client-name> <socket>\n";
}

bool parse_onelab_arguments(int argc, char** argv, OnelabArguments& parsed_arguments)
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

onelab::number get_or_create_number(
        onelab::remoteNetworkClient& client,
        std::string const& name,
        double default_value,
        std::string const& label,
        std::string const& help,
        double min_value,
        double max_value,
        double step)
{
    std::vector<onelab::number> parameters;
    client.get(parameters, name);
    onelab::number parameter = parameters.empty() ? onelab::number(name, default_value, label, help)
                                                  : parameters.front();
    if (parameters.empty()) {
        parameter.setMin(min_value);
        parameter.setMax(max_value);
        parameter.setStep(step);
        client.set(parameter);
    }
    return parameter;
}

onelab::string get_or_create_string(
        onelab::remoteNetworkClient& client,
        std::string const& name,
        std::string const& default_value,
        std::string const& label,
        std::string const& help)
{
    std::vector<onelab::string> parameters;
    client.get(parameters, name);
    onelab::string parameter = parameters.empty() ? onelab::string(name, default_value, label, help)
                                                  : parameters.front();
    if (parameters.empty()) {
        client.set(parameter);
    }
    return parameter;
}

void publish_interface_parameters(onelab::remoteNetworkClient& client)
{
    onelab::number is_metamodel("IsMetamodel", 1.0);
    is_metamodel.setNeverChanged(true);
    client.set(is_metamodel);

    onelab::number scale = get_or_create_number(
            client,
            "0Modules/SimiLie/0Control/Require rectilinear mesh",
            1.0,
            "Require rectilinear mesh",
            "Reject any incoming mesh that is not a full rectilinear hexahedral grid.",
            0.0,
            1.0,
            1.0);
    scale.setChoices({0.0, 1.0});
    scale.setValueLabels({{0.0, "No"}, {1.0, "Yes"}});
    client.set(scale);

    onelab::number export_view = get_or_create_number(
            client,
            "0Modules/SimiLie/0Control/Export fake view",
            1.0,
            "Export fake view",
            "When enabled, the interface writes a .pos file and asks Gmsh to merge it.",
            0.0,
            1.0,
            1.0);
    export_view.setChoices({0.0, 1.0});
    export_view.setValueLabels({{0.0, "No"}, {1.0, "Yes"}});
    client.set(export_view);

    onelab::string output_file = get_or_create_string(
            client,
            "0Modules/SimiLie/0Control/Output view file",
            (std::filesystem::path(SIMILIE_ONELAB_DEFAULT_OUTPUT_DIR)
             / "similie_rectilinear_positions.pos")
                    .string(),
            "Output view file",
            "Absolute path of the node-position .pos result produced by the SimiLie ONELAB interface.");
    output_file.setKind("file");
    client.set(output_file);

    onelab::string input_mesh_file = get_or_create_string(
            client,
            "0Modules/SimiLie/0Control/Input mesh file",
            "",
            "Input mesh file",
            "Path to the Gmsh .msh file that the SimiLie ONELAB interface should validate.");
    input_mesh_file.setKind("file");
    client.set(input_mesh_file);
}

similie::onelab_interface::FakeRunConfig read_run_configuration(onelab::remoteNetworkClient& client)
{
    std::vector<onelab::number> number_parameters;
    std::vector<onelab::string> string_parameters;

    similie::onelab_interface::FakeRunConfig config;

    client.get(number_parameters, "0Modules/SimiLie/0Control/Export fake view");
    if (!number_parameters.empty()) {
        config.export_view = (number_parameters.front().getValue() != 0.0);
    }

    client.get(string_parameters, "0Modules/SimiLie/0Control/Output view file");
    if (!string_parameters.empty()) {
        config.output_file = string_parameters.front().getValue();
    }

    client.get(string_parameters, "0Modules/SimiLie/0Control/Input mesh file");
    if (!string_parameters.empty()) {
        config.input_mesh_file = string_parameters.front().getValue();
    }

    return config;
}

void publish_result(
        onelab::remoteNetworkClient& client,
        similie::onelab_interface::FakeRunResult const& result,
        bool export_view)
{
    onelab::string
            status("0Modules/SimiLie/1Output/Last status",
                   result.status,
                   "Last status",
                   "Status reported by the fake SimiLie ONELAB run.");
    status.setReadOnly(true);
    client.set(status);

    onelab::number checksum(
            "0Modules/SimiLie/1Output/Position tensor checksum",
            result.checksum,
            "Position tensor checksum",
            "Checksum of the tensor that stores the accepted rectilinear node positions.");
    checksum.setReadOnly(true);
    client.set(checksum);

    onelab::number nx("0Modules/SimiLie/1Output/Grid points in x",
                      static_cast<double>(result.nx),
                      "Grid points in x",
                      "Number of grid points along the x axis.");
    nx.setReadOnly(true);
    client.set(nx);

    onelab::number ny("0Modules/SimiLie/1Output/Grid points in y",
                      static_cast<double>(result.ny),
                      "Grid points in y",
                      "Number of grid points along the y axis.");
    ny.setReadOnly(true);
    client.set(ny);

    onelab::number nz("0Modules/SimiLie/1Output/Grid points in z",
                      static_cast<double>(result.nz),
                      "Grid points in z",
                      "Number of grid points along the z axis.");
    nz.setReadOnly(true);
    client.set(nz);

    onelab::number node_count("0Modules/SimiLie/1Output/Number of nodes",
                              static_cast<double>(result.num_nodes),
                              "Number of nodes",
                              "Number of accepted mesh nodes.");
    node_count.setReadOnly(true);
    client.set(node_count);

    onelab::number cell_count("0Modules/SimiLie/1Output/Number of hexahedra",
                              static_cast<double>(result.num_cells),
                              "Number of hexahedra",
                              "Number of accepted hexahedral cells.");
    cell_count.setReadOnly(true);
    client.set(cell_count);

    onelab::string output_file(
            "0Modules/SimiLie/1Output/Result view file",
            result.output_file.string(),
            "Result view file",
            "The latest fake .pos result generated by the SimiLie ONELAB interface.");
    output_file.setKind("file");
    output_file.setReadOnly(true);
    client.set(output_file);

    if (export_view) {
        client.sendMergeFileRequest(result.output_file.string());
    }
}

} // namespace

int main(int argc, char** argv)
{
    OnelabArguments parsed_arguments;
    if (!parse_onelab_arguments(argc, argv, parsed_arguments)) {
        print_usage(argv[0]);
        return 1;
    }

    Kokkos::ScopeGuard const kokkos_scope(argc, argv);
    ddc::ScopeGuard const ddc_scope(argc, argv);

    onelab::remoteNetworkClient
            client(parsed_arguments.client_name, parsed_arguments.socket_address);
    if (client.getGmshClient() == nullptr) {
        std::cerr << "failed to connect to the ONELAB server at " << parsed_arguments.socket_address
                  << '\n';
        return 2;
    }

    client.sendInfo("SimiLie ONELAB interface connected");
    publish_interface_parameters(client);

    similie::onelab_interface::FakeRunConfig const config = read_run_configuration(client);

    try {
        client.sendProgress("SimiLie ONELAB interface: validating rectilinear mesh");
        similie::onelab_interface::FakeRunResult const result
                = similie::onelab_interface::run_fake_similie_job(config);
        publish_result(client, result, config.export_view);
        client.sendInfo("SimiLie ONELAB interface finished");
    } catch (std::exception const& exception) {
        client.sendError(exception.what());
        return 3;
    }

    return 0;
}
