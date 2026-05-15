// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: MIT

#pragma once

#include <filesystem>

#include "base_onelab_interface.hpp"

namespace similie::onelab_interface {

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
    }
    void run_module() override
    {
        client().sendProgress(module_name() + " ONELAB interface: exporting mesh for linear magnetostatics");
        std::filesystem::path const mesh_file = export_input_mesh_from_gmsh();

        publish_output_string(
                "Input mesh file",
                mesh_file.string(),
                "Input mesh file",
                "Mesh file exported by Gmsh for the linear magnetostatics interface.",
                "file");
        publish_status("linear magnetostatics interface ready; solver implementation is pending");
        client().sendWarning("Linear magnetostatics solver implementation is pending.");
    }
};

} // namespace similie::onelab_interface
