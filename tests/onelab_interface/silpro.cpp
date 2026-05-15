// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: MIT

#include <filesystem>

#include <gtest/gtest.h>

#include "onelab_interface.hpp"

namespace {

std::filesystem::path test_file(char const* name)
{
    return std::filesystem::path(SIMILIE_ONELAB_TEST_DIR) / name;
}

} // namespace

TEST(OnelabInterface, ParseMagnetostaticsSilpro)
{
    similie::onelab_interface::SilproProblem const problem
            = similie::onelab_interface::OnelabInterface::parse_silpro_file(
                    test_file("magnetostatics.silpro"));

    EXPECT_EQ(problem.name, "Test magnetostatics problem");
    EXPECT_EQ(problem.physics, similie::onelab_interface::SupportedPhysics::Magnetostatics);
    EXPECT_EQ(
            problem.solver,
            similie::onelab_interface::SupportedSolver::MinimizeStrongFormulationResidual);
    EXPECT_EQ(problem.solver_settings.max_iterations, 4321U);
    EXPECT_DOUBLE_EQ(problem.solver_settings.relative_tolerance, 1e-8);
    EXPECT_EQ(problem.solver_settings.jacobi_max_block_size, 3U);
    EXPECT_FALSE(problem.solver_settings.use_matrix_free);

    EXPECT_EQ(problem.magnetostatics.current_rms_parameter, "Input/Test Current");
    EXPECT_EQ(problem.magnetostatics.number_of_turns_parameter, "Input/Test Turns");
    EXPECT_DOUBLE_EQ(problem.magnetostatics.current_rms, 12.5);
    EXPECT_DOUBLE_EQ(problem.magnetostatics.number_of_turns, 144.0);
    EXPECT_DOUBLE_EQ(problem.magnetostatics.core_relative_permeability, 1500.0);
    EXPECT_DOUBLE_EQ(problem.magnetostatics.coil_width, 0.02);
    EXPECT_DOUBLE_EQ(problem.magnetostatics.coil_height, 0.08);
    EXPECT_TRUE(problem.magnetostatics.export_input_fields_view);
    EXPECT_FALSE(problem.magnetostatics.merge_result_view_in_gmsh);
    EXPECT_EQ(problem.magnetostatics.input_fields_view_file, "test_result.pos");
}

TEST(OnelabInterface, ParseScalarFieldSilpro)
{
    similie::onelab_interface::SilproProblem const problem
            = similie::onelab_interface::OnelabInterface::parse_silpro_file(
                    test_file("scalar_field.silpro"));

    EXPECT_EQ(problem.name, "Test scalar field problem");
    EXPECT_EQ(
            problem.physics,
            similie::onelab_interface::SupportedPhysics::ScalarFieldWithPowerCoupling);
    EXPECT_EQ(problem.solver_settings.max_iterations, 250U);
    EXPECT_DOUBLE_EQ(problem.solver_settings.relative_tolerance, 1e-6);
    EXPECT_TRUE(problem.solver_settings.use_matrix_free);
    EXPECT_DOUBLE_EQ(problem.scalar_field.mass, 2.5);
    EXPECT_DOUBLE_EQ(problem.scalar_field.coupling_constant, 3.25);
    EXPECT_DOUBLE_EQ(problem.scalar_field.coupling_power, 6.0);
}
