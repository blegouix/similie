// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#include <sstream>

#include <gtest/gtest.h>

#include "potential_flow_onelab.hpp"

namespace potential_flow = similie::onelab_interface::potential_flow_onelab;

TEST(PotentialFlow, ExtrapolatedPatchGradientsAndCirculation)
{
    std::size_t const nx = 6;
    std::size_t const ny = 3;
    std::vector<double> potential(nx * ny);
    for (std::size_t j = 0; j < ny; ++j)
        for (std::size_t i = 0; i < nx; ++i)
            potential[j * nx + i] = static_cast<double>(i * i + 2 * j);
    std::array<std::size_t, 2> const starts {0, 3};
    std::array<std::size_t, 2> const cells {3, 3};
    std::vector<potential_flow::TraceConnection> const connections {
            {0, potential_flow::TraceSide::UpperX, 1, potential_flow::TraceSide::LowerX, -1.0},
            {1, potential_flow::TraceSide::UpperX, 0, potential_flow::TraceSide::LowerX, 0.0}};
    potential_flow::PotentialFlowSamples const samples = potential_flow::
            sample_potential_flow_field(potential, 5.0, nx, ny, starts, cells, connections);
    for (std::size_t j = 0; j < ny; ++j)
        for (std::size_t i = 0; i < nx; ++i) {
            std::size_t const next = (i + 1) % nx;
            double const jump = i == 2 ? -5.0 : 0.0;
            double const expected_x = potential[j * nx + next] - potential[j * nx + i] + jump;
            EXPECT_DOUBLE_EQ(samples.differences[j * nx + i][0], expected_x);
            EXPECT_DOUBLE_EQ(samples.differences[j * nx + i][1], 2.0);
            if (j + 1 < ny) {
                std::array<double, 4> const& cell = samples.cell_potentials[j * nx + i];
                EXPECT_DOUBLE_EQ(cell[0], potential[j * nx + i]);
                EXPECT_DOUBLE_EQ(cell[1], potential[j * nx + next] + jump);
                EXPECT_DOUBLE_EQ(cell[2], potential[(j + 1) * nx + next] + jump);
                EXPECT_DOUBLE_EQ(cell[3], potential[(j + 1) * nx + i]);
                EXPECT_DOUBLE_EQ(cell[1] - cell[0], samples.differences[j * nx + i][0]);
            }
        }
}

TEST(PotentialFlow, CoupledCylinderSolve)
{
    std::filesystem::path const mesh_file = "potential_flow_test.msh";
    std::filesystem::path const result_file = "potential_flow_test.pos";
    // Four mapped patches, two angular cells per patch and two radial cells.
    // Keep the same physical tags and cell ordering as the ONELAB example.
    std::ofstream mesh(mesh_file);
    mesh << std::setprecision(17) << "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$Nodes\n24\n";
    for (std::size_t j = 0; j < 3; ++j)
        for (std::size_t i = 0; i < 8; ++i) {
            double const angle = i * std::acos(-1.0) / 4.0;
            mesh << 8 * j + i + 1 << ' ' << (1.0 + j) * std::cos(angle) << ' '
                 << (1.0 + j) * std::sin(angle) << " 0\n";
        }
    mesh << "$EndNodes\n$Elements\n30\n";
    std::size_t element = 1;
    for (std::size_t i = 0; i < 8; ++i)
        mesh << element++ << " 1 2 12 1 " << i + 1 << ' ' << (i + 1) % 8 + 1 << '\n';
    for (std::size_t i : {3, 4, 7, 0})
        mesh << element++ << " 1 2 " << (i == 3 || i == 4 ? 10 : 11) << " 8 " << 16 + i + 1 << ' '
             << 16 + (i + 1) % 8 + 1 << '\n';
    for (std::size_t j = 0; j < 2; ++j)
        mesh << element++ << " 1 2 13 5 " << 8 * j + 7 << ' ' << 8 * (j + 1) + 7 << '\n';
    for (std::size_t patch = 0; patch < 4; ++patch)
        for (std::size_t a = 0; a < 2; ++a)
            for (std::size_t j = 0; j < 2; ++j) {
                std::size_t const i = 2 * patch + a;
                mesh << element++ << " 3 2 " << (patch == 0 || patch == 3 ? 2 : 3) << ' '
                     << 24 + patch << ' ' << 8 * j + i + 1 << ' ' << 8 * j + (i + 1) % 8 + 1 << ' '
                     << 8 * (j + 1) + (i + 1) % 8 + 1 << ' ' << 8 * (j + 1) + i + 1 << '\n';
            }
    mesh << "$EndElements\n";
    mesh.close();
    potential_flow::Inputs inputs;
    inputs.airfoil = false;
    inputs.impose_circulation = true;
    inputs.circulation = -2.0;
    inputs.velocity = 1.0;
    inputs.box_size = 6.0;
    similie::solvers::StrongFormulationSolverSettings settings;
    settings.relative_tolerance = 1.0e-10;
    settings.max_iterations = 1000;
    potential_flow::Result const result
            = potential_flow::run(mesh_file, result_file, inputs, settings);
    EXPECT_EQ(result.node_count, 24);
    EXPECT_EQ(result.cell_count, 16);
    EXPECT_DOUBLE_EQ(result.circulation, -2.0);
    EXPECT_TRUE(std::isfinite(result.mass_flow_rate));
    EXPECT_TRUE(std::isfinite(result.max_speed));
    EXPECT_GT(result.max_speed, 0.0);
    EXPECT_GT(std::filesystem::file_size(result_file), 0);
    std::ifstream output(result_file);
    std::string line;
    std::getline(output, line); // Potential view header.
    for (std::size_t i = 0; i < 8; ++i)
        for (std::size_t j = 0; j < 2; ++j) {
            ASSERT_TRUE(bool(std::getline(output, line)));
            std::istringstream values(line.substr(line.find("{ ") + 2));
            std::array<double, 4> potential;
            char separator;
            values >> potential[0] >> separator >> potential[1] >> separator >> potential[2]
                    >> separator >> potential[3];
            ASSERT_TRUE(bool(values));
            if (j == 1 && (i == 3 || i == 4 || i == 7 || i == 0)) {
                double const prescribed = i == 3 || i == 4 ? 0.0 : 6.0;
                EXPECT_NEAR(potential[2], prescribed, 1.0e-8);
                EXPECT_NEAR(potential[3], prescribed, 1.0e-8);
            }
        }
    output.close();
    settings.use_matrix_free = false;
    potential_flow::Result const assembled
            = potential_flow::run(mesh_file, result_file, inputs, settings);
    EXPECT_NEAR(assembled.mass_flow_rate, result.mass_flow_rate, 1.0e-8);
    EXPECT_NEAR(assembled.max_speed, result.max_speed, 1.0e-8);
    std::filesystem::remove(mesh_file);
    std::filesystem::remove(result_file);
}
