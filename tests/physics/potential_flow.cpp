// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

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
    for (std::size_t i : {2, 3, 6, 7})
        mesh << element++ << " 1 2 " << (i < 4 ? 10 : 11) << " 8 " << 16 + i + 1 << ' '
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
    std::filesystem::remove(mesh_file);
    std::filesystem::remove(result_file);
}

TEST(PotentialFlow, MappedHarmonicPotential)
{
    std::size_t const nx = 9;
    std::size_t const ny = 7;
    std::vector<std::array<double, 2>> positions(nx * ny);
    for (std::size_t j = 0; j < ny; ++j)
        for (std::size_t i = 0; i < nx; ++i)
            positions[j * nx + i] = {static_cast<double>(i) + 0.3 * j, static_cast<double>(j)};
    potential_flow::TensorLaplacianStencils const stencils
            = potential_flow::conservative_tensor_laplacian_rows(positions, nx, ny);
    for (std::size_t j = 1; j + 1 < ny; ++j)
        for (std::size_t i = 1; i + 1 < nx; ++i) {
            double residual = 0.0;
            for (auto const& [column, coefficient] : stencils.rows[j * nx + i])
                residual += coefficient * positions[column][0] * positions[column][1];
            // x*y is harmonic in physical coordinates, including on a skew grid.
            EXPECT_NEAR(residual, 0.0, 1.0e-12) << "i=" << i << " j=" << j;
        }
    for (std::size_t row = 0; row < stencils.rows.size(); ++row) {
        double constant_residual = 0.0;
        for (auto const& [column, coefficient] : stencils.rows[row]) {
            constant_residual += coefficient;
            EXPECT_NEAR(coefficient, stencils.rows[column].at(row), 1.0e-12);
        }
        EXPECT_NEAR(constant_residual, 0.0, 1.0e-12);
    }
}

TEST(PotentialFlow, AffineWallFlux)
{
    std::size_t const nx = 9;
    std::size_t const ny = 7;
    std::vector<potential_flow::PotentialFlowPatch> domains(2);
    for (std::size_t side = 0; side < domains.size(); ++side) {
        potential_flow::PotentialFlowPatch& domain = domains[side];
        domain.nodes_x = nx;
        domain.nodes_y = ny;
        domain.positions.resize(nx * ny);
        domain.lower_y.resize(nx, potential_flow::NormalFlux {});
        domain.upper_y.resize(nx, potential_flow::NormalFlux {});
        for (std::size_t j = 0; j < ny; ++j)
            for (std::size_t i = 0; i < nx; ++i)
                domain.positions[j * nx + i]
                        = {static_cast<double>(side == 0 ? i : nx - 1 - i) + 0.3 * j,
                           static_cast<double>(j)};
        potential_flow::TensorLaplacianStencils const stencils
                = potential_flow::tensor_laplacian_rows(domain.positions, nx, ny);
        domain.laplacian_rows = stencils.rows;
        domain.lower_x_flux_rows = stencils.lower_x_flux_rows;
        domain.upper_x_flux_rows = stencils.upper_x_flux_rows;
        domain.lower_y_flux_rows = stencils.lower_y_flux_rows;
        domain.upper_y_flux_rows = stencils.upper_y_flux_rows;
    }
    std::vector<potential_flow::TraceConnection> const connections {
            {0, potential_flow::TraceSide::UpperX, 1, potential_flow::TraceSide::LowerX, 0.0},
            {1, potential_flow::TraceSide::UpperX, 0, potential_flow::TraceSide::LowerX, 0.0}};
    potential_flow::FreeScalarFieldHamiltonian const hamiltonian(0.0, 0.0, 2.0);
    similie::physics::HamiltonEquations const equations(hamiltonian);
    potential_flow::PotentialFlowSystem const system
            = potential_flow::assemble_potential_flow_system<
                    potential_flow::X,
                    potential_flow::Y>(domains, connections, equations, 1.0);
    std::vector<double> potential(system.rows.size());
    for (std::size_t side = 0; side < domains.size(); ++side)
        for (std::size_t local = 0; local < nx * ny; ++local)
            potential[system.global_index[side][local]] = domains[side].positions[local][0];
    for (std::size_t row = 0; row < system.rows.size(); ++row) {
        double residual = 0.0;
        for (auto const& [column, coefficient] : system.rows[row])
            residual += coefficient * potential[column];
        EXPECT_NEAR(residual, system.base_rhs[row], 1.0e-12) << "row=" << row;
    }
    for (potential_flow::PotentialFlowPatch& domain : domains) {
        domain.lower_y.assign(nx, potential_flow::NormalFlux {-1.0});
        domain.upper_y.assign(nx, potential_flow::NormalFlux {1.0});
    }
    potential_flow::PotentialFlowSystem const nonzero_flux_system
            = potential_flow::assemble_potential_flow_system<
                    potential_flow::X,
                    potential_flow::Y>(domains, connections, equations, 1.0);
    for (std::size_t side = 0; side < domains.size(); ++side)
        for (std::size_t local = 0; local < nx * ny; ++local)
            potential[nonzero_flux_system.global_index[side][local]]
                    = domains[side].positions[local][1];
    for (std::size_t row = 0; row < nonzero_flux_system.rows.size(); ++row) {
        double residual = 0.0;
        for (auto const& [column, coefficient] : nonzero_flux_system.rows[row])
            residual += coefficient * potential[column];
        EXPECT_NEAR(residual, nonzero_flux_system.base_rhs[row], 1.0e-12) << "row=" << row;
    }
}

class PotentialFlowWallJunctionTest : public ::testing::TestWithParam<bool>
{
};

TEST_P(PotentialFlowWallJunctionTest, WallConditionAtConnectedTrace)
{
    std::vector<potential_flow::PotentialFlowPatch> domains(4);
    std::vector<potential_flow::TraceConnection> connections;
    for (std::size_t side = 0; side < domains.size(); ++side) {
        potential_flow::PotentialFlowPatch& domain = domains[side];
        domain.nodes_x = 3;
        domain.nodes_y = 3;
        domain.positions.resize(9);
        domain.laplacian_rows.resize(9);
        domain.lower_x_flux_rows.resize(3);
        domain.upper_x_flux_rows.resize(3);
        domain.lower_y_flux_rows.resize(3);
        domain.upper_y_flux_rows.resize(3);
        domain.lower_y.resize(3, potential_flow::NormalFlux {});
        domain.upper_y.resize(3, potential_flow::NormalFlux {});
        domain.conservative_laplacian_rows = GetParam();
        if (domain.conservative_laplacian_rows) {
            domain.lower_y.assign(3, potential_flow::NaturalBoundary {});
            domain.upper_y.assign(3, potential_flow::NaturalBoundary {});
        }
        for (std::size_t j = 0; j < 3; ++j) {
            for (std::size_t i = 0; i < 3; ++i) {
                double const angle = (2 * side + i) * std::acos(-1.0) / 4.0;
                domain.positions[3 * j + i]
                        = {(1.0 + j) * std::cos(angle), (1.0 + j) * std::sin(angle)};
                domain.laplacian_rows[3 * j + i][3 * j + i] = 1.0;
            }
            domain.lower_x_flux_rows[j] = {{3 * j, -1.0}, {3 * j + 1, 1.0}};
            domain.upper_x_flux_rows[j] = {{3 * j + 1, -1.0}, {3 * j + 2, 1.0}};
        }
        for (std::size_t i = 0; i < 3; ++i) {
            domain.lower_y_flux_rows[i] = {{i, -1.0}, {i + 3, 1.0}};
            domain.upper_y_flux_rows[i] = {{i + 3, -1.0}, {i + 6, 1.0}};
        }
        connections.push_back(
                {side,
                 potential_flow::TraceSide::UpperX,
                 (side + 1) % domains.size(),
                 potential_flow::TraceSide::LowerX,
                 0.0});
    }
    potential_flow::FreeScalarFieldHamiltonian const hamiltonian(0.0, 0.0, 2.0);
    similie::physics::HamiltonEquations const equations(hamiltonian);
    potential_flow::PotentialFlowSystem const system
            = potential_flow::assemble_potential_flow_system<
                    potential_flow::X,
                    potential_flow::Y>(domains, connections, equations, 1.0);
    std::vector<double> radial(system.rows.size());
    for (std::size_t side = 0; side < domains.size(); ++side)
        for (std::size_t local = 0; local < 9; ++local)
            radial[system.global_index[side][local]] = 1.0 + local / 3;
    for (std::size_t side = 0; side < domains.size(); ++side)
        for (std::size_t j : {std::size_t(0), std::size_t(2)}) {
            std::size_t const row = system.global_index[side][3 * j + 2];
            double residual = 0.0;
            for (auto const& [column, coefficient] : system.rows[row])
                residual += coefficient * radial[column];
            // A radial potential violates impermeability, including at patch junctions.
            EXPECT_GT(std::abs(residual), 0.1) << "side=" << side << " j=" << j;
        }
    // At a Dirichlet/Neumann corner the shared potential is prescribed;
    // the Neumann condition still applies along the neighbouring wall.
    domains[0].upper_y[2] = potential_flow::PrescribedPotential {7.0};
    potential_flow::PotentialFlowSystem const prescribed_system
            = potential_flow::assemble_potential_flow_system<
                    potential_flow::X,
                    potential_flow::Y>(domains, connections, equations, 1.0);
    std::size_t const corner = prescribed_system.global_index[0][8];
    ASSERT_EQ(prescribed_system.rows[corner].size(), 1);
    EXPECT_DOUBLE_EQ(prescribed_system.rows[corner].at(corner), 1.0);
    EXPECT_DOUBLE_EQ(prescribed_system.base_rhs[corner], 7.0);
}

INSTANTIATE_TEST_SUITE_P(ScalarRowConventions, PotentialFlowWallJunctionTest, ::testing::Bool());
