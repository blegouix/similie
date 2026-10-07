// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#include <gtest/gtest.h>

#include "potential_flow_onelab.hpp"

TEST(PotentialFlow, MappedHarmonicPotential)
{
    std::size_t const nx = 9;
    std::size_t const ny = 7;
    std::vector<std::array<double, 2>> positions(nx * ny);
    for (std::size_t j = 0; j < ny; ++j)
        for (std::size_t i = 0; i < nx; ++i)
            positions[j * nx + i] = {static_cast<double>(i) + 0.3 * j, static_cast<double>(j)};
    similie::onelab_interface::potential_flow_onelab::TensorLaplacianStencils const stencils
            = similie::onelab_interface::potential_flow_onelab::
                    conservative_tensor_laplacian_rows(positions, nx, ny);
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
    std::vector<sil::exterior::ScalarTensorDomain2D> domains(2);
    for (std::size_t side = 0; side < domains.size(); ++side) {
        sil::exterior::ScalarTensorDomain2D& domain = domains[side];
        domain.nodes_x = nx;
        domain.nodes_y = ny;
        domain.positions.resize(nx * ny);
        domain.lower_y.resize(nx, sil::exterior::NormalScalarFluxExtrapolationRule {});
        domain.upper_y.resize(nx, sil::exterior::NormalScalarFluxExtrapolationRule {});
        for (std::size_t j = 0; j < ny; ++j)
            for (std::size_t i = 0; i < nx; ++i)
                domain.positions[j * nx + i]
                        = {static_cast<double>(side == 0 ? i : nx - 1 - i) + 0.3 * j,
                           static_cast<double>(j)};
        similie::onelab_interface::potential_flow_onelab::TensorLaplacianStencils const stencils
                = similie::onelab_interface::potential_flow_onelab::
                        tensor_laplacian_rows(domain.positions, nx, ny);
        domain.laplacian_rows = stencils.rows;
        domain.lower_x_flux_rows = stencils.lower_x_flux_rows;
        domain.upper_x_flux_rows = stencils.upper_x_flux_rows;
        domain.lower_y_flux_rows = stencils.lower_y_flux_rows;
        domain.upper_y_flux_rows = stencils.upper_y_flux_rows;
    }
    std::vector<sil::exterior::ConnectedScalarExtrapolationRule> const connections {
            {0,
             sil::exterior::ScalarTraceSide::UpperX,
             1,
             sil::exterior::ScalarTraceSide::LowerX,
             0.0},
            {1,
             sil::exterior::ScalarTraceSide::UpperX,
             0,
             sil::exterior::ScalarTraceSide::LowerX,
             0.0}};
    similie::onelab_interface::potential_flow_onelab::FreeScalarFieldHamiltonian const
            hamiltonian(0.0, 0.0, 2.0);
    similie::physics::HamiltonEquations const equations(hamiltonian);
    sil::exterior::CoupledScalarLaplacian2D const system
            = sil::exterior::assemble_coupled_scalar_laplacian<
                    similie::onelab_interface::potential_flow_onelab::X,
                    similie::onelab_interface::potential_flow_onelab::
                            Y>(domains, connections, equations, 1.0);
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
    for (sil::exterior::ScalarTensorDomain2D& domain : domains) {
        domain.lower_y.assign(nx, sil::exterior::NormalScalarFluxExtrapolationRule {-1.0});
        domain.upper_y.assign(nx, sil::exterior::NormalScalarFluxExtrapolationRule {1.0});
    }
    sil::exterior::CoupledScalarLaplacian2D const nonzero_flux_system
            = sil::exterior::assemble_coupled_scalar_laplacian<
                    similie::onelab_interface::potential_flow_onelab::X,
                    similie::onelab_interface::potential_flow_onelab::
                            Y>(domains, connections, equations, 1.0);
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
    std::vector<sil::exterior::ScalarTensorDomain2D> domains(4);
    std::vector<sil::exterior::ConnectedScalarExtrapolationRule> connections;
    for (std::size_t side = 0; side < domains.size(); ++side) {
        sil::exterior::ScalarTensorDomain2D& domain = domains[side];
        domain.nodes_x = 3;
        domain.nodes_y = 3;
        domain.positions.resize(9);
        domain.laplacian_rows.resize(9);
        domain.lower_x_flux_rows.resize(3);
        domain.upper_x_flux_rows.resize(3);
        domain.lower_y_flux_rows.resize(3);
        domain.upper_y_flux_rows.resize(3);
        domain.lower_y.resize(3, sil::exterior::NormalScalarFluxExtrapolationRule {});
        domain.upper_y.resize(3, sil::exterior::NormalScalarFluxExtrapolationRule {});
        domain.conservative_laplacian_rows = GetParam();
        if (domain.conservative_laplacian_rows) {
            domain.lower_y.assign(3, sil::exterior::NaturalScalarExtrapolationRule {});
            domain.upper_y.assign(3, sil::exterior::NaturalScalarExtrapolationRule {});
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
                 sil::exterior::ScalarTraceSide::UpperX,
                 (side + 1) % domains.size(),
                 sil::exterior::ScalarTraceSide::LowerX,
                 0.0});
    }
    similie::onelab_interface::potential_flow_onelab::FreeScalarFieldHamiltonian const
            hamiltonian(0.0, 0.0, 2.0);
    similie::physics::HamiltonEquations const equations(hamiltonian);
    sil::exterior::CoupledScalarLaplacian2D const system
            = sil::exterior::assemble_coupled_scalar_laplacian<
                    similie::onelab_interface::potential_flow_onelab::X,
                    similie::onelab_interface::potential_flow_onelab::
                            Y>(domains, connections, equations, 1.0);
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
    domains[0].upper_y[2] = sil::exterior::PrescribedScalarExtrapolationRule {7.0};
    sil::exterior::CoupledScalarLaplacian2D const prescribed_system
            = sil::exterior::assemble_coupled_scalar_laplacian<
                    similie::onelab_interface::potential_flow_onelab::X,
                    similie::onelab_interface::potential_flow_onelab::
                            Y>(domains, connections, equations, 1.0);
    std::size_t const corner = prescribed_system.global_index[0][8];
    ASSERT_EQ(prescribed_system.rows[corner].size(), 1);
    EXPECT_DOUBLE_EQ(prescribed_system.rows[corner].at(corner), 1.0);
    EXPECT_DOUBLE_EQ(prescribed_system.base_rhs[corner], 7.0);
}

INSTANTIATE_TEST_SUITE_P(ScalarRowConventions, PotentialFlowWallJunctionTest, ::testing::Bool());
