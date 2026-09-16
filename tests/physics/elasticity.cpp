// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// AI-GENERATED
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <gtest/gtest.h>
#include <similie/physics/magnetostatics/nonlinear_magnetostatics.hpp>

#include "elasticity_onelab.hpp"

namespace elasticity = similie::onelab_interface::elasticity_onelab;

TEST(Elasticity, PlaneStressAndTensorialShear)
{
    auto const equations = similie::physics::HamiltonEquations {
            similie::physics::elasticity::LinearElasticityHamiltonian<>(200.0, 0.3)};
    similie::physics::elasticity::Strain2D const strain {.xx = 0.02, .yy = -0.01, .xy = 0.03};
    auto const stress = elasticity::detail::linear_elasticity_stress(equations, strain);
    EXPECT_NEAR(stress.xx, 200.0 / (1 - 0.3 * 0.3) * (0.02 - 0.3 * 0.01), 1e-13);
    EXPECT_NEAR(stress.yy, 200.0 / (1 - 0.3 * 0.3) * (-0.01 + 0.3 * 0.02), 1e-13);
    EXPECT_NEAR(stress.xy, 200.0 / (1 + 0.3) * 0.03, 1e-13);
}

TEST(Elasticity, EnergyAdjointAndBackendAgreement)
{
    using memory_space = Kokkos::DefaultExecutionSpace::memory_space;
    Kokkos::View<double*, memory_space> x("x", 9), y("y", 9);
    Kokkos::View<int*, memory_space> active("active", 9), clamped("clamped", 9);
    auto xh = Kokkos::create_mirror_view(x), yh = Kokkos::create_mirror_view(y);
    for (int j = 0; j < 3; ++j) {
        for (int i = 0; i < 3; ++i) {
            xh(i + 3 * j) = i + 0.2 * j + 0.1 * i * j;
            yh(i + 3 * j) = j + 0.3 * i;
        }
    }
    Kokkos::deep_copy(x, xh);
    Kokkos::deep_copy(y, yh);
    Kokkos::deep_copy(active, 1);
    auto const equations = similie::physics::HamiltonEquations {
            similie::physics::elasticity::LinearElasticityHamiltonian<>(200.0, 0.3)};
    elasticity::detail::ElasticityOperator2D<memory_space, decltype(equations)> const
            op(3, 3, x, y, equations, active, clamped);
    auto const data = elasticity::detail::assemble_matrix_data(op);
    std::array<std::array<double, 18>, 18> matrix {};
    for (auto const& entry : data.nonzeros) {
        matrix[entry.row][entry.column] += entry.value;
    }
    Kokkos::View<double**> u("u", 18, 1), residual("residual", 18, 1);
    auto uh = Kokkos::create_mirror_view(u);
    for (int a = 0; a < 9; ++a) {
        // A rigid rotation plus translation must have zero strain and force.
        uh(2 * a, 0) = 2 - yh(a);
        uh(2 * a + 1, 0) = 3 + xh(a);
    }
    Kokkos::deep_copy(u, uh);
    op.apply(Kokkos::DefaultExecutionSpace(), u, residual);
    auto rh = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), residual);
    for (int r = 0; r < 18; ++r) {
        EXPECT_NEAR(rh(r, 0), 0, 1e-11);
    }
    for (int r = 0; r < 18; ++r) {
        uh(r, 0) = std::sin(0.7 * r);
    }
    Kokkos::deep_copy(u, uh);
    op.apply(Kokkos::DefaultExecutionSpace(), u, residual);
    Kokkos::deep_copy(rh, residual);
    double energy = 0;
    for (int r = 0; r < 18; ++r) {
        double expected = 0;
        for (int c = 0; c < 18; ++c) {
            EXPECT_NEAR(matrix[r][c], matrix[c][r], 1e-12);
            expected += matrix[r][c] * uh(c, 0);
        }
        EXPECT_NEAR(rh(r, 0), expected, 1e-11);
        energy += 0.5 * uh(r, 0) * rh(r, 0);
    }
    EXPECT_GT(energy, 0);
    // Independent quadrature of W = mu eps:eps + lambda/2 tr(eps)^2.
    double integrated_energy = 0;
    for (int j = 0; j < 2; ++j)
        for (int i = 0; i < 2; ++i) {
            std::array<int, 4> const
                    nodes {i + 3 * j, i + 1 + 3 * j, i + 3 * (j + 1), i + 1 + 3 * (j + 1)};
            std::array<std::array<double, 2>, 4> positions;
            for (int a = 0; a < 4; ++a)
                positions[a] = {xh(nodes[a]), yh(nodes[a])};
            for (int q = 0; q < 4; ++q) {
                sil::exterior::BilinearQuadrilateralGradient2D const
                        d(positions,
                          0.5 + (q % 2 ? 1 : -1) / std::sqrt(12.0),
                          0.5 + (q / 2 ? 1 : -1) / std::sqrt(12.0));
                double xx = 0, yy = 0, xy = 0;
                for (int a = 0; a < 4; ++a) {
                    xx += uh(2 * nodes[a], 0) * d.gradient[a][0];
                    yy += uh(2 * nodes[a] + 1, 0) * d.gradient[a][1];
                    xy += 0.5
                          * (uh(2 * nodes[a], 0) * d.gradient[a][1]
                             + uh(2 * nodes[a] + 1, 0) * d.gradient[a][0]);
                }
                integrated_energy
                        += 0.25 * d.measure
                           * (200 / (2 * 1.3) * (xx * xx + yy * yy + 2 * xy * xy)
                              + 0.5 * 200 * 0.3 / (1 - 0.3 * 0.3) * (xx + yy) * (xx + yy));
            }
        }
    EXPECT_NEAR(energy, integrated_energy, 1e-10);
}

TEST(MagneticEnergy, DerivativeMatchesNonlinearConstitutiveLaw)
{
    similie::physics::magnetostatics::InterpolatedNonlinearBHCurve<4> const
            curve(std::array<double, 4> {0.0, 1.0, 2.0, 3.0},
                  std::array<double, 4> {0.0, 2.0, 7.0, 20.0});
    EXPECT_DOUBLE_EQ(curve.magnetic_energy_from_q(0.0), 0.0);
    // Include the extrapolated interval as well as every tabulated segment.
    for (double b : {0.25, 1.25, 2.5, 3.5}) {
        double const step = 1e-5;
        double const derivative = (curve.magnetic_energy_from_q((b + step) * (b + step))
                                   - curve.magnetic_energy_from_q((b - step) * (b - step)))
                                  / (2 * step);
        EXPECT_NEAR(derivative, curve.h_from_b(b), 1e-8);
    }
}
