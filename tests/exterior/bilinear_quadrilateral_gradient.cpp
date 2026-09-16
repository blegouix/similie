// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// AI-GENERATED
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <gtest/gtest.h>
#include <similie/exterior/bilinear_quadrilateral_gradient.hpp>

TEST(BilinearQuadrilateralGradient, AffineFieldsOnDistortedCell)
{
    std::array<std::array<double, 2>, 4> const positions {{{0, 0}, {2, 0.3}, {0.4, 1}, {2.8, 1.6}}};
    for (double xi : {0.0, 0.2113248654, 0.7886751346, 1.0}) {
        for (double eta : {0.0, 0.2113248654, 0.7886751346, 1.0}) {
            sil::exterior::BilinearQuadrilateralGradient2D const derivative(positions, xi, eta);
            std::array<double, 2> translation {};
            std::array<double, 2> linear {};
            for (int a = 0; a < 4; ++a) {
                for (int d = 0; d < 2; ++d) {
                    translation[d] += derivative.gradient[a][d];
                    linear[d] += derivative.gradient[a][d]
                                 * (3 * positions[a][0] - 2 * positions[a][1] + 7);
                }
            }
            EXPECT_NEAR(translation[0], 0, 1e-14);
            EXPECT_NEAR(translation[1], 0, 1e-14);
            EXPECT_NEAR(linear[0], 3, 1e-14);
            EXPECT_NEAR(linear[1], -2, 1e-14);
        }
    }
}

TEST(BilinearQuadrilateralGradient, IntegratesBilinearMode)
{
    std::array<std::array<double, 2>, 4> const positions {{{0, 0}, {1, 0}, {0, 1}, {1, 1}}};
    // u=xi*eta: integral |grad u|^2 = 2/3, not zero (corner) or 1/2 (center).
    double energy = 0;
    double area = 0;
    for (int q = 0; q < 4; ++q) {
        sil::exterior::BilinearQuadrilateralGradient2D const derivative(
                positions,
                0.5 + (q % 2 ? 1 : -1) / std::sqrt(12.0),
                0.5 + (q / 2 ? 1 : -1) / std::sqrt(12.0));
        area += 0.25 * derivative.measure;
        energy += 0.25 * derivative.measure
                  * (derivative.gradient[3][0] * derivative.gradient[3][0]
                     + derivative.gradient[3][1] * derivative.gradient[3][1]);
    }
    EXPECT_NEAR(area, 1, 1e-14);
    EXPECT_NEAR(energy, 2.0 / 3, 1e-14);
}

TEST(BilinearQuadrilateralGradient, RejectsFoldedCell)
{
    std::array<std::array<double, 2>, 4> const positions {{{0, 0}, {1, 0}, {0, 1}, {-0.1, 0.5}}};
    EXPECT_THROW(
            (sil::exterior::BilinearQuadrilateralGradient2D(positions, 0.2, 0.2)),
            std::runtime_error);
}
