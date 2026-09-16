// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// AI-GENERATED
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

#include <array>
#include <cmath>
#include <stdexcept>

namespace sil::exterior {

/** Pointwise gradient reconstruction on a bilinear quadrilateral. This is the
 * geometry/interpolation policy used after d^nabla in a fixed Cartesian frame;
 * it is not a separate differential operator. Nodes are ordered
 * (0,0), (1,0), (0,1), (1,1).
 * The reference derivative interpolates the oriented edge coboundaries:
 * du/dxi = (1-eta)(u10-u00) + eta(u11-u01), and similarly for eta.
 * Multiplication by the inverse geometry Jacobian returns the full physical
 * gradient. The measure belongs to the same geometry and quadrature point.
 */
struct BilinearQuadrilateralGradient2D
{
    std::array<std::array<double, 2>, 4> gradient;
    double measure;

    BilinearQuadrilateralGradient2D(
            std::array<std::array<double, 2>, 4> const& position,
            double xi,
            double eta)
    {
        std::array<std::array<double, 2>, 4> const reference {{
                {-(1 - eta), -(1 - xi)},
                {1 - eta, -xi},
                {-eta, 1 - xi},
                {eta, xi},
        }};
        std::array<std::array<double, 2>, 2> jacobian {};
        for (std::size_t a = 0; a < 4; ++a) {
            for (std::size_t i = 0; i < 2; ++i) {
                for (std::size_t j = 0; j < 2; ++j) {
                    jacobian[i][j] += position[a][i] * reference[a][j];
                }
            }
        }
        double const determinant
                = jacobian[0][0] * jacobian[1][1] - jacobian[0][1] * jacobian[1][0];
        measure = std::abs(determinant);
        if (!(measure > 0.0) || !std::isfinite(measure)) {
            throw std::runtime_error("singular bilinear quadrilateral geometry");
        }
        // det J is affine in (xi,eta), so checking the four corners rules
        // out a folded map everywhere in the reference cell. Either global
        // orientation is valid, but it must stay consistent across the cell.
        for (int corner = 0; corner < 4; ++corner) {
            int const i = corner % 2;
            int const j = corner / 2;
            double const dx_dxi = position[1 + 2 * j][0] - position[2 * j][0];
            double const dy_dxi = position[1 + 2 * j][1] - position[2 * j][1];
            double const dx_deta = position[2 + i][0] - position[i][0];
            double const dy_deta = position[2 + i][1] - position[i][1];
            if (!((dx_dxi * dy_deta - dx_deta * dy_dxi) * determinant > 0.0)) {
                throw std::runtime_error("folded or degenerate bilinear quadrilateral geometry");
            }
        }
        for (std::size_t a = 0; a < 4; ++a) {
            gradient[a][0] = (reference[a][0] * jacobian[1][1] - reference[a][1] * jacobian[1][0])
                             / determinant;
            gradient[a][1] = (-reference[a][0] * jacobian[0][1] + reference[a][1] * jacobian[0][0])
                             / determinant;
        }
    }
};

} // namespace sil::exterior
