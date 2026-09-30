// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>

#include <similie/exterior/covariant_derivative.hpp>

#include "elasticity_quantities.hpp"

namespace similie::physics::elasticity {

/** Cell-local map from four vector edge differences to integrated dual forces.
 * Edges are 0->1, 2->3, 0->2, 1->3 in bit-mask vertex order. The matching
 * dual segments run from each edge midpoint to the cell centre.
 */
class ElasticMaterialHodge2D
{
public:
    using Vector = std::array<double, 8>;
    using Matrix = std::array<std::array<double, 8>, 8>;
    using Positions = std::array<std::array<double, 2>, 4>;

private:
    std::array<std::array<double, 8>, 4> m_affine_recovery {};
    Matrix m_matrix {};

public:
    template <class StressLaw>
    ElasticMaterialHodge2D(Positions const& positions, StressLaw stress_law)
    {
        double const signed_area
                = 0.5
                  * (positions[0][0] * positions[1][1] - positions[0][1] * positions[1][0]
                     + positions[1][0] * positions[3][1] - positions[1][1] * positions[3][0]
                     + positions[3][0] * positions[2][1] - positions[3][1] * positions[2][0]
                     + positions[2][0] * positions[0][1] - positions[2][1] * positions[0][0]);
        if (!(signed_area > 0.0))
            throw std::runtime_error("elasticity cell has nonpositive area");
        for (int corner = 0; corner < 4; ++corner) {
            int const next_x = corner ^ 1;
            int const next_y = corner ^ 2;
            double const dx_x
                    = (corner & 1 ? -1.0 : 1.0) * (positions[next_x][0] - positions[corner][0]);
            double const dx_y
                    = (corner & 1 ? -1.0 : 1.0) * (positions[next_x][1] - positions[corner][1]);
            double const dy_x
                    = (corner & 2 ? -1.0 : 1.0) * (positions[next_y][0] - positions[corner][0]);
            double const dy_y
                    = (corner & 2 ? -1.0 : 1.0) * (positions[next_y][1] - positions[corner][1]);
            if (!(dx_x * dy_y - dx_y * dy_x > 0.0))
                throw std::runtime_error("elasticity cell has folded geometry");
        }

        constexpr std::array<std::array<int, 2>, 4> endpoints {{{0, 1}, {2, 3}, {0, 2}, {1, 3}}};
        std::array<std::array<double, 4>, 8> affine {};
        std::array<std::array<double, 4>, 8> physical_force {};
        std::array<double, 2> centre {};
        for (int vertex = 0; vertex < 4; ++vertex)
            for (int axis = 0; axis < 2; ++axis)
                centre[axis] += 0.25 * positions[vertex][axis];
        for (int edge = 0; edge < 4; ++edge) {
            int const tail = endpoints[edge][0];
            int const head = endpoints[edge][1];
            double const dx = positions[head][0] - positions[tail][0];
            double const dy = positions[head][1] - positions[tail][1];
            affine[2 * edge] = {dx, dy, 0.0, 0.0};
            affine[2 * edge + 1] = {0.0, 0.0, dx, dy};
            double const mx = 0.5 * (positions[tail][0] + positions[head][0]);
            double const my = 0.5 * (positions[tail][1] + positions[head][1]);
            double const vx = (edge == 0 || edge == 2 ? 1.0 : -1.0) * (centre[0] - mx);
            double const vy = (edge == 0 || edge == 2 ? 1.0 : -1.0) * (centre[1] - my);
            double const nx = edge < 2 ? vy : -vy;
            double const ny = edge < 2 ? -vx : vx;
            for (int gradient = 0; gradient < 4; ++gradient) {
                Strain2D const strain = DisplacementToStrain::from_gradient(
                        gradient == 0 ? 1.0 : 0.0,
                        gradient == 3 ? 1.0 : 0.0,
                        gradient == 1 ? 1.0 : 0.0,
                        gradient == 2 ? 1.0 : 0.0);
                CauchyStress2D const stress = stress_law(strain);
                physical_force[2 * edge][gradient] = stress.xx * nx + stress.xy * ny;
                physical_force[2 * edge + 1][gradient] = stress.xy * nx + stress.yy * ny;
            }
        }

        std::array<std::array<double, 8>, 4> augmented {};
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j)
                for (int edge_component = 0; edge_component < 8; ++edge_component)
                    augmented[i][j] += affine[edge_component][i] * affine[edge_component][j];
            augmented[i][4 + i] = 1.0;
        }
        for (int pivot = 0; pivot < 4; ++pivot) {
            int best = pivot;
            for (int row = pivot + 1; row < 4; ++row)
                if (std::abs(augmented[row][pivot]) > std::abs(augmented[best][pivot]))
                    best = row;
            if (!(std::abs(augmented[best][pivot]) > 1e-25 * signed_area))
                throw std::runtime_error("degenerate elasticity cell");
            std::swap(augmented[best], augmented[pivot]);
            double const divisor = augmented[pivot][pivot];
            for (int column = 0; column < 8; ++column)
                augmented[pivot][column] /= divisor;
            for (int row = 0; row < 4; ++row) {
                if (row == pivot)
                    continue;
                double const factor = augmented[row][pivot];
                for (int column = 0; column < 8; ++column)
                    augmented[row][column] -= factor * augmented[pivot][column];
            }
        }
        for (int i = 0; i < 4; ++i)
            for (int edge_component = 0; edge_component < 8; ++edge_component)
                for (int j = 0; j < 4; ++j)
                    m_affine_recovery[i][edge_component]
                            += augmented[i][4 + j] * affine[edge_component][j];

        std::array<std::array<double, 4>, 4> affine_energy {};
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                for (int component = 0; component < 8; ++component)
                    affine_energy[i][j] += affine[component][i] * physical_force[component][j];
        // Symmetric extension of H E = B. It supplies exact affine tractions,
        // while the complementary projector controls unresolved edge modes.
        double scale = 0.0;
        for (int i = 0; i < 8; ++i)
            for (int j = 0; j < 8; ++j) {
                for (int k = 0; k < 4; ++k) {
                    m_matrix[i][j] += physical_force[i][k] * m_affine_recovery[k][j]
                                      + m_affine_recovery[k][i] * physical_force[j][k];
                    for (int l = 0; l < 4; ++l)
                        m_matrix[i][j] -= m_affine_recovery[k][i] * affine_energy[k][l]
                                          * m_affine_recovery[l][j];
                }
                scale = std::max(scale, std::abs(m_matrix[i][j]));
            }
        Matrix complement {};
        for (int i = 0; i < 8; ++i)
            for (int j = 0; j < 8; ++j) {
                complement[i][j] = i == j ? 1.0 : 0.0;
                for (int k = 0; k < 4; ++k)
                    complement[i][j] -= affine[i][k] * m_affine_recovery[k][j];
            }
        double coupling_norm_squared = 0.0;
        for (int i = 0; i < 8; ++i)
            for (int gradient = 0; gradient < 4; ++gradient) {
                double coupling = 0.0;
                for (int j = 0; j < 8; ++j)
                    coupling += complement[i][j] * physical_force[j][gradient];
                coupling_norm_squared += coupling * coupling;
            }
        double const shear_modulus = 0.5 * stress_law(Strain2D {.xy = 1.0}).xy;
        if (!(shear_modulus > 0.0))
            throw std::runtime_error("elasticity material has nonpositive shear modulus");
        // The second bound dominates the affine/complement coupling in the
        // Schur complement; the first controls the otherwise free modes.
        double const stabilization = std::
                max(0.25 * scale,
                    1.01 * coupling_norm_squared / (2.0 * shear_modulus * signed_area));
        for (int i = 0; i < 8; ++i)
            for (int j = 0; j < 8; ++j)
                m_matrix[i][j] += stabilization * complement[i][j];
    }

    [[nodiscard]] Matrix const& matrix() const
    {
        return m_matrix;
    }

    [[nodiscard]] std::array<double, 4> recover_gradient(Vector const& differences) const
    {
        std::array<double, 4> gradient {};
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 8; ++j)
                gradient[i] += m_affine_recovery[i][j] * differences[j];
        return gradient;
    }

    template <class SpatialX, class SpatialY>
    [[nodiscard]] static Vector edge_differences(std::array<std::array<double, 2>, 4> const& nodal)
    {
        Vector differences {};
        constexpr std::array<std::array<std::size_t, 2>, 4> edges {
                {{0, 0}, {0, 2}, {1, 0}, {1, 1}}};
        for (int edge = 0; edge < 4; ++edge) {
            std::array<double, 2> const difference = sil::exterior::
                    CovariantDerivative<SpatialX, SpatialY>::template cochain_value<0, 2>(
                            {edges[edge][0]},
                            edges[edge][1],
                            [&](std::size_t, std::size_t vertex) { return nodal[vertex]; });
            differences[2 * edge] = difference[0];
            differences[2 * edge + 1] = difference[1];
        }
        return differences;
    }
};

} // namespace similie::physics::elasticity
