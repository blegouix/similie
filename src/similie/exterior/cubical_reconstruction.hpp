// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#pragma once

#include <array>
#include <cmath>
#include <stdexcept>

#include <similie/misc/small_matrix.hpp>

namespace sil::exterior {

/** Tensor-product Whitney reconstruction on a mapped n-cube.
 * Vertex ids are bit masks: bit i is reference coordinate i. A k-cell is
 * identified by its increasing coordinate directions and its lower vertex.
 * Geometry is the reconstruction of the coordinate 0-cochains. Form components
 * transform by minors of its inverse Jacobian; no metric or physics is assumed.
 * The caller must supply a regular, injective cell map. valid() checks the
 * Jacobian at the evaluation point, not global injectivity in higher dimensions.
 */
template <std::size_t Dimension>
class CubicalReconstruction
{
    static_assert(Dimension > 0);
    std::array<double, Dimension> m_point;
    std::array<double, Dimension * Dimension> m_inverse {};
    double m_determinant = 0;

public:
    static constexpr std::size_t vertex_count = std::size_t(1) << Dimension;

    KOKKOS_FUNCTION CubicalReconstruction(
            std::array<std::array<double, Dimension>, vertex_count> const& positions,
            std::array<double, Dimension> const& point)
        : m_point(point)
    {
        std::array<double, Dimension * Dimension> jacobian {};
        for (std::size_t d = 0; d < Dimension; ++d) {
            for (std::size_t v = 0; v < vertex_count; ++v) {
                if (v & (std::size_t(1) << d))
                    continue;
                double const weight = reference_basis(std::size_t(1) << d, v);
                for (std::size_t i = 0; i < Dimension; ++i) {
                    jacobian[i * Dimension + d]
                            += weight * (positions[v | (std::size_t(1) << d)][i] - positions[v][i]);
                }
            }
        }
        auto determinant_work = jacobian;
        m_determinant = misc::math::determinant(
                misc::math::matrix_view<
                        double,
                        Kokkos::AnonymousSpace>(determinant_work.data(), Dimension, Dimension));
        if (valid()) {
            std::array<double, Dimension * Dimension> workspace {};
            misc::math::
                    invert(misc::math::matrix_view<
                                   double,
                                   Kokkos::AnonymousSpace>(m_inverse.data(), Dimension, Dimension),
                           misc::math::matrix_view<
                                   double,
                                   Kokkos::AnonymousSpace>(jacobian.data(), Dimension, Dimension),
                           misc::math::vector_view<
                                   double,
                                   Kokkos::AnonymousSpace>(workspace.data(), workspace.size()));
        }
    }

    [[nodiscard]] KOKKOS_FUNCTION bool valid() const
    {
        return m_determinant != 0 && Kokkos::isfinite(m_determinant);
    }

    [[nodiscard]] KOKKOS_FUNCTION double signed_measure() const
    {
        return m_determinant;
    }
    [[nodiscard]] KOKKOS_FUNCTION double measure() const
    {
        return Kokkos::abs(m_determinant);
    }

    /** Reference Whitney basis: constant in tangent directions, linear in
     * transverse directions. Its integral on the associated oriented cell is 1.
     */
    [[nodiscard]] KOKKOS_FUNCTION double reference_basis(std::size_t directions, std::size_t vertex)
            const
    {
        double result = 1;
        for (std::size_t i = 0; i < Dimension; ++i) {
            if (!(directions & (std::size_t(1) << i))) {
                result *= vertex & (std::size_t(1) << i) ? m_point[i] : 1 - m_point[i];
            }
        }
        return result;
    }

    template <std::size_t Degree>
    [[nodiscard]] KOKKOS_FUNCTION double basis(
            std::array<std::size_t, Degree> const& directions,
            std::size_t vertex,
            std::array<std::size_t, Degree> const& physical_directions) const
    {
        static_assert(Degree <= Dimension);
        assert(valid());
        std::size_t mask = 0;
        std::array<double, Degree * Degree> minor {};
        for (std::size_t i = 0; i < Degree; ++i) {
            assert(directions[i] < Dimension && physical_directions[i] < Dimension);
            mask |= std::size_t(1) << directions[i];
            for (std::size_t j = 0; j < Degree; ++j) {
                minor[i * Degree + j]
                        = m_inverse[directions[i] * Dimension + physical_directions[j]];
            }
        }
        assert((vertex & mask) == 0);
        return reference_basis(mask, vertex)
               * misc::math::determinant(
                       misc::math::matrix_view<
                               double,
                               Kokkos::AnonymousSpace>(minor.data(), Degree, Degree));
    }

    /** Host-side mesh check, independent of the field/operator. In 2D the
     * determinant is affine, so the corner test certifies its sign throughout
     * the cell. In dimensions >2 this is only a necessary validity check.
     */
    static void check_orientation(
            std::array<std::array<double, Dimension>, vertex_count> const& positions)
    {
        double sign = 0;
        for (std::size_t v = 0; v < vertex_count; ++v) {
            std::array<double, Dimension> point {};
            for (std::size_t d = 0; d < Dimension; ++d)
                point[d] = (v >> d) & 1;
            CubicalReconstruction const map(positions, point);
            if (!map.valid() || (sign != 0 && (map.signed_measure() > 0) != (sign > 0))) {
                throw std::runtime_error("singular or inconsistently oriented cell map");
            }
            sign = map.signed_measure();
        }
    }
};

} // namespace sil::exterior
