// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <map>
#include <stdexcept>
#include <vector>

#include <ginkgo/core/base/matrix_data.hpp>

#include <Kokkos_Core.hpp>

namespace similie::physics::scalar_field {

struct TriangleStiffness
{
    double area;
    std::array<std::array<double, 2>, 3> gradients;
    std::array<std::array<double, 3>, 3> matrix;
};

/** Piecewise-linear triangle stiffness for div(coefficient grad(phi)) = 0.
 *
 * On each triangle phi = sum_a phi_a N_a. The weak-form contribution is
 * K_ab = coefficient * area * grad(N_a) dot grad(N_b). Natural zero-flux
 * boundaries require no extra term; prescribed values are eliminated by
 * the caller before passing rows to TriangularLaplaceOperator.
 */
inline TriangleStiffness triangular_laplace_stiffness(
        std::array<std::array<double, 2>, 3> const& points,
        double coefficient)
{
    double const determinant = (points[1][0] - points[0][0]) * (points[2][1] - points[0][1])
                               - (points[2][0] - points[0][0]) * (points[1][1] - points[0][1]);
    if (std::abs(determinant) < 1.0e-18 || !(coefficient > 0.0)) {
        throw std::runtime_error("degenerate potential-flow triangle or invalid density");
    }
    TriangleStiffness element {};
    element.area = 0.5 * std::abs(determinant);
    for (int a = 0; a < 3; ++a) {
        int const b = (a + 1) % 3;
        int const c = (a + 2) % 3;
        element.gradients[a]
                = {(points[b][1] - points[c][1]) / determinant,
                   (points[c][0] - points[b][0]) / determinant};
    }
    for (int a = 0; a < 3; ++a) {
        for (int b = 0; b < 3; ++b) {
            element.matrix[a][b] = coefficient * element.area
                                   * (element.gradients[a][0] * element.gradients[b][0]
                                      + element.gradients[a][1] * element.gradients[b][1]);
        }
    }
    return element;
}

class TriangularLaplaceOperator
{
    Kokkos::View<int*> m_offsets;
    Kokkos::View<int*> m_columns;
    Kokkos::View<double*> m_values;
    std::size_t m_size;

public:
    static constexpr bool IS_LINEAR = true;
    static constexpr bool IS_SYMMETRIC = true;

    explicit TriangularLaplaceOperator(std::vector<std::map<std::size_t, double>> const& rows)
        : m_offsets("similie_potential_flow_offsets", rows.size() + 1)
        , m_columns(
                  "similie_potential_flow_columns",
                  [&]() {
                      std::size_t count = 0;
                      for (std::map<std::size_t, double> const& row : rows)
                          count += row.size();
                      return count;
                  }())
        , m_values("similie_potential_flow_values", m_columns.extent(0))
        , m_size(rows.size())
    {
        auto offsets = Kokkos::create_mirror_view(m_offsets);
        auto columns = Kokkos::create_mirror_view(m_columns);
        auto values = Kokkos::create_mirror_view(m_values);
        std::size_t slot = 0;
        for (std::size_t row = 0; row < rows.size(); ++row) {
            offsets(row) = static_cast<int>(slot);
            for (auto const& [column, value] : rows[row]) {
                columns(slot) = static_cast<int>(column);
                values(slot) = value;
                ++slot;
            }
        }
        offsets(rows.size()) = static_cast<int>(slot);
        Kokkos::deep_copy(m_offsets, offsets);
        Kokkos::deep_copy(m_columns, columns);
        Kokkos::deep_copy(m_values, values);
    }

    [[nodiscard]] std::size_t size() const
    {
        return m_size;
    }

    template <class ExecSpace, class InputView, class OutputView>
    void apply(ExecSpace exec_space, InputView input, OutputView output) const
    {
        auto const offsets = m_offsets;
        auto const columns = m_columns;
        auto const values = m_values;
        Kokkos::parallel_for(
                "similie_triangular_laplace",
                Kokkos::RangePolicy<ExecSpace>(exec_space, 0, m_size),
                KOKKOS_LAMBDA(std::size_t row) {
                    double sum = 0.0;
                    for (int slot = offsets(row); slot < offsets(row + 1); ++slot) {
                        sum += values(slot) * input(columns(slot), 0);
                    }
                    output(row, 0) = sum;
                });
        exec_space.fence();
    }

    friend gko::matrix_data<double, gko::int32> assemble_matrix_data(
            TriangularLaplaceOperator const& operator_model)
    {
        gko::matrix_data<double, gko::int32> data(
                gko::dim<2>(operator_model.m_size, operator_model.m_size));
        auto const offsets = Kokkos::
                create_mirror_view_and_copy(Kokkos::HostSpace(), operator_model.m_offsets);
        auto const columns = Kokkos::
                create_mirror_view_and_copy(Kokkos::HostSpace(), operator_model.m_columns);
        auto const values
                = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), operator_model.m_values);
        for (std::size_t row = 0; row < operator_model.m_size; ++row) {
            for (int slot = offsets(row); slot < offsets(row + 1); ++slot) {
                data.nonzeros
                        .emplace_back(static_cast<gko::int32>(row), columns(slot), values(slot));
            }
        }
        return data;
    }
};

} // namespace similie::physics::scalar_field
