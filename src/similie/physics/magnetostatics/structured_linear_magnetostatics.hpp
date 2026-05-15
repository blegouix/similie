// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>

#include <Kokkos_Core.hpp>

namespace similie::physics::magnetostatics {

template <class MemorySpace>
class StructuredScalarPoissonStrongFormOperator2D
{
public:
    using memory_space = MemorySpace;
    using coord_view_type = Kokkos::View<double const*, memory_space>;

private:
    coord_view_type m_x_coords;
    coord_view_type m_y_coords;
    std::size_t m_nx;
    std::size_t m_ny;

public:
    StructuredScalarPoissonStrongFormOperator2D(
            coord_view_type x_coords,
            coord_view_type y_coords)
        : m_x_coords(x_coords)
        , m_y_coords(y_coords)
        , m_nx(x_coords.extent(0))
        , m_ny(y_coords.extent(0))
    {
    }

    StructuredScalarPoissonStrongFormOperator2D()
        : m_x_coords()
        , m_y_coords()
        , m_nx(0)
        , m_ny(0)
    {
    }

    [[nodiscard]] std::size_t size() const
    {
        return m_nx * m_ny;
    }

    [[nodiscard]] std::size_t nx() const
    {
        return m_nx;
    }

    [[nodiscard]] std::size_t ny() const
    {
        return m_ny;
    }

    [[nodiscard]] coord_view_type x_coords() const
    {
        return m_x_coords;
    }

    [[nodiscard]] coord_view_type y_coords() const
    {
        return m_y_coords;
    }

    template <class InputView, class OutputView>
    KOKKOS_INLINE_FUNCTION void apply_at(OutputView output, InputView input, std::size_t row) const
    {
        std::size_t const j = row / m_nx;
        std::size_t const i = row % m_nx;

        bool const boundary = (i == 0 || j == 0 || i + 1 == m_nx || j + 1 == m_ny);

        if (boundary) {
            output(row, 0) = input(row, 0);
            return;
        }

        double const dxm = m_x_coords(i) - m_x_coords(i - 1);
        double const dxp = m_x_coords(i + 1) - m_x_coords(i);
        double const dym = m_y_coords(j) - m_y_coords(j - 1);
        double const dyp = m_y_coords(j + 1) - m_y_coords(j);

        auto value = [&](std::size_t ii, std::size_t jj) { return input(flat_index(ii, jj), 0); };

        double const second_x
                = 2.0 * ((value(i + 1, j) - value(i, j)) / dxp
                         - (value(i, j) - value(i - 1, j)) / dxm)
                  / (dxm + dxp);
        double const second_y
                = 2.0 * ((value(i, j + 1) - value(i, j)) / dyp
                         - (value(i, j) - value(i, j - 1)) / dym)
                  / (dym + dyp);

        output(row, 0) = -(second_x + second_y);
    }

private:
    KOKKOS_INLINE_FUNCTION std::size_t flat_index(std::size_t i, std::size_t j) const
    {
        return i + m_nx * j;
    }
};

} // namespace similie::physics::magnetostatics
