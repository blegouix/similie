// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>

#include <Kokkos_Core.hpp>

namespace similie::physics::magnetostatics {

template <class MemorySpace>
class StructuredVectorPoissonStrongFormOperator
{
public:
    using memory_space = MemorySpace;
    using coord_view_type = Kokkos::View<double const*, memory_space>;

private:
    coord_view_type m_x_coords;
    coord_view_type m_y_coords;
    coord_view_type m_z_coords;
    std::size_t m_nx;
    std::size_t m_ny;
    std::size_t m_nz;

public:
    StructuredVectorPoissonStrongFormOperator(
            coord_view_type x_coords,
            coord_view_type y_coords,
            coord_view_type z_coords)
        : m_x_coords(x_coords)
        , m_y_coords(y_coords)
        , m_z_coords(z_coords)
        , m_nx(x_coords.extent(0))
        , m_ny(y_coords.extent(0))
        , m_nz(z_coords.extent(0))
    {
    }

    StructuredVectorPoissonStrongFormOperator()
        : m_x_coords()
        , m_y_coords()
        , m_z_coords()
        , m_nx(0)
        , m_ny(0)
        , m_nz(0)
    {
    }

    [[nodiscard]] std::size_t size() const
    {
        return m_nx * m_ny * m_nz;
    }

    template <class InputView, class OutputView>
    KOKKOS_INLINE_FUNCTION void apply_at(OutputView output, InputView input, std::size_t row) const
    {
        std::size_t const plane = m_nx * m_ny;
        std::size_t const k = row / plane;
        std::size_t const in_plane = row % plane;
        std::size_t const j = in_plane / m_nx;
        std::size_t const i = in_plane % m_nx;

        bool const boundary = (i == 0 || j == 0 || k == 0 || i + 1 == m_nx || j + 1 == m_ny
                               || k + 1 == m_nz);

        for (std::size_t component = 0; component < 3; ++component) {
            if (boundary) {
                output(row, component) = input(row, component);
                continue;
            }

            double const dxm = m_x_coords(i) - m_x_coords(i - 1);
            double const dxp = m_x_coords(i + 1) - m_x_coords(i);
            double const dym = m_y_coords(j) - m_y_coords(j - 1);
            double const dyp = m_y_coords(j + 1) - m_y_coords(j);
            double const dzm = m_z_coords(k) - m_z_coords(k - 1);
            double const dzp = m_z_coords(k + 1) - m_z_coords(k);

            auto value = [&](std::size_t ii, std::size_t jj, std::size_t kk) {
                return input(flat_index(ii, jj, kk), component);
            };

            double const second_x
                    = 2.0 * ((value(i + 1, j, k) - value(i, j, k)) / dxp
                             - (value(i, j, k) - value(i - 1, j, k)) / dxm)
                      / (dxm + dxp);
            double const second_y
                    = 2.0 * ((value(i, j + 1, k) - value(i, j, k)) / dyp
                             - (value(i, j, k) - value(i, j - 1, k)) / dym)
                      / (dym + dyp);
            double const second_z
                    = 2.0 * ((value(i, j, k + 1) - value(i, j, k)) / dzp
                             - (value(i, j, k) - value(i, j, k - 1)) / dzm)
                      / (dzm + dzp);

            output(row, component) = -(second_x + second_y + second_z);
        }
    }

private:
    KOKKOS_INLINE_FUNCTION std::size_t flat_index(std::size_t i, std::size_t j, std::size_t k) const
    {
        return i + m_nx * (j + m_ny * k);
    }
};

} // namespace similie::physics::magnetostatics
