// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cstddef>
#include <type_traits>

#include <ginkgo/core/base/matrix_data.hpp>
#include <ginkgo/core/base/types.hpp>
#include <similie/physics/hamilton_equations.hpp>
#include <similie/physics/magnetostatics/magnetostatics_indices.hpp>
#include <similie/physics/stationary_equations_operator.hpp>

#include <Kokkos_Core.hpp>

namespace similie::physics::magnetostatics {

namespace detail {

struct LocalMagneticFieldTensor
{
    std::array<double, 3> values {0.0, 0.0, 0.0};

    template <class Dim>
    KOKKOS_INLINE_FUNCTION static constexpr std::size_t access_element()
    {
        if constexpr (std::is_same_v<Dim, X>) {
            return 0;
        } else if constexpr (std::is_same_v<Dim, Y>) {
            return 1;
        } else {
            return 2;
        }
    }

    KOKKOS_INLINE_FUNCTION double& operator()(std::size_t idx)
    {
        return values[idx];
    }

    KOKKOS_INLINE_FUNCTION double operator()(std::size_t idx) const
    {
        return values[idx];
    }
};

} // namespace detail

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
    StructuredScalarPoissonStrongFormOperator2D(coord_view_type x_coords, coord_view_type y_coords)
        : m_x_coords(x_coords)
        , m_y_coords(y_coords)
        , m_nx(x_coords.extent(0))
        , m_ny(y_coords.extent(0))
    {
    }

    StructuredScalarPoissonStrongFormOperator2D() : m_x_coords(), m_y_coords(), m_nx(0), m_ny(0) {}

    [[nodiscard]] KOKKOS_INLINE_FUNCTION std::size_t size() const
    {
        return m_nx * m_ny;
    }

    [[nodiscard]] KOKKOS_INLINE_FUNCTION std::size_t nx() const
    {
        return m_nx;
    }

    [[nodiscard]] KOKKOS_INLINE_FUNCTION std::size_t ny() const
    {
        return m_ny;
    }

    [[nodiscard]] KOKKOS_INLINE_FUNCTION coord_view_type x_coords() const
    {
        return m_x_coords;
    }

    [[nodiscard]] KOKKOS_INLINE_FUNCTION coord_view_type y_coords() const
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
                = 2.0
                  * ((value(i + 1, j) - value(i, j)) / dxp - (value(i, j) - value(i - 1, j)) / dxm)
                  / (dxm + dxp);
        double const second_y
                = 2.0
                  * ((value(i, j + 1) - value(i, j)) / dyp - (value(i, j) - value(i, j - 1)) / dym)
                  / (dym + dyp);

        output(row, 0) = -(second_x + second_y);
    }

private:
    KOKKOS_INLINE_FUNCTION std::size_t flat_index(std::size_t i, std::size_t j) const
    {
        return i + m_nx * j;
    }
};

template <class MemorySpace>
gko::matrix_data<double, gko::int32> assemble_matrix_data(
        StructuredScalarPoissonStrongFormOperator2D<MemorySpace> const& operator_model)
{
    auto const x_coords_host
            = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), operator_model.x_coords());
    auto const y_coords_host
            = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), operator_model.y_coords());

    std::size_t const nx = operator_model.nx();
    std::size_t const ny = operator_model.ny();

    gko::matrix_data<double, gko::int32> matrix_data(
            gko::dim<2>(operator_model.size(), operator_model.size()));
    matrix_data.nonzeros.reserve(5 * operator_model.size());

    auto flat_index
            = [nx](std::size_t i, std::size_t j) { return static_cast<gko::int32>(i + nx * j); };

    for (std::size_t j = 0; j < ny; ++j) {
        for (std::size_t i = 0; i < nx; ++i) {
            gko::int32 const row = flat_index(i, j);
            bool const boundary = (i == 0 || j == 0 || i + 1 == nx || j + 1 == ny);
            if (boundary) {
                matrix_data.nonzeros.emplace_back(row, row, 1.0);
                continue;
            }

            double const dxm = x_coords_host(i) - x_coords_host(i - 1);
            double const dxp = x_coords_host(i + 1) - x_coords_host(i);
            double const dym = y_coords_host(j) - y_coords_host(j - 1);
            double const dyp = y_coords_host(j + 1) - y_coords_host(j);

            double const coeff_im1 = -2.0 / (dxm * (dxm + dxp));
            double const coeff_ip1 = -2.0 / (dxp * (dxm + dxp));
            double const coeff_jm1 = -2.0 / (dym * (dym + dyp));
            double const coeff_jp1 = -2.0 / (dyp * (dym + dyp));
            double const coeff_center = -coeff_im1 - coeff_ip1 - coeff_jm1 - coeff_jp1;

            matrix_data.nonzeros.emplace_back(row, flat_index(i - 1, j), coeff_im1);
            matrix_data.nonzeros.emplace_back(row, flat_index(i + 1, j), coeff_ip1);
            matrix_data.nonzeros.emplace_back(row, flat_index(i, j - 1), coeff_jm1);
            matrix_data.nonzeros.emplace_back(row, flat_index(i, j + 1), coeff_jp1);
            matrix_data.nonzeros.emplace_back(row, row, coeff_center);
        }
    }

    matrix_data.sort_row_major();
    return matrix_data;
}

template <class Hamiltonian, class MemorySpace, class InputView, class OutputView>
KOKKOS_INLINE_FUNCTION void apply_stationary_equations_at(
        OutputView output,
        InputView input,
        std::size_t row,
        physics::HamiltonEquations<Hamiltonian> const& equations,
        StructuredScalarPoissonStrongFormOperator2D<MemorySpace> const& operator_model)
{
    // This matrix-free action is equations-driven: from A_z we reconstruct B,
    // evaluate H=dH/dB through the Hamiltonian, then discretize curl(H).
    std::size_t const nx = operator_model.nx();
    std::size_t const ny = operator_model.ny();
    std::size_t const j = row / nx;
    std::size_t const i = row % nx;

    bool const boundary = (i == 0 || j == 0 || i + 1 == nx || j + 1 == ny);
    if (boundary) {
        output(row, 0) = input(row, 0);
        return;
    }

    auto flat_index = [nx](std::size_t ii, std::size_t jj) { return ii + nx * jj; };
    auto value = [&](std::size_t ii, std::size_t jj) { return input(flat_index(ii, jj), 0); };

    auto const x_coords = operator_model.x_coords();
    auto const y_coords = operator_model.y_coords();
    double const dxm = x_coords(i) - x_coords(i - 1);
    double const dxp = x_coords(i + 1) - x_coords(i);
    double const dym = y_coords(j) - y_coords(j - 1);
    double const dyp = y_coords(j + 1) - y_coords(j);

    auto magnetic_field_from_induction
            = [&](double bx, double by, double bz, std::size_t component) {
                  if (component == detail::LocalMagneticFieldTensor::template access_element<X>()) {
                      return equations.template dpotential_dt<0>(bx);
                  }
                  if (component == detail::LocalMagneticFieldTensor::template access_element<Y>()) {
                      return equations.template dpotential_dt<1>(by);
                  }
                  return equations.template dpotential_dt<2>(bz);
              };

    double const bx_s = (value(i, j) - value(i, j - 1)) / dym;
    double const bx_n = (value(i, j + 1) - value(i, j)) / dyp;
    double const by_w = -(value(i, j) - value(i - 1, j)) / dxm;
    double const by_e = -(value(i + 1, j) - value(i, j)) / dxp;

    double const hx_s = magnetic_field_from_induction(
            bx_s,
            0.0,
            0.0,
            detail::LocalMagneticFieldTensor::template access_element<X>());
    double const hx_n = magnetic_field_from_induction(
            bx_n,
            0.0,
            0.0,
            detail::LocalMagneticFieldTensor::template access_element<X>());
    double const hy_w = magnetic_field_from_induction(
            0.0,
            by_w,
            0.0,
            detail::LocalMagneticFieldTensor::template access_element<Y>());
    double const hy_e = magnetic_field_from_induction(
            0.0,
            by_e,
            0.0,
            detail::LocalMagneticFieldTensor::template access_element<Y>());

    double const curl_h_z = 2.0 * (hy_e - hy_w) / (dxm + dxp) - 2.0 * (hx_n - hx_s) / (dym + dyp);
    output(row, 0) = curl_h_z;
}

} // namespace similie::physics::magnetostatics
