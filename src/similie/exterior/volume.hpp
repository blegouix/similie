// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: MIT

#pragma once

#include <array>

#include <ddc/ddc.hpp>

#include <similie/misc/binomial_coefficient.hpp>
#include <similie/tensor/gram_matrix.hpp>

namespace sil {

namespace exterior {

enum class DualStrategy {
    Circumcentric,
    Barycentric,
};

namespace detail {

template <std::size_t N>
KOKKOS_FUNCTION std::size_t count_active(std::array<bool, N> const& active_dims)
{
    std::size_t count = 0;
    for (bool const active_dim : active_dims) {
        count += active_dim ? 1 : 0;
    }
    return count;
}

template <std::size_t N>
KOKKOS_FUNCTION double determinant_from_mask(
        std::array<double, N * N> const& matrix,
        std::array<bool, N> const& active_dims)
{
    std::array<double, N * N> reduced {};
    std::size_t const size = count_active(active_dims);

    if (size == 0) {
        return 1.;
    }

    std::size_t row_id = 0;
    for (std::size_t i = 0; i < N; ++i) {
        if (!active_dims[i]) {
            continue;
        }
        std::size_t col_id = 0;
        for (std::size_t j = 0; j < N; ++j) {
            if (!active_dims[j]) {
                continue;
            }
            reduced[row_id * N + col_id] = matrix[i * N + j];
            ++col_id;
        }
        ++row_id;
    }

    double det = 1.;
    int sign = 1;

    for (std::size_t i = 0; i < size; ++i) {
        std::size_t pivot = i;
        double max_value = Kokkos::abs(reduced[i * N + i]);
        for (std::size_t j = i + 1; j < size; ++j) {
            double const candidate = Kokkos::abs(reduced[j * N + i]);
            if (candidate > max_value) {
                pivot = j;
                max_value = candidate;
            }
        }

        if (max_value == 0.) {
            return 0.;
        }

        if (pivot != i) {
            sign *= -1;
            for (std::size_t j = 0; j < size; ++j) {
                Kokkos::kokkos_swap(reduced[i * N + j], reduced[pivot * N + j]);
            }
        }

        double const diagonal = reduced[i * N + i];
        det *= diagonal;
        for (std::size_t j = i + 1; j < size; ++j) {
            double const factor = reduced[j * N + i] / diagonal;
            for (std::size_t k = i + 1; k < size; ++k) {
                reduced[j * N + k] -= factor * reduced[i * N + k];
            }
        }
    }

    return sign * det;
}

template <std::size_t N>
KOKKOS_FUNCTION std::array<bool, N> complement(std::array<bool, N> const& active_dims)
{
    std::array<bool, N> complement_dims {};
    for (std::size_t i = 0; i < N; ++i) {
        complement_dims[i] = !active_dims[i];
    }
    return complement_dims;
}

template <DualStrategy Strategy, std::size_t N>
KOKKOS_FUNCTION double dual_volume_factor(std::array<bool, N> const& active_dims)
{
    if constexpr (Strategy == DualStrategy::Circumcentric) {
        return 1.;
    } else {
        std::size_t const k = count_active(active_dims);
        return 1. / static_cast<double>(misc::binomial_coefficient(N, k));
    }
}

} // namespace detail

template <std::size_t N, class MetricType, class PositionType, class BatchElem>
KOKKOS_FUNCTION double simplex_volume(
        MetricType metric,
        PositionType position,
        BatchElem elem,
        std::array<bool, N> const& active_dims)
{
    double const det = detail::determinant_from_mask(
            tensor::gram_matrix<
                    MetricType,
                    PositionType,
                    BatchElem,
                    N>(metric, position, elem, active_dims),
            active_dims);
    return Kokkos::sqrt(Kokkos::abs(det));
}

template <
        DualStrategy Strategy = DualStrategy::Circumcentric,
        std::size_t N,
        class MetricType,
        class PositionType,
        class BatchElem>
KOKKOS_FUNCTION double dual_simplex_volume(
        MetricType metric,
        PositionType position,
        BatchElem elem,
        std::array<bool, N> const& active_dims)
{
    return detail::dual_volume_factor<Strategy>(active_dims)
           * simplex_volume<N>(metric, position, elem, detail::complement(active_dims));
}

} // namespace exterior

} // namespace sil
