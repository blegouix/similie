// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cstddef>

#include <ddc/ddc.hpp>

namespace sil::misc::math {

template <class Scalar, class MemorySpace>
using unmanaged_matrix_view_t = Kokkos::
        View<Scalar**, Kokkos::LayoutRight, MemorySpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

template <class Scalar, class MemorySpace>
using unmanaged_vector_view_t = Kokkos::
        View<Scalar*, Kokkos::LayoutRight, MemorySpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

template <class Scalar, class MemorySpace>
KOKKOS_FUNCTION unmanaged_matrix_view_t<Scalar, MemorySpace> matrix_view(
        Scalar* data,
        std::size_t rows,
        std::size_t cols)
{
    return unmanaged_matrix_view_t<Scalar, MemorySpace>(data, rows, cols);
}

template <class Scalar, class MemorySpace>
KOKKOS_FUNCTION unmanaged_vector_view_t<Scalar, MemorySpace> vector_view(
        Scalar* data,
        std::size_t n)
{
    return unmanaged_vector_view_t<Scalar, MemorySpace>(data, n);
}

template <class MatrixView>
KOKKOS_FUNCTION void fill_identity(MatrixView matrix)
{
    for (std::size_t i = 0; i < matrix.extent(0); ++i) {
        for (std::size_t j = 0; j < matrix.extent(1); ++j) {
            matrix(i, j) = (i == j) ? 1. : 0.;
        }
    }
}

template <class MatrixView1, class MatrixView2>
KOKKOS_FUNCTION void copy_matrix(MatrixView1 dst, MatrixView2 src)
{
    assert(dst.extent(0) == src.extent(0) && dst.extent(1) == src.extent(1));
    for (std::size_t i = 0; i < dst.extent(0); ++i) {
        for (std::size_t j = 0; j < dst.extent(1); ++j) {
            dst(i, j) = src(i, j);
        }
    }
}

template <class MatrixView, class RowIds, class ColIds, class SubmatrixView>
KOKKOS_FUNCTION void extract_submatrix(
        SubmatrixView submatrix,
        MatrixView matrix,
        RowIds const& row_ids,
        ColIds const& col_ids)
{
    assert(submatrix.extent(0) == row_ids.size() && submatrix.extent(1) == col_ids.size());
    for (std::size_t i = 0; i < row_ids.size(); ++i) {
        for (std::size_t j = 0; j < col_ids.size(); ++j) {
            submatrix(i, j) = matrix(row_ids[i], col_ids[j]);
        }
    }
}

template <class MatrixView>
KOKKOS_FUNCTION typename MatrixView::non_const_value_type determinant(MatrixView matrix)
{
    assert(matrix.extent(0) == matrix.extent(1) && "Matrix should be square.");
    if (matrix.extent(0) == 0) {
        return 1.;
    }
    typename MatrixView::non_const_value_type det = 1.;
    int sign = 1;

    for (std::size_t i = 0; i < matrix.extent(0); ++i) {
        std::size_t pivot = i;
        auto pivot_abs = Kokkos::abs(matrix(i, i));
        for (std::size_t row = i + 1; row < matrix.extent(0); ++row) {
            auto const candidate_abs = Kokkos::abs(matrix(row, i));
            if (candidate_abs > pivot_abs) {
                pivot = row;
                pivot_abs = candidate_abs;
            }
        }

        if (pivot_abs == 0.) {
            return 0.;
        }

        if (pivot != i) {
            sign *= -1;
            for (std::size_t col = 0; col < matrix.extent(1); ++col) {
                Kokkos::kokkos_swap(matrix(i, col), matrix(pivot, col));
            }
        }

        auto const diagonal = matrix(i, i);
        det *= diagonal;
        for (std::size_t row = i + 1; row < matrix.extent(0); ++row) {
            auto const factor = matrix(row, i) / diagonal;
            for (std::size_t col = i + 1; col < matrix.extent(1); ++col) {
                matrix(row, col) -= factor * matrix(i, col);
            }
            matrix(row, i) = 0.;
        }
    }
    return sign * det;
}

template <class InverseView, class MatrixView, class WorkspaceView>
KOKKOS_FUNCTION bool invert(InverseView inverse, MatrixView matrix, WorkspaceView workspace)
{
    assert(inverse.extent(0) == inverse.extent(1) && "Inverse target should be square.");
    assert(matrix.extent(0) == matrix.extent(1) && "Input matrix should be square.");
    assert(inverse.extent(0) == matrix.extent(0) && "Input/output matrix sizes should match.");
    assert(workspace.extent(0) >= matrix.extent(0) * matrix.extent(1) && "Workspace is too small.");

    auto matrix_work = matrix_view<
            typename MatrixView::non_const_value_type,
            typename MatrixView::
                    memory_space>(workspace.data(), matrix.extent(0), matrix.extent(1));
    copy_matrix(matrix_work, matrix);
    fill_identity(inverse);

    for (std::size_t i = 0; i < matrix.extent(0); ++i) {
        std::size_t pivot = i;
        auto pivot_abs = Kokkos::abs(matrix_work(i, i));
        for (std::size_t row = i + 1; row < matrix.extent(0); ++row) {
            auto const candidate_abs = Kokkos::abs(matrix_work(row, i));
            if (candidate_abs > pivot_abs) {
                pivot = row;
                pivot_abs = candidate_abs;
            }
        }

        if (pivot_abs == 0.) {
            for (std::size_t row = 0; row < inverse.extent(0); ++row) {
                for (std::size_t col = 0; col < inverse.extent(1); ++col) {
                    inverse(row, col) = 0.;
                }
            }
            return false;
        }

        if (pivot != i) {
            for (std::size_t col = 0; col < matrix_work.extent(1); ++col) {
                Kokkos::kokkos_swap(matrix_work(i, col), matrix_work(pivot, col));
                Kokkos::kokkos_swap(inverse(i, col), inverse(pivot, col));
            }
        }

        auto const diagonal = matrix_work(i, i);
        for (std::size_t col = 0; col < matrix_work.extent(1); ++col) {
            matrix_work(i, col) /= diagonal;
            inverse(i, col) /= diagonal;
        }

        for (std::size_t row = 0; row < matrix_work.extent(0); ++row) {
            if (row == i) {
                continue;
            }
            auto const factor = matrix_work(row, i);
            for (std::size_t col = 0; col < matrix_work.extent(1); ++col) {
                matrix_work(row, col) -= factor * matrix_work(i, col);
                inverse(row, col) -= factor * inverse(i, col);
            }
        }
    }

    return true;
}

template <std::size_t N, class MatrixView, class RowIds, class ColIds>
KOKKOS_FUNCTION double submatrix_determinant(
        MatrixView matrix,
        RowIds const& row_ids,
        ColIds const& col_ids,
        std::array<double, N * N>& submatrix_alloc)
{
    if constexpr (N == 0) {
        return 1.;
    } else {
        auto submatrix = matrix_view<
                double,
                typename MatrixView::memory_space>(submatrix_alloc.data(), N, N);
        extract_submatrix(submatrix, matrix, row_ids, col_ids);
        return determinant(submatrix);
    }
}

} // namespace sil::misc::math
