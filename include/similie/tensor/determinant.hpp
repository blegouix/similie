// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <ddc/ddc.hpp>

#include <similie/misc/specialization.hpp>

#include "tensor_impl.hpp"

namespace sil {

namespace tensor {

namespace detail {

// ChatGPT-generated, TODO rely on KokkosKernels LU factorization
template <misc::Specialization<Kokkos::View> ViewType>
KOKKOS_FUNCTION typename ViewType::value_type determinant(const ViewType& matrix)
{
    const std::size_t N = matrix.extent(0);

    // Create a copy of the input matrix (LU decomposition modifies the matrix)
    ViewType LU("LU", N, N);
    Kokkos::deep_copy(LU, matrix);

    // Permutation sign (to account for row swaps)
    int permutation_sign = 1;

    // Perform LU decomposition with partial pivoting
    for (int i = 0; i < N; ++i) {
        // Pivoting (find the largest element in the current column)
        int pivot = i;
        typename ViewType::value_type max_val = std::abs(LU(i, i));
        for (int j = i + 1; j < N; ++j) {
            if (std::abs(LU(j, i)) > max_val) {
                pivot = j;
                max_val = std::abs(LU(j, i));
            }
        }

        // Swap rows if necessary
        if (pivot != i) {
            permutation_sign *= -1; // Track the effect of row swaps
            for (int k = 0; k < N; ++k) {
                std::swap(LU(i, k), LU(pivot, k));
            }
        }

        // Check for singular matrix
        if (LU(i, i) == 0.0) {
            return 0.0; // Determinant is zero for singular matrices
        }

        // Perform elimination below the pivot
        for (int j = i + 1; j < N; ++j) {
            LU(j, i) /= LU(i, i);
            for (int k = i + 1; k < N; ++k) {
                LU(j, k) -= LU(j, i) * LU(i, k);
            }
        }
    }

    // Compute the determinant as the product of the diagonal elements
    typename ViewType::value_type det = permutation_sign;
    for (int i = 0; i < N; ++i) {
        det *= LU(i, i);
    }

    return det;
}

} // namespace detail

template <misc::Specialization<Tensor> TensorType>
KOKKOS_FUNCTION TensorType::element_type determinant(TensorType tensor)
{
    static_assert(TensorType::natural_domain_t::rank() == 2);
    auto extents = ddc::detail::array(tensor.natural_domain().extents());
    assert(extents[0] == extents[1] && "Matrix should be squared to compute its determinant");
    const std::size_t n = extents[0];

    Kokkos::View<typename TensorType::element_type**, Kokkos::LayoutRight, Kokkos::HostSpace>
            buffer("determinant_buffer", n, n);
    ddc::for_each(tensor.accessor().natural_domain(), [&](auto index) {
        buffer(ddc::detail::array(index)[0], ddc::detail::array(index)[1])
                = tensor(tensor.access_element(index));
    });
    return detail::determinant(buffer);
}

} // namespace tensor

} // namespace sil
