// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: MIT

#pragma once

#include <ddc/ddc.hpp>

#include <similie/misc/macros.hpp>
#include <similie/misc/specialization.hpp>

#include "tensor_impl.hpp"

namespace sil {

namespace tensor {

// ChatGPT-generated, TODO rely on KokkosKernels LU factorization. It overwrite matrix
template <misc::Specialization<Kokkos::View> ViewType>
KOKKOS_FUNCTION typename ViewType::value_type determinant(const ViewType& matrix)
{
    assert(matrix.extent(0) == matrix.extent(1)
           && "Matrix should be squared to compute its determinant");

    const std::size_t N = matrix.extent(0);

    // Permutation sign (to account for row swaps)
    int permutation_sign = 1;

    // Perform LU decomposition with partial pivoting
    for (std::size_t i = 0; i < N; ++i) {
        // Pivoting (find the largest element in the current column)
        std::size_t pivot = i;
        typename ViewType::value_type max_val = Kokkos::abs(matrix(i, i));
        for (std::size_t j = i + 1; j < N; ++j) {
            if (Kokkos::abs(matrix(j, i)) > max_val) {
                pivot = j;
                max_val = Kokkos::abs(matrix(j, i));
            }
        }

        // Swap rows if necessary
        if (pivot != i) {
            permutation_sign *= -1; // Track the effect of row swaps
            for (std::size_t k = 0; k < N; ++k) {
                Kokkos::kokkos_swap(matrix(i, k), matrix(pivot, k));
            }
        }

        // Check for singular matrix
        if (matrix(i, i) == 0.0) {
            return 0.0; // Determinant is zero for singular matrices
        }

        // Perform elimination below the pivot
        for (std::size_t j = i + 1; j < N; ++j) {
            matrix(j, i) /= matrix(i, i);
            for (std::size_t k = i + 1; k < N; ++k) {
                matrix(j, k) -= matrix(j, i) * matrix(i, k);
            }
        }
    }

    // Compute the determinant as the product of the diagonal elements
    typename ViewType::value_type det = permutation_sign;
    for (std::size_t i = 0; i < N; ++i) {
        det *= matrix(i, i);
    }

    return det;
}

// TODO port on GPU
template <misc::Specialization<Tensor> TensorType>
TensorType::element_type determinant(TensorType tensor)
{
    static_assert(TensorType::natural_domain_t::rank() == 2);
    auto extents = ddc::detail::array(tensor.natural_domain().extents());
    assert(extents[0] == extents[1] && "Matrix should be squared to compute its determinant");
    const std::size_t n = extents[0];

    Kokkos::View<typename TensorType::element_type**, Kokkos::LayoutRight, Kokkos::HostSpace>
            buffer("determinant_buffer", n, n);
    SIMILIE_DEBUG_LOG("similie_compute_determinant");
    ddc::parallel_for_each(
            Kokkos::DefaultHostExecutionSpace(),
            "similie_compute_determinant",
            tensor.accessor().natural_domain(),
            [&](auto index) {
                buffer(ddc::detail::array(index)[0], ddc::detail::array(index)[1])
                        = tensor(tensor.access_element(index));
            });
    return determinant(buffer);
}

} // namespace tensor

} // namespace sil
