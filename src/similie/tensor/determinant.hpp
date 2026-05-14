// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: MIT

#pragma once

#include <ddc/ddc.hpp>

#include <similie/misc/macros.hpp>
#include <similie/misc/small_matrix.hpp>
#include <similie/misc/specialization.hpp>

#include "tensor_impl.hpp"

namespace sil::tensor {

template <misc::Specialization<Kokkos::View> ViewType>
KOKKOS_FUNCTION typename ViewType::value_type determinant(const ViewType& matrix)
{
    return misc::math::determinant(matrix);
}

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
            "similie_compute_determinant",
            Kokkos::DefaultHostExecutionSpace(),
            tensor.accessor().natural_domain(),
            [&](auto index) {
                buffer(ddc::detail::array(index)[0], ddc::detail::array(index)[1])
                        = tensor(tensor.access_element(index));
            });
    return determinant(buffer);
}

} // namespace sil::tensor
