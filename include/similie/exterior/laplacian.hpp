// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <ddc/ddc.hpp>

#include <similie/misc/specialization.hpp>
#include <similie/tensor/character.hpp>
#include <similie/tensor/tensor_impl.hpp>

#include "coboundary.hpp"
#include "codifferential.hpp"


namespace sil {

namespace exterior {

template <
        tensor::TensorIndex MetricIndex,
        tensor::TensorNatIndex LaplacianDummyIndex,
        tensor::TensorIndex CochainTag,
        misc::Specialization<tensor::Tensor> TensorType,
        misc::Specialization<tensor::Tensor> MetricType,
        class ExecSpace>
TensorType laplacian(
        ExecSpace const& exec_space,
        TensorType laplacian_tensor,
        TensorType tensor,
        MetricType inv_metric)
{
    static_assert(CochainTag::rank() == 0); // TODO support higher forms
    static_assert(tensor::is_covariant_v<LaplacianDummyIndex>);

    // Derivative
    [[maybe_unused]] tensor::TensorAccessor<coboundary_index_t<LaplacianDummyIndex, CochainTag>>
            derivative_accessor;
    ddc::cartesian_prod_t<
            typename TensorType::non_indices_domain_t,
            ddc::DiscreteDomain<coboundary_index_t<LaplacianDummyIndex, CochainTag>>>
            derivative_tensor_dom(tensor.non_indices_domain(), derivative_accessor.mem_domain());
    ddc::Chunk derivative_tensor_alloc(
            derivative_tensor_dom,
            ddc::KokkosAllocator<double, typename ExecSpace::memory_space>());
    sil::tensor::Tensor derivative_tensor(derivative_tensor_alloc);

    sil::exterior::deriv<LaplacianDummyIndex, CochainTag>(exec_space, derivative_tensor, tensor);
    Kokkos::fence();

    // Codifferential
    sil::exterior::codifferential<
            MetricIndex,
            LaplacianDummyIndex,
            coboundary_index_t<
                    LaplacianDummyIndex,
                    CochainTag>>(exec_space, laplacian_tensor, derivative_tensor, inv_metric);
    Kokkos::fence();

    return laplacian_tensor;
}

} // namespace exterior

} // namespace sil
