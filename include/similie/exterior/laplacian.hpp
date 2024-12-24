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

namespace detail {

template <
        tensor::TensorIndex MetricIndex,
        tensor::TensorNatIndex LaplacianDummyIndex,
        tensor::TensorIndex CochainTag,
        misc::Specialization<tensor::Tensor> TensorType,
        misc::Specialization<tensor::Tensor> MetricType,
        class ExecSpace>
TensorType codifferential_of_coboundary(
        ExecSpace const& exec_space,
        TensorType out_tensor,
        TensorType tensor,
        MetricType inv_metric)
{
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
                    CochainTag>>(exec_space, out_tensor, derivative_tensor, inv_metric);
    Kokkos::fence();

    return out_tensor;
}

template <
        tensor::TensorIndex MetricIndex,
        tensor::TensorNatIndex LaplacianDummyIndex,
        tensor::TensorIndex CochainTag,
        misc::Specialization<tensor::Tensor> TensorType,
        misc::Specialization<tensor::Tensor> MetricType,
        class ExecSpace>
TensorType coboundary_of_codifferential(
        ExecSpace const& exec_space,
        TensorType out_tensor,
        TensorType tensor,
        MetricType inv_metric)
{
    // Codifferential
    [[maybe_unused]] tensor::TensorAccessor<codifferential_index_t<LaplacianDummyIndex, CochainTag>>
            codifferential_accessor;
    ddc::cartesian_prod_t<
            typename TensorType::non_indices_domain_t,
            ddc::DiscreteDomain<codifferential_index_t<LaplacianDummyIndex, CochainTag>>>
            codifferential_tensor_dom(
                    tensor.non_indices_domain(),
                    codifferential_accessor.mem_domain());
    ddc::Chunk codifferential_tensor_alloc(
            codifferential_tensor_dom,
            ddc::KokkosAllocator<double, typename ExecSpace::memory_space>());
    sil::tensor::Tensor codifferential_tensor(codifferential_tensor_alloc);

    sil::exterior::codifferential<
            MetricIndex,
            LaplacianDummyIndex,
            coboundary_index_t<
                    LaplacianDummyIndex,
                    CochainTag>>(exec_space, codifferential_tensor, tensor, inv_metric);
    Kokkos::fence();

    // Derivative
    sil::exterior::
            deriv<LaplacianDummyIndex, CochainTag>(exec_space, out_tensor, codifferential_tensor);
    Kokkos::fence();

    return out_tensor;
}

} // namespace detail

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
    static_assert(tensor::is_covariant_v<LaplacianDummyIndex>);

    if constexpr (CochainTag::rank() == 0) {
        detail::codifferential_of_coboundary<
                MetricIndex,
                LaplacianDummyIndex,
                coboundary_index_t<
                        LaplacianDummyIndex,
                        CochainTag>>(exec_space, laplacian_tensor, tensor, inv_metric);
        Kokkos::fence();

    } else if (CochainTag::rank() == LaplacianDummyIndex::size()) {
        detail::coboundary_of_codifferential<
                MetricIndex,
                LaplacianDummyIndex,
                coboundary_index_t<
                        LaplacianDummyIndex,
                        CochainTag>>(exec_space, laplacian_tensor, tensor, inv_metric);
        Kokkos::fence();
    } else {
        detail::codifferential_of_coboundary<
                MetricIndex,
                LaplacianDummyIndex,
                coboundary_index_t<
                        LaplacianDummyIndex,
                        CochainTag>>(exec_space, laplacian_tensor, tensor, inv_metric);

        auto tmp_alloc = ddc::create_mirror_view(laplacian_tensor);
        tensor::Tensor tmp(tmp_alloc);
        detail::coboundary_of_codifferential<
                MetricIndex,
                LaplacianDummyIndex,
                coboundary_index_t<
                        LaplacianDummyIndex,
                        CochainTag>>(exec_space, tmp, tensor, inv_metric);
        Kokkos::fence();

        ddc::parallel_for_each(
                exec_space,
                laplacian_tensor.domain(),
                KOKKOS_LAMBDA(typename TensorType::discrete_element_type elem) {
                    laplacian_tensor(elem) += tmp(elem);
                });
    }
    return laplacian_tensor;
}

} // namespace exterior

} // namespace sil
