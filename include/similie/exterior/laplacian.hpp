// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: MIT

#pragma once

#include <ddc/ddc.hpp>

#include <similie/misc/macros.hpp>
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
    // Coboundary
    [[maybe_unused]] tensor::TensorAccessor<coboundary_index_t<LaplacianDummyIndex, CochainTag>>
            derivative_accessor;
    ddc::cartesian_prod_t<
            typename TensorType::non_indices_domain_t,
            ddc::DiscreteDomain<coboundary_index_t<LaplacianDummyIndex, CochainTag>>>
            derivative_tensor_dom(tensor.non_indices_domain(), derivative_accessor.domain());
    ddc::Chunk derivative_tensor_alloc(
            derivative_tensor_dom,
            ddc::KokkosAllocator<double, typename ExecSpace::memory_space>());
    sil::tensor::Tensor derivative_tensor(derivative_tensor_alloc);

    sil::exterior::deriv<LaplacianDummyIndex, CochainTag>(exec_space, derivative_tensor, tensor);

    // Codifferential
    sil::exterior::codifferential<
            MetricIndex,
            LaplacianDummyIndex,
            coboundary_index_t<
                    LaplacianDummyIndex,
                    CochainTag>>(exec_space, out_tensor, derivative_tensor, inv_metric);

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
                    codifferential_accessor.domain());
    ddc::Chunk codifferential_tensor_alloc(
            codifferential_tensor_dom,
            ddc::KokkosAllocator<double, typename ExecSpace::memory_space>());
    sil::tensor::Tensor codifferential_tensor(codifferential_tensor_alloc);

    sil::exterior::codifferential<
            MetricIndex,
            LaplacianDummyIndex,
            CochainTag>(exec_space, codifferential_tensor, tensor, inv_metric);

    // Coboundary
    sil::exterior::deriv<
            LaplacianDummyIndex,
            codifferential_index_t<
                    LaplacianDummyIndex,
                    CochainTag>>(exec_space, out_tensor, codifferential_tensor);

    return out_tensor;
}

template <class T>
struct LaplacianDummy2 : T
{
};

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
    using LaplacianDummyIndex2 = tensor::Covariant<
            detail::LaplacianDummy2<tensor::uncharacterize_t<LaplacianDummyIndex>>>;

    if constexpr (CochainTag::rank() == 0) {
        detail::codifferential_of_coboundary<
                MetricIndex,
                LaplacianDummyIndex2,
                CochainTag>(exec_space, laplacian_tensor, tensor, inv_metric);
    } else if constexpr (CochainTag::rank() < LaplacianDummyIndex::size()) {
        auto tmp_alloc = ddc::create_mirror(exec_space, laplacian_tensor);
        tensor::Tensor tmp(tmp_alloc);

        auto exec_spaces = Kokkos::Experimental::partition_space(exec_space, 1, 1);

        detail::codifferential_of_coboundary<
                MetricIndex,
                LaplacianDummyIndex2,
                CochainTag>(exec_spaces[0], laplacian_tensor, tensor, inv_metric);
        detail::coboundary_of_codifferential<
                MetricIndex,
                LaplacianDummyIndex,
                CochainTag>(exec_spaces[1], tmp, tensor, inv_metric);
        exec_spaces[0].fence();
        exec_spaces[1].fence();

        SIMILIE_DEBUG_LOG("similie_add_coboundary_of_codifferential_contribution_to_laplacian");
        ddc::parallel_for_each(
                "similie_add_coboundary_of_codifferential_contribution_to_laplacian",
                exec_space,
                laplacian_tensor.domain(),
                KOKKOS_LAMBDA(typename TensorType::discrete_element_type elem) {
                    laplacian_tensor(elem) += tmp(elem);
                });
    } else if constexpr (CochainTag::rank() == LaplacianDummyIndex::size()) {
        detail::coboundary_of_codifferential<
                MetricIndex,
                LaplacianDummyIndex,
                CochainTag>(exec_space, laplacian_tensor, tensor, inv_metric);
    } else {
        assert(false && "Unsupported differential form in Laplacian operator");
    }

    return laplacian_tensor;
}

} // namespace exterior

} // namespace sil
