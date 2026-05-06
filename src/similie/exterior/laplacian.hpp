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
        tensor::TensorNatIndex TagToRemoveFromCochain,
        tensor::TensorIndex CochainTag,
        misc::Specialization<tensor::Tensor> TensorType,
        misc::Specialization<tensor::Tensor> MetricType,
        misc::Specialization<tensor::Tensor> PositionType,
        class ExecSpace>
codifferential_tensor_t<TagToRemoveFromCochain, CochainTag, TensorType>
apply_prefilled_codifferential(
        ExecSpace const& exec_space,
        codifferential_tensor_t<TagToRemoveFromCochain, CochainTag, TensorType>
                codifferential_tensor,
        TensorType tensor,
        MetricType metric,
        PositionType position)
{
    using MuUpSeq = tensor::upper_t<ddc::to_type_seq_t<tensor::natural_domain_t<CochainTag>>>;
    using NuLowSeq = typename detail::CodifferentialDummyIndexSeq<
            TagToRemoveFromCochain::size() - CochainTag::rank(),
            TagToRemoveFromCochain>::type;
    using RhoLowSeq = ddc::type_seq_merge_t<ddc::detail::TypeSeq<TagToRemoveFromCochain>, NuLowSeq>;
    using RhoUpSeq = tensor::upper_t<RhoLowSeq>;
    using SigmaLowSeq = ddc::type_seq_remove_t<
            tensor::lower_t<MuUpSeq>,
            ddc::detail::TypeSeq<TagToRemoveFromCochain>>;
    using DualIndex = misc::convert_type_seq_to_t<tensor::TensorAntisymmetricIndex, NuLowSeq>;
    using DualCodifferentialIndex
            = misc::convert_type_seq_to_t<tensor::TensorAntisymmetricIndex, RhoLowSeq>;

    [[maybe_unused]] sil::tensor::tensor_accessor_for_domain_t<
            hodge_star_domain_t<MuUpSeq, NuLowSeq>> hodge_star_accessor;
    ddc::cartesian_prod_t<
            typename MetricType::non_indices_domain_t,
            hodge_star_domain_t<MuUpSeq, NuLowSeq>>
            hodge_star_dom(metric.non_indices_domain(), hodge_star_accessor.domain());
    ddc::Chunk hodge_star_alloc(
            hodge_star_dom,
            ddc::KokkosAllocator<double, typename ExecSpace::memory_space>());
    sil::tensor::Tensor hodge_star(hodge_star_alloc);

    [[maybe_unused]] sil::tensor::tensor_accessor_for_domain_t<
            hodge_star_domain_t<RhoUpSeq, SigmaLowSeq>> dual_hodge_star_accessor;
    ddc::cartesian_prod_t<
            typename MetricType::non_indices_domain_t,
            hodge_star_domain_t<RhoUpSeq, SigmaLowSeq>>
            dual_hodge_star_dom(metric.non_indices_domain(), dual_hodge_star_accessor.domain());
    ddc::Chunk dual_hodge_star_alloc(
            dual_hodge_star_dom,
            ddc::KokkosAllocator<double, typename ExecSpace::memory_space>());
    sil::tensor::Tensor dual_hodge_star(dual_hodge_star_alloc);

    [[maybe_unused]] tensor::TensorAccessor<DualIndex> dual_tensor_accessor;
    ddc::cartesian_prod_t<typename TensorType::non_indices_domain_t, ddc::DiscreteDomain<DualIndex>>
            dual_tensor_dom(tensor.non_indices_domain(), dual_tensor_accessor.domain());
    ddc::Chunk dual_tensor_alloc(
            dual_tensor_dom,
            ddc::KokkosAllocator<double, typename ExecSpace::memory_space>());
    sil::tensor::Tensor dual_tensor_buffer(dual_tensor_alloc);

    [[maybe_unused]] tensor::TensorAccessor<DualCodifferentialIndex> dual_codifferential_accessor;
    ddc::cartesian_prod_t<
            typename TensorType::non_indices_domain_t,
            ddc::DiscreteDomain<DualCodifferentialIndex>>
            dual_codifferential_dom(
                    tensor.non_indices_domain(),
                    dual_codifferential_accessor.domain());
    ddc::Chunk dual_codifferential_alloc(
            dual_codifferential_dom,
            ddc::KokkosAllocator<double, typename ExecSpace::memory_space>());
    sil::tensor::Tensor dual_codifferential_buffer(dual_codifferential_alloc);

    fill_discrete_hodge_star<MuUpSeq, NuLowSeq>(exec_space, hodge_star, metric, position);
    fill_discrete_hodge_star<RhoUpSeq, SigmaLowSeq>(exec_space, dual_hodge_star, metric, position);

    return sil::exterior::codifferential<MetricIndex, TagToRemoveFromCochain, CochainTag>(
            exec_space,
            codifferential_tensor,
            tensor,
            hodge_star,
            dual_hodge_star,
            dual_tensor_buffer,
            dual_codifferential_buffer);
}

template <
        tensor::TensorIndex MetricIndex,
        tensor::TensorNatIndex LaplacianDummyIndex,
        tensor::TensorIndex CochainTag,
        misc::Specialization<tensor::Tensor> TensorType,
        misc::Specialization<tensor::Tensor> MetricType,
        misc::Specialization<tensor::Tensor> PositionType,
        class ExecSpace>
TensorType codifferential_of_coboundary(
        ExecSpace const& exec_space,
        TensorType out_tensor,
        TensorType tensor,
        MetricType metric,
        PositionType position)
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
    detail::apply_prefilled_codifferential<
            MetricIndex,
            LaplacianDummyIndex,
            coboundary_index_t<
                    LaplacianDummyIndex,
                    CochainTag>>(exec_space, out_tensor, derivative_tensor, metric, position);

    return out_tensor;
}

template <
        tensor::TensorIndex MetricIndex,
        tensor::TensorNatIndex LaplacianDummyIndex,
        tensor::TensorIndex CochainTag,
        misc::Specialization<tensor::Tensor> TensorType,
        misc::Specialization<tensor::Tensor> MetricType,
        misc::Specialization<tensor::Tensor> PositionType,
        class ExecSpace>
TensorType coboundary_of_codifferential(
        ExecSpace const& exec_space,
        TensorType out_tensor,
        TensorType tensor,
        MetricType metric,
        PositionType position)
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

    detail::apply_prefilled_codifferential<
            MetricIndex,
            LaplacianDummyIndex,
            CochainTag>(exec_space, codifferential_tensor, tensor, metric, position);

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
        misc::Specialization<tensor::Tensor> PositionType,
        class ExecSpace>
TensorType laplacian(
        ExecSpace const& exec_space,
        TensorType laplacian_tensor,
        TensorType tensor,
        MetricType metric,
        PositionType position)
{
    static_assert(tensor::is_covariant_v<LaplacianDummyIndex>);
    using LaplacianDummyIndex2 = tensor::Covariant<
            detail::LaplacianDummy2<tensor::uncharacterize_t<LaplacianDummyIndex>>>;

    if constexpr (CochainTag::rank() == 0) {
        detail::codifferential_of_coboundary<
                MetricIndex,
                LaplacianDummyIndex2,
                CochainTag>(exec_space, laplacian_tensor, tensor, metric, position);
    } else if constexpr (CochainTag::rank() < LaplacianDummyIndex::size()) {
        auto tmp_alloc = ddc::create_mirror(exec_space, laplacian_tensor);
        tensor::Tensor tmp(tmp_alloc);

        auto exec_spaces = Kokkos::Experimental::partition_space(exec_space, 1, 1);

        detail::codifferential_of_coboundary<
                MetricIndex,
                LaplacianDummyIndex2,
                CochainTag>(exec_spaces[0], laplacian_tensor, tensor, metric, position);
        detail::coboundary_of_codifferential<
                MetricIndex,
                LaplacianDummyIndex,
                CochainTag>(exec_spaces[1], tmp, tensor, metric, position);
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
                CochainTag>(exec_space, laplacian_tensor, tensor, metric, position);
    } else {
        assert(false && "Unsupported differential form in Laplacian operator");
    }

    return laplacian_tensor;
}

} // namespace exterior

} // namespace sil
