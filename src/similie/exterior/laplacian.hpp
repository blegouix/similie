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
        misc::Specialization<tensor::Tensor> DualTensorBufferType,
        misc::Specialization<tensor::Tensor> TensorType,
        misc::Specialization<tensor::Tensor> HodgeStarType,
        misc::Specialization<tensor::Tensor> DualHodgeStarType,
        class ExecSpace>
TensorType codifferential_of_coboundary(
        ExecSpace const& exec_space,
        TensorType out_tensor,
        TensorType tensor,
        HodgeStarType hodge_star,
        DualHodgeStarType dual_hodge_star,
        DualTensorBufferType dual_tensor_buffer)
{
    using DerivativeIndex = coboundary_index_t<LaplacianDummyIndex, CochainTag>;
    using DualDummySeq = typename detail::CodifferentialDummyIndexSeq<
            LaplacianDummyIndex::size() - DerivativeIndex::rank(),
            LaplacianDummyIndex>::type;
    using DualIndex = misc::convert_type_seq_to_t<tensor::TensorAntisymmetricIndex, DualDummySeq>;
    using RhoLowSeq
            = ddc::type_seq_merge_t<ddc::detail::TypeSeq<LaplacianDummyIndex>, DualDummySeq>;
    using DualCodifferentialIndex
            = misc::convert_type_seq_to_t<tensor::TensorAntisymmetricIndex, RhoLowSeq>;
    auto chain = tangent_basis<
            CochainTag::rank() + 1,
            typename detail::NonSpectatorDimension<
                    LaplacianDummyIndex,
                    typename TensorType::non_indices_domain_t>::type>(exec_space);
    auto lower_chain = tangent_basis<
            CochainTag::rank(),
            typename detail::NonSpectatorDimension<
                    LaplacianDummyIndex,
                    typename TensorType::non_indices_domain_t>::type>(exec_space);
    auto dual_chain = tangent_basis<
            DualIndex::rank() + 1,
            typename detail::NonSpectatorDimension<
                    LaplacianDummyIndex,
                    typename TensorType::non_indices_domain_t>::type>(exec_space);
    auto dual_lower_chain = tangent_basis<
            DualIndex::rank(),
            typename detail::NonSpectatorDimension<
                    LaplacianDummyIndex,
                    typename TensorType::non_indices_domain_t>::type>(exec_space);

    SIMILIE_DEBUG_LOG("similie_deriv_and_apply_first_hodge_star_for_codifferential_of_coboundary");
    ddc::parallel_for_each(
            "similie_deriv_and_apply_first_hodge_star_for_codifferential_of_coboundary",
            exec_space,
            dual_tensor_buffer.non_indices_domain(),
            KOKKOS_LAMBDA(typename TensorType::non_indices_domain_t::discrete_element_type elem) {
                [[maybe_unused]] tensor::TensorAccessor<DerivativeIndex> derivative_accessor;
                std::array<double, DerivativeIndex::access_size()> derivative_alloc {};
                ddc::ChunkSpan<
                        double,
                        ddc::DiscreteDomain<DerivativeIndex>,
                        Kokkos::layout_right,
                        typename TensorType::memory_space>
                        derivative_span(derivative_alloc.data(), derivative_accessor.domain());
                sil::tensor::Tensor derivative_tensor(derivative_span);

                Coboundary<LaplacianDummyIndex, CochainTag>::run(
                        derivative_tensor,
                        // TODO this is an assumption on boundary condition (free boundary), needs to be generalized.
                        [&](auto sampled_elem, auto cochain_elem) {
                            auto const clamped_elem = misc::
                                    clamp_to_domain(tensor.non_indices_domain(), sampled_elem);
                            return tensor.mem(clamped_elem, cochain_elem);
                        },
                        chain,
                        lower_chain,
                        elem);

                sil::tensor::
                        tensor_prod(dual_tensor_buffer[elem], derivative_tensor, hodge_star[elem]);
            });

    SIMILIE_DEBUG_LOG("similie_deriv_and_apply_second_hodge_star_for_codifferential_of_coboundary");
    ddc::parallel_for_each(
            "similie_deriv_and_apply_second_hodge_star_for_codifferential_of_coboundary",
            exec_space,
            out_tensor.non_indices_domain(),
            KOKKOS_LAMBDA(typename TensorType::non_indices_domain_t::discrete_element_type elem) {
                [[maybe_unused]] tensor::TensorAccessor<DualCodifferentialIndex>
                        dual_codifferential_accessor;
                std::array<double, DualCodifferentialIndex::access_size()>
                        dual_codifferential_alloc {};
                ddc::ChunkSpan<
                        double,
                        ddc::DiscreteDomain<DualCodifferentialIndex>,
                        Kokkos::layout_right,
                        typename TensorType::memory_space>
                        dual_codifferential_span(
                                dual_codifferential_alloc.data(),
                                dual_codifferential_accessor.domain());
                sil::tensor::Tensor dual_codifferential(dual_codifferential_span);

                Coboundary<LaplacianDummyIndex, DualIndex>::run(
                        dual_codifferential,
                        // TODO this is an assumption on boundary condition (free boundary), needs to be generalized.
                        [&](auto sampled_elem, auto dual_elem) {
                            auto const clamped_elem = misc::clamp_to_domain(
                                    dual_tensor_buffer.non_indices_domain(),
                                    sampled_elem);
                            return dual_tensor_buffer.mem(clamped_elem, dual_elem);
                        },
                        dual_chain,
                        dual_lower_chain,
                        elem);

                sil::tensor::
                        tensor_prod(out_tensor[elem], dual_codifferential, dual_hodge_star[elem]);
                if constexpr (
                        (LaplacianDummyIndex::size() * (DerivativeIndex::rank() + 1) + 1) % 2
                        == 1) {
                    out_tensor[elem] *= -1;
                }
            });

    return out_tensor;
}

template <class T>
struct LaplacianDummy2 : T
{
};

template <class LaplacianDummyIndex, class CochainTag>
concept ZeroRankLaplacianCochain = CochainTag::rank() == 0;

template <class LaplacianDummyIndex, class CochainTag>
concept IntermediateRankLaplacianCochain
        = CochainTag::rank() > 0 && CochainTag::rank() < LaplacianDummyIndex::size();

template <class LaplacianDummyIndex, class CochainTag>
concept TopRankLaplacianCochain = CochainTag::rank() == LaplacianDummyIndex::size();

} // namespace detail

template <
        tensor::TensorIndex MetricIndex,
        tensor::TensorNatIndex LaplacianDummyIndex,
        tensor::TensorIndex CochainTag,
        misc::Specialization<tensor::Tensor> TensorType,
        misc::Specialization<tensor::Tensor> DerivativeHodgeStarType,
        misc::Specialization<tensor::Tensor> DualDerivativeHodgeStarType,
        misc::Specialization<tensor::Tensor> DerivativeDualTensorBufferType,
        class ExecSpace>
    requires(detail::ZeroRankLaplacianCochain<LaplacianDummyIndex, CochainTag>)
TensorType laplacian(
        ExecSpace const& exec_space,
        TensorType laplacian_tensor,
        TensorType tensor,
        DerivativeHodgeStarType hodge_star,
        DualDerivativeHodgeStarType dual_hodge_star,
        DerivativeDualTensorBufferType dual_tensor_buffer)
{
    using LaplacianDummyIndex2 = tensor::Covariant<
            detail::LaplacianDummy2<tensor::uncharacterize_t<LaplacianDummyIndex>>>;

    return detail::codifferential_of_coboundary<MetricIndex, LaplacianDummyIndex2, CochainTag>(
            exec_space,
            laplacian_tensor,
            tensor,
            hodge_star,
            dual_hodge_star,
            dual_tensor_buffer);
}

template <
        tensor::TensorIndex MetricIndex,
        tensor::TensorNatIndex LaplacianDummyIndex,
        tensor::TensorIndex CochainTag,
        misc::Specialization<tensor::Tensor> TensorType,
        misc::Specialization<tensor::Tensor> DerivativeHodgeStarType,
        misc::Specialization<tensor::Tensor> DualDerivativeHodgeStarType,
        misc::Specialization<tensor::Tensor> DerivativeDualTensorBufferType,
        misc::Specialization<tensor::Tensor> HodgeStarType,
        misc::Specialization<tensor::Tensor> DualHodgeStarType,
        misc::Specialization<tensor::Tensor> DualTensorBufferType,
        misc::Specialization<tensor::Tensor> CodifferentialTensorBufferType,
        misc::Specialization<tensor::Tensor> CoboundaryOfCodifferentialBufferType,
        class ExecSpace>
    requires(detail::IntermediateRankLaplacianCochain<LaplacianDummyIndex, CochainTag>)
TensorType laplacian(
        ExecSpace const& exec_space,
        TensorType laplacian_tensor,
        TensorType tensor,
        DerivativeHodgeStarType derivative_hodge_star,
        DualDerivativeHodgeStarType dual_derivative_hodge_star,
        DerivativeDualTensorBufferType derivative_dual_tensor_buffer,
        HodgeStarType hodge_star,
        DualHodgeStarType dual_hodge_star,
        DualTensorBufferType dual_tensor_buffer,
        CodifferentialTensorBufferType codifferential_tensor_buffer,
        CoboundaryOfCodifferentialBufferType coboundary_of_codifferential_buffer)
{
    using LaplacianDummyIndex2 = tensor::Covariant<
            detail::LaplacianDummy2<tensor::uncharacterize_t<LaplacianDummyIndex>>>;

    auto exec_spaces = Kokkos::Experimental::partition_space(exec_space, 1, 1);

    detail::codifferential_of_coboundary<MetricIndex, LaplacianDummyIndex2, CochainTag>(
            exec_spaces[0],
            laplacian_tensor,
            tensor,
            derivative_hodge_star,
            dual_derivative_hodge_star,
            derivative_dual_tensor_buffer);

    sil::exterior::codifferential<MetricIndex, LaplacianDummyIndex, CochainTag>(
            exec_spaces[1],
            codifferential_tensor_buffer,
            tensor,
            hodge_star,
            dual_hodge_star,
            dual_tensor_buffer);
    sil::exterior::
            deriv<LaplacianDummyIndex, codifferential_index_t<LaplacianDummyIndex, CochainTag>>(
                    exec_spaces[1],
                    coboundary_of_codifferential_buffer,
                    codifferential_tensor_buffer);

    exec_spaces[0].fence();
    exec_spaces[1].fence();

    SIMILIE_DEBUG_LOG("similie_add_coboundary_of_codifferential_contribution_to_laplacian");
    ddc::parallel_for_each(
            "similie_add_coboundary_of_codifferential_contribution_to_laplacian",
            exec_space,
            laplacian_tensor.domain(),
            KOKKOS_LAMBDA(typename TensorType::discrete_element_type elem) {
                laplacian_tensor.mem(elem) += coboundary_of_codifferential_buffer.mem(elem);
            });

    return laplacian_tensor;
}

template <
        tensor::TensorIndex MetricIndex,
        tensor::TensorNatIndex LaplacianDummyIndex,
        tensor::TensorIndex CochainTag,
        misc::Specialization<tensor::Tensor> TensorType,
        misc::Specialization<tensor::Tensor> HodgeStarType,
        misc::Specialization<tensor::Tensor> DualHodgeStarType,
        misc::Specialization<tensor::Tensor> DualTensorBufferType,
        misc::Specialization<tensor::Tensor> CodifferentialTensorBufferType,
        class ExecSpace>
    requires(detail::TopRankLaplacianCochain<LaplacianDummyIndex, CochainTag>)
TensorType laplacian(
        ExecSpace const& exec_space,
        TensorType laplacian_tensor,
        TensorType tensor,
        HodgeStarType hodge_star,
        DualHodgeStarType dual_hodge_star,
        DualTensorBufferType dual_tensor_buffer,
        CodifferentialTensorBufferType codifferential_tensor_buffer)
{
    sil::exterior::codifferential<MetricIndex, LaplacianDummyIndex, CochainTag>(
            exec_space,
            codifferential_tensor_buffer,
            tensor,
            hodge_star,
            dual_hodge_star,
            dual_tensor_buffer);
    return sil::exterior::deriv<
            LaplacianDummyIndex,
            codifferential_index_t<
                    LaplacianDummyIndex,
                    CochainTag>>(exec_space, laplacian_tensor, codifferential_tensor_buffer);
}

} // namespace exterior

} // namespace sil
