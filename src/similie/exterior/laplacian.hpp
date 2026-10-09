// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

#include <optional>
#include <stdexcept>
#include <utility>

#include <ddc/ddc.hpp>

#include <similie/misc/domain_contains.hpp>
#include <similie/misc/macros.hpp>
#include <similie/misc/specialization.hpp>
#include <similie/misc/type_seq_ext.hpp>
#include <similie/tensor/character.hpp>
#include <similie/tensor/tensor_impl.hpp>

#include "coboundary.hpp"
#include "codifferential.hpp"
#include "scalar_extrapolation_rules.hpp"


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
        class ExecSpace,
        class PrimalExtrapolationRule = ClampCochainExtrapolationRule,
        class DualExtrapolationRule = ZeroCochainExtrapolationRule>
TensorType codifferential_of_coboundary(
        ExecSpace const& exec_space,
        TensorType out_tensor,
        TensorType tensor,
        HodgeStarType hodge_star,
        DualHodgeStarType dual_hodge_star,
        DualTensorBufferType dual_tensor_buffer,
        PrimalExtrapolationRule primal_extrapolation = {},
        DualExtrapolationRule dual_extrapolation = {})
{
    auto const primal_extrapolation_rules = [&]() {
        if constexpr (misc::Specialization<PrimalExtrapolationRule, ExtrapolationRules>)
            return primal_extrapolation;
        else
            return make_extrapolation_rules(tensor, primal_extrapolation);
    }();
    auto const dual_extrapolation_rules = [&]() {
        if constexpr (misc::Specialization<DualExtrapolationRule, ExtrapolationRules>)
            return dual_extrapolation;
        else
            return make_extrapolation_rules(dual_tensor_buffer, dual_extrapolation);
    }();
    using coboundary_output_index = coboundary_index_t<LaplacianDummyIndex, CochainTag>;
    using codifferential_hodge_output_indices = codifferential_hodge_output_indices_t<
            LaplacianDummyIndex::size() - coboundary_output_index::rank(),
            LaplacianDummyIndex>;
    using coboundary_dual_tensor_index = misc::convert_type_seq_to_t<
            tensor::TensorAntisymmetricIndex,
            codifferential_hodge_output_indices>;
    using dual_codifferential_hodge_input_indices = ddc::type_seq_merge_t<
            ddc::TypeSeq<LaplacianDummyIndex>,
            codifferential_hodge_output_indices>;
    using dual_codifferential_index = misc::convert_type_seq_to_t<
            tensor::TensorAntisymmetricIndex,
            dual_codifferential_hodge_input_indices>;
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
            coboundary_dual_tensor_index::rank() + 1,
            typename detail::NonSpectatorDimension<
                    LaplacianDummyIndex,
                    typename TensorType::non_indices_domain_t>::type>(exec_space);
    auto dual_lower_chain = tangent_basis<
            coboundary_dual_tensor_index::rank(),
            typename detail::NonSpectatorDimension<
                    LaplacianDummyIndex,
                    typename TensorType::non_indices_domain_t>::type>(exec_space);

    SIMILIE_DEBUG_LOG("similie_deriv_and_apply_first_hodge_star_for_codifferential_of_coboundary");
    ddc::parallel_for_each(
            "similie_deriv_and_apply_first_hodge_star_for_codifferential_of_coboundary",
            exec_space,
            dual_tensor_buffer.non_indices_domain(),
            KOKKOS_LAMBDA(typename TensorType::non_indices_domain_t::discrete_element_type elem) {
                [[maybe_unused]] tensor::TensorAccessor<coboundary_output_index>
                        derivative_accessor;
                std::array<double, coboundary_output_index::access_size()> derivative_alloc {};
                ddc::ChunkSpan<
                        double,
                        ddc::DiscreteDomain<coboundary_output_index>,
                        Kokkos::layout_right,
                        typename TensorType::memory_space>
                        derivative_span(derivative_alloc.data(), derivative_accessor.domain());
                sil::tensor::Tensor derivative_tensor(derivative_span);

                Coboundary<LaplacianDummyIndex, CochainTag>::operator()(
                        derivative_tensor,
                        [&](auto sampled_elem, auto cochain_elem) {
                            return primal_extrapolation_rules(tensor, sampled_elem, cochain_elem);
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
                [[maybe_unused]] tensor::TensorAccessor<dual_codifferential_index>
                        dual_codifferential_accessor;
                std::array<double, dual_codifferential_index::access_size()>
                        dual_codifferential_alloc {};
                ddc::ChunkSpan<
                        double,
                        ddc::DiscreteDomain<dual_codifferential_index>,
                        Kokkos::layout_right,
                        typename TensorType::memory_space>
                        dual_codifferential_span(
                                dual_codifferential_alloc.data(),
                                dual_codifferential_accessor.domain());
                sil::tensor::Tensor dual_codifferential(dual_codifferential_span);

                TransposedCoboundary<LaplacianDummyIndex, coboundary_dual_tensor_index>::operator()(
                        dual_codifferential,
                        [&](auto sampled_elem, auto dual_elem) {
                            return dual_extrapolation_rules(
                                    dual_tensor_buffer,
                                    sampled_elem,
                                    dual_elem);
                        },
                        dual_chain,
                        dual_lower_chain,
                        elem);

                sil::tensor::
                        tensor_prod(out_tensor[elem], dual_codifferential, dual_hodge_star[elem]);
                if constexpr (
                        (LaplacianDummyIndex::size() * (coboundary_output_index::rank() + 1) + 1)
                                % 2
                        == 1) {
                    out_tensor[elem] *= -1;
                }
            });

    return out_tensor;
}

template <class LaplacianDummyIndex, class CochainTag>
concept ZeroRankLaplacianCochain = CochainTag::rank() == 0;

template <class LaplacianDummyIndex, class CochainTag>
concept IntermediateRankLaplacianCochain
        = CochainTag::rank() > 0 && CochainTag::rank() < LaplacianDummyIndex::size();

template <class LaplacianDummyIndex, class CochainTag>
concept TopRankLaplacianCochain = CochainTag::rank() == LaplacianDummyIndex::size();

} // namespace detail

template <class T>
struct IndexForCodifferentialOfCoboundaryInLaplacian : T
{
};

template <class... Args>
class StagedLaplacian;

template <
        tensor::TensorIndex MetricIndex,
        tensor::TensorNatIndex LaplacianDummyIndex,
        tensor::TensorIndex CochainTag,
        misc::Specialization<tensor::Tensor> TensorType,
        misc::Specialization<tensor::Tensor> MetricType,
        misc::Specialization<tensor::Tensor> PositionType,
        class ExecSpace>
    requires(detail::ZeroRankLaplacianCochain<LaplacianDummyIndex, CochainTag>)
class StagedLaplacian<
        MetricIndex,
        LaplacianDummyIndex,
        CochainTag,
        TensorType,
        MetricType,
        PositionType,
        ExecSpace>
{
    using MemorySpace = typename TensorType::memory_space;
    using AllocatorType = ddc::KokkosAllocator<double, MemorySpace>;
    using CodifferentialOfCoboundaryIndex
            = tensor::Covariant<IndexForCodifferentialOfCoboundaryInLaplacian<
                    tensor::uncharacterize_t<LaplacianDummyIndex>>>;
    using CoboundaryOutputIndex = coboundary_index_t<CodifferentialOfCoboundaryIndex, CochainTag>;
    using CoboundaryHodgeInputIndices
            = tensor::upper_t<ddc::to_type_seq_t<tensor::natural_domain_t<CoboundaryOutputIndex>>>;
    using CoboundaryHodgeOutputIndices = codifferential_hodge_output_indices_t<
            CodifferentialOfCoboundaryIndex::size() - CoboundaryOutputIndex::rank(),
            CodifferentialOfCoboundaryIndex>;
    using DualCoboundaryHodgeInputIndices = ddc::type_seq_merge_t<
            ddc::TypeSeq<CodifferentialOfCoboundaryIndex>,
            CoboundaryHodgeOutputIndices>;
    using DualCoboundaryHodgeOutputIndices = ddc::type_seq_remove_t<
            tensor::lower_t<CoboundaryHodgeInputIndices>,
            ddc::TypeSeq<CodifferentialOfCoboundaryIndex>>;
    using CoboundaryDualTensorIndex = misc::
            convert_type_seq_to_t<tensor::TensorAntisymmetricIndex, CoboundaryHodgeOutputIndices>;

    using DerivativeHodgeStarDomainType = sil::misc::cartesian_prod_t<
            typename MetricType::non_indices_domain_t,
            hodge_star_domain_t<CoboundaryHodgeInputIndices, CoboundaryHodgeOutputIndices>>;
    using DualDerivativeHodgeStarDomainType = sil::misc::cartesian_prod_t<
            typename MetricType::non_indices_domain_t,
            hodge_star_domain_t<
                    tensor::upper_t<DualCoboundaryHodgeInputIndices>,
                    DualCoboundaryHodgeOutputIndices>>;
    using DerivativeDualTensorDomainType = sil::misc::cartesian_prod_t<
            typename TensorType::non_indices_domain_t,
            ddc::DiscreteDomain<CoboundaryDualTensorIndex>>;

    using DerivativeHodgeStarAllocType
            = ddc::Chunk<double, DerivativeHodgeStarDomainType, AllocatorType>;
    using DualDerivativeHodgeStarAllocType
            = ddc::Chunk<double, DualDerivativeHodgeStarDomainType, AllocatorType>;
    using DerivativeDualTensorAllocType
            = ddc::Chunk<double, DerivativeDualTensorDomainType, AllocatorType>;

    using DerivativeHodgeStarTensorType = tensor::
            Tensor<double, DerivativeHodgeStarDomainType, Kokkos::layout_right, MemorySpace>;
    using DualDerivativeHodgeStarTensorType = tensor::
            Tensor<double, DualDerivativeHodgeStarDomainType, Kokkos::layout_right, MemorySpace>;
    using DerivativeDualTensorType = tensor::
            Tensor<double, DerivativeDualTensorDomainType, Kokkos::layout_right, MemorySpace>;
    ExecSpace m_exec_space;
    std::optional<DerivativeHodgeStarAllocType> m_derivative_hodge_star_alloc;
    std::optional<DualDerivativeHodgeStarAllocType> m_dual_derivative_hodge_star_alloc;
    std::optional<DerivativeDualTensorAllocType> m_derivative_dual_tensor_alloc;
    std::optional<DerivativeHodgeStarTensorType> m_derivative_hodge_star;
    std::optional<DualDerivativeHodgeStarTensorType> m_dual_derivative_hodge_star;
    std::optional<DerivativeDualTensorType> m_derivative_dual_tensor_buffer;

public:
    /**
     * Access the dual one-cochain produced by the first Hodge stage.
     * \important This operator and documentation are fully AI-generated.
     *
     * The buffer contains the constitutive flux after the staged Laplacian
     * has evaluated a primal scalar cochain. Connected tensor domains can
     * balance its trace values without constructing a separate Hodge star.
     */
    DerivativeDualTensorType derivative_dual_tensor_buffer() const
    {
        return *m_derivative_dual_tensor_buffer;
    }

    DualDerivativeHodgeStarTensorType dual_derivative_hodge_star() const
    {
        return *m_dual_derivative_hodge_star;
    }

    StagedLaplacian(
            ExecSpace const& exec_space,
            DerivativeHodgeStarTensorType&& derivative_hodge_star,
            DualDerivativeHodgeStarTensorType&& dual_derivative_hodge_star,
            DerivativeDualTensorType&& derivative_dual_tensor_buffer)
        : m_exec_space(exec_space)
        , m_derivative_hodge_star(std::move(derivative_hodge_star))
        , m_dual_derivative_hodge_star(std::move(dual_derivative_hodge_star))
        , m_derivative_dual_tensor_buffer(std::move(derivative_dual_tensor_buffer))
    {
    }

    StagedLaplacian(
            ExecSpace const& exec_space,
            TensorType,
            TensorType tensor,
            MetricType metric,
            PositionType position)
        : m_exec_space(exec_space)
    {
        [[maybe_unused]] tensor::tensor_accessor_for_domain_t<
                hodge_star_domain_t<CoboundaryHodgeInputIndices, CoboundaryHodgeOutputIndices>>
                derivative_hodge_star_accessor;
        [[maybe_unused]] tensor::tensor_accessor_for_domain_t<hodge_star_domain_t<
                tensor::upper_t<DualCoboundaryHodgeInputIndices>,
                DualCoboundaryHodgeOutputIndices>> dual_derivative_hodge_star_accessor;
        [[maybe_unused]] tensor::TensorAccessor<CoboundaryDualTensorIndex>
                derivative_dual_tensor_accessor;

        m_derivative_hodge_star_alloc.emplace(
                DerivativeHodgeStarDomainType(
                        metric.non_indices_domain(),
                        derivative_hodge_star_accessor.domain()),
                AllocatorType());
        m_dual_derivative_hodge_star_alloc.emplace(
                DualDerivativeHodgeStarDomainType(
                        metric.non_indices_domain(),
                        dual_derivative_hodge_star_accessor.domain()),
                AllocatorType());
        m_derivative_dual_tensor_alloc.emplace(
                DerivativeDualTensorDomainType(
                        tensor.non_indices_domain(),
                        derivative_dual_tensor_accessor.domain()),
                AllocatorType());

        m_derivative_hodge_star.emplace(*m_derivative_hodge_star_alloc);
        m_dual_derivative_hodge_star.emplace(*m_dual_derivative_hodge_star_alloc);
        m_derivative_dual_tensor_buffer.emplace(*m_derivative_dual_tensor_alloc);

        fill_discrete_hodge_star<CoboundaryHodgeInputIndices, CoboundaryHodgeOutputIndices>(
                exec_space,
                *m_derivative_hodge_star,
                metric,
                position);
        fill_discrete_hodge_star<
                tensor::upper_t<DualCoboundaryHodgeInputIndices>,
                DualCoboundaryHodgeOutputIndices>(
                exec_space,
                *m_dual_derivative_hodge_star,
                metric,
                position);
    }

    template <
            class PrimalExtrapolationRule = ClampCochainExtrapolationRule,
            class DualExtrapolationRule = ZeroCochainExtrapolationRule>
    TensorType operator()(
            TensorType laplacian_tensor,
            TensorType tensor,
            PrimalExtrapolationRule primal_extrapolation = {},
            DualExtrapolationRule dual_extrapolation = {})
    {
        return detail::codifferential_of_coboundary<
                MetricIndex,
                CodifferentialOfCoboundaryIndex,
                CochainTag>(
                m_exec_space,
                laplacian_tensor,
                tensor,
                *m_derivative_hodge_star,
                *m_dual_derivative_hodge_star,
                *m_derivative_dual_tensor_buffer,
                primal_extrapolation,
                dual_extrapolation);
    }
};

template <
        tensor::TensorIndex MetricIndex,
        tensor::TensorNatIndex LaplacianDummyIndex,
        tensor::TensorIndex CochainTag,
        class ExecSpace,
        misc::Specialization<tensor::Tensor> TensorType,
        misc::Specialization<tensor::Tensor> MetricType,
        misc::Specialization<tensor::Tensor> PositionType>
StagedLaplacian<
        MetricIndex,
        LaplacianDummyIndex,
        CochainTag,
        TensorType,
        MetricType,
        PositionType,
        ExecSpace>
make_staged_laplacian(
        ExecSpace const& exec_space,
        TensorType laplacian_tensor,
        TensorType tensor,
        MetricType metric,
        PositionType position)
{
    return StagedLaplacian<
            MetricIndex,
            LaplacianDummyIndex,
            CochainTag,
            TensorType,
            MetricType,
            PositionType,
            ExecSpace>(exec_space, laplacian_tensor, tensor, metric, position);
}

template <
        tensor::TensorIndex MetricIndex,
        tensor::TensorNatIndex LaplacianDummyIndex,
        tensor::TensorIndex CochainTag,
        misc::Specialization<tensor::Tensor> TensorType,
        misc::Specialization<tensor::Tensor> MetricType,
        misc::Specialization<tensor::Tensor> PositionType,
        class ExecSpace>
    requires(detail::IntermediateRankLaplacianCochain<LaplacianDummyIndex, CochainTag>)
class StagedLaplacian<
        MetricIndex,
        LaplacianDummyIndex,
        CochainTag,
        TensorType,
        MetricType,
        PositionType,
        ExecSpace>
{
    using MemorySpace = typename TensorType::memory_space;
    using AllocatorType = ddc::KokkosAllocator<double, MemorySpace>;
    using CodifferentialOfCoboundaryIndex
            = tensor::Covariant<IndexForCodifferentialOfCoboundaryInLaplacian<
                    tensor::uncharacterize_t<LaplacianDummyIndex>>>;
    using CoboundaryOutputIndex = coboundary_index_t<CodifferentialOfCoboundaryIndex, CochainTag>;
    using CoboundaryHodgeInputIndices
            = tensor::upper_t<ddc::to_type_seq_t<tensor::natural_domain_t<CoboundaryOutputIndex>>>;
    using CoboundaryHodgeOutputIndices = codifferential_hodge_output_indices_t<
            CodifferentialOfCoboundaryIndex::size() - CoboundaryOutputIndex::rank(),
            CodifferentialOfCoboundaryIndex>;
    using DualCoboundaryHodgeInputIndices = ddc::type_seq_merge_t<
            ddc::TypeSeq<CodifferentialOfCoboundaryIndex>,
            CoboundaryHodgeOutputIndices>;
    using DualCoboundaryHodgeOutputIndices = ddc::type_seq_remove_t<
            tensor::lower_t<CoboundaryHodgeInputIndices>,
            ddc::TypeSeq<CodifferentialOfCoboundaryIndex>>;
    using CoboundaryDualTensorIndex = misc::
            convert_type_seq_to_t<tensor::TensorAntisymmetricIndex, CoboundaryHodgeOutputIndices>;

    using CodifferentialHodgeInputIndices
            = tensor::upper_t<ddc::to_type_seq_t<tensor::natural_domain_t<CochainTag>>>;
    using CodifferentialHodgeOutputIndices = codifferential_hodge_output_indices_t<
            LaplacianDummyIndex::size() - CochainTag::rank(),
            LaplacianDummyIndex>;
    using DualCodifferentialHodgeInputIndices = ddc::
            type_seq_merge_t<ddc::TypeSeq<LaplacianDummyIndex>, CodifferentialHodgeOutputIndices>;
    using DualCodifferentialHodgeOutputIndices = ddc::type_seq_remove_t<
            tensor::lower_t<CodifferentialHodgeInputIndices>,
            ddc::TypeSeq<LaplacianDummyIndex>>;
    using CodifferentialDualTensorIndex = misc::convert_type_seq_to_t<
            tensor::TensorAntisymmetricIndex,
            CodifferentialHodgeOutputIndices>;
    using CodifferentialOutputIndex = codifferential_index_t<LaplacianDummyIndex, CochainTag>;

    using DerivativeHodgeStarDomainType = sil::misc::cartesian_prod_t<
            typename MetricType::non_indices_domain_t,
            hodge_star_domain_t<CoboundaryHodgeInputIndices, CoboundaryHodgeOutputIndices>>;
    using DualDerivativeHodgeStarDomainType = sil::misc::cartesian_prod_t<
            typename MetricType::non_indices_domain_t,
            hodge_star_domain_t<
                    tensor::upper_t<DualCoboundaryHodgeInputIndices>,
                    DualCoboundaryHodgeOutputIndices>>;
    using DerivativeDualTensorDomainType = sil::misc::cartesian_prod_t<
            typename TensorType::non_indices_domain_t,
            ddc::DiscreteDomain<CoboundaryDualTensorIndex>>;
    using HodgeStarDomainType = sil::misc::cartesian_prod_t<
            typename MetricType::non_indices_domain_t,
            hodge_star_domain_t<CodifferentialHodgeInputIndices, CodifferentialHodgeOutputIndices>>;
    using DualHodgeStarDomainType = sil::misc::cartesian_prod_t<
            typename MetricType::non_indices_domain_t,
            hodge_star_domain_t<
                    tensor::upper_t<DualCodifferentialHodgeInputIndices>,
                    DualCodifferentialHodgeOutputIndices>>;
    using DualTensorDomainType = sil::misc::cartesian_prod_t<
            typename TensorType::non_indices_domain_t,
            ddc::DiscreteDomain<CodifferentialDualTensorIndex>>;
    using CodifferentialDomainType = sil::misc::cartesian_prod_t<
            typename TensorType::non_indices_domain_t,
            ddc::DiscreteDomain<CodifferentialOutputIndex>>;

    using DerivativeHodgeStarAllocType
            = ddc::Chunk<double, DerivativeHodgeStarDomainType, AllocatorType>;
    using DualDerivativeHodgeStarAllocType
            = ddc::Chunk<double, DualDerivativeHodgeStarDomainType, AllocatorType>;
    using DerivativeDualTensorAllocType
            = ddc::Chunk<double, DerivativeDualTensorDomainType, AllocatorType>;
    using HodgeStarAllocType = ddc::Chunk<double, HodgeStarDomainType, AllocatorType>;
    using DualHodgeStarAllocType = ddc::Chunk<double, DualHodgeStarDomainType, AllocatorType>;
    using DualTensorAllocType = ddc::Chunk<double, DualTensorDomainType, AllocatorType>;
    using CodifferentialAllocType = ddc::Chunk<double, CodifferentialDomainType, AllocatorType>;
    using CoboundaryOfCodifferentialAllocType
            = ddc::Chunk<double, typename TensorType::discrete_domain_type, AllocatorType>;

    using DerivativeHodgeStarTensorType = tensor::
            Tensor<double, DerivativeHodgeStarDomainType, Kokkos::layout_right, MemorySpace>;
    using DualDerivativeHodgeStarTensorType = tensor::
            Tensor<double, DualDerivativeHodgeStarDomainType, Kokkos::layout_right, MemorySpace>;
    using DerivativeDualTensorType = tensor::
            Tensor<double, DerivativeDualTensorDomainType, Kokkos::layout_right, MemorySpace>;
    using HodgeStarTensorType
            = tensor::Tensor<double, HodgeStarDomainType, Kokkos::layout_right, MemorySpace>;
    using DualHodgeStarTensorType
            = tensor::Tensor<double, DualHodgeStarDomainType, Kokkos::layout_right, MemorySpace>;
    using DualTensorType
            = tensor::Tensor<double, DualTensorDomainType, Kokkos::layout_right, MemorySpace>;
    using CodifferentialTensorType
            = tensor::Tensor<double, CodifferentialDomainType, Kokkos::layout_right, MemorySpace>;
    ExecSpace m_exec_space;
    std::optional<DerivativeHodgeStarAllocType> m_derivative_hodge_star_alloc;
    std::optional<DualDerivativeHodgeStarAllocType> m_dual_derivative_hodge_star_alloc;
    std::optional<DerivativeDualTensorAllocType> m_derivative_dual_tensor_alloc;
    std::optional<HodgeStarAllocType> m_hodge_star_alloc;
    std::optional<DualHodgeStarAllocType> m_dual_hodge_star_alloc;
    std::optional<DualTensorAllocType> m_dual_tensor_alloc;
    std::optional<CodifferentialAllocType> m_codifferential_alloc;
    std::optional<CoboundaryOfCodifferentialAllocType> m_coboundary_of_codifferential_alloc;
    std::optional<DerivativeHodgeStarTensorType> m_derivative_hodge_star;
    std::optional<DualDerivativeHodgeStarTensorType> m_dual_derivative_hodge_star;
    std::optional<DerivativeDualTensorType> m_derivative_dual_tensor_buffer;
    std::optional<HodgeStarTensorType> m_hodge_star;
    std::optional<DualHodgeStarTensorType> m_dual_hodge_star;
    std::optional<DualTensorType> m_dual_tensor_buffer;
    std::optional<CodifferentialTensorType> m_codifferential_tensor_buffer;
    std::optional<TensorType> m_coboundary_of_codifferential_buffer;

public:
    StagedLaplacian(
            ExecSpace const& exec_space,
            DerivativeHodgeStarTensorType derivative_hodge_star,
            DualDerivativeHodgeStarTensorType dual_derivative_hodge_star,
            DerivativeDualTensorType derivative_dual_tensor_buffer,
            HodgeStarTensorType hodge_star,
            DualHodgeStarTensorType dual_hodge_star,
            DualTensorType dual_tensor_buffer,
            CodifferentialTensorType codifferential_tensor_buffer,
            TensorType coboundary_of_codifferential_buffer)
        : m_exec_space(exec_space)
        , m_derivative_hodge_star(derivative_hodge_star)
        , m_dual_derivative_hodge_star(dual_derivative_hodge_star)
        , m_derivative_dual_tensor_buffer(derivative_dual_tensor_buffer)
        , m_hodge_star(hodge_star)
        , m_dual_hodge_star(dual_hodge_star)
        , m_dual_tensor_buffer(dual_tensor_buffer)
        , m_codifferential_tensor_buffer(codifferential_tensor_buffer)
        , m_coboundary_of_codifferential_buffer(coboundary_of_codifferential_buffer)
    {
    }

    StagedLaplacian(
            ExecSpace const& exec_space,
            TensorType laplacian_tensor,
            TensorType tensor,
            MetricType metric,
            PositionType position)
        : m_exec_space(exec_space)
    {
        [[maybe_unused]] tensor::tensor_accessor_for_domain_t<
                hodge_star_domain_t<CoboundaryHodgeInputIndices, CoboundaryHodgeOutputIndices>>
                derivative_hodge_star_accessor;
        [[maybe_unused]] tensor::tensor_accessor_for_domain_t<hodge_star_domain_t<
                tensor::upper_t<DualCoboundaryHodgeInputIndices>,
                DualCoboundaryHodgeOutputIndices>> dual_derivative_hodge_star_accessor;
        [[maybe_unused]] tensor::TensorAccessor<CoboundaryDualTensorIndex>
                derivative_dual_tensor_accessor;
        [[maybe_unused]] tensor::tensor_accessor_for_domain_t<hodge_star_domain_t<
                CodifferentialHodgeInputIndices,
                CodifferentialHodgeOutputIndices>> hodge_star_accessor;
        [[maybe_unused]] tensor::tensor_accessor_for_domain_t<hodge_star_domain_t<
                tensor::upper_t<DualCodifferentialHodgeInputIndices>,
                DualCodifferentialHodgeOutputIndices>> dual_hodge_star_accessor;
        [[maybe_unused]] tensor::TensorAccessor<CodifferentialDualTensorIndex> dual_tensor_accessor;
        [[maybe_unused]] tensor::TensorAccessor<CodifferentialOutputIndex> codifferential_accessor;

        m_derivative_hodge_star_alloc.emplace(
                DerivativeHodgeStarDomainType(
                        metric.non_indices_domain(),
                        derivative_hodge_star_accessor.domain()),
                AllocatorType());
        m_dual_derivative_hodge_star_alloc.emplace(
                DualDerivativeHodgeStarDomainType(
                        metric.non_indices_domain(),
                        dual_derivative_hodge_star_accessor.domain()),
                AllocatorType());
        m_derivative_dual_tensor_alloc.emplace(
                DerivativeDualTensorDomainType(
                        tensor.non_indices_domain(),
                        derivative_dual_tensor_accessor.domain()),
                AllocatorType());
        m_hodge_star_alloc.emplace(
                HodgeStarDomainType(metric.non_indices_domain(), hodge_star_accessor.domain()),
                AllocatorType());
        m_dual_hodge_star_alloc.emplace(
                DualHodgeStarDomainType(
                        metric.non_indices_domain(),
                        dual_hodge_star_accessor.domain()),
                AllocatorType());
        m_dual_tensor_alloc.emplace(
                DualTensorDomainType(tensor.non_indices_domain(), dual_tensor_accessor.domain()),
                AllocatorType());
        m_codifferential_alloc.emplace(
                CodifferentialDomainType(
                        tensor.non_indices_domain(),
                        codifferential_accessor.domain()),
                AllocatorType());
        m_coboundary_of_codifferential_alloc.emplace(laplacian_tensor.domain(), AllocatorType());

        m_derivative_hodge_star.emplace(*m_derivative_hodge_star_alloc);
        m_dual_derivative_hodge_star.emplace(*m_dual_derivative_hodge_star_alloc);
        m_derivative_dual_tensor_buffer.emplace(*m_derivative_dual_tensor_alloc);
        m_hodge_star.emplace(*m_hodge_star_alloc);
        m_dual_hodge_star.emplace(*m_dual_hodge_star_alloc);
        m_dual_tensor_buffer.emplace(*m_dual_tensor_alloc);
        m_codifferential_tensor_buffer.emplace(*m_codifferential_alloc);
        m_coboundary_of_codifferential_buffer.emplace(*m_coboundary_of_codifferential_alloc);

        fill_discrete_hodge_star<CoboundaryHodgeInputIndices, CoboundaryHodgeOutputIndices>(
                exec_space,
                *m_derivative_hodge_star,
                metric,
                position);
        fill_discrete_hodge_star<
                tensor::upper_t<DualCoboundaryHodgeInputIndices>,
                DualCoboundaryHodgeOutputIndices>(
                exec_space,
                *m_dual_derivative_hodge_star,
                metric,
                position);
        fill_discrete_hodge_star<
                CodifferentialHodgeInputIndices,
                CodifferentialHodgeOutputIndices>(exec_space, *m_hodge_star, metric, position);
        fill_discrete_hodge_star<
                tensor::upper_t<DualCodifferentialHodgeInputIndices>,
                DualCodifferentialHodgeOutputIndices>(
                exec_space,
                *m_dual_hodge_star,
                metric,
                position);
    }

    template <
            class PrimalExtrapolationRule = ClampCochainExtrapolationRule,
            class DualExtrapolationRule = ZeroCochainExtrapolationRule>
    TensorType operator()(
            TensorType laplacian_tensor,
            TensorType tensor,
            PrimalExtrapolationRule primal_extrapolation = {},
            DualExtrapolationRule dual_extrapolation = {})
    {
        auto exec_spaces = Kokkos::Experimental::partition_space(m_exec_space, 1, 1);

        detail::codifferential_of_coboundary<
                MetricIndex,
                CodifferentialOfCoboundaryIndex,
                CochainTag>(
                exec_spaces[0],
                laplacian_tensor,
                tensor,
                *m_derivative_hodge_star,
                *m_dual_derivative_hodge_star,
                *m_derivative_dual_tensor_buffer,
                primal_extrapolation,
                dual_extrapolation);

        StagedCodifferential<
                MetricIndex,
                LaplacianDummyIndex,
                CochainTag,
                TensorType,
                MetricType,
                PositionType,
                ExecSpace>(
                exec_spaces[1],
                std::move(*m_hodge_star),
                std::move(*m_dual_hodge_star),
                std::move(*m_dual_tensor_buffer))(
                *m_codifferential_tensor_buffer,
                tensor,
                dual_extrapolation);
        sil::exterior::
                deriv<LaplacianDummyIndex, codifferential_index_t<LaplacianDummyIndex, CochainTag>>(
                        exec_spaces[1],
                        *m_coboundary_of_codifferential_buffer,
                        *m_codifferential_tensor_buffer,
                        primal_extrapolation);

        exec_spaces[0].fence();
        exec_spaces[1].fence();

        auto coboundary_of_codifferential_buffer = *m_coboundary_of_codifferential_buffer;
        SIMILIE_DEBUG_LOG("similie_add_coboundary_of_codifferential_contribution_to_laplacian");
        ddc::parallel_for_each(
                "similie_add_coboundary_of_codifferential_contribution_to_laplacian",
                m_exec_space,
                laplacian_tensor.domain(),
                KOKKOS_LAMBDA(typename TensorType::discrete_element_type elem) {
                    laplacian_tensor.mem(elem) += coboundary_of_codifferential_buffer.mem(elem);
                });

        return laplacian_tensor;
    }
};

template <
        tensor::TensorIndex MetricIndex,
        tensor::TensorNatIndex LaplacianDummyIndex,
        tensor::TensorIndex CochainTag,
        misc::Specialization<tensor::Tensor> TensorType,
        misc::Specialization<tensor::Tensor> MetricType,
        misc::Specialization<tensor::Tensor> PositionType,
        class ExecSpace>
    requires(detail::TopRankLaplacianCochain<LaplacianDummyIndex, CochainTag>)
class StagedLaplacian<
        MetricIndex,
        LaplacianDummyIndex,
        CochainTag,
        TensorType,
        MetricType,
        PositionType,
        ExecSpace>
{
private:
    using MemorySpace = typename TensorType::memory_space;
    using AllocatorType = ddc::KokkosAllocator<double, MemorySpace>;
    using CodifferentialHodgeInputIndices
            = tensor::upper_t<ddc::to_type_seq_t<tensor::natural_domain_t<CochainTag>>>;
    using CodifferentialHodgeOutputIndices = codifferential_hodge_output_indices_t<
            LaplacianDummyIndex::size() - CochainTag::rank(),
            LaplacianDummyIndex>;
    using DualCodifferentialHodgeInputIndices = ddc::
            type_seq_merge_t<ddc::TypeSeq<LaplacianDummyIndex>, CodifferentialHodgeOutputIndices>;
    using DualCodifferentialHodgeOutputIndices = ddc::type_seq_remove_t<
            tensor::lower_t<CodifferentialHodgeInputIndices>,
            ddc::TypeSeq<LaplacianDummyIndex>>;
    using CodifferentialDualTensorIndex = misc::convert_type_seq_to_t<
            tensor::TensorAntisymmetricIndex,
            CodifferentialHodgeOutputIndices>;
    using CodifferentialOutputIndex = codifferential_index_t<LaplacianDummyIndex, CochainTag>;

    using HodgeStarDomainType = sil::misc::cartesian_prod_t<
            typename MetricType::non_indices_domain_t,
            hodge_star_domain_t<CodifferentialHodgeInputIndices, CodifferentialHodgeOutputIndices>>;
    using DualHodgeStarDomainType = sil::misc::cartesian_prod_t<
            typename MetricType::non_indices_domain_t,
            hodge_star_domain_t<
                    tensor::upper_t<DualCodifferentialHodgeInputIndices>,
                    DualCodifferentialHodgeOutputIndices>>;
    using DualTensorDomainType = sil::misc::cartesian_prod_t<
            typename TensorType::non_indices_domain_t,
            ddc::DiscreteDomain<CodifferentialDualTensorIndex>>;
    using CodifferentialDomainType = sil::misc::cartesian_prod_t<
            typename TensorType::non_indices_domain_t,
            ddc::DiscreteDomain<CodifferentialOutputIndex>>;

    using HodgeStarAllocType = ddc::Chunk<double, HodgeStarDomainType, AllocatorType>;
    using DualHodgeStarAllocType = ddc::Chunk<double, DualHodgeStarDomainType, AllocatorType>;
    using DualTensorAllocType = ddc::Chunk<double, DualTensorDomainType, AllocatorType>;
    using CodifferentialAllocType = ddc::Chunk<double, CodifferentialDomainType, AllocatorType>;

    using HodgeStarTensorType
            = tensor::Tensor<double, HodgeStarDomainType, Kokkos::layout_right, MemorySpace>;
    using DualHodgeStarTensorType
            = tensor::Tensor<double, DualHodgeStarDomainType, Kokkos::layout_right, MemorySpace>;
    using DualTensorType
            = tensor::Tensor<double, DualTensorDomainType, Kokkos::layout_right, MemorySpace>;
    using CodifferentialTensorType
            = tensor::Tensor<double, CodifferentialDomainType, Kokkos::layout_right, MemorySpace>;
    ExecSpace m_exec_space;
    std::optional<HodgeStarAllocType> m_hodge_star_alloc;
    std::optional<DualHodgeStarAllocType> m_dual_hodge_star_alloc;
    std::optional<DualTensorAllocType> m_dual_tensor_alloc;
    std::optional<CodifferentialAllocType> m_codifferential_alloc;
    std::optional<HodgeStarTensorType> m_hodge_star;
    std::optional<DualHodgeStarTensorType> m_dual_hodge_star;
    std::optional<DualTensorType> m_dual_tensor_buffer;
    std::optional<CodifferentialTensorType> m_codifferential_tensor_buffer;

public:
    StagedLaplacian(
            ExecSpace const& exec_space,
            HodgeStarTensorType&& hodge_star,
            DualHodgeStarTensorType&& dual_hodge_star,
            DualTensorType&& dual_tensor_buffer,
            CodifferentialTensorType&& codifferential_tensor_buffer)
        : m_exec_space(exec_space)
        , m_hodge_star(std::move(hodge_star))
        , m_dual_hodge_star(std::move(dual_hodge_star))
        , m_dual_tensor_buffer(std::move(dual_tensor_buffer))
        , m_codifferential_tensor_buffer(std::move(codifferential_tensor_buffer))
    {
    }

    StagedLaplacian(
            ExecSpace const& exec_space,
            TensorType,
            TensorType tensor,
            MetricType metric,
            PositionType position)
        : m_exec_space(exec_space)
    {
        [[maybe_unused]] tensor::tensor_accessor_for_domain_t<hodge_star_domain_t<
                CodifferentialHodgeInputIndices,
                CodifferentialHodgeOutputIndices>> hodge_star_accessor;
        [[maybe_unused]] tensor::tensor_accessor_for_domain_t<hodge_star_domain_t<
                tensor::upper_t<DualCodifferentialHodgeInputIndices>,
                DualCodifferentialHodgeOutputIndices>> dual_hodge_star_accessor;
        [[maybe_unused]] tensor::TensorAccessor<CodifferentialDualTensorIndex> dual_tensor_accessor;
        [[maybe_unused]] tensor::TensorAccessor<CodifferentialOutputIndex> codifferential_accessor;

        m_hodge_star_alloc.emplace(
                HodgeStarDomainType(metric.non_indices_domain(), hodge_star_accessor.domain()),
                AllocatorType());
        m_dual_hodge_star_alloc.emplace(
                DualHodgeStarDomainType(
                        metric.non_indices_domain(),
                        dual_hodge_star_accessor.domain()),
                AllocatorType());
        m_dual_tensor_alloc.emplace(
                DualTensorDomainType(tensor.non_indices_domain(), dual_tensor_accessor.domain()),
                AllocatorType());
        m_codifferential_alloc.emplace(
                CodifferentialDomainType(
                        tensor.non_indices_domain(),
                        codifferential_accessor.domain()),
                AllocatorType());

        m_hodge_star.emplace(*m_hodge_star_alloc);
        m_dual_hodge_star.emplace(*m_dual_hodge_star_alloc);
        m_dual_tensor_buffer.emplace(*m_dual_tensor_alloc);
        m_codifferential_tensor_buffer.emplace(*m_codifferential_alloc);

        fill_discrete_hodge_star<
                CodifferentialHodgeInputIndices,
                CodifferentialHodgeOutputIndices>(exec_space, *m_hodge_star, metric, position);
        fill_discrete_hodge_star<
                tensor::upper_t<DualCodifferentialHodgeInputIndices>,
                DualCodifferentialHodgeOutputIndices>(
                exec_space,
                *m_dual_hodge_star,
                metric,
                position);
    }

    template <
            class PrimalExtrapolationRule = ClampCochainExtrapolationRule,
            class DualExtrapolationRule = ZeroCochainExtrapolationRule>
    TensorType operator()(
            TensorType laplacian_tensor,
            TensorType tensor,
            PrimalExtrapolationRule primal_extrapolation = {},
            DualExtrapolationRule dual_extrapolation = {})
    {
        StagedCodifferential<
                MetricIndex,
                LaplacianDummyIndex,
                CochainTag,
                TensorType,
                MetricType,
                PositionType,
                ExecSpace>(
                m_exec_space,
                std::move(*m_hodge_star),
                std::move(*m_dual_hodge_star),
                std::move(*m_dual_tensor_buffer))(
                *m_codifferential_tensor_buffer,
                tensor,
                dual_extrapolation);
        return sil::exterior::
                deriv<LaplacianDummyIndex, codifferential_index_t<LaplacianDummyIndex, CochainTag>>(
                        m_exec_space,
                        laplacian_tensor,
                        *m_codifferential_tensor_buffer,
                        primal_extrapolation);
    }
};

template <
        tensor::TensorIndex MetricIndex,
        tensor::TensorNatIndex LaplacianDummyIndex,
        tensor::TensorIndex CochainTag,
        misc::Specialization<tensor::Tensor> TensorType,
        misc::Specialization<tensor::Tensor> DerivativeHodgeStarType,
        misc::Specialization<tensor::Tensor> DualDerivativeHodgeStarType,
        misc::Specialization<tensor::Tensor> DerivativeDualTensorBufferType,
        class ExecSpace,
        class PrimalExtrapolationRule = ClampCochainExtrapolationRule,
        class DualExtrapolationRule = ZeroCochainExtrapolationRule>
    requires(detail::ZeroRankLaplacianCochain<LaplacianDummyIndex, CochainTag>)
TensorType laplacian(
        ExecSpace const& exec_space,
        TensorType laplacian_tensor,
        TensorType tensor,
        DerivativeHodgeStarType hodge_star,
        DualDerivativeHodgeStarType dual_hodge_star,
        DerivativeDualTensorBufferType dual_tensor_buffer,
        PrimalExtrapolationRule primal_extrapolation = {},
        DualExtrapolationRule dual_extrapolation = {})
{
    using codifferential_of_coboundary_index
            = tensor::Covariant<IndexForCodifferentialOfCoboundaryInLaplacian<
                    tensor::uncharacterize_t<LaplacianDummyIndex>>>;
    return detail::codifferential_of_coboundary<
            MetricIndex,
            codifferential_of_coboundary_index,
            CochainTag>(
            exec_space,
            laplacian_tensor,
            tensor,
            hodge_star,
            dual_hodge_star,
            dual_tensor_buffer,
            primal_extrapolation,
            dual_extrapolation);
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
        class ExecSpace,
        class PrimalExtrapolationRule = ClampCochainExtrapolationRule,
        class DualExtrapolationRule = ZeroCochainExtrapolationRule>
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
        CoboundaryOfCodifferentialBufferType coboundary_of_codifferential_buffer,
        PrimalExtrapolationRule primal_extrapolation = {},
        DualExtrapolationRule dual_extrapolation = {})
{
    using codifferential_of_coboundary_index
            = tensor::Covariant<IndexForCodifferentialOfCoboundaryInLaplacian<
                    tensor::uncharacterize_t<LaplacianDummyIndex>>>;
    auto exec_spaces = Kokkos::Experimental::partition_space(exec_space, 1, 1);

    detail::codifferential_of_coboundary<
            MetricIndex,
            codifferential_of_coboundary_index,
            CochainTag>(
            exec_spaces[0],
            laplacian_tensor,
            tensor,
            derivative_hodge_star,
            dual_derivative_hodge_star,
            derivative_dual_tensor_buffer,
            primal_extrapolation,
            dual_extrapolation);

    sil::exterior::codifferential<MetricIndex, LaplacianDummyIndex, CochainTag>(
            exec_spaces[1],
            codifferential_tensor_buffer,
            tensor,
            hodge_star,
            dual_hodge_star,
            dual_tensor_buffer,
            dual_extrapolation);
    sil::exterior::
            deriv<LaplacianDummyIndex, codifferential_index_t<LaplacianDummyIndex, CochainTag>>(
                    exec_spaces[1],
                    coboundary_of_codifferential_buffer,
                    codifferential_tensor_buffer,
                    primal_extrapolation);

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
        class ExecSpace,
        class PrimalExtrapolationRule = ClampCochainExtrapolationRule,
        class DualExtrapolationRule = ZeroCochainExtrapolationRule>
    requires(detail::TopRankLaplacianCochain<LaplacianDummyIndex, CochainTag>)
TensorType laplacian(
        ExecSpace const& exec_space,
        TensorType laplacian_tensor,
        TensorType tensor,
        HodgeStarType hodge_star,
        DualHodgeStarType dual_hodge_star,
        DualTensorBufferType dual_tensor_buffer,
        CodifferentialTensorBufferType codifferential_tensor_buffer,
        PrimalExtrapolationRule primal_extrapolation = {},
        DualExtrapolationRule dual_extrapolation = {})
{
    sil::exterior::codifferential<MetricIndex, LaplacianDummyIndex, CochainTag>(
            exec_space,
            codifferential_tensor_buffer,
            tensor,
            hodge_star,
            dual_hodge_star,
            dual_tensor_buffer,
            dual_extrapolation);
    return sil::exterior::
            deriv<LaplacianDummyIndex, codifferential_index_t<LaplacianDummyIndex, CochainTag>>(
                    exec_space,
                    laplacian_tensor,
                    codifferential_tensor_buffer,
                    primal_extrapolation);
}

} // namespace exterior

} // namespace sil

namespace sil::exterior {
namespace detail {
template <class Rule>
inline constexpr bool is_interface_rule_v = [] {
    if constexpr (requires { Rule::IS_INTERFACE; })
        return Rule::IS_INTERFACE;
    else
        return false;
}();

template <class Memory>
struct LocalExecution
{
    using memory_space = Memory;
};

struct ZeroScalarSampler
{
    template <class Field, class Element, class Component>
    KOKKOS_FUNCTION double operator()(Field const&, Element, Component) const
    {
        return 0.0;
    }
};

template <std::size_t Axis, class Position, class Rules, class Element>
KOKKOS_FUNCTION void geometry_front(
        Position const& position,
        Rules const& rules,
        Element elem,
        Element& front)
{
    using Upper = std::remove_cvref_t<decltype(rules.template boundary_rule<Axis, true>())>;
    if constexpr (!is_interface_rule_v<Upper>) {
        if (ddc::detail::array(elem)[Axis]
                    == ddc::detail::array(position.non_indices_domain().back())[Axis]
            && ddc::detail::array(elem)[Axis]
                       > ddc::detail::array(position.non_indices_domain().front())[Axis])
            --ddc::detail::array(front)[Axis];
    }
}

template <class Position, class Rules, class Element, std::size_t... Axis>
KOKKOS_FUNCTION auto local_geometry(
        Position const& position,
        Rules const& rules,
        Element elem,
        std::index_sequence<Axis...>)
{
    using Index
            = ddc::type_seq_element_t<0, ddc::to_type_seq_t<typename Position::indices_domain_t>>;
    Element front = elem;
    (geometry_front<Axis>(position, rules, elem, front), ...);
    auto geometry = make_stencil<typename Position::memory_space, Index>(front);
    ddc::device_for_each(geometry.domain(), [&](auto sample) {
        geometry.mem(sample)
                = rules(position, Element(sample), ddc::DiscreteElement<Index>(sample));
    });
    return geometry;
}

template <class MemorySpace, class... Coordinates>
KOKKOS_FUNCTION auto euclidean_metric(ddc::TypeSeq<Coordinates...>)
{
    using Index = tensor::TensorIdentityIndex<
            tensor::Covariant<tensor::MetricIndex1<Coordinates...>>,
            tensor::Covariant<tensor::MetricIndex2<Coordinates...>>>;
    tensor::TensorAccessor<Index> accessor;
    // Identity tensors have no stored entries.
    return tensor::Tensor(
            ddc::ChunkSpan<
                    double,
                    ddc::DiscreteDomain<Index>,
                    Kokkos::layout_right,
                    MemorySpace>(nullptr, accessor.domain()));
}

template <class Sampler>
struct FluxSampler
{
    Sampler primal;
    template <class Flux, class Element, class Component>
    KOKKOS_FUNCTION double operator()(Flux const& flux, Element elem, Component component) const
    {
        return flux.value(primal, elem, component);
    }
};
} // namespace detail

/** Lazy constitutive dual cochain *d(phi) in Cartesian physical coordinates.
 * \important This operator and documentation is fully AI-generated.
 * Uses the ordinary Coboundary and DiscreteHodgeStar operators. Geometry samples
 * cross interfaces through their rules. At physical upper faces the outgoing
 * dual sample comes from the physical flux policy, including the zero wall law.
 */
template <
        class Direction,
        class Field,
        class Position,
        class PrimalRules,
        class GeometryRules,
        class BoundaryFluxRules>
struct ScalarLaplacianFlux
{
    using source_indices = tensor::upper_t<ddc::to_type_seq_t<tensor::natural_domain_t<Direction>>>;
    using dual_indices = codifferential_hodge_output_indices_t<Direction::size() - 1, Direction>;
    using component_type
            = misc::convert_type_seq_to_t<tensor::TensorAntisymmetricIndex, dual_indices>;
    using non_indices_domain_t = typename Field::non_indices_domain_t;
    using discrete_domain_type
            = misc::cartesian_prod_t<non_indices_domain_t, ddc::DiscreteDomain<component_type>>;
    Field field;
    Position position;
    PrimalRules primal;
    GeometryRules geometry;
    BoundaryFluxRules boundary_flux;
    double diffusivity;

    KOKKOS_FUNCTION non_indices_domain_t non_indices_domain() const
    {
        return field.non_indices_domain();
    }

    template <std::size_t Axis, class Element, class Component>
    KOKKOS_FUNCTION bool upper_flux(
            Element elem,
            Component component,
            std::size_t normal,
            double& result) const
    {
        using Rule = std::remove_cvref_t<decltype(primal.template boundary_rule<Axis, true>())>;
        if constexpr (!detail::is_interface_rule_v<Rule>) {
            if (normal == Axis
                && ddc::detail::array(elem)[Axis]
                           == ddc::detail::array(non_indices_domain().back())[Axis]) {
                ++ddc::detail::array(elem)[Axis];
                result = detail::EvaluateRuleValue<
                        detail::ZeroScalarSampler> {detail::ZeroScalarSampler {}}(
                        boundary_flux.template boundary_rule<Axis, true>(),
                        *this,
                        elem,
                        component);
                return true;
            }
        }
        return false;
    }

    template <class Sampler, class Element, class Component, std::size_t... Axis>
    KOKKOS_FUNCTION double evaluate(
            Sampler sampler,
            Element elem,
            Component component,
            std::index_sequence<Axis...> axes) const
    {
        tensor::TensorAccessor<component_type> dual_accessor;
        auto const ids
                = detail::flat_natural_elem_ids(dual_accessor.canonical_natural_element(component));
        std::size_t normal = 0;
        for (; normal < Direction::size(); ++normal) {
            bool found = false;
            for (std::size_t id : ids)
                found = found || id == normal;
            if (!found)
                break;
        }
        double prescribed = 0.0;
        if ((upper_flux<Axis>(elem, component, normal, prescribed) || ...))
            return prescribed;
        auto positions = detail::local_geometry(position, geometry, elem, axes);
        tensor::Tensor<
                double,
                typename Position::discrete_domain_type,
                Kokkos::layout_right,
                typename Position::memory_space>
                local_position(positions);
        auto metric = detail::euclidean_metric<typename Position::memory_space>(
                typename tensor::uncharacterize_t<Direction>::type_seq_dimensions {});
        auto derivative = detail::make_stencil<typename Position::memory_space, Direction>(
                ddc::DiscreteElement<>());
        auto flux = detail::make_stencil<typename Position::memory_space, component_type>(
                ddc::DiscreteElement<>());
        auto const chain = tangent_basis<1, non_indices_domain_t>(
                detail::LocalExecution<typename Position::memory_space> {});
        auto const lower = tangent_basis<0, non_indices_domain_t>(
                detail::LocalExecution<typename Position::memory_space> {});
        using Derivative = tensor::Tensor<
                double,
                ddc::DiscreteDomain<Direction>,
                Kokkos::layout_right,
                typename Position::memory_space>;
        using Flux = tensor::Tensor<
                double,
                ddc::DiscreteDomain<component_type>,
                Kokkos::layout_right,
                typename Position::memory_space>;
        Coboundary<
                Direction,
                ddc::type_seq_element_t<0, ddc::to_type_seq_t<typename Field::indices_domain_t>>>::
        operator()(
                Derivative(derivative),
                [&](auto sample, auto index) {
                    return primal.value(field, sampler, sample, index);
                },
                chain,
                lower,
                elem);
        DiscreteHodgeStar<
                CellComplex::CircumcentricDual,
                source_indices,
                dual_indices,
                decltype(metric),
                decltype(local_position),
                Element>::
        operator()(Flux(flux), Derivative(derivative), metric, local_position, elem);
        return diffusivity * flux.mem(component);
    }

    template <class Sampler, class Element, class Component>
    KOKKOS_FUNCTION double value(Sampler sampler, Element elem, Component component) const
    {
        return evaluate(
                sampler,
                elem,
                component,
                std::make_index_sequence<
                        ddc::type_seq_size_v<ddc::to_type_seq_t<non_indices_domain_t>>> {});
    }
    template <class Element, class Component>
    KOKKOS_FUNCTION double mem(Element elem, Component component) const
    {
        return value(StoredCochainSampler {}, elem, component);
    }
};

/** Normalize single policies locally, as for the staged differential operators.
 * \important This operator and documentation is fully AI-generated.
 */
template <class Direction, class Field, class Position, class Primal, class Geometry, class Flux>
auto make_scalar_laplacian_flux(
        Field field,
        Position const& position,
        Primal primal_rule,
        Geometry geometry_rule,
        Flux flux_rule,
        double diffusivity = 1.0)
{
#if defined(KOKKOS_ENABLE_CUDA)
    if constexpr (Kokkos::SpaceAccessibility<Kokkos::Cuda, typename Position::memory_space>::
                          accessible) {
        // Debug builds cannot always infer the full sampling/Hodge call depth.
        std::size_t stack_size = 0;
        if (cudaDeviceGetLimit(&stack_size, cudaLimitStackSize) != cudaSuccess
            || (stack_size < 32768 && cudaDeviceSetLimit(cudaLimitStackSize, 32768) != cudaSuccess))
            throw std::runtime_error("cannot reserve the CUDA stack for scalar DEC sampling");
    }
#endif
    auto const primal = [&] {
        if constexpr (misc::Specialization<Primal, ExtrapolationRules>)
            return primal_rule;
        else
            return make_extrapolation_rules(field, primal_rule);
    }();
    auto const geometry = [&] {
        if constexpr (misc::Specialization<Geometry, ExtrapolationRules>)
            return geometry_rule;
        else
            return make_extrapolation_rules(position, geometry_rule);
    }();
    auto const flux = [&] {
        if constexpr (misc::Specialization<Flux, ExtrapolationRules>)
            return flux_rule;
        else
            return make_extrapolation_rules(field, flux_rule);
    }();
    return ScalarLaplacianFlux<
            Direction,
            Field,
            Position,
            std::remove_cvref_t<decltype(primal)>,
            std::remove_cvref_t<decltype(geometry)>,
            std::remove_cvref_t<
                    decltype(flux)>> {field, position, primal, geometry, flux, diffusivity};
}

/** Point evaluation of the scalar DEC Laplacian, sharing the staged operator's
 * incidence, Hodge stars and codifferential sign convention.
 * \important This operator and documentation is fully AI-generated.
 * value(sampler, element) accepts a global basis sampler for sparse assembly;
 * operator()(element) evaluates stored cochains through the identical path.
 * Dual rules map to lazy donor fluxes, whose primal rules resolve donor unknowns
 * and affine jumps. Thus both stages follow the topology rather than a separate
 * interface assembly formula.
 */
template <class Direction, class Flux, class DualRules>
struct ScalarLaplacian
{
    Flux flux;
    DualRules dual;
    template <class Sampler, class Element>
    KOKKOS_FUNCTION double value(Sampler sampler, Element elem) const
    {
        using DualIndex = typename Flux::component_type;
        using TopIndices
                = ddc::type_seq_merge_t<ddc::TypeSeq<Direction>, typename Flux::dual_indices>;
        using TopIndex = misc::convert_type_seq_to_t<tensor::TensorAntisymmetricIndex, TopIndices>;
        using Scalar = tensor::Covariant<tensor::ScalarIndex>;
        using Memory = typename decltype(flux.position)::memory_space;
        auto top = detail::make_stencil<Memory, TopIndex>(ddc::DiscreteElement<>());
        auto scalar = detail::make_stencil<Memory, Scalar>(ddc::DiscreteElement<>());
        auto const chain = tangent_basis<Direction::size(), typename Flux::non_indices_domain_t>(
                detail::LocalExecution<Memory> {});
        auto const lower
                = tangent_basis<Direction::size() - 1, typename Flux::non_indices_domain_t>(
                        detail::LocalExecution<Memory> {});
        using TopTensor = tensor::
                Tensor<double, ddc::DiscreteDomain<TopIndex>, Kokkos::layout_right, Memory>;
        using ScalarTensor
                = tensor::Tensor<double, ddc::DiscreteDomain<Scalar>, Kokkos::layout_right, Memory>;
        TransposedCoboundary<Direction, DualIndex>::operator()(
                TopTensor(top),
                [&](auto sample, auto component) {
                    return dual
                            .value(flux, detail::FluxSampler<Sampler> {sampler}, sample, component);
                },
                chain,
                lower,
                elem);
        auto positions = detail::local_geometry(
                flux.position,
                flux.geometry,
                elem,
                std::make_index_sequence<Direction::size()> {});
        using Position = std::remove_cvref_t<decltype(flux.position)>;
        tensor::Tensor<
                double,
                typename Position::discrete_domain_type,
                Kokkos::layout_right,
                Memory>
                local_position(positions);
        auto metric = detail::euclidean_metric<Memory>(
                typename tensor::uncharacterize_t<Direction>::type_seq_dimensions {});
        DiscreteHodgeStar<
                CellComplex::CircumcentricDual,
                tensor::upper_t<TopIndices>,
                ddc::TypeSeq<>,
                decltype(metric),
                decltype(local_position),
                Element>::
        operator()(ScalarTensor(scalar), TopTensor(top), metric, local_position, elem);
        // The scalar codifferential sign is (-1)^(2*N + 1) in every dimension.
        return -scalar.mem(ddc::DiscreteElement<Scalar>(0));
    }
    template <class Element>
    KOKKOS_FUNCTION double operator()(Element elem) const
    {
        return value(StoredCochainSampler {}, elem);
    }
};
} // namespace sil::exterior
