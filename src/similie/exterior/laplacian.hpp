// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: MIT

#pragma once

#include <optional>

#include <ddc/ddc.hpp>

#include <similie/misc/domain_contains.hpp>
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
    using coboundary_output_index = coboundary_index_t<LaplacianDummyIndex, CochainTag>;
    using codifferential_hodge_output_indices = typename detail::CodifferentialDummyIndexSeq<
            LaplacianDummyIndex::size() - coboundary_output_index::rank(),
            LaplacianDummyIndex>::type;
    using coboundary_dual_tensor_index = misc::convert_type_seq_to_t<
            tensor::TensorAntisymmetricIndex,
            codifferential_hodge_output_indices>;
    using dual_codifferential_hodge_input_indices = ddc::type_seq_merge_t<
            ddc::detail::TypeSeq<LaplacianDummyIndex>,
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

                Coboundary<LaplacianDummyIndex, CochainTag>::run(
                        derivative_tensor,
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

                TransposedCoboundary<LaplacianDummyIndex, coboundary_dual_tensor_index>::run(
                        dual_codifferential,
                        [&](auto sampled_elem, auto dual_elem) {
                            if (!misc::domain_contains(
                                        dual_tensor_buffer.non_indices_domain(),
                                        sampled_elem)) {
                                return 0.0;
                            }
                            return dual_tensor_buffer.mem(sampled_elem, dual_elem);
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
    using CoboundaryCodifferentialIndex = tensor::Covariant<
            detail::LaplacianDummy2<tensor::uncharacterize_t<LaplacianDummyIndex>>>;
    using CoboundaryOutputIndex = coboundary_index_t<CoboundaryCodifferentialIndex, CochainTag>;
    using CoboundaryHodgeInputIndices
            = tensor::upper_t<ddc::to_type_seq_t<tensor::natural_domain_t<CoboundaryOutputIndex>>>;
    using CoboundaryHodgeOutputIndices = typename detail::CodifferentialDummyIndexSeq<
            CoboundaryCodifferentialIndex::size() - CoboundaryOutputIndex::rank(),
            CoboundaryCodifferentialIndex>::type;
    using DualCoboundaryHodgeInputIndices = ddc::type_seq_merge_t<
            ddc::detail::TypeSeq<CoboundaryCodifferentialIndex>,
            CoboundaryHodgeOutputIndices>;
    using DualCoboundaryHodgeOutputIndices = ddc::type_seq_remove_t<
            tensor::lower_t<CoboundaryHodgeInputIndices>,
            ddc::detail::TypeSeq<CoboundaryCodifferentialIndex>>;
    using CoboundaryDualTensorIndex = misc::
            convert_type_seq_to_t<tensor::TensorAntisymmetricIndex, CoboundaryHodgeOutputIndices>;

    using DerivativeHodgeStarDomainType = ddc::cartesian_prod_t<
            typename MetricType::non_indices_domain_t,
            hodge_star_domain_t<CoboundaryHodgeInputIndices, CoboundaryHodgeOutputIndices>>;
    using DualDerivativeHodgeStarDomainType = ddc::cartesian_prod_t<
            typename MetricType::non_indices_domain_t,
            hodge_star_domain_t<
                    tensor::upper_t<DualCoboundaryHodgeInputIndices>,
                    DualCoboundaryHodgeOutputIndices>>;
    using DerivativeDualTensorDomainType = ddc::cartesian_prod_t<
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
    StagedLaplacian(
            ExecSpace const& exec_space,
            DerivativeHodgeStarTensorType derivative_hodge_star,
            DualDerivativeHodgeStarTensorType dual_derivative_hodge_star,
            DerivativeDualTensorType derivative_dual_tensor_buffer)
        : m_exec_space(exec_space)
        , m_derivative_hodge_star(derivative_hodge_star)
        , m_dual_derivative_hodge_star(dual_derivative_hodge_star)
        , m_derivative_dual_tensor_buffer(derivative_dual_tensor_buffer)
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

    TensorType run(TensorType laplacian_tensor, TensorType tensor)
    {
        return detail::codifferential_of_coboundary<
                MetricIndex,
                CoboundaryCodifferentialIndex,
                CochainTag>(
                m_exec_space,
                laplacian_tensor,
                tensor,
                *m_derivative_hodge_star,
                *m_dual_derivative_hodge_star,
                *m_derivative_dual_tensor_buffer);
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
    using CoboundaryCodifferentialIndex = tensor::Covariant<
            detail::LaplacianDummy2<tensor::uncharacterize_t<LaplacianDummyIndex>>>;
    using CoboundaryOutputIndex = coboundary_index_t<CoboundaryCodifferentialIndex, CochainTag>;
    using CoboundaryHodgeInputIndices
            = tensor::upper_t<ddc::to_type_seq_t<tensor::natural_domain_t<CoboundaryOutputIndex>>>;
    using CoboundaryHodgeOutputIndices = typename detail::CodifferentialDummyIndexSeq<
            CoboundaryCodifferentialIndex::size() - CoboundaryOutputIndex::rank(),
            CoboundaryCodifferentialIndex>::type;
    using DualCoboundaryHodgeInputIndices = ddc::type_seq_merge_t<
            ddc::detail::TypeSeq<CoboundaryCodifferentialIndex>,
            CoboundaryHodgeOutputIndices>;
    using DualCoboundaryHodgeOutputIndices = ddc::type_seq_remove_t<
            tensor::lower_t<CoboundaryHodgeInputIndices>,
            ddc::detail::TypeSeq<CoboundaryCodifferentialIndex>>;
    using CoboundaryDualTensorIndex = misc::
            convert_type_seq_to_t<tensor::TensorAntisymmetricIndex, CoboundaryHodgeOutputIndices>;

    using CodifferentialHodgeInputIndices
            = tensor::upper_t<ddc::to_type_seq_t<tensor::natural_domain_t<CochainTag>>>;
    using CodifferentialHodgeOutputIndices = typename detail::CodifferentialDummyIndexSeq<
            LaplacianDummyIndex::size() - CochainTag::rank(),
            LaplacianDummyIndex>::type;
    using DualCodifferentialHodgeInputIndices = ddc::type_seq_merge_t<
            ddc::detail::TypeSeq<LaplacianDummyIndex>,
            CodifferentialHodgeOutputIndices>;
    using DualCodifferentialHodgeOutputIndices = ddc::type_seq_remove_t<
            tensor::lower_t<CodifferentialHodgeInputIndices>,
            ddc::detail::TypeSeq<LaplacianDummyIndex>>;
    using CodifferentialDualTensorIndex = misc::convert_type_seq_to_t<
            tensor::TensorAntisymmetricIndex,
            CodifferentialHodgeOutputIndices>;
    using CodifferentialOutputIndex = codifferential_index_t<LaplacianDummyIndex, CochainTag>;

    using DerivativeHodgeStarDomainType = ddc::cartesian_prod_t<
            typename MetricType::non_indices_domain_t,
            hodge_star_domain_t<CoboundaryHodgeInputIndices, CoboundaryHodgeOutputIndices>>;
    using DualDerivativeHodgeStarDomainType = ddc::cartesian_prod_t<
            typename MetricType::non_indices_domain_t,
            hodge_star_domain_t<
                    tensor::upper_t<DualCoboundaryHodgeInputIndices>,
                    DualCoboundaryHodgeOutputIndices>>;
    using DerivativeDualTensorDomainType = ddc::cartesian_prod_t<
            typename TensorType::non_indices_domain_t,
            ddc::DiscreteDomain<CoboundaryDualTensorIndex>>;
    using HodgeStarDomainType = ddc::cartesian_prod_t<
            typename MetricType::non_indices_domain_t,
            hodge_star_domain_t<CodifferentialHodgeInputIndices, CodifferentialHodgeOutputIndices>>;
    using DualHodgeStarDomainType = ddc::cartesian_prod_t<
            typename MetricType::non_indices_domain_t,
            hodge_star_domain_t<
                    tensor::upper_t<DualCodifferentialHodgeInputIndices>,
                    DualCodifferentialHodgeOutputIndices>>;
    using DualTensorDomainType = ddc::cartesian_prod_t<
            typename TensorType::non_indices_domain_t,
            ddc::DiscreteDomain<CodifferentialDualTensorIndex>>;
    using CodifferentialDomainType = ddc::cartesian_prod_t<
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

    TensorType run(TensorType laplacian_tensor, TensorType tensor)
    {
        auto exec_spaces = Kokkos::Experimental::partition_space(m_exec_space, 1, 1);

        detail::codifferential_of_coboundary<
                MetricIndex,
                CoboundaryCodifferentialIndex,
                CochainTag>(
                exec_spaces[0],
                laplacian_tensor,
                tensor,
                *m_derivative_hodge_star,
                *m_dual_derivative_hodge_star,
                *m_derivative_dual_tensor_buffer);

        StagedCodifferential<
                MetricIndex,
                LaplacianDummyIndex,
                CochainTag,
                TensorType,
                MetricType,
                PositionType,
                ExecSpace>(exec_spaces[1], *m_hodge_star, *m_dual_hodge_star, *m_dual_tensor_buffer)
                .run(*m_codifferential_tensor_buffer, tensor);
        sil::exterior::
                deriv<LaplacianDummyIndex, codifferential_index_t<LaplacianDummyIndex, CochainTag>>(
                        exec_spaces[1],
                        *m_coboundary_of_codifferential_buffer,
                        *m_codifferential_tensor_buffer);

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
    using CodifferentialHodgeOutputIndices = typename detail::CodifferentialDummyIndexSeq<
            LaplacianDummyIndex::size() - CochainTag::rank(),
            LaplacianDummyIndex>::type;
    using DualCodifferentialHodgeInputIndices = ddc::type_seq_merge_t<
            ddc::detail::TypeSeq<LaplacianDummyIndex>,
            CodifferentialHodgeOutputIndices>;
    using DualCodifferentialHodgeOutputIndices = ddc::type_seq_remove_t<
            tensor::lower_t<CodifferentialHodgeInputIndices>,
            ddc::detail::TypeSeq<LaplacianDummyIndex>>;
    using CodifferentialDualTensorIndex = misc::convert_type_seq_to_t<
            tensor::TensorAntisymmetricIndex,
            CodifferentialHodgeOutputIndices>;
    using CodifferentialOutputIndex = codifferential_index_t<LaplacianDummyIndex, CochainTag>;

    using HodgeStarDomainType = ddc::cartesian_prod_t<
            typename MetricType::non_indices_domain_t,
            hodge_star_domain_t<CodifferentialHodgeInputIndices, CodifferentialHodgeOutputIndices>>;
    using DualHodgeStarDomainType = ddc::cartesian_prod_t<
            typename MetricType::non_indices_domain_t,
            hodge_star_domain_t<
                    tensor::upper_t<DualCodifferentialHodgeInputIndices>,
                    DualCodifferentialHodgeOutputIndices>>;
    using DualTensorDomainType = ddc::cartesian_prod_t<
            typename TensorType::non_indices_domain_t,
            ddc::DiscreteDomain<CodifferentialDualTensorIndex>>;
    using CodifferentialDomainType = ddc::cartesian_prod_t<
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
            HodgeStarTensorType hodge_star,
            DualHodgeStarTensorType dual_hodge_star,
            DualTensorType dual_tensor_buffer,
            CodifferentialTensorType codifferential_tensor_buffer)
        : m_exec_space(exec_space)
        , m_hodge_star(hodge_star)
        , m_dual_hodge_star(dual_hodge_star)
        , m_dual_tensor_buffer(dual_tensor_buffer)
        , m_codifferential_tensor_buffer(codifferential_tensor_buffer)
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

    TensorType run(TensorType laplacian_tensor, TensorType tensor)
    {
        StagedCodifferential<
                MetricIndex,
                LaplacianDummyIndex,
                CochainTag,
                TensorType,
                MetricType,
                PositionType,
                ExecSpace>(m_exec_space, *m_hodge_star, *m_dual_hodge_star, *m_dual_tensor_buffer)
                .run(*m_codifferential_tensor_buffer, tensor);
        return sil::exterior::
                deriv<LaplacianDummyIndex, codifferential_index_t<LaplacianDummyIndex, CochainTag>>(
                        m_exec_space,
                        laplacian_tensor,
                        *m_codifferential_tensor_buffer);
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
    using coboundary_codifferential_index = tensor::Covariant<
            detail::LaplacianDummy2<tensor::uncharacterize_t<LaplacianDummyIndex>>>;
    return detail::
            codifferential_of_coboundary<MetricIndex, coboundary_codifferential_index, CochainTag>(
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
    using coboundary_codifferential_index = tensor::Covariant<
            detail::LaplacianDummy2<tensor::uncharacterize_t<LaplacianDummyIndex>>>;
    auto exec_spaces = Kokkos::Experimental::partition_space(exec_space, 1, 1);

    detail::codifferential_of_coboundary<MetricIndex, coboundary_codifferential_index, CochainTag>(
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
