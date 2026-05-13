// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <optional>

#include <ddc/ddc.hpp>

#include <similie/exterior/hodge_star.hpp>
#include <similie/misc/domain_contains.hpp>
#include <similie/misc/macros.hpp>
#include <similie/misc/specialization.hpp>
#include <similie/tensor/character.hpp>
#include <similie/tensor/tensor_impl.hpp>

#include <Kokkos_StdAlgorithms.hpp>

#include "coboundary.hpp"
#include "cochain.hpp"
#include "cosimplex.hpp"


namespace sil {

namespace exterior {

namespace detail {

template <class T>
struct CodifferentialType;

template <
        std::size_t K,
        class... Tag,
        class ElementType,
        class LayoutStridedPolicy1,
        class LayoutStridedPolicy2,
        class ExecSpace>
struct CodifferentialType<
        Cochain<Chain<Simplex<K, Tag...>, LayoutStridedPolicy1, ExecSpace>,
                ElementType,
                LayoutStridedPolicy2>>
{
    using type = Cosimplex<Simplex<K - 1, Tag...>, ElementType>;
};

} // namespace detail

template <misc::Specialization<Cochain> CochainType>
using codifferential_t = typename detail::CodifferentialType<CochainType>::type;

namespace detail {

template <class TagToRemoveFromCochain, class CochainTag>
struct CodifferentialIndex;

template <tensor::TensorNatIndex TagToRemoveFromCochain, tensor::TensorNatIndex CochainTag>
    requires(CochainTag::rank() == 1 && std::is_same_v<TagToRemoveFromCochain, CochainTag>)
struct CodifferentialIndex<TagToRemoveFromCochain, CochainTag>
{
    using type = tensor::Covariant<tensor::ScalarIndex>;
};

template <tensor::TensorNatIndex TagToRemoveFromCochain, tensor::TensorNatIndex Tag>
struct CodifferentialIndex<
        TagToRemoveFromCochain,
        tensor::TensorAntisymmetricIndex<TagToRemoveFromCochain, Tag>>
{
    using type = Tag;
};

template <tensor::TensorNatIndex TagToRemoveFromCochain, tensor::TensorNatIndex... Tag>
    requires(sizeof...(Tag) > 1)
struct CodifferentialIndex<
        TagToRemoveFromCochain,
        tensor::TensorAntisymmetricIndex<TagToRemoveFromCochain, Tag...>>
{
    using type = tensor::TensorAntisymmetricIndex<Tag...>;
};

} // namespace detail

template <class TagToRemoveFromCochain, class CochainTag>
using codifferential_index_t =
        typename detail::CodifferentialIndex<TagToRemoveFromCochain, CochainTag>::type;

namespace detail {

template <
        tensor::TensorNatIndex TagToRemoveFromCochain,
        tensor::TensorIndex CochainTag,
        misc::Specialization<tensor::Tensor> TensorType>
struct CodifferentialTensorType;

template <
        tensor::TensorNatIndex TagToRemoveFromCochain,
        tensor::TensorIndex CochainIndex,
        class ElementType,
        class... DDim,
        class SupportType,
        class MemorySpace>
    requires(ddc::type_seq_contains_v<
             ddc::detail::TypeSeq<CochainIndex>,
             ddc::detail::TypeSeq<DDim...>>)
struct CodifferentialTensorType<
        TagToRemoveFromCochain,
        CochainIndex,
        tensor::Tensor<ElementType, ddc::DiscreteDomain<DDim...>, SupportType, MemorySpace>>
{
    using type = tensor::Tensor<
            ElementType,
            ddc::replace_dim_of_t<
                    ddc::DiscreteDomain<DDim...>,
                    CochainIndex,
                    codifferential_index_t<TagToRemoveFromCochain, CochainIndex>>,
            SupportType,
            MemorySpace>;
};

} // namespace detail

template <
        tensor::TensorNatIndex TagToRemoveFromCochain,
        tensor::TensorIndex CochainTag,
        misc::Specialization<tensor::Tensor> TensorType>
using codifferential_tensor_t = typename detail::
        CodifferentialTensorType<TagToRemoveFromCochain, CochainTag, TensorType>::type;

namespace detail {

template <std::size_t I, class T>
struct CodifferentialDummyIndex : tensor::uncharacterize_t<T>
{
};

template <class Ids, class T>
struct CodifferentialDummyIndexSeq_;

template <std::size_t... Id, class T>
struct CodifferentialDummyIndexSeq_<std::index_sequence<Id...>, T>
{
    using type = ddc::detail::TypeSeq<tensor::Covariant<CodifferentialDummyIndex<Id, T>>...>;
};

template <std::size_t EndId, class T>
struct CodifferentialDummyIndexSeq;

template <std::size_t EndId, class T>
    requires(EndId == 0)
struct CodifferentialDummyIndexSeq<EndId, T>
{
    using type = ddc::detail::TypeSeq<>;
};

template <std::size_t EndId, class T>
    requires(EndId > 0)
struct CodifferentialDummyIndexSeq<EndId, T>
{
    using type = typename CodifferentialDummyIndexSeq_<std::make_index_sequence<EndId>, T>::type;
};

} // namespace detail

template <class... Args>
struct Codifferential;

template <
        tensor::TensorIndex MetricIndex,
        tensor::TensorNatIndex TagToRemoveFromCochain,
        tensor::TensorIndex CochainTag,
        misc::Specialization<tensor::Tensor> TensorType,
        misc::Specialization<tensor::Tensor> MetricType,
        misc::Specialization<tensor::Tensor> PositionType>
struct Codifferential<
        MetricIndex,
        TagToRemoveFromCochain,
        CochainTag,
        TensorType,
        MetricType,
        PositionType>
{
    KOKKOS_FUNCTION static void run(
            auto codifferential_tensor,
            TensorType tensor,
            MetricType metric,
            PositionType position,
            auto chain,
            auto lower_chain,
            typename TensorType::non_indices_domain_t::discrete_element_type elem)
    {
        using source_hodge_input_indices
                = tensor::upper_t<ddc::to_type_seq_t<tensor::natural_domain_t<CochainTag>>>;
        using source_hodge_output_indices = typename detail::CodifferentialDummyIndexSeq<
                TagToRemoveFromCochain::size() - CochainTag::rank(),
                TagToRemoveFromCochain>::type;
        using target_hodge_input_indices = ddc::type_seq_merge_t<
                ddc::detail::TypeSeq<TagToRemoveFromCochain>,
                source_hodge_output_indices>;
        using dual_tensor_index = misc::convert_type_seq_to_t<
                tensor::TensorAntisymmetricIndex,
                source_hodge_output_indices>;
        using dual_codifferential_index = misc::
                convert_type_seq_to_t<tensor::TensorAntisymmetricIndex, target_hodge_input_indices>;
        [[maybe_unused]] tensor::TensorAccessor<dual_codifferential_index>
                dual_codifferential_accessor;

        std::array<double, dual_codifferential_index::access_size()> dual_codifferential_alloc {};
        ddc::ChunkSpan<
                double,
                ddc::DiscreteDomain<dual_codifferential_index>,
                Kokkos::layout_right,
                typename TensorType::memory_space>
                dual_codifferential_span(
                        dual_codifferential_alloc.data(),
                        dual_codifferential_accessor.domain());

        sil::tensor::Tensor dual_codifferential(dual_codifferential_span);

        auto dual_evaluator = [&](auto sampled_elem, auto dual_elem) {
            if (!misc::domain_contains(tensor.non_indices_domain(), sampled_elem)) {
                return 0.0;
            }
            [[maybe_unused]] tensor::TensorAccessor<dual_tensor_index> dual_tensor_accessor;
            static constexpr std::size_t DUAL_TENSOR_SIZE = dual_tensor_index::access_size();
            std::array<double, DUAL_TENSOR_SIZE> dual_tensor_alloc {};
            ddc::ChunkSpan<
                    double,
                    ddc::DiscreteDomain<dual_tensor_index>,
                    Kokkos::layout_right,
                    typename TensorType::memory_space>
                    dual_tensor_span(dual_tensor_alloc.data(), dual_tensor_accessor.domain());
            sil::tensor::Tensor dual_tensor(dual_tensor_span);

            DiscreteHodgeStar<
                    CellComplex::CircumcentricDual,
                    source_hodge_input_indices,
                    source_hodge_output_indices,
                    MetricType,
                    PositionType,
                    typename TensorType::non_indices_domain_t::discrete_element_type>::
                    run(dual_tensor, tensor[sampled_elem], metric, position, sampled_elem);
            return dual_tensor.mem(dual_elem);
        };

        TransposedCoboundary<TagToRemoveFromCochain, dual_tensor_index>::
                run(dual_codifferential, dual_evaluator, chain, lower_chain, elem);

        DiscreteHodgeStar<
                CellComplex::CircumcentricDual,
                tensor::upper_t<target_hodge_input_indices>,
                ddc::type_seq_remove_t<
                        tensor::lower_t<source_hodge_input_indices>,
                        ddc::detail::TypeSeq<TagToRemoveFromCochain>>,
                MetricType,
                PositionType,
                typename TensorType::non_indices_domain_t::discrete_element_type>::
                run(codifferential_tensor, dual_codifferential, metric, position, elem);
        if constexpr ((TagToRemoveFromCochain::size() * (CochainTag::rank() + 1) + 1) % 2 == 1) {
            codifferential_tensor *= -1;
        }
    }
};

template <
        tensor::TensorIndex MetricIndex,
        tensor::TensorNatIndex TagToRemoveFromCochain,
        tensor::TensorIndex CochainTag,
        misc::Specialization<tensor::Tensor> TensorType,
        misc::Specialization<tensor::Tensor> MetricType,
        misc::Specialization<tensor::Tensor> PositionType,
        class ExecSpace>
class StagedCodifferential
{
    using CodifferentialTensorType
            = codifferential_tensor_t<TagToRemoveFromCochain, CochainTag, TensorType>;
    using MemorySpace = typename TensorType::memory_space;
    using AllocatorType = ddc::KokkosAllocator<double, MemorySpace>;
    using SourceHodgeInputIndices
            = tensor::upper_t<ddc::to_type_seq_t<tensor::natural_domain_t<CochainTag>>>;
    using SourceHodgeOutputIndices = typename detail::CodifferentialDummyIndexSeq<
            TagToRemoveFromCochain::size() - CochainTag::rank(),
            TagToRemoveFromCochain>::type;
    using TargetHodgeInputIndices = ddc::type_seq_merge_t<
            ddc::detail::TypeSeq<TagToRemoveFromCochain>,
            SourceHodgeOutputIndices>;
    using TargetHodgeOutputIndices = ddc::type_seq_remove_t<
            tensor::lower_t<SourceHodgeInputIndices>,
            ddc::detail::TypeSeq<TagToRemoveFromCochain>>;
    using DualTensorIndex = misc::
            convert_type_seq_to_t<tensor::TensorAntisymmetricIndex, SourceHodgeOutputIndices>;
    using DualCodifferentialIndex = misc::
            convert_type_seq_to_t<tensor::TensorAntisymmetricIndex, TargetHodgeInputIndices>;
    using NonSpectatorDimensions = typename detail::NonSpectatorDimension<
            TagToRemoveFromCochain,
            typename TensorType::non_indices_domain_t>::type;
    using ChainType = decltype(tangent_basis<DualTensorIndex::rank() + 1, NonSpectatorDimensions>(
            std::declval<ExecSpace const&>()));
    using LowerChainType = decltype(tangent_basis<DualTensorIndex::rank(), NonSpectatorDimensions>(
            std::declval<ExecSpace const&>()));

    using HodgeStarDomainType = ddc::cartesian_prod_t<
            typename MetricType::non_indices_domain_t,
            hodge_star_domain_t<SourceHodgeInputIndices, SourceHodgeOutputIndices>>;
    using DualHodgeStarDomainType = ddc::cartesian_prod_t<
            typename MetricType::non_indices_domain_t,
            hodge_star_domain_t<
                    tensor::upper_t<TargetHodgeInputIndices>,
                    TargetHodgeOutputIndices>>;
    using DualTensorDomainType = ddc::cartesian_prod_t<
            typename TensorType::non_indices_domain_t,
            ddc::DiscreteDomain<DualTensorIndex>>;

    using HodgeStarAllocType = ddc::Chunk<double, HodgeStarDomainType, AllocatorType>;
    using DualHodgeStarAllocType = ddc::Chunk<double, DualHodgeStarDomainType, AllocatorType>;
    using DualTensorAllocType = ddc::Chunk<double, DualTensorDomainType, AllocatorType>;

    using HodgeStarTensorType
            = tensor::Tensor<double, HodgeStarDomainType, Kokkos::layout_right, MemorySpace>;
    using DualHodgeStarTensorType
            = tensor::Tensor<double, DualHodgeStarDomainType, Kokkos::layout_right, MemorySpace>;
    using DualTensorType
            = tensor::Tensor<double, DualTensorDomainType, Kokkos::layout_right, MemorySpace>;

    ExecSpace m_exec_space;
    std::optional<HodgeStarAllocType> m_hodge_star_alloc;
    std::optional<DualHodgeStarAllocType> m_dual_hodge_star_alloc;
    std::optional<DualTensorAllocType> m_dual_tensor_alloc;
    std::optional<HodgeStarTensorType> m_hodge_star;
    std::optional<DualHodgeStarTensorType> m_dual_hodge_star;
    std::optional<DualTensorType> m_dual_tensor_buffer;
    ChainType m_chain;
    LowerChainType m_lower_chain;

public:
    StagedCodifferential(
            ExecSpace const& exec_space,
            HodgeStarTensorType hodge_star,
            DualHodgeStarTensorType dual_hodge_star,
            DualTensorType dual_tensor_buffer)
        : m_exec_space(exec_space)
        , m_hodge_star(hodge_star)
        , m_dual_hodge_star(dual_hodge_star)
        , m_dual_tensor_buffer(dual_tensor_buffer)
        , m_chain(tangent_basis<DualTensorIndex::rank() + 1, NonSpectatorDimensions>(exec_space))
        , m_lower_chain(tangent_basis<DualTensorIndex::rank(), NonSpectatorDimensions>(exec_space))
    {
    }

    StagedCodifferential(
            ExecSpace const& exec_space,
            TensorType tensor,
            MetricType metric,
            PositionType position)
        : m_exec_space(exec_space)
        , m_chain(tangent_basis<DualTensorIndex::rank() + 1, NonSpectatorDimensions>(exec_space))
        , m_lower_chain(tangent_basis<DualTensorIndex::rank(), NonSpectatorDimensions>(exec_space))
    {
        [[maybe_unused]] tensor::tensor_accessor_for_domain_t<
                hodge_star_domain_t<SourceHodgeInputIndices, SourceHodgeOutputIndices>>
                hodge_star_accessor;
        [[maybe_unused]] tensor::tensor_accessor_for_domain_t<hodge_star_domain_t<
                tensor::upper_t<TargetHodgeInputIndices>,
                TargetHodgeOutputIndices>> dual_hodge_star_accessor;
        [[maybe_unused]] tensor::TensorAccessor<DualTensorIndex> dual_tensor_accessor;

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

        m_hodge_star.emplace(*m_hodge_star_alloc);
        m_dual_hodge_star.emplace(*m_dual_hodge_star_alloc);
        m_dual_tensor_buffer.emplace(*m_dual_tensor_alloc);

        fill_discrete_hodge_star<
                SourceHodgeInputIndices,
                SourceHodgeOutputIndices>(exec_space, *m_hodge_star, metric, position);
        fill_discrete_hodge_star<
                tensor::upper_t<TargetHodgeInputIndices>,
                TargetHodgeOutputIndices>(exec_space, *m_dual_hodge_star, metric, position);
    }

    CodifferentialTensorType run(CodifferentialTensorType codifferential_tensor, TensorType tensor)
            const
    {
        auto exec_space = m_exec_space;
        auto hodge_star = *m_hodge_star;
        auto dual_hodge_star = *m_dual_hodge_star;
        auto dual_tensor_buffer = *m_dual_tensor_buffer;
        auto chain = m_chain;
        auto lower_chain = m_lower_chain;

        SIMILIE_DEBUG_LOG("similie_apply_first_hodge_star_for_codifferential");
        ddc::parallel_for_each(
                "similie_apply_first_hodge_star_for_codifferential",
                exec_space,
                dual_tensor_buffer.non_indices_domain(),
                KOKKOS_LAMBDA(
                        typename TensorType::non_indices_domain_t::discrete_element_type elem) {
                    sil::tensor::
                            tensor_prod(dual_tensor_buffer[elem], tensor[elem], hodge_star[elem]);
                });

        SIMILIE_DEBUG_LOG("similie_apply_second_hodge_star_for_codifferential");
        ddc::parallel_for_each(
                "similie_apply_second_hodge_star_for_codifferential",
                exec_space,
                codifferential_tensor.non_indices_domain(),
                KOKKOS_LAMBDA(
                        typename TensorType::non_indices_domain_t::discrete_element_type elem) {
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

                    TransposedCoboundary<TagToRemoveFromCochain, DualTensorIndex>::run(
                            dual_codifferential,
                            [&](auto sampled_elem, auto dual_elem) {
                                if (!misc::domain_contains(
                                            dual_tensor_buffer.non_indices_domain(),
                                            sampled_elem)) {
                                    return 0.0;
                                }
                                return dual_tensor_buffer.mem(sampled_elem, dual_elem);
                            },
                            chain,
                            lower_chain,
                            elem);

                    sil::tensor::tensor_prod(
                            codifferential_tensor[elem],
                            dual_codifferential,
                            dual_hodge_star[elem]);
                    if constexpr (
                            (TagToRemoveFromCochain::size() * (CochainTag::rank() + 1) + 1) % 2
                            == 1) {
                        codifferential_tensor[elem] *= -1;
                    }
                });

        return codifferential_tensor;
    }
};

template <
        tensor::TensorIndex MetricIndex,
        tensor::TensorNatIndex TagToRemoveFromCochain,
        tensor::TensorIndex CochainTag,
        misc::Specialization<tensor::Tensor> DualTensorType,
        misc::Specialization<tensor::Tensor> TensorType,
        misc::Specialization<tensor::Tensor> HodgeStarType,
        misc::Specialization<tensor::Tensor> DualHodgeStarType,
        class ExecSpace>
codifferential_tensor_t<TagToRemoveFromCochain, CochainTag, TensorType> codifferential(
        ExecSpace const& exec_space,
        codifferential_tensor_t<TagToRemoveFromCochain, CochainTag, TensorType>
                codifferential_tensor,
        TensorType tensor,
        HodgeStarType hodge_star,
        DualHodgeStarType dual_hodge_star,
        DualTensorType dual_tensor_buffer)
{
    static_assert(tensor::is_covariant_v<TagToRemoveFromCochain>);
    using source_hodge_output_indices = typename detail::CodifferentialDummyIndexSeq<
            TagToRemoveFromCochain::size() - CochainTag::rank(),
            TagToRemoveFromCochain>::type;
    using dual_tensor_index = misc::
            convert_type_seq_to_t<tensor::TensorAntisymmetricIndex, source_hodge_output_indices>;
    using target_hodge_input_indices = ddc::type_seq_merge_t<
            ddc::detail::TypeSeq<TagToRemoveFromCochain>,
            source_hodge_output_indices>;
    using dual_codifferential_index = misc::
            convert_type_seq_to_t<tensor::TensorAntisymmetricIndex, target_hodge_input_indices>;
    using non_spectator_dimensions = typename detail::NonSpectatorDimension<
            TagToRemoveFromCochain,
            typename TensorType::non_indices_domain_t>::type;
    auto chain = tangent_basis<dual_tensor_index::rank() + 1, non_spectator_dimensions>(exec_space);
    auto lower_chain
            = tangent_basis<dual_tensor_index::rank(), non_spectator_dimensions>(exec_space);

    SIMILIE_DEBUG_LOG("similie_apply_first_hodge_star_for_codifferential");
    ddc::parallel_for_each(
            "similie_apply_first_hodge_star_for_codifferential",
            exec_space,
            dual_tensor_buffer.non_indices_domain(),
            KOKKOS_LAMBDA(typename TensorType::non_indices_domain_t::discrete_element_type elem) {
                sil::tensor::tensor_prod(dual_tensor_buffer[elem], tensor[elem], hodge_star[elem]);
            });

    SIMILIE_DEBUG_LOG("similie_apply_second_hodge_star_for_codifferential");
    ddc::parallel_for_each(
            "similie_apply_second_hodge_star_for_codifferential",
            exec_space,
            codifferential_tensor.non_indices_domain(),
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

                TransposedCoboundary<TagToRemoveFromCochain, dual_tensor_index>::run(
                        dual_codifferential,
                        [&](auto sampled_elem, auto dual_elem) {
                            if (!misc::domain_contains(
                                        dual_tensor_buffer.non_indices_domain(),
                                        sampled_elem)) {
                                return 0.0;
                            }
                            return dual_tensor_buffer.mem(sampled_elem, dual_elem);
                        },
                        chain,
                        lower_chain,
                        elem);

                sil::tensor::tensor_prod(
                        codifferential_tensor[elem],
                        dual_codifferential,
                        dual_hodge_star[elem]);
                if constexpr (
                        (TagToRemoveFromCochain::size() * (CochainTag::rank() + 1) + 1) % 2 == 1) {
                    codifferential_tensor[elem] *= -1;
                }
            });

    return codifferential_tensor;
}

template <
        tensor::TensorIndex MetricIndex,
        tensor::TensorNatIndex TagToRemoveFromCochain,
        tensor::TensorIndex CochainTag,
        misc::Specialization<tensor::Tensor> TensorType,
        misc::Specialization<tensor::Tensor> MetricType,
        misc::Specialization<tensor::Tensor> PositionType,
        class ExecSpace>
codifferential_tensor_t<TagToRemoveFromCochain, CochainTag, TensorType> codifferential(
        ExecSpace const& exec_space,
        codifferential_tensor_t<TagToRemoveFromCochain, CochainTag, TensorType>
                codifferential_tensor,
        TensorType tensor,
        MetricType metric,
        PositionType position)
{
    static_assert(tensor::is_covariant_v<TagToRemoveFromCochain>);
    return StagedCodifferential<
                   MetricIndex,
                   TagToRemoveFromCochain,
                   CochainTag,
                   TensorType,
                   MetricType,
                   PositionType,
                   ExecSpace>(exec_space, tensor, metric, position)
            .run(codifferential_tensor, tensor);
}

} // namespace exterior

} // namespace sil
