// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: MIT

#pragma once

#include <array>

#include <ddc/ddc.hpp>

#include <similie/exterior/hodge_star.hpp>
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
struct CodifferentialTensorType<
        TagToRemoveFromCochain,
        CochainIndex,
        tensor::Tensor<ElementType, ddc::DiscreteDomain<DDim...>, SupportType, MemorySpace>>
{
    static_assert(ddc::type_seq_contains_v<
                  ddc::detail::TypeSeq<CochainIndex>,
                  ddc::detail::TypeSeq<DDim...>>);
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
        using MuUpSeq = tensor::upper_t<ddc::to_type_seq_t<tensor::natural_domain_t<CochainTag>>>;
        using NuLowSeq = typename detail::CodifferentialDummyIndexSeq<
                TagToRemoveFromCochain::size() - CochainTag::rank(),
                TagToRemoveFromCochain>::type;
        using RhoLowSeq
                = ddc::type_seq_merge_t<ddc::detail::TypeSeq<TagToRemoveFromCochain>, NuLowSeq>;
        using DualIndex = misc::convert_type_seq_to_t<tensor::TensorAntisymmetricIndex, NuLowSeq>;
        using DualCodifferentialIndex
                = misc::convert_type_seq_to_t<tensor::TensorAntisymmetricIndex, RhoLowSeq>;
        [[maybe_unused]] tensor::TensorAccessor<DualCodifferentialIndex>
                dual_codifferential_accessor;

        std::array<double, DualCodifferentialIndex::access_size()> dual_codifferential_alloc {};
        ddc::ChunkSpan<
                double,
                ddc::DiscreteDomain<DualCodifferentialIndex>,
                Kokkos::layout_right,
                typename TensorType::memory_space>
                dual_codifferential_span(
                        dual_codifferential_alloc.data(),
                        dual_codifferential_accessor.domain());

        sil::tensor::Tensor dual_codifferential(dual_codifferential_span);

        auto dual_evaluator = [&](auto sampled_elem, auto dual_elem) {
            auto const clamped_elem
                    = misc::clamp_to_domain(tensor.non_indices_domain(), sampled_elem);
            [[maybe_unused]] tensor::TensorAccessor<DualIndex> dual_tensor_accessor;
            static constexpr std::size_t DUAL_TENSOR_SIZE = DualIndex::access_size();
            std::array<double, DUAL_TENSOR_SIZE> dual_tensor_alloc {};
            ddc::ChunkSpan<
                    double,
                    ddc::DiscreteDomain<DualIndex>,
                    Kokkos::layout_right,
                    typename TensorType::memory_space>
                    dual_tensor_span(dual_tensor_alloc.data(), dual_tensor_accessor.domain());
            sil::tensor::Tensor dual_tensor(dual_tensor_span);

            DiscreteHodgeStar<
                    DualStrategy::Circumcentric,
                    MuUpSeq,
                    NuLowSeq,
                    MetricType,
                    PositionType,
                    typename TensorType::non_indices_domain_t::discrete_element_type>::
                    run(dual_tensor,
                        tensor[clamped_elem],
                        metric,
                        position,
                        clamped_elem); // Warning: there is redundancy here (neighbors threads apply hodge star to same tensor elements) but it is the only way to be able to compute the coboundary afterward without synchronization.
            return dual_tensor.mem(dual_elem);
        };

        Coboundary<TagToRemoveFromCochain, DualIndex>::
                run(dual_codifferential, dual_evaluator, chain, lower_chain, elem);

        DiscreteHodgeStar<
                DualStrategy::Circumcentric,
                tensor::upper_t<RhoLowSeq>,
                ddc::type_seq_remove_t<
                        tensor::lower_t<MuUpSeq>,
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
        misc::Specialization<tensor::Tensor> DualTensorType,
        misc::Specialization<tensor::Tensor> DualCodifferentialTensorType,
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
        DualTensorType dual_tensor_buffer,
        DualCodifferentialTensorType dual_codifferential_buffer)
{
    static_assert(tensor::is_covariant_v<TagToRemoveFromCochain>);
    using DualDummySeq = typename detail::CodifferentialDummyIndexSeq<
            TagToRemoveFromCochain::size() - CochainTag::rank(),
            TagToRemoveFromCochain>::type;
    using DualIndex = misc::convert_type_seq_to_t<tensor::TensorAntisymmetricIndex, DualDummySeq>;

    SIMILIE_DEBUG_LOG("similie_apply_first_hodge_star_for_codifferential");
    ddc::parallel_for_each(
            "similie_apply_first_hodge_star_for_codifferential",
            exec_space,
            dual_tensor_buffer.non_indices_domain(),
            KOKKOS_LAMBDA(typename TensorType::non_indices_domain_t::discrete_element_type elem) {
                sil::tensor::tensor_prod(dual_tensor_buffer[elem], tensor[elem], hodge_star[elem]);
            });

    sil::exterior::coboundary<
            TagToRemoveFromCochain,
            DualIndex>(exec_space, dual_codifferential_buffer, dual_tensor_buffer);

    SIMILIE_DEBUG_LOG("similie_apply_second_hodge_star_for_codifferential");
    ddc::parallel_for_each(
            "similie_apply_second_hodge_star_for_codifferential",
            exec_space,
            codifferential_tensor.non_indices_domain(),
            KOKKOS_LAMBDA(typename TensorType::non_indices_domain_t::discrete_element_type elem) {
                sil::tensor::tensor_prod(
                        codifferential_tensor[elem],
                        dual_codifferential_buffer[elem],
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
    using DualDummySeq = typename detail::CodifferentialDummyIndexSeq<
            TagToRemoveFromCochain::size() - CochainTag::rank(),
            TagToRemoveFromCochain>::type;
    using DualIndex = misc::convert_type_seq_to_t<tensor::TensorAntisymmetricIndex, DualDummySeq>;
    auto chain = tangent_basis<
            DualIndex::rank() + 1,
            typename detail::NonSpectatorDimension<
                    TagToRemoveFromCochain,
                    typename TensorType::non_indices_domain_t>::type>(exec_space);
    auto lower_chain = tangent_basis<
            DualIndex::rank(),
            typename detail::NonSpectatorDimension<
                    TagToRemoveFromCochain,
                    typename TensorType::non_indices_domain_t>::type>(exec_space);

    SIMILIE_DEBUG_LOG("similie_compute_codifferential");
    ddc::parallel_for_each(
            "similie_compute_codifferential",
            exec_space,
            codifferential_tensor.non_indices_domain(),
            KOKKOS_LAMBDA(typename TensorType::non_indices_domain_t::discrete_element_type elem) {
                Codifferential<
                        MetricIndex,
                        TagToRemoveFromCochain,
                        CochainTag,
                        TensorType,
                        MetricType,
                        PositionType>::
                        run(codifferential_tensor[elem],
                            tensor,
                            metric,
                            position,
                            chain,
                            lower_chain,
                            elem);
            });

    return codifferential_tensor;
}

} // namespace exterior

} // namespace sil
