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

namespace detail {

template <tensor::TensorNatIndex TagToRemoveFromCochain, tensor::TensorIndex CochainTag>
struct CodifferentialValue
{
    KOKKOS_FUNCTION static void run(auto output, auto dual_codifferential, auto hodge_star)
    {
        sil::tensor::tensor_prod(output, dual_codifferential, hodge_star);
        if constexpr ((TagToRemoveFromCochain::size() * (CochainTag::rank() + 1) + 1) % 2 == 1) {
            output *= -1;
        }
    }
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
        misc::Specialization<tensor::Tensor> PositionType,
        class ExecSpace>
struct Codifferential<
        MetricIndex,
        TagToRemoveFromCochain,
        CochainTag,
        TensorType,
        MetricType,
        PositionType,
        ExecSpace>
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
        using RhoUpSeq = tensor::upper_t<RhoLowSeq>;
        using SigmaLowSeq = ddc::type_seq_remove_t<
                tensor::lower_t<MuUpSeq>,
                ddc::detail::TypeSeq<TagToRemoveFromCochain>>;

        using HodgeStarDomain = sil::exterior::hodge_star_domain_t<MuUpSeq, NuLowSeq>;
        using HodgeStarDomain2 = sil::exterior::hodge_star_domain_t<RhoUpSeq, SigmaLowSeq>;
        using HodgeStarIndex = misc::convert_type_seq_to_t<tensor::TensorFullIndex, MuUpSeq>;
        using HodgeStarIndex2
                = misc::convert_type_seq_to_t<tensor::TensorAntisymmetricIndex, NuLowSeq>;
        using HodgeStarIndex3 = misc::convert_type_seq_to_t<tensor::TensorFullIndex, RhoUpSeq>;
        using HodgeStarIndex4
                = misc::convert_type_seq_to_t<tensor::TensorAntisymmetricIndex, SigmaLowSeq>;
        using DualIndex = misc::convert_type_seq_to_t<tensor::TensorAntisymmetricIndex, NuLowSeq>;
        using DualCodifferentialIndex
                = misc::convert_type_seq_to_t<tensor::TensorAntisymmetricIndex, RhoLowSeq>;
        using MemorySpace = typename TensorType::memory_space;

        [[maybe_unused]] sil::tensor::tensor_accessor_for_domain_t<HodgeStarDomain2>
                hodge_star_accessor2;
        [[maybe_unused]] tensor::TensorAccessor<DualCodifferentialIndex>
                dual_codifferential_accessor;
        static constexpr std::size_t HODGE_STAR_SIZE_2
                = HodgeStarIndex3::access_size() * HodgeStarIndex4::access_size();
        static constexpr std::size_t DUAL_CODIFFERENTIAL_SIZE
                = DualCodifferentialIndex::access_size();

        std::array<double, HODGE_STAR_SIZE_2> hodge_star_storage2 {};
        std::array<double, DUAL_CODIFFERENTIAL_SIZE> dual_codifferential_storage {};

        ddc::ChunkSpan<double, HodgeStarDomain2, Kokkos::layout_right, MemorySpace>
                hodge_star_span2(hodge_star_storage2.data(), hodge_star_accessor2.domain());
        ddc::ChunkSpan<
                double,
                ddc::DiscreteDomain<DualCodifferentialIndex>,
                Kokkos::layout_right,
                MemorySpace>
                dual_codifferential_span(
                        dual_codifferential_storage.data(),
                        dual_codifferential_accessor.domain());

        sil::tensor::Tensor hodge_star2(hodge_star_span2);
        sil::tensor::Tensor dual_codifferential(dual_codifferential_span);

        auto dual_value_at = [&](auto sampled_elem, auto dual_elem) {
            [[maybe_unused]] sil::tensor::tensor_accessor_for_domain_t<HodgeStarDomain>
                    hodge_star_accessor;
            [[maybe_unused]] tensor::TensorAccessor<DualIndex> dual_tensor_accessor;
            static constexpr std::size_t HODGE_STAR_SIZE
                    = HodgeStarIndex::access_size() * HodgeStarIndex2::access_size();
            static constexpr std::size_t DUAL_TENSOR_SIZE = DualIndex::access_size();
            std::array<double, HODGE_STAR_SIZE> hodge_star_storage {};
            std::array<double, DUAL_TENSOR_SIZE> dual_tensor_storage {};
            ddc::ChunkSpan<double, HodgeStarDomain, Kokkos::layout_right, MemorySpace>
                    hodge_star_span(hodge_star_storage.data(), hodge_star_accessor.domain());
            ddc::ChunkSpan<
                    double,
                    ddc::DiscreteDomain<DualIndex>,
                    Kokkos::layout_right,
                    MemorySpace>
                    dual_tensor_span(dual_tensor_storage.data(), dual_tensor_accessor.domain());
            sil::tensor::Tensor hodge_star(hodge_star_span);
            sil::tensor::Tensor dual_tensor(dual_tensor_span);

            ddc::device_for_each(hodge_star.domain(), [&](auto it) {
                hodge_star.mem(it) = DiscreteHodgeStar<
                        DualStrategy::Circumcentric,
                        MuUpSeq,
                        NuLowSeq,
                        MetricType,
                        PositionType,
                        typename TensorType::non_indices_domain_t::discrete_element_type>::
                        value(metric,
                              position,
                              sampled_elem,
                              hodge_star.canonical_natural_element(it));
            });
            sil::tensor::tensor_prod(dual_tensor, tensor[sampled_elem], hodge_star);
            return dual_tensor.mem(dual_elem);
        };

        detail::PointwiseCoboundary<TagToRemoveFromCochain, DualIndex>::
                run(dual_codifferential,
                    dual_value_at,
                    tensor.non_indices_domain(),
                    chain,
                    lower_chain,
                    elem);

        ddc::device_for_each(hodge_star2.domain(), [&](auto it) {
            hodge_star2.mem(it) = DiscreteHodgeStar<
                    DualStrategy::Circumcentric,
                    RhoUpSeq,
                    SigmaLowSeq,
                    MetricType,
                    PositionType,
                    typename TensorType::non_indices_domain_t::discrete_element_type>::
                    value(metric, position, elem, hodge_star2.canonical_natural_element(it));
        });
        detail::CodifferentialValue<TagToRemoveFromCochain, CochainTag>::
                run(codifferential_tensor, dual_codifferential, hodge_star2);
    }

    static codifferential_tensor_t<TagToRemoveFromCochain, CochainTag, TensorType> run(
            ExecSpace const& exec_space,
            codifferential_tensor_t<TagToRemoveFromCochain, CochainTag, TensorType>
                    codifferential_tensor,
            TensorType tensor,
            MetricType metric,
            PositionType position)
    {
        static_assert(tensor::is_covariant_v<TagToRemoveFromCochain>);
        using MuUpSeq = tensor::upper_t<ddc::to_type_seq_t<tensor::natural_domain_t<CochainTag>>>;
        using NuLowSeq = typename detail::CodifferentialDummyIndexSeq<
                TagToRemoveFromCochain::size() - CochainTag::rank(),
                TagToRemoveFromCochain>::type;
        using RhoLowSeq
                = ddc::type_seq_merge_t<ddc::detail::TypeSeq<TagToRemoveFromCochain>, NuLowSeq>;
        using RhoUpSeq = tensor::upper_t<RhoLowSeq>;
        using SigmaLowSeq = ddc::type_seq_remove_t<
                tensor::lower_t<MuUpSeq>,
                ddc::detail::TypeSeq<TagToRemoveFromCochain>>;

        using DualIndex = misc::convert_type_seq_to_t<tensor::TensorAntisymmetricIndex, NuLowSeq>;
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

        // Codifferential
        SIMILIE_DEBUG_LOG("similie_compute_codifferential");
        ddc::parallel_for_each(
                "similie_compute_codifferential",
                exec_space,
                codifferential_tensor.non_indices_domain(),
                KOKKOS_LAMBDA(
                        typename TensorType::non_indices_domain_t::discrete_element_type elem) {
                    Codifferential::
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
};

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
    return Codifferential<
            MetricIndex,
            TagToRemoveFromCochain,
            CochainTag,
            TensorType,
            MetricType,
            PositionType,
            ExecSpace>::run(exec_space, codifferential_tensor, tensor, metric, position);
}

} // namespace exterior

} // namespace sil
