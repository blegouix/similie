// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <ddc/ddc.hpp>

#include <similie/exterior/hodge_star.hpp>
#include <similie/misc/specialization.hpp>
#include <similie/tensor/character.hpp>
#include <similie/tensor/tensor_impl.hpp>

#include <Kokkos_StdAlgorithms.hpp>

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
struct CodifferentialDummyIndex : tensor::uncharacterize<T>
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

template <
        tensor::TensorIndex MetricIndex,
        tensor::TensorNatIndex TagToRemoveFromCochain,
        tensor::TensorIndex CochainTag,
        misc::Specialization<tensor::Tensor> TensorType,
        misc::Specialization<tensor::Tensor> MetricType,
        class ExecSpace>
codifferential_tensor_t<TagToRemoveFromCochain, CochainTag, TensorType> codifferential(
        ExecSpace const& exec_space,
        codifferential_tensor_t<TagToRemoveFromCochain, CochainTag, TensorType>
                codifferential_tensor,
        TensorType tensor,
        MetricType inv_metric)
{
    static_assert(tensor::is_covariant_v<TagToRemoveFromCochain>);
    using MuUpSeq = tensor::upper<ddc::to_type_seq_t<tensor::natural_domain_t<CochainTag>>>;
    using NuLowSeq = typename detail::CodifferentialDummyIndexSeq<
            TagToRemoveFromCochain::size() - CochainTag::rank(),
            TagToRemoveFromCochain>::type;
    using RhoLowSeq = ddc::type_seq_merge_t<ddc::detail::TypeSeq<TagToRemoveFromCochain>, NuLowSeq>;
    using RhoUpSeq = tensor::upper<RhoLowSeq>;
    using SigmaLowSeq = ddc::
            type_seq_remove_t<tensor::lower<MuUpSeq>, ddc::detail::TypeSeq<TagToRemoveFromCochain>>;

    using HodgeStarDomain = sil::exterior::hodge_star_domain_t<MuUpSeq, NuLowSeq>;
    using HodgeStarDomain2 = sil::exterior::hodge_star_domain_t<RhoUpSeq, SigmaLowSeq>;

    // Hodge star
    [[maybe_unused]] sil::tensor::tensor_accessor_for_domain_t<HodgeStarDomain> hodge_star_accessor;
    ddc::cartesian_prod_t<typename MetricType::non_indices_domain_t, HodgeStarDomain>
            hodge_star_dom(inv_metric.non_indices_domain(), hodge_star_accessor.mem_domain());
    ddc::Chunk hodge_star_alloc(
            hodge_star_dom,
            ddc::KokkosAllocator<double, typename ExecSpace::memory_space>());
    sil::tensor::Tensor hodge_star(hodge_star_alloc);

    sil::exterior::fill_hodge_star<
            sil::tensor::upper<MetricIndex>,
            MuUpSeq,
            NuLowSeq>(exec_space, hodge_star, inv_metric);

    // Dual tensor
    [[maybe_unused]] tensor::TensorAccessor<
            misc::convert_type_seq_to_t<tensor::TensorAntisymmetricIndex, NuLowSeq>>
            dual_tensor_accessor;
    ddc::cartesian_prod_t<
            typename TensorType::non_indices_domain_t,
            ddc::DiscreteDomain<
                    misc::convert_type_seq_to_t<tensor::TensorAntisymmetricIndex, NuLowSeq>>>
            dual_tensor_dom(tensor.non_indices_domain(), dual_tensor_accessor.mem_domain());
    ddc::Chunk dual_tensor_alloc(
            dual_tensor_dom,
            ddc::KokkosAllocator<double, typename ExecSpace::memory_space>());
    sil::tensor::Tensor dual_tensor(dual_tensor_alloc);

    ddc::parallel_for_each(
            exec_space,
            dual_tensor.non_indices_domain(),
            KOKKOS_LAMBDA(typename TensorType::non_indices_domain_t::discrete_element_type elem) {
                sil::tensor::tensor_prod(dual_tensor[elem], tensor[elem], hodge_star[elem]);
            });

    // Dual codifferential
    [[maybe_unused]] tensor::TensorAccessor<
            misc::convert_type_seq_to_t<tensor::TensorAntisymmetricIndex, RhoLowSeq>>
            dual_codifferential_accessor;
    ddc::cartesian_prod_t<
            typename TensorType::non_indices_domain_t,
            ddc::DiscreteDomain<
                    misc::convert_type_seq_to_t<tensor::TensorAntisymmetricIndex, RhoLowSeq>>>
            dual_codifferential_dom(
                    tensor.non_indices_domain(),
                    dual_codifferential_accessor.mem_domain());
    ddc::Chunk dual_codifferential_alloc(
            dual_codifferential_dom,
            ddc::KokkosAllocator<double, typename ExecSpace::memory_space>());
    sil::tensor::Tensor dual_codifferential(dual_codifferential_alloc);
    sil::exterior::deriv<
            TagToRemoveFromCochain,
            misc::convert_type_seq_to_t<
                    tensor::TensorAntisymmetricIndex,
                    NuLowSeq>>(exec_space, dual_codifferential, dual_tensor);

    // Hodge star 2
    [[maybe_unused]] sil::tensor::tensor_accessor_for_domain_t<HodgeStarDomain2>
            hodge_star_accessor2;
    ddc::cartesian_prod_t<typename MetricType::non_indices_domain_t, HodgeStarDomain2>
            hodge_star_dom2(inv_metric.non_indices_domain(), hodge_star_accessor2.mem_domain());
    ddc::Chunk hodge_star_alloc2(
            hodge_star_dom2,
            ddc::KokkosAllocator<double, typename ExecSpace::memory_space>());
    sil::tensor::Tensor hodge_star2(hodge_star_alloc2);

    sil::exterior::fill_hodge_star<
            sil::tensor::upper<MetricIndex>,
            RhoUpSeq,
            SigmaLowSeq>(exec_space, hodge_star2, inv_metric);

    // Codifferential
    /*
    ddc::parallel_for_each(
            exec_space,
            codifferential_tensor.non_indices_domain(),
            KOKKOS_LAMBDA(typename TensorType::non_indices_domain_t::discrete_element_type elem) {
                sil::tensor::tensor_prod(
                        codifferential_tensor[elem],
                        dual_codifferential[elem],
                        hodge_star2[elem]);
                if constexpr (
                        (TagToRemoveFromCochain::size() * (CochainTag::rank() + 1) + 1) % 2 == 1) {
                    codifferential_tensor[elem] *= -1;
                }
            });
    */

    return codifferential_tensor;
}

} // namespace exterior

} // namespace sil
