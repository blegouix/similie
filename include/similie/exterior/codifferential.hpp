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
    using type = tensor::TensorCovariantNaturalIndex<tensor::TensorNaturalIndex<>>;
};

template <tensor::TensorNatIndex TagToRemoveFromCochain, tensor::TensorNatIndex... Tag>
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

template <std::size_t I, class T>
struct CodifferentialDummyIndex : tensor::uncharacterize<T>
{
};

template <class Ids, class T>
struct CodifferentialDummyIndexSeq;

template <std::size_t... Id, class T>
struct CodifferentialDummyIndexSeq<std::index_sequence<Id...>, T>
{
    using type = ddc::detail::TypeSeq<
            tensor::TensorCovariantNaturalIndex<CodifferentialDummyIndex<Id, T>>...>;
};

} // namespace detail

template <
        tensor::TensorNatIndex TagToRemoveFromCochain,
        tensor::TensorIndex CochainTag,
        misc::Specialization<tensor::Tensor> TensorType>
using codifferential_tensor_t = typename detail::
        CodifferentialTensorType<TagToRemoveFromCochain, CochainTag, TensorType>::type;

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
    using MuLowSeq = ddc::to_type_seq_t<tensor::natural_domain_t<CochainTag>>;
    using MuUpSeq = tensor::upper<MuLowSeq>;
    using NuLowSeq = typename detail::CodifferentialDummyIndexSeq<
            std::make_index_sequence<TagToRemoveFromCochain::size() - CochainTag::rank()>,
            TagToRemoveFromCochain>::type;
    using NuUpSeq = tensor::upper<NuLowSeq>;

    using HodgeStarDomain = sil::exterior::hodge_star_domain_t<MuUpSeq, NuLowSeq>;
    /*
        using HodgeStarDomain2 = sil::exterior::
                hodge_star_domain_t<ddc::detail::TypeSeq<RhoUp, NuUp>, ddc::detail::TypeSeq<>>;
		*/

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
    Kokkos::fence();

    // Dual tensor
    [[maybe_unused]] tensor::tensor_accessor_for_domain_t<
            ddc::detail::convert_type_seq_to_discrete_domain_t<NuLowSeq>> dual_tensor_accessor;
    ddc::cartesian_prod_t<
            typename TensorType::non_indices_domain_t,
            ddc::detail::convert_type_seq_to_discrete_domain_t<NuLowSeq>>
            dual_tensor_dom(tensor.non_indices_domain(), dual_tensor_accessor.mem_domain());
    ddc::Chunk dual_tensor_alloc(
            dual_tensor_dom,
            ddc::KokkosAllocator<double, typename ExecSpace::memory_space>());
    sil::tensor::Tensor dual_tensor(dual_tensor_alloc);

    ddc::parallel_for_each(
            Kokkos::DefaultExecutionSpace(),
            dual_tensor.non_indices_domain(),
            KOKKOS_LAMBDA(typename TensorType::non_indices_domain_t::discrete_element_type elem) {
                sil::tensor::tensor_prod(dual_tensor[elem], tensor[elem], hodge_star[elem]);
            });
    Kokkos::fence();

    std::cout << dual_tensor;
    /*
        // Dual Laplacian
        [[maybe_unused]] sil::tensor::TensorAccessor<
                sil::tensor::TensorAntisymmetricIndex<RhoLow, NuLow>> dual_laplacian_accessor;
        ddc::DiscreteDomain<DDimX, DDimY, sil::tensor::TensorAntisymmetricIndex<RhoLow, NuLow>>
                dual_laplacian_dom(
                        mesh_xy.remove_last(ddc::DiscreteVector<DDimX, DDimY> {1, 1}),
                        dual_laplacian_accessor.mem_domain());
        ddc::Chunk dual_laplacian_alloc(dual_laplacian_dom, ddc::DeviceAllocator<double>());
        sil::tensor::Tensor dual_laplacian(dual_laplacian_alloc);
        sil::exterior::deriv<
                RhoLow,
                NuLow>(Kokkos::DefaultExecutionSpace(), dual_laplacian, dual_tensor);
        Kokkos::fence();

        // Hodge star 2
        [[maybe_unused]] sil::tensor::tensor_accessor_for_domain_t<HodgeStarDomain2>
                hodge_star_accessor2;
        ddc::cartesian_prod_t<ddc::DiscreteDomain<DDimX, DDimY>, HodgeStarDomain2>
                hodge_star_dom2(metric.non_indices_domain(), hodge_star_accessor2.mem_domain());
        ddc::Chunk hodge_star_alloc2(hodge_star_dom2, ddc::DeviceAllocator<double>());
        sil::tensor::Tensor hodge_star2(hodge_star_alloc2);

        sil::exterior::fill_hodge_star<
                sil::tensor::upper<MetricIndex>,
                ddc::detail::TypeSeq<RhoUp, NuUp>,
                ddc::detail::TypeSeq<>>(Kokkos::DefaultExecutionSpace(), hodge_star2, inv_metric);
        Kokkos::fence();

        // Laplacian
        [[maybe_unused]] sil::tensor::TensorAccessor<CodifferentialDummyIndex> laplacian_accessor;
        ddc::DiscreteDomain<DDimX, DDimY, CodifferentialDummyIndex> laplacian_dom(
                mesh_xy.remove_last(ddc::DiscreteVector<DDimX, DDimY> {1, 1}),
                laplacian_accessor.mem_domain());
        ddc::Chunk laplacian_alloc(laplacian_dom, ddc::DeviceAllocator<double>());
        sil::tensor::Tensor laplacian(laplacian_alloc);
    */
    return codifferential_tensor;
}

} // namespace exterior

} // namespace sil
