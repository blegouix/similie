// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: MIT

#pragma once

#include <ddc/ddc.hpp>

#include <similie/exterior/coboundary.hpp>
#include <similie/exterior/hodge_star.hpp>
#include <similie/misc/macros.hpp>
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

template <
        class MetricComponentIndex,
        class Axis,
        misc::Specialization<tensor::Tensor> MetricType,
        class Elem>
KOKKOS_FUNCTION double centered_metric_factor(MetricType const& inv_metric, Elem const& elem)
{
    if constexpr (misc::Specialization<MetricComponentIndex, tensor::TensorIdentityIndex>) {
        return 1.;
    } else {
        return inv_metric(
                elem,
                inv_metric.accessor().template access_element<Axis, Axis>());
    }
}

template <
        class NaturalDims,
        class MetricComponentIndex,
        class OutTensorType,
        class TensorType,
        class MetricType,
        class Elem,
        class InIndex,
        class OutIndex>
struct FillCenteredCodifferential;

template <
        class Axis,
        class... TailAxes,
        class MetricComponentIndex,
        class OutTensorType,
        class TensorType,
        class MetricType,
        class Elem,
        class InIndex,
        class OutIndex>
struct FillCenteredCodifferential<
        ddc::detail::TypeSeq<Axis, TailAxes...>,
        MetricComponentIndex,
        OutTensorType,
        TensorType,
        MetricType,
        Elem,
        InIndex,
        OutIndex>
{
    KOKKOS_FUNCTION static void run(
            OutTensorType out_tensor,
            TensorType tensor,
            MetricType inv_metric,
            Elem elem)
    {
        using d_dim_t = discrete_dimension_for_t<Axis, typename TensorType::non_indices_domain_t>;
        static_assert(!std::is_void_v<d_dim_t>);
        constexpr std::size_t comp = ddc::type_seq_rank_v<Axis, typename InIndex::type_seq_dimensions>;

        auto const left_elem = mirrored_left_elem<d_dim_t>(tensor.non_indices_domain(), elem);
        auto const right_elem = mirrored_right_elem<d_dim_t>(tensor.non_indices_domain(), elem);
        double const denom = centered_step<d_dim_t>(tensor.non_indices_domain(), elem);

        out_tensor.mem(elem, ddc::DiscreteElement<OutIndex>(0))
                += centered_metric_factor<MetricComponentIndex, Axis>(inv_metric, elem)
                   * (tensor.get(right_elem, ddc::DiscreteElement<InIndex>(comp))
                      - tensor.get(left_elem, ddc::DiscreteElement<InIndex>(comp)))
                   / denom;

        FillCenteredCodifferential<
                ddc::detail::TypeSeq<TailAxes...>,
                MetricComponentIndex,
                OutTensorType,
                TensorType,
                MetricType,
                Elem,
                InIndex,
                OutIndex>::run(out_tensor, tensor, inv_metric, elem);
    }
};

template <
        class MetricComponentIndex,
        class OutTensorType,
        class TensorType,
        class MetricType,
        class Elem,
        class InIndex,
        class OutIndex>
struct FillCenteredCodifferential<
        ddc::detail::TypeSeq<>,
        MetricComponentIndex,
        OutTensorType,
        TensorType,
        MetricType,
        Elem,
        InIndex,
        OutIndex>
{
    KOKKOS_FUNCTION static void run(
            [[maybe_unused]] OutTensorType out_tensor,
            [[maybe_unused]] TensorType tensor,
            [[maybe_unused]] MetricType inv_metric,
            [[maybe_unused]] Elem elem)
    {
    }
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
    using MuUpSeq = tensor::upper_t<ddc::to_type_seq_t<tensor::natural_domain_t<CochainTag>>>;
    using NuLowSeq = typename detail::CodifferentialDummyIndexSeq<
            TagToRemoveFromCochain::size() - CochainTag::rank(),
            TagToRemoveFromCochain>::type;
    using RhoLowSeq = ddc::type_seq_merge_t<ddc::detail::TypeSeq<TagToRemoveFromCochain>, NuLowSeq>;
    using RhoUpSeq = tensor::upper_t<RhoLowSeq>;
    using SigmaLowSeq = ddc::type_seq_remove_t<
            tensor::lower_t<MuUpSeq>,
            ddc::detail::TypeSeq<TagToRemoveFromCochain>>;

    using HodgeStarDomain = sil::exterior::hodge_star_domain_t<MuUpSeq, NuLowSeq>;
    using HodgeStarDomain2 = sil::exterior::hodge_star_domain_t<RhoUpSeq, SigmaLowSeq>;

    // Hodge star
    [[maybe_unused]] sil::tensor::tensor_accessor_for_domain_t<HodgeStarDomain> hodge_star_accessor;
    ddc::cartesian_prod_t<typename MetricType::non_indices_domain_t, HodgeStarDomain>
            hodge_star_dom(inv_metric.non_indices_domain(), hodge_star_accessor.domain());
    ddc::Chunk hodge_star_alloc(
            hodge_star_dom,
            ddc::KokkosAllocator<double, typename ExecSpace::memory_space>());
    sil::tensor::Tensor hodge_star(hodge_star_alloc);

    sil::exterior::fill_hodge_star<
            sil::tensor::upper_t<MetricIndex>,
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
            dual_tensor_dom(tensor.non_indices_domain(), dual_tensor_accessor.domain());
    ddc::Chunk dual_tensor_alloc(
            dual_tensor_dom,
            ddc::KokkosAllocator<double, typename ExecSpace::memory_space>());
    sil::tensor::Tensor dual_tensor(dual_tensor_alloc);

    SIMILIE_DEBUG_LOG("similie_compute_dual_tensor");
    ddc::parallel_for_each(
            "similie_compute_dual_tensor",
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
                    dual_codifferential_accessor.domain());
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
            hodge_star_dom2(inv_metric.non_indices_domain(), hodge_star_accessor2.domain());
    ddc::Chunk hodge_star_alloc2(
            hodge_star_dom2,
            ddc::KokkosAllocator<double, typename ExecSpace::memory_space>());
    sil::tensor::Tensor hodge_star2(hodge_star_alloc2);

    sil::exterior::fill_hodge_star<
            sil::tensor::upper_t<MetricIndex>,
            RhoUpSeq,
            SigmaLowSeq>(exec_space, hodge_star2, inv_metric);

    // Codifferential
    SIMILIE_DEBUG_LOG("similie_compute_codifferential");
    ddc::parallel_for_each(
            "similie_compute_codifferential",
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

    return codifferential_tensor;
}

template <
        tensor::TensorIndex MetricIndex,
        tensor::TensorNatIndex TagToRemoveFromCochain,
        tensor::TensorIndex CochainTag,
        misc::Specialization<tensor::Tensor> TensorType,
        misc::Specialization<tensor::Tensor> MetricType,
        class ExecSpace>
    requires(CochainTag::rank() == 1 && TagToRemoveFromCochain::rank() == 1)
codifferential_tensor_t<TagToRemoveFromCochain, CochainTag, TensorType> codifferential(
        ExecSpace const& exec_space,
        codifferential_tensor_t<TagToRemoveFromCochain, CochainTag, TensorType>
                codifferential_tensor,
        TensorType tensor,
        MetricType inv_metric,
        CenteredMirroredBoundary)
{
    using metric_component_index_t = ddc::type_seq_element_t<
            0,
            ddc::to_type_seq_t<typename MetricType::indices_domain_t>>;
    using in_index_t = ddc::type_seq_element_t<
            0,
            ddc::to_type_seq_t<typename TensorType::indices_domain_t>>;
    using out_index_t = ddc::type_seq_element_t<
            0,
            ddc::to_type_seq_t<typename decltype(codifferential_tensor)::indices_domain_t>>;
    static_assert(in_index_t::rank() == 1);
    static_assert(out_index_t::rank() == 0);
    static_assert(
            misc::Specialization<metric_component_index_t, tensor::TensorIdentityIndex>
            && "Centered mirrored codifferential currently supports identity metric only.");

    ddc::parallel_for_each(
            "similie_compute_centered_mirrored_codifferential",
            exec_space,
            codifferential_tensor.non_indices_domain(),
            KOKKOS_LAMBDA(typename decltype(codifferential_tensor)::non_indices_domain_t::discrete_element_type elem) {
                codifferential_tensor.mem(elem, ddc::DiscreteElement<out_index_t>(0)) = 0.;
                detail::FillCenteredCodifferential<
                        typename in_index_t::type_seq_dimensions,
                        metric_component_index_t,
                        decltype(codifferential_tensor),
                        TensorType,
                        MetricType,
                        decltype(elem),
                        in_index_t,
                        out_index_t>::run(codifferential_tensor, tensor, inv_metric, elem);
            });

    return codifferential_tensor;
}

} // namespace exterior

} // namespace sil
