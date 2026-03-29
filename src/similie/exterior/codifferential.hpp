// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: MIT

#pragma once

#include <ddc/ddc.hpp>

#include <similie/exterior/hodge_star.hpp>
#include <similie/mesher/dualizer.hpp>
#include <similie/misc/macros.hpp>
#include <similie/misc/specialization.hpp>
#include <similie/tensor/character.hpp>
#include <similie/tensor/tensor_impl.hpp>

#include <Kokkos_StdAlgorithms.hpp>

#include "cochain.hpp"
#include "cosimplex.hpp"
#include "form.hpp"


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

template <class DDim>
constexpr bool is_dual_discrete_dimension_v
        = !std::is_same_v<DDim, mesher::detail::primal_discrete_dimension_t<DDim>>;

template <
        class AxisTag,
        misc::Specialization<tensor::Tensor> OutTensorType,
        misc::Specialization<tensor::Tensor> TensorType>
KOKKOS_FUNCTION double centered_form_derivative(
        typename OutTensorType::non_indices_domain_t::discrete_element_type out_elem,
        TensorType tensor)
{
    using out_d_dim_t = mesher::detail::
            discrete_dimension_for_t<AxisTag, typename OutTensorType::non_indices_domain_t>;
    using comp_d_dim_t = mesher::detail::
            discrete_dimension_for_t<AxisTag, typename TensorType::non_indices_domain_t>;
    using comp_elem_t = typename TensorType::non_indices_domain_t::discrete_element_type;
    using comp_index_t = ddc::type_seq_element_t<
            0,
            ddc::to_type_seq_t<typename TensorType::indices_domain_t>>;
    static_assert(!std::is_void_v<out_d_dim_t>);
    static_assert(!std::is_void_v<comp_d_dim_t>);
    static_assert(comp_index_t::rank() == 0);

    comp_elem_t const aligned_elem = mesher::detail::dualizer_remap_element<comp_elem_t>(out_elem);
    if constexpr (is_dual_discrete_dimension_v<out_d_dim_t>) {
        auto const left = aligned_elem;
        auto const right = left + ddc::DiscreteVector<comp_d_dim_t>(1);
        double const dx = static_cast<double>(
                ddc::coordinate(ddc::DiscreteElement<comp_d_dim_t>(right))
                - ddc::coordinate(ddc::DiscreteElement<comp_d_dim_t>(left)));
        return (tensor.get(right, ddc::DiscreteElement<comp_index_t>(0))
                - tensor.get(left, ddc::DiscreteElement<comp_index_t>(0)))
               / dx;
    } else {
        auto const right = aligned_elem;
        auto const left = right - ddc::DiscreteVector<comp_d_dim_t>(1);
        double const dx = static_cast<double>(
                ddc::coordinate(ddc::DiscreteElement<comp_d_dim_t>(right))
                - ddc::coordinate(ddc::DiscreteElement<comp_d_dim_t>(left)));
        return (tensor.get(right, ddc::DiscreteElement<comp_index_t>(0))
                - tensor.get(left, ddc::DiscreteElement<comp_index_t>(0)))
               / dx;
    }
}

template <
        misc::Specialization<tensor::Tensor> OutTensorType,
        class SupportTag,
        class FirstComponent,
        class SecondComponent,
        class ExecSpace>
OutTensorType exterior_derivative_of_tensor_form_2d(
        ExecSpace const& exec_space,
        OutTensorType out_tensor,
        TensorForm<SupportTag, FirstComponent, SecondComponent> tensor_form)
{
    using out_index_t = ddc::type_seq_element_t<
            0,
            ddc::to_type_seq_t<typename OutTensorType::indices_domain_t>>;
    using first_out_d_dim_t = mesher::detail::
            discrete_dimension_for_t<typename FirstComponent::tag, typename OutTensorType::non_indices_domain_t>;
    using second_out_d_dim_t = mesher::detail::
            discrete_dimension_for_t<typename SecondComponent::tag, typename OutTensorType::non_indices_domain_t>;
    static_assert(out_index_t::rank() == 0);

    auto const first_tensor = tensor_form.template component<typename FirstComponent::tag>();
    auto const second_tensor = tensor_form.template component<typename SecondComponent::tag>();

    ddc::parallel_for_each(
            "similie_compute_tensor_form_exterior_derivative_2d",
            exec_space,
            out_tensor.non_indices_domain(),
            KOKKOS_LAMBDA(typename OutTensorType::non_indices_domain_t::discrete_element_type elem) {
                if constexpr (!is_dual_discrete_dimension_v<first_out_d_dim_t>) {
                    ddc::DiscreteDomain<first_out_d_dim_t> dim_dom(out_tensor.non_indices_domain());
                    ddc::DiscreteElement<first_out_d_dim_t> dim_elem(elem);
                    if (dim_elem.uid() == dim_dom.front().uid()
                        || dim_elem.uid() == dim_dom.back().uid()) {
                        return;
                    }
                }
                if constexpr (!is_dual_discrete_dimension_v<second_out_d_dim_t>) {
                    ddc::DiscreteDomain<second_out_d_dim_t> dim_dom(out_tensor.non_indices_domain());
                    ddc::DiscreteElement<second_out_d_dim_t> dim_elem(elem);
                    if (dim_elem.uid() == dim_dom.front().uid()
                        || dim_elem.uid() == dim_dom.back().uid()) {
                        return;
                    }
                }
                out_tensor.mem(elem, ddc::DiscreteElement<out_index_t>(0))
                        += centered_form_derivative<typename FirstComponent::tag, OutTensorType>(
                                   elem,
                                   second_tensor)
                           - centered_form_derivative<typename SecondComponent::tag, OutTensorType>(
                                   elem,
                                   first_tensor);
            });
    return out_tensor;
}

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
        class TagToRemoveFromCochain,
        tensor::TensorIndex CochainTag,
        misc::Specialization<tensor::Tensor> OutTensorType,
        misc::Specialization<tensor::Tensor> TensorType,
        misc::Specialization<tensor::Tensor> MetricType,
        class Dualizer,
        class ExecSpace>
OutTensorType codifferential(
        ExecSpace const& exec_space,
        OutTensorType out_tensor,
        TensorType tensor,
        MetricType inv_metric,
        Dualizer const& dualizer)
{
    using out_index_t = ddc::type_seq_element_t<
            0,
            ddc::to_type_seq_t<typename OutTensorType::indices_domain_t>>;
    using metric_component_index_t = ddc::type_seq_element_t<
            0,
            ddc::to_type_seq_t<typename MetricType::indices_domain_t>>;
    using out_d_dim_t = mesher::detail::
            discrete_dimension_for_t<TagToRemoveFromCochain, typename OutTensorType::non_indices_domain_t>;
    using flux_d_dim_t = mesher::detail::
            discrete_dimension_for_t<TagToRemoveFromCochain, typename TensorType::non_indices_domain_t>;
    using out_elem_t = typename OutTensorType::non_indices_domain_t::discrete_element_type;
    using flux_elem_t = typename TensorType::non_indices_domain_t::discrete_element_type;
    using flux_index_t = ddc::type_seq_element_t<
            0,
            ddc::to_type_seq_t<typename TensorType::indices_domain_t>>;
    static_assert(out_index_t::rank() == 0);
    static_assert(!std::is_void_v<out_d_dim_t>);
    static_assert(!std::is_void_v<flux_d_dim_t>);

    ddc::parallel_for_each(
            "similie_compute_centered_dualized_codifferential",
            exec_space,
            out_tensor.non_indices_domain(),
            KOKKOS_LAMBDA(out_elem_t out_elem) {
                ddc::DiscreteDomain<out_d_dim_t> dim_dom(out_tensor.non_indices_domain());
                ddc::DiscreteElement<out_d_dim_t> dim_elem(out_elem);
                if (dim_elem.uid() == dim_dom.front().uid() || dim_elem.uid() == dim_dom.back().uid()) {
                    return;
                }

                flux_elem_t const right_face = dualizer(out_elem);
                flux_elem_t const left_face = right_face - ddc::DiscreteVector<flux_d_dim_t>(1);
                double const dx = static_cast<double>(
                        ddc::coordinate(ddc::DiscreteElement<flux_d_dim_t>(right_face))
                        - ddc::coordinate(ddc::DiscreteElement<flux_d_dim_t>(left_face)));
                double const deriv = (tensor.get(right_face, ddc::DiscreteElement<flux_index_t>(0))
                                      - tensor.get(left_face, ddc::DiscreteElement<flux_index_t>(0)))
                                     / dx;
                double const metric_factor = [&]() -> double {
                    if constexpr (
                            misc::Specialization<metric_component_index_t, tensor::TensorIdentityIndex>) {
                        return 1.;
                    } else {
                        return inv_metric(
                                out_elem,
                                inv_metric.accessor().template access_element<
                                        TagToRemoveFromCochain,
                                        TagToRemoveFromCochain>());
                    }
                }();
                out_tensor.mem(out_elem, ddc::DiscreteElement<out_index_t>(0))
                        += metric_factor * deriv;
            });
    return out_tensor;
}

template <
        tensor::TensorIndex MetricIndex,
        misc::Specialization<tensor::Tensor> OutTensorType,
        class SupportTag,
        class FirstComponent,
        class SecondComponent,
        misc::Specialization<tensor::Tensor> MetricType,
        class ExecSpace>
OutTensorType codifferential(
        ExecSpace const& exec_space,
        OutTensorType out_tensor,
        TensorForm<SupportTag, FirstComponent, SecondComponent> tensor_form,
        MetricType inv_metric)
{
    ddc::Chunk dual_first_alloc(
            typename SecondComponent::tensor_type::discrete_domain_type(
                    tensor_form.template component<typename SecondComponent::tag>().domain()),
            ddc::KokkosAllocator<double, typename ExecSpace::memory_space>());
    sil::tensor::Tensor dual_first(dual_first_alloc);
    ddc::Chunk dual_second_alloc(
            typename FirstComponent::tensor_type::discrete_domain_type(
                    tensor_form.template component<typename FirstComponent::tag>().domain()),
            ddc::KokkosAllocator<double, typename ExecSpace::memory_space>());
    sil::tensor::Tensor dual_second(dual_second_alloc);
    auto dual_form = make_tensor_form<hodge_dual_support_t<SupportTag>>(
            component<typename FirstComponent::tag>(dual_first),
            component<typename SecondComponent::tag>(dual_second));

    sil::exterior::hodge_star(exec_space, dual_form, tensor_form, inv_metric);
    return detail::exterior_derivative_of_tensor_form_2d(exec_space, out_tensor, dual_form);
}

} // namespace exterior

} // namespace sil
