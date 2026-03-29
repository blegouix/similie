// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: MIT

#pragma once

#include <ddc/ddc.hpp>

#include <similie/misc/factorial.hpp>
#include <similie/misc/macros.hpp>
#include <similie/mesher/dualizer.hpp>
#include <similie/misc/specialization.hpp>
#include <similie/misc/type_seq_conversion.hpp>
#include <similie/tensor/antisymmetric_tensor.hpp>
#include <similie/tensor/character.hpp>
#include <similie/tensor/determinant.hpp>
#include <similie/tensor/full_tensor.hpp>
#include <similie/tensor/levi_civita_tensor.hpp>
#include <similie/tensor/metric.hpp>
#include <similie/tensor/prime.hpp>

#include "form.hpp"

namespace sil {

namespace exterior {

namespace detail {

template <class Axis1, class Axis2, misc::Specialization<tensor::Tensor> MetricType, class Elem>
KOKKOS_FUNCTION double inverse_metric_component(MetricType metric, Elem elem)
{
    using metric_component_index_t = ddc::type_seq_element_t<
            0,
            ddc::to_type_seq_t<typename MetricType::indices_domain_t>>;
    if constexpr (misc::Specialization<metric_component_index_t, tensor::TensorIdentityIndex>) {
        return std::is_same_v<Axis1, Axis2> ? 1. : 0.;
    } else if constexpr (
            misc::Specialization<metric_component_index_t, tensor::TensorDiagonalIndex>) {
        if constexpr (std::is_same_v<Axis1, Axis2>) {
            return metric(
                    elem,
                    metric.accessor().template access_element<Axis1, Axis2>());
        } else {
            return 0.;
        }
    } else {
        return metric(
                elem,
                metric.accessor().template access_element<Axis1, Axis2>());
    }
}

template <tensor::TensorIndex MetricIndex, misc::Specialization<tensor::Tensor> MetricType, class Elem>
KOKKOS_FUNCTION double primal_volume_factor(MetricType metric, Elem elem)
{
    using metric_index_1_t = tensor::metric_index_1<tensor::upper_t<MetricIndex>>;
    using axis1_t = ddc::type_seq_element_t<0, typename metric_index_1_t::type_seq_dimensions>;
    using axis2_t = ddc::type_seq_element_t<1, typename metric_index_1_t::type_seq_dimensions>;
    double const gxx = inverse_metric_component<axis1_t, axis1_t>(metric, elem);
    double const gxy = inverse_metric_component<axis1_t, axis2_t>(metric, elem);
    double const gyx = inverse_metric_component<axis2_t, axis1_t>(metric, elem);
    double const gyy = inverse_metric_component<axis2_t, axis2_t>(metric, elem);
    double const det_inv_metric = gxx * gyy - gxy * gyx;
    return Kokkos::sqrt(Kokkos::abs(1. / det_inv_metric));
}

template <class Dom>
struct IsFullyDualDomain;

template <class... DDims>
struct IsFullyDualDomain<ddc::DiscreteDomain<DDims...>>
{
    static constexpr bool value
            = (true
               && ... && !std::is_same_v<DDims, sil::mesher::detail::primal_discrete_dimension_t<DDims>>);
};

template <class Dom>
constexpr bool is_fully_dual_domain_v = IsFullyDualDomain<Dom>::value;

} // namespace detail

template <
        misc::Specialization<ddc::detail::TypeSeq> Indices1,
        misc::Specialization<ddc::detail::TypeSeq> Indices2>
using hodge_star_domain_t
        = ddc::detail::convert_type_seq_to_discrete_domain_t<ddc::type_seq_merge_t<
                ddc::detail::TypeSeq<misc::convert_type_seq_to_t<
                        tensor::TensorFullIndex,
                        Indices1>>, // TODO clarify Antisymmetric
                std::conditional_t<
                        (ddc::type_seq_size_v<Indices2> == 0),
                        ddc::detail::TypeSeq<>,
                        ddc::detail::TypeSeq<misc::convert_type_seq_to_t<
                                tensor::TensorAntisymmetricIndex,
                                Indices2>>>>>;

template <
        misc::Specialization<ddc::detail::TypeSeq> Indices1,
        misc::Specialization<ddc::detail::TypeSeq> Indices2,
        misc::Specialization<tensor::Tensor> HodgeStarType,
        class MetricDeterminantType,
        misc::Specialization<tensor::Tensor> MetricProdType,
        class ExecSpace>
    requires(
            misc::Specialization<MetricDeterminantType, ddc::ChunkSpan>
            || misc::Specialization<MetricDeterminantType, tensor::Tensor>)
HodgeStarType fill_hodge_star(
        ExecSpace const& exec_space,
        HodgeStarType hodge_star,
        MetricDeterminantType metric_determinant,
        MetricProdType metric_prod)
{
    sil::tensor::TensorAccessor<misc::convert_type_seq_to_t<
            tensor::TensorLeviCivitaIndex,
            ddc::type_seq_merge_t<tensor::primes<tensor::lower_t<Indices1>>, Indices2>>>
            levi_civita_accessor;
    ddc::DiscreteDomain<misc::convert_type_seq_to_t<
            tensor::TensorLeviCivitaIndex,
            ddc::type_seq_merge_t<tensor::primes<tensor::lower_t<Indices1>>, Indices2>>>
            levi_civita_dom(levi_civita_accessor.domain());
    ddc::Chunk levi_civita_alloc(
            levi_civita_dom,
            ddc::KokkosAllocator<
                    double,
                    typename ExecSpace::memory_space>()); // TODO consider avoid allocation
    sil::tensor::Tensor levi_civita(levi_civita_alloc);

    SIMILIE_DEBUG_LOG("similie_compute_hodge_star");
    ddc::parallel_for_each(
            "similie_compute_hodge_star",
            exec_space,
            hodge_star.non_indices_domain(),
            KOKKOS_LAMBDA(
                    typename HodgeStarType::non_indices_domain_t::discrete_element_type elem) {
                tensor_prod(hodge_star[elem], metric_prod[elem], levi_civita);
                hodge_star[elem] *= Kokkos::sqrt(Kokkos::abs(metric_determinant(elem)))
                                    / misc::factorial(ddc::type_seq_size_v<Indices1>);
            });
    return hodge_star;
}

template <
        tensor::TensorIndex MetricIndex,
        misc::Specialization<ddc::detail::TypeSeq> Indices1,
        misc::Specialization<ddc::detail::TypeSeq> Indices2,
        misc::Specialization<tensor::Tensor> HodgeStarType,
        misc::Specialization<tensor::Tensor> MetricType,
        class ExecSpace>
HodgeStarType fill_hodge_star(
        ExecSpace const& exec_space,
        HodgeStarType hodge_star,
        MetricType metric)
{
    static_assert(tensor::are_contravariant_v<
                  ddc::to_type_seq_t<typename MetricIndex::subindices_domain_t>>);
    static_assert(tensor::are_contravariant_v<
                  ddc::to_type_seq_t<typename MetricType::accessor_t::natural_domain_t>>);
    static_assert(tensor::are_contravariant_v<Indices1>);
    static_assert(tensor::are_covariant_v<Indices2>);

    // Allocate metric_det to receive metric field determinant values
    ddc::Chunk metric_det_alloc(
            metric.non_indices_domain(),
            ddc::KokkosAllocator<double, typename ExecSpace::memory_space>());
    ddc::ChunkSpan metric_det(metric_det_alloc);
    // Allocate a buffer mirroring the metric as a full matrix, it will be overwritten by tensor::determinant() which involves a LU decomposition
    ddc::Chunk buffer_alloc(
            ddc::cartesian_prod_t<
                    typename MetricType::non_indices_domain_t,
                    ddc::DiscreteDomain<
                            tensor::metric_index_1<MetricIndex>,
                            tensor::metric_index_2<MetricIndex>>>(
                    metric.non_indices_domain(),
                    ddc::DiscreteDomain<
                            tensor::metric_index_1<MetricIndex>,
                            tensor::metric_index_2<MetricIndex>>(metric.natural_domain())),
            ddc::KokkosAllocator<double, typename ExecSpace::memory_space>());
    ddc::ChunkSpan buffer(buffer_alloc);
    // Compute determinants
    SIMILIE_DEBUG_LOG("similie_compute_metric_determinant");
    ddc::parallel_for_each(
            "similie_compute_metric_determinant",
            exec_space,
            metric.non_indices_domain(),
            KOKKOS_LAMBDA(typename MetricType::non_indices_domain_t::discrete_element_type elem) {
                ddc::device_for_each(
                        ddc::DiscreteDomain<
                                tensor::metric_index_1<MetricIndex>,
                                tensor::metric_index_2<MetricIndex>>(metric.natural_domain()),
                        [&](ddc::DiscreteElement<
                                tensor::metric_index_1<MetricIndex>,
                                tensor::metric_index_2<MetricIndex>> index) {
                            buffer(elem, index) = metric.get(metric.access_element(
                                    elem,
                                    index)); // TODO: triggers a "nvlink warning : Stack size for entry function cannot be statically determined"
                        });
                metric_det(elem) = 1. / tensor::determinant(buffer[elem].allocation_kokkos_view());
            });

    // Allocate & compute the product of metrics
    tensor::tensor_accessor_for_domain_t<
            tensor::metric_prod_domain_t<MetricIndex, Indices1, tensor::primes<Indices1>>>
            metric_prod_accessor;
    ddc::Chunk metric_prod_alloc(
            ddc::cartesian_prod_t<
                    typename MetricType::non_indices_domain_t,
                    tensor::metric_prod_domain_t<MetricIndex, Indices1, tensor::primes<Indices1>>>(
                    metric.non_indices_domain(),
                    metric_prod_accessor.domain()),
            ddc::KokkosAllocator<double, typename ExecSpace::memory_space>());
    tensor::Tensor metric_prod(metric_prod_alloc);

    fill_metric_prod<
            MetricIndex,
            Indices1,
            tensor::primes<Indices1>>(exec_space, metric_prod, metric);

    // Compute Hodge star
    return fill_hodge_star<Indices1, Indices2>(exec_space, hodge_star, metric_det, metric_prod);
}

template <
        class SupportTag,
        class... OutComponents,
        class FirstComponent,
        class SecondComponent,
        class MetricType,
        class ExecSpace>
auto hodge_star(
        ExecSpace const& exec_space,
        TensorForm<hodge_dual_support_t<SupportTag>, OutComponents...> out_form,
        TensorForm<SupportTag, FirstComponent, SecondComponent> in_form,
        MetricType const&)
{
    using first_in_tensor_t = typename FirstComponent::tensor_type;
    using second_in_tensor_t = typename SecondComponent::tensor_type;
    using out_first_component_t = typename detail::
            FormComponentByTag<typename FirstComponent::tag, OutComponents...>::type;
    using out_second_component_t = typename detail::
            FormComponentByTag<typename SecondComponent::tag, OutComponents...>::type;
    using first_in_index_t = ddc::type_seq_element_t<
            0,
            ddc::to_type_seq_t<typename first_in_tensor_t::indices_domain_t>>;
    using second_in_index_t = ddc::type_seq_element_t<
            0,
            ddc::to_type_seq_t<typename second_in_tensor_t::indices_domain_t>>;
    using first_out_index_t = ddc::type_seq_element_t<
            0,
            ddc::to_type_seq_t<typename second_in_tensor_t::indices_domain_t>>;
    using second_out_index_t = ddc::type_seq_element_t<
            0,
            ddc::to_type_seq_t<typename out_second_component_t::tensor_type::indices_domain_t>>;
    static_assert(!std::is_void_v<out_first_component_t>);
    static_assert(!std::is_void_v<out_second_component_t>);
    static_assert(std::is_same_v<typename out_first_component_t::tensor_type, second_in_tensor_t>);
    static_assert(std::is_same_v<typename out_second_component_t::tensor_type, first_in_tensor_t>);
    static_assert(first_in_index_t::rank() == 0);
    static_assert(second_in_index_t::rank() == 0);
    static_assert(first_out_index_t::rank() == 0);
    static_assert(second_out_index_t::rank() == 0);

    auto const in_first = in_form.template component<typename FirstComponent::tag>();
    auto const in_second = in_form.template component<typename SecondComponent::tag>();
    auto const out_first = out_form.template component<typename FirstComponent::tag>();
    auto const out_second = out_form.template component<typename SecondComponent::tag>();

    ddc::parallel_for_each(
            "similie_compute_tensor_form_hodge_star_first_component",
            exec_space,
            out_first.non_indices_domain(),
            KOKKOS_LAMBDA(typename second_in_tensor_t::non_indices_domain_t::discrete_element_type elem) {
                out_first.mem(elem, ddc::DiscreteElement<first_out_index_t>(0))
                        = -in_second.get(elem, ddc::DiscreteElement<second_in_index_t>(0));
            });
    ddc::parallel_for_each(
            "similie_compute_tensor_form_hodge_star_second_component",
            exec_space,
            out_second.non_indices_domain(),
            KOKKOS_LAMBDA(typename first_in_tensor_t::non_indices_domain_t::discrete_element_type elem) {
                out_second.mem(elem, ddc::DiscreteElement<second_out_index_t>(0))
                        = in_first.get(elem, ddc::DiscreteElement<first_in_index_t>(0));
            });
    return out_form;
}

template <
        tensor::TensorIndex MetricIndex,
        misc::Specialization<tensor::Tensor> OutTensorType,
        misc::Specialization<tensor::Tensor> InTensorType,
        misc::Specialization<tensor::Tensor> MetricType,
        class ExecSpace>
    requires(
            ddc::type_seq_element_t<0, ddc::to_type_seq_t<typename OutTensorType::indices_domain_t>>::rank()
                    == 0
            && ddc::type_seq_element_t<
                               0,
                               ddc::to_type_seq_t<typename InTensorType::indices_domain_t>>::rank()
                       == 0
            && OutTensorType::non_indices_domain_t::rank() == 2
            && InTensorType::non_indices_domain_t::rank() == 2)
OutTensorType hodge_star(
        ExecSpace const& exec_space,
        OutTensorType out_tensor,
        InTensorType in_tensor,
        MetricType metric)
{
    using out_index_t = ddc::type_seq_element_t<
            0,
            ddc::to_type_seq_t<typename OutTensorType::indices_domain_t>>;
    using in_index_t = ddc::type_seq_element_t<
            0,
            ddc::to_type_seq_t<typename InTensorType::indices_domain_t>>;
    using out_elem_t = typename OutTensorType::non_indices_domain_t::discrete_element_type;
    using in_elem_t = typename InTensorType::non_indices_domain_t::discrete_element_type;
    using metric_elem_t = typename MetricType::non_indices_domain_t::discrete_element_type;
    constexpr bool in_is_fully_dual = detail::is_fully_dual_domain_v<typename InTensorType::non_indices_domain_t>;

    ddc::parallel_for_each(
            "similie_compute_scalar_tensor_hodge_star_2d",
            exec_space,
            out_tensor.non_indices_domain(),
            KOKKOS_LAMBDA(out_elem_t out_elem) {
                in_elem_t const in_elem
                        = sil::mesher::detail::dualizer_remap_element<in_elem_t>(out_elem);
                metric_elem_t const metric_elem
                        = sil::mesher::detail::dualizer_remap_element<metric_elem_t>(out_elem);
                double const volume_factor = detail::primal_volume_factor<MetricIndex>(metric, metric_elem);
                double const scale = in_is_fully_dual ? 1. / volume_factor : volume_factor;
                out_tensor.mem(out_elem, ddc::DiscreteElement<out_index_t>(0))
                        = scale * in_tensor.get(in_elem, ddc::DiscreteElement<in_index_t>(0));
            });
    return out_tensor;
}

} // namespace exterior

} // namespace sil
