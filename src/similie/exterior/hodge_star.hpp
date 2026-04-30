// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: MIT

#pragma once

#include <ddc/ddc.hpp>

#include <similie/misc/factorial.hpp>
#include <similie/misc/macros.hpp>
#include <similie/misc/specialization.hpp>
#include <similie/misc/type_seq_conversion.hpp>
#include <similie/tensor/antisymmetric_tensor.hpp>
#include <similie/tensor/character.hpp>
#include <similie/tensor/determinant.hpp>
#include <similie/tensor/full_tensor.hpp>
#include <similie/tensor/levi_civita_tensor.hpp>
#include <similie/tensor/metric.hpp>
#include <similie/tensor/prime.hpp>

namespace sil {

namespace exterior {

namespace detail {

template <class... DDim>
bool can_use_cochain_measure_scaling(ddc::DiscreteDomain<DDim...> const&)
{
    return (ddc::is_discrete_space_initialized<DDim>() && ...);
}

template <class DDim, class Domain>
KOKKOS_FUNCTION double primal_length(Domain const& domain)
{
    auto const subdomain = ddc::DiscreteDomain<DDim>(domain);
    if (subdomain.template extent<DDim>() <= 1) {
        return 1.;
    }
    auto const first = subdomain.front();
    auto const second = first + ddc::DiscreteVector<DDim>(1);
    return static_cast<double>(ddc::coordinate(second) - ddc::coordinate(first));
}

template <class DDim, class Domain, class Elem>
KOKKOS_FUNCTION double dual_length(Domain const& domain, Elem const& elem)
{
    double const h = primal_length<DDim>(domain);
    auto const subdomain = ddc::DiscreteDomain<DDim>(domain);
    auto const uid = ddc::uid<DDim>(elem);
    if (uid == subdomain.front().template uid<DDim>()
        || uid == subdomain.back().template uid<DDim>()) {
        return 0.5 * h;
    }
    return h;
}

template <std::size_t Pos, class... Index>
KOKKOS_FUNCTION bool orientation_contains(ddc::DiscreteElement<Index...> const& natural_elem)
{
    return ((natural_elem.template uid<Index>() == Pos) || ...);
}

template <std::size_t Pos, class DDimSeq>
struct ForwardMeasureRatio;

template <std::size_t Pos>
struct ForwardMeasureRatio<Pos, ddc::detail::TypeSeq<>>
{
    template <class Domain, class Elem, class NaturalElem>
    KOKKOS_FUNCTION static double run(Domain const&, Elem const&, NaturalElem const&)
    {
        return 1.;
    }
};

template <std::size_t Pos, class DDim, class... Tail>
struct ForwardMeasureRatio<Pos, ddc::detail::TypeSeq<DDim, Tail...>>
{
    template <class Domain, class Elem, class NaturalElem>
    KOKKOS_FUNCTION static double run(
            Domain const& domain,
            Elem const& elem,
            NaturalElem const& natural_elem)
    {
        double const factor = orientation_contains<Pos>(natural_elem)
                                      ? 1. / primal_length<DDim>(domain)
                                      : dual_length<DDim>(domain, elem);
        return factor
               * ForwardMeasureRatio<Pos + 1, ddc::detail::TypeSeq<Tail...>>::
                       run(domain, elem, natural_elem);
    }
};

template <std::size_t Pos, class DDimSeq>
struct InverseMeasureRatio;

template <std::size_t Pos>
struct InverseMeasureRatio<Pos, ddc::detail::TypeSeq<>>
{
    template <class Domain, class Elem, class NaturalElem>
    KOKKOS_FUNCTION static double run(Domain const&, Elem const&, NaturalElem const&)
    {
        return 1.;
    }
};

template <std::size_t Pos, class DDim, class... Tail>
struct InverseMeasureRatio<Pos, ddc::detail::TypeSeq<DDim, Tail...>>
{
    template <class Domain, class Elem, class NaturalElem>
    KOKKOS_FUNCTION static double run(
            Domain const& domain,
            Elem const& elem,
            NaturalElem const& natural_elem)
    {
        double const factor
                = orientation_contains<Pos>(natural_elem)
                          ? 1.
                          : primal_length<DDim>(domain) / dual_length<DDim>(domain, elem);
        return factor
               * InverseMeasureRatio<Pos + 1, ddc::detail::TypeSeq<Tail...>>::
                       run(domain, elem, natural_elem);
    }
};

template <
        misc::Specialization<ddc::detail::TypeSeq> Indices1,
        misc::Specialization<tensor::Tensor> HodgeStarType,
        class ExecSpace>
void apply_forward_cochain_measure_scaling(ExecSpace const& exec_space, HodgeStarType hodge_star)
{
    using input_index_t = misc::convert_type_seq_to_t<tensor::TensorFullIndex, Indices1>;
    tensor::TensorAccessor<input_index_t> input_accessor;
    ddc::parallel_for_each(
            "similie_apply_forward_cochain_measure_scaling",
            exec_space,
            hodge_star.domain(),
            KOKKOS_LAMBDA(typename HodgeStarType::discrete_element_type elem) {
                auto const batch_elem =
                        typename HodgeStarType::non_indices_domain_t::discrete_element_type(elem);
                auto const input_elem = ddc::DiscreteElement<input_index_t>(elem);
                auto const input_natural_elem
                        = input_accessor.canonical_natural_element(input_elem);
                hodge_star(elem) *= ForwardMeasureRatio<
                        0,
                        ddc::to_type_seq_t<typename HodgeStarType::non_indices_domain_t>>::
                        run(hodge_star.non_indices_domain(), batch_elem, input_natural_elem);
            });
}

template <
        misc::Specialization<ddc::detail::TypeSeq> Indices1,
        misc::Specialization<tensor::Tensor> HodgeStarType,
        class ExecSpace>
void apply_inverse_cochain_measure_scaling(ExecSpace const& exec_space, HodgeStarType hodge_star)
{
    using input_index_t = misc::convert_type_seq_to_t<tensor::TensorFullIndex, Indices1>;
    tensor::TensorAccessor<input_index_t> input_accessor;
    ddc::parallel_for_each(
            "similie_apply_inverse_cochain_measure_scaling",
            exec_space,
            hodge_star.domain(),
            KOKKOS_LAMBDA(typename HodgeStarType::discrete_element_type elem) {
                auto const batch_elem =
                        typename HodgeStarType::non_indices_domain_t::discrete_element_type(elem);
                auto const input_elem = ddc::DiscreteElement<input_index_t>(elem);
                auto const input_natural_elem
                        = input_accessor.canonical_natural_element(input_elem);
                hodge_star(elem) *= InverseMeasureRatio<
                        0,
                        ddc::to_type_seq_t<typename HodgeStarType::non_indices_domain_t>>::
                        run(hodge_star.non_indices_domain(), batch_elem, input_natural_elem);
            });
}

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
    fill_hodge_star<Indices1, Indices2>(exec_space, hodge_star, metric_det, metric_prod);

    if constexpr (misc::Specialization<MetricIndex, tensor::TensorIdentityIndex>) {
        if (detail::can_use_cochain_measure_scaling(hodge_star.non_indices_domain())) {
            detail::apply_forward_cochain_measure_scaling<Indices1>(exec_space, hodge_star);
        }
    }

    return hodge_star;
}

template <
        tensor::TensorIndex MetricIndex,
        misc::Specialization<ddc::detail::TypeSeq> Indices1,
        misc::Specialization<ddc::detail::TypeSeq> Indices2,
        misc::Specialization<tensor::Tensor> HodgeStarType,
        misc::Specialization<tensor::Tensor> MetricType,
        class ExecSpace>
HodgeStarType fill_inverse_hodge_star(
        ExecSpace const& exec_space,
        HodgeStarType inverse_hodge_star,
        MetricType metric)
{
    fill_hodge_star<MetricIndex, Indices1, Indices2>(exec_space, inverse_hodge_star, metric);

    if constexpr (misc::Specialization<MetricIndex, tensor::TensorIdentityIndex>) {
        if (detail::can_use_cochain_measure_scaling(inverse_hodge_star.non_indices_domain())) {
            detail::apply_inverse_cochain_measure_scaling<Indices1>(exec_space, inverse_hodge_star);
        }
    }

    constexpr std::size_t in_degree = ddc::type_seq_size_v<Indices1>;
    constexpr std::size_t out_degree = ddc::type_seq_size_v<Indices2>;
    if constexpr ((in_degree * out_degree) % 2 == 1) {
        ddc::parallel_for_each(
                "similie_apply_inverse_hodge_star_sign",
                exec_space,
                inverse_hodge_star.domain(),
                KOKKOS_LAMBDA(typename HodgeStarType::discrete_element_type elem) {
                    inverse_hodge_star(elem) *= -1.;
                });
    }

    return inverse_hodge_star;
}

} // namespace exterior

} // namespace sil
