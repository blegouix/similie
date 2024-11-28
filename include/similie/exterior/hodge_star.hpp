// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <ddc/ddc.hpp>

#include <similie/misc/factorial.hpp>
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

template <
        misc::Specialization<ddc::detail::TypeSeq> Indices1,
        misc::Specialization<ddc::detail::TypeSeq> Indices2>
using hodge_star_domain_t
        = ddc::detail::convert_type_seq_to_discrete_domain_t<ddc::type_seq_merge_t<
                ddc::detail::TypeSeq<
                        misc::convert_type_seq_to_t<tensor::TensorFullIndex, Indices1>>,
                ddc::detail::TypeSeq<
                        misc::convert_type_seq_to_t<tensor::TensorAntisymmetricIndex, Indices2>>>>;

template <
        misc::Specialization<ddc::detail::TypeSeq> Indices1,
        misc::Specialization<ddc::detail::TypeSeq> Indices2,
        misc::Specialization<tensor::Tensor> HodgeStarType,
        class MetricDeterminantType,
        misc::Specialization<tensor::Tensor> MetricProdType>
    requires(
            misc::Specialization<MetricDeterminantType, ddc::ChunkSpan>
            || misc::Specialization<MetricDeterminantType, tensor::Tensor>)
HodgeStarType fill_hodge_star(
        HodgeStarType hodge_star,
        MetricDeterminantType metric_determinant,
        MetricProdType metric_prod)
{
    sil::tensor::TensorAccessor<misc::convert_type_seq_to_t<
            tensor::TensorLeviCivitaIndex,
            ddc::type_seq_merge_t<tensor::primes<Indices1>, Indices2>>>
            levi_civita_accessor;
    ddc::DiscreteDomain<misc::convert_type_seq_to_t<
            tensor::TensorLeviCivitaIndex,
            ddc::type_seq_merge_t<tensor::primes<Indices1>, Indices2>>>
            levi_civita_dom(levi_civita_accessor.mem_domain());
    ddc::Chunk levi_civita_alloc(
            levi_civita_dom,
            ddc::HostAllocator<double>()); // TODO consider avoid allocation
    sil::tensor::Tensor<
            double,
            ddc::DiscreteDomain<misc::convert_type_seq_to_t<
                    tensor::TensorLeviCivitaIndex,
                    ddc::type_seq_merge_t<tensor::primes<Indices1>, Indices2>>>,
            Kokkos::layout_right,
            Kokkos::DefaultHostExecutionSpace::memory_space>
            levi_civita(levi_civita_alloc);

    ddc::parallel_for_each(
            Kokkos::DefaultHostExecutionSpace(),
            hodge_star.non_indices_domain(),
            [&](auto elem) {
                tensor_prod(hodge_star[elem], metric_prod[elem], levi_civita);
                hodge_star[elem] *= 1. / Kokkos::sqrt(Kokkos::abs(metric_determinant(elem)))
                                    / misc::factorial(ddc::type_seq_size_v<Indices1>);
            });
    return hodge_star;
}

template <
        tensor::TensorIndex MetricIndex,
        misc::Specialization<ddc::detail::TypeSeq> Indices1,
        misc::Specialization<ddc::detail::TypeSeq> Indices2,
        misc::Specialization<tensor::Tensor> HodgeStarType,
        misc::Specialization<tensor::Tensor> MetricType>
HodgeStarType fill_hodge_star(HodgeStarType hodge_star, MetricType metric)
{
    static_assert(tensor::are_contravariant_v<Indices1>);
    static_assert(tensor::are_covariant_v<Indices2>);
    ddc::Chunk metric_det_alloc(metric.non_indices_domain(), ddc::HostAllocator<double>());
    ddc::ChunkSpan<
            double,
            typename MetricType::non_indices_domain_t,
            Kokkos::layout_right,
            Kokkos::DefaultHostExecutionSpace::memory_space>
            metric_det(metric_det_alloc);
    ddc::for_each( // TODO parallel_for_each (weird lock)
            metric_det.domain(),
            [&](auto elem) { metric_det(elem) = tensor::determinant(metric[elem]); });

    tensor::tensor_accessor_for_domain_t<
            tensor::metric_prod_domain_t<MetricIndex, Indices1, tensor::primes<Indices1>>>
            metric_prod_accessor;
    ddc::Chunk metric_prod_alloc(
            ddc::cartesian_prod_t<
                    typename MetricType::non_indices_domain_t,
                    tensor::metric_prod_domain_t<MetricIndex, Indices1, tensor::primes<Indices1>>>(
                    metric.non_indices_domain(),
                    metric_prod_accessor.mem_domain()),
            ddc::HostAllocator<double>());
    tensor::Tensor<
            double,
            ddc::cartesian_prod_t<
                    typename MetricType::non_indices_domain_t,
                    tensor::metric_prod_domain_t<MetricIndex, Indices1, tensor::primes<Indices1>>>,
            Kokkos::layout_right,
            Kokkos::DefaultHostExecutionSpace::memory_space>
            metric_prod(metric_prod_alloc);

    fill_metric_prod<MetricIndex, Indices1, tensor::primes<Indices1>>(metric_prod, metric);

    return fill_hodge_star<Indices1, Indices2>(hodge_star, metric_det, metric_prod);
}

} // namespace exterior

} // namespace sil
