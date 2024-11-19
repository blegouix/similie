// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <ddc/ddc.hpp>

#include "tensor.hpp"

namespace sil {

namespace normalization {

template <
        misc::Specialization<MetricIndex> MetricIndex,
        misc::Specialization<Tensor> TensorType,
        misc::Specialization<ChunkSpan> PositionType,
        misc::Specialization<Tensor> MetricType>
KOKKOS_FUNCTION TensorType inplace_normalize(TensorType tensor, PositionType pos, MetricType metric)
{
    assert(tensor.non_indices_domain() == metric.non_indices_domain());
    tensor::tensor_accessor_for_domain_t<metric_prod_domain_t<
            MetricIndex,
            primes<ddc::detail::convert_discrete_domain_to_type_seq_t<
                    typename TensorType::non_indices_domain_t>>,
            second<ddc::detail::convert_discrete_domain_to_type_seq_t<
                    typename TensorType::non_indices_domain_t>>>>
            metric_prod_accessor;
    ddc::Chunk metric_prod_alloc(
            ddc::cartesian_prod_t<
                    typename TensorType::non_indices_domain_t,
                    metric_prod_domain_t<
                            MetricIndex,
                            primes<ddc::detail::convert_discrete_domain_to_type_seq_t<
                                    typename TensorType::non_indices_domain_t>>,
                            seconds<ddc::detail::convert_discrete_domain_to_type_seq_t<
                                    typename TensorType::non_indices_domain_t>>>>(
                    tensor.non_indices_domain(),
                    metric_prod_accessor.mem_domain()),
            ddc::HostAllocator<double>());
    tensor::Tensor<
            double,
            ddc::cartesian_prod_t<
                    typename TensorType::non_indices_domain_t,
                    metric_prod_domain_t<
                            MetricIndex,
                            primes<ddc::detail::convert_discrete_domain_to_type_seq_t<
                                    typename TensorType::non_indices_domain_t>>,
                            seconds<ddc::detail::convert_discrete_domain_to_type_seq_t<
                                    typename TensorType::non_indices_domain_t>>>>,
            Kokkos::layout_right,
            Kokkos::DefaultHostExecutionSpace::memory_space>
            metric_prod(metric_prod_alloc);

    fill_metric_prod<
            MetricIndex,
            primes<ddc::detail::convert_discrete_domain_to_type_seq_t<
                    typename TensorType::non_indices_domain_t>>,
            seconds<ddc::detail::convert_discrete_domain_to_type_seq_t<
                    typename TensorType::non_indices_domain_t>>>(metric_prod, metric);

    ddc::parallel_for_each(
            Kokkos::DefaultHostExecutionSpace(),
            tensor.domain(),
            [&](typename TensorType::discrete_element_type elem) {
                tensor[elem]
                        *= tensor::tensor_prod(metric_prod[elem], pos[elem] - pos[elem]); // TODO
            });
    return tensor;
}

} // namespace normalization

} // namespace sil
