// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: GPL-3.0

#pragma once

#include <ddc/ddc.hpp>

#include "symmetric_tensor.hpp"

namespace sil {

namespace tensor {

// Abstract indexes used to characterize a generic metric
struct MetricIndex1
{
};

struct MetricIndex2
{
};

// Helpers to metrify or unmetrify an index
namespace detail {
template <class IndexToMetrify>
struct Metrify;

template <
        template <class..., class, class>
        class TensorIndexType,
        class TensorIndex1,
        class TensorIndex2,
        class... HeadArgs>
struct Metrify<TensorIndexType<HeadArgs..., TensorIndex1, TensorIndex2>>
    : TensorIndexType<HeadArgs..., MetricIndex1, MetricIndex2>
{
};

template <class IndexToUnmetrify, class TensorIndex1, class TensorIndex2>
struct Unmetrify;

template <
        template <class..., class, class>
        class TensorIndexType,
        class... HeadArgs,
        class TensorIndex1,
        class TensorIndex2>
struct Unmetrify<
        TensorIndexType<HeadArgs..., TensorIndex1, TensorIndex2>,
        TensorIndex1,
        TensorIndex2> : TensorIndexType<HeadArgs..., TensorIndex1, TensorIndex2>
{
};

} // namespace detail

// Relabelize metric
template <class MetricType, class MetricIndex, class Index1, class Index2>
using relabelize_metric_t = relabelize_index_of_t<
        relabelize_index_of_t<MetricType, MetricIndex1, Index1>,
        MetricIndex2,
        Index2>;

template <class MetricIndex, class Index1, class Index2, class MetricType>
relabelize_metric_t<MetricType, MetricIndex, Index1, Index2> relabelize_metric(MetricType metric)
{
    return relabelize_index_of<MetricIndex2, Index2>(
            relabelize_index_of<MetricIndex1, Index1>(metric));
}

// Inplace apply metric
template <class MetricIndex, class Index1, class Index2, class MetricType, class TensorType>
relabelize_index_of_t<TensorType, Index2, Index1> inplace_apply_metric(
        MetricType metric,
        TensorType tensor)
{
    ddc::Chunk result_alloc = ddc::create_mirror(tensor);
    relabelize_index_of_t<TensorType, Index2, Index1> result
            = relabelize_index_of<Index2, Index1>(TensorType(result_alloc));
    tensor_prod3(result, relabelize_metric<MetricIndex, Index1, Index2>(metric), tensor);

    Kokkos::deep_copy(
            tensor.allocation_kokkos_view(),
            result.allocation_kokkos_view()); // We rely on Kokkos::deep_copy in place of ddc::parallel_deepcopy to avoid type verification of the type dimensions

    return relabelize_index_of<Index2, Index1>(tensor);
}

namespace detail {

// Inplace lower index
template <class MetricIndex, class Indexes1, class Indexes2>
struct InplaceApplyMetrics;

template <class MetricIndex>
struct InplaceApplyMetrics<MetricIndex, ddc::detail::TypeSeq<>, ddc::detail::TypeSeq<>>
{
    template <class KokkosViewType, class MetricType, class TensorType>
    static TensorType run(KokkosViewType result_kokkos_view, MetricType metric, TensorType tensor)
    {
        return tensor;
    }
};

template <
        class MetricIndex,
        class HeadIndex1,
        class... TailIndex1,
        class HeadIndex2,
        class... TailIndex2>
struct InplaceApplyMetrics<
        MetricIndex,
        ddc::detail::TypeSeq<HeadIndex1, TailIndex1...>,
        ddc::detail::TypeSeq<HeadIndex2, TailIndex2...>>
{
    template <class KokkosViewType, class MetricType, class TensorType>
    static auto run(KokkosViewType result_kokkos_view, MetricType metric, TensorType tensor)
    {
        relabelize_index_of_t<TensorType, HeadIndex2, HeadIndex1>
                result(result_kokkos_view,
                       relabelize_index_of<HeadIndex2, HeadIndex1>(tensor).domain());
        tensor_prod3(
                result,
                relabelize_metric<MetricIndex, HeadIndex1, HeadIndex2>(metric),
                tensor);

        Kokkos::deep_copy(
                tensor.allocation_kokkos_view(),
                result.allocation_kokkos_view()); // We rely on Kokkos::deep_copy in place of ddc::parallel_deepcopy to avoid type verification of the type dimensions
        // TODO use swap ?

        return InplaceApplyMetrics<
                MetricIndex,
                ddc::detail::TypeSeq<TailIndex1...>,
                ddc::detail::TypeSeq<TailIndex2...>>::
                run(result_kokkos_view,
                    metric,
                    relabelize_index_of<HeadIndex2, HeadIndex1>(tensor));
    }
};

} // namespace detail

template <class MetricIndex, class Indexes1, class Indexes2, class MetricType, class TensorType>
relabelize_indexes_of_t<TensorType, Indexes2, Indexes1>
inplace_apply_metrics( // TODO avoid metricS by using concepts
        MetricType metric,
        TensorType tensor)
{
    ddc::Chunk result_alloc = ddc::create_mirror(tensor);
    return detail::InplaceApplyMetrics<MetricIndex, Indexes1, Indexes2>::
            run(result_alloc.allocation_kokkos_view(), metric, tensor);
}

} // namespace tensor

} // namespace sil
