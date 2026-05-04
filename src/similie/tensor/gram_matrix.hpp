// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: MIT

#pragma once

#include <array>

#include <ddc/ddc.hpp>

#include <similie/misc/domain_contains.hpp>

#include "metric.hpp"

namespace sil {

namespace tensor {

namespace detail {

template <class PositionIndex, class... DDim>
KOKKOS_FUNCTION constexpr ddc::DiscreteElement<DDim...> shift_along_dimension(
        ddc::DiscreteElement<DDim...> elem,
        std::size_t const dim_id,
        std::ptrdiff_t const offset)
{
    int const dummy[]
            = {((dim_id
                 == ddc::type_seq_rank_v<
                         typename DDim::continuous_dimension_type,
                         typename PositionIndex::type_seq_dimensions>)
                        ? (elem.template uid<DDim>() += offset, 0)
                        : 0)...};
    (void)dummy;
    return elem;
}

template <class PositionIndex, class PositionType, class BatchElem>
KOKKOS_FUNCTION std::array<double, PositionIndex::size()> edge_vector(
        PositionType position,
        BatchElem elem,
        std::size_t const dim_id)
{
    std::array<double, PositionIndex::size()> vector {};
    BatchElem const forward = shift_along_dimension<PositionIndex>(elem, dim_id, 1);
    BatchElem const backward = shift_along_dimension<PositionIndex>(elem, dim_id, -1);

    bool const has_forward = misc::domain_contains(position.non_indices_domain(), forward);
    bool const has_backward = misc::domain_contains(position.non_indices_domain(), backward);

    for (std::size_t comp = 0; comp < PositionIndex::size(); ++comp) {
        ddc::DiscreteElement<PositionIndex> const component(comp);
        double const center = position.get(position.access_element(elem, component));
        if (has_forward) {
            vector[comp] = position.get(position.access_element(forward, component)) - center;
        } else if (has_backward) {
            vector[comp] = center - position.get(position.access_element(backward, component));
        } else {
            vector[comp] = 0.;
        }
    }
    return vector;
}

template <class MetricIndex, class MetricType, class BatchElem>
KOKKOS_FUNCTION double metric_component(
        MetricType metric,
        BatchElem elem,
        std::size_t const i,
        std::size_t const j)
{
    using MetricIndex1 = tensor::metric_index_1<MetricIndex>;
    using MetricIndex2 = tensor::metric_index_2<MetricIndex>;
    return metric.get(
            metric.access_element(elem, ddc::DiscreteElement<MetricIndex1, MetricIndex2>(i, j)));
}

template <std::size_t N>
KOKKOS_FUNCTION double dot(
        std::array<double, N> const& lhs,
        std::array<double, N> const& rhs,
        std::array<double, N * N> const& local_metric)
{
    double result = 0.;
    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = 0; j < N; ++j) {
            result += lhs[i] * local_metric[i * N + j] * rhs[j];
        }
    }
    return result;
}

template <std::size_t N, class MetricType, class BatchElem>
KOKKOS_FUNCTION std::array<double, N * N> local_metric(MetricType metric, BatchElem elem)
{
    std::array<double, N * N> local_metric_values {};
    using MetricIndex
            = ddc::type_seq_element_t<0, ddc::to_type_seq_t<typename MetricType::indices_domain_t>>;
    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = 0; j < N; ++j) {
            local_metric_values[i * N + j] = metric_component<MetricIndex>(metric, elem, i, j);
        }
    }
    return local_metric_values;
}

template <class DDimSeq, class PositionIndex, class PositionType, class BatchElem, std::size_t N>
struct GramMatrixBuilder;

template <class... DDim, class PositionIndex, class PositionType, class BatchElem, std::size_t N>
struct GramMatrixBuilder<ddc::detail::TypeSeq<DDim...>, PositionIndex, PositionType, BatchElem, N>
{
    KOKKOS_FUNCTION static std::array<double, N * N> run(
            auto metric,
            PositionType position,
            BatchElem elem,
            std::array<bool, N> const& active_dims)
    {
        static_assert(sizeof...(DDim) == N);

        std::array<double, N * N> gram {};
        std::array<double, N * N> const local_metric_values = local_metric<N>(metric, elem);
        std::array<std::array<double, N>, N> edges {};

        for (std::size_t i = 0; i < N; ++i) {
            if (active_dims[i]) {
                edges[i] = edge_vector<PositionIndex>(position, elem, i);
            }
        }

        for (std::size_t i = 0; i < N; ++i) {
            if (!active_dims[i]) {
                continue;
            }
            for (std::size_t j = i; j < N; ++j) {
                if (!active_dims[j]) {
                    continue;
                }
                double const value = dot(edges[i], edges[j], local_metric_values);
                gram[i * N + j] = value;
                gram[j * N + i] = value;
            }
        }

        return gram;
    }
};

} // namespace detail

template <class MetricType, class PositionType, class BatchElem, std::size_t N>
KOKKOS_FUNCTION std::array<double, N * N> gram_matrix(
        MetricType metric,
        PositionType position,
        BatchElem elem,
        std::array<bool, N> const& active_dims)
{
    using PositionIndex = ddc::
            type_seq_element_t<0, ddc::to_type_seq_t<typename PositionType::indices_domain_t>>;
    return detail::GramMatrixBuilder<
            ddc::to_type_seq_t<typename PositionType::non_indices_domain_t>,
            PositionIndex,
            PositionType,
            BatchElem,
            N>::run(metric, position, elem, active_dims);
}

} // namespace tensor

} // namespace sil
