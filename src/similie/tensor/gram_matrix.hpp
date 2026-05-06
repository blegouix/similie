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

template <class PositionIndex, class PositionType, class BatchElem>
KOKKOS_FUNCTION std::array<double, PositionIndex::size()> edge_vector(
        PositionType position,
        BatchElem elem,
        std::size_t const dim_id)
{
    std::array<double, PositionIndex::size()> vector {};
    BatchElem forward = elem;
    BatchElem backward = elem;
    ddc::detail::array(forward)[dim_id] += 1;
    ddc::detail::array(backward)[dim_id] -= 1;

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

} // namespace detail

template <class MetricType, class PositionType, class BatchElem, std::size_t N>
struct GramMatrix
{
    template <std::size_t K>
    KOKKOS_FUNCTION static std::array<double, K * K> value(
            MetricType metric,
            PositionType position,
            BatchElem elem,
            std::array<std::size_t, K> const& ids)
    {
        using PositionIndex = ddc::
                type_seq_element_t<0, ddc::to_type_seq_t<typename PositionType::indices_domain_t>>;
        using MetricIndex = ddc::
                type_seq_element_t<0, ddc::to_type_seq_t<typename MetricType::indices_domain_t>>;
        using MetricIndex1 = tensor::metric_index_1<MetricIndex>;
        using MetricIndex2 = tensor::metric_index_2<MetricIndex>;

        std::array<double, K * K> gram {};
        std::array<double, N * N> local_metric {};
        std::array<std::array<double, N>, N> edges {};

        for (std::size_t i = 0; i < N; ++i) {
            for (std::size_t j = 0; j < N; ++j) {
                local_metric[i * N + j] = metric.get(metric.access_element(
                        elem,
                        ddc::DiscreteElement<MetricIndex1, MetricIndex2>(i, j)));
            }
        }

        for (std::size_t id : ids) {
            edges[id] = detail::edge_vector<PositionIndex>(position, elem, id);
        }

        for (std::size_t i = 0; i < K; ++i) {
            for (std::size_t j = i; j < K; ++j) {
                double const value = detail::dot(edges[ids[i]], edges[ids[j]], local_metric);
                gram[i * K + j] = value;
                gram[j * K + i] = value;
            }
        }

        return gram;
    }
};

} // namespace tensor

} // namespace sil
