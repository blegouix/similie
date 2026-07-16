// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

#include <array>

#include <ddc/ddc.hpp>

#include <similie/misc/binomial_coefficient.hpp>
#include <similie/misc/clamp_to_domain.hpp>
#include <similie/misc/domain_contains.hpp>
#include <similie/misc/factorial.hpp>
#include <similie/misc/small_matrix.hpp>
#include <similie/tensor/metric.hpp>

namespace sil {

namespace exterior {

enum class CellComplex {
    Primal,
    CircumcentricDual,
    BarycentricDual,
};

namespace detail {
template <std::size_t N, std::size_t K>
KOKKOS_FUNCTION std::array<std::size_t, N - K> complement(std::array<std::size_t, K> const& ids)
{
    std::array<std::size_t, N - K> complement_ids {};
    std::size_t complement_id = 0;
    for (std::size_t i = 0; i < N; ++i) {
        bool found = false;
        for (std::size_t id : ids) {
            found = found || id == i;
        }
        if (!found) {
            complement_ids[complement_id++] = i;
        }
    }
    return complement_ids;
}

template <CellComplex Complex, std::size_t N>
KOKKOS_FUNCTION double complex_volume_factor(std::size_t const simplex_dim)
{
    if constexpr (Complex == CellComplex::Primal || Complex == CellComplex::CircumcentricDual) {
        return 1.;
    } else {
        return 1. / static_cast<double>(misc::binomial_coefficient(N, simplex_dim));
    }
}

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

template <CellComplex Complex, std::size_t N, class MetricType, class PositionType, class BatchElem>
struct SimplexVolume
{
    template <std::size_t K>
    KOKKOS_FUNCTION static double run(
            MetricType metric,
            PositionType position,
            BatchElem elem,
            std::array<std::size_t, K> const& ids)
    {
        if constexpr (K == 0) {
            return 1.;
        } else {
            using PositionIndex = ddc::type_seq_element_t<
                    0,
                    ddc::to_type_seq_t<typename PositionType::indices_domain_t>>;
            using MetricIndex = ddc::type_seq_element_t<
                    0,
                    ddc::to_type_seq_t<typename MetricType::indices_domain_t>>;
            using MetricIndex1 = tensor::metric_index_1<MetricIndex>;
            using MetricIndex2 = tensor::metric_index_2<MetricIndex>;

            std::array<double, N * N> local_metric {};
            for (std::size_t i = 0; i < N; ++i) {
                for (std::size_t j = 0; j < N; ++j) {
                    local_metric[i * N + j] = metric.get(metric.access_element(
                            elem,
                            ddc::DiscreteElement<MetricIndex1, MetricIndex2>(i, j)));
                }
            }

            std::array<std::array<double, N>, K> edges {};
            for (std::size_t i = 0; i < K; ++i) {
                edges[i] = detail::edge_vector<PositionIndex>(position, elem, ids[i]);
            }

            std::array<double, K * K> gram_matrix {};
            for (std::size_t i = 0; i < K; ++i) {
                for (std::size_t j = i; j < K; ++j) {
                    double const value = detail::dot(edges[i], edges[j], local_metric);
                    gram_matrix[i * K + j] = value;
                    gram_matrix[j * K + i] = value;
                }
            }

            return Kokkos::sqrt(Kokkos::abs([&]() {
                std::array<double, K * K> determinant_alloc = gram_matrix;
                auto determinant_view = misc::math::matrix_view<
                        double,
                        typename MetricType::memory_space>(determinant_alloc.data(), K, K);
                return misc::math::determinant(determinant_view);
            }()));
        }
    }
};

template <CellComplex Complex, std::size_t N, class MetricType, class PositionType, class BatchElem>
struct DualSimplexVolume
{
    template <std::size_t K>
    KOKKOS_FUNCTION static double run(
            MetricType metric,
            PositionType position,
            BatchElem elem,
            std::array<std::size_t, K> const& ids)
    {
        static_assert(
                Complex != CellComplex::Primal,
                "DualSimplexVolume must use a dual cell complex.");
        return SimplexVolume<Complex, N, MetricType, PositionType, BatchElem>::template run<
                N - K>(metric, position, elem, detail::complement<N>(ids));
    }
};

} // namespace exterior

} // namespace sil
