// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: MIT

#pragma once

#include <array>

#include <ddc/ddc.hpp>

#include <similie/misc/binomial_coefficient.hpp>
#include <similie/tensor/gram_matrix.hpp>

#include <KokkosBatched_LU_Decl.hpp>

namespace sil {

namespace exterior {

enum class DualStrategy {
    Circumcentric,
    Barycentric,
};

namespace detail {

struct VolumeMatrixIndex
{
};

struct VolumePrimeMatrixIndex
{
};

template <std::size_t K, class MemorySpace>
KOKKOS_FUNCTION double determinant(std::array<double, K * K> const& matrix)
{
    if constexpr (K == 0) {
        return 1.;
    } else {
        std::array<double, K * K> reduced_alloc {};
        ddc::DiscreteDomain<VolumeMatrixIndex, VolumePrimeMatrixIndex> reduced_domain(
                ddc::DiscreteElement<VolumeMatrixIndex, VolumePrimeMatrixIndex>(0, 0),
                ddc::DiscreteVector<VolumeMatrixIndex, VolumePrimeMatrixIndex>(K, K));
        ddc::ChunkSpan<
                double,
                ddc::DiscreteDomain<VolumeMatrixIndex, VolumePrimeMatrixIndex>,
                Kokkos::layout_right,
                MemorySpace>
                reduced(reduced_alloc.data(), reduced_domain);
        auto reduced_view = reduced.allocation_kokkos_view();

        for (std::size_t i = 0; i < K; ++i) {
            for (std::size_t j = 0; j < K; ++j) {
                reduced_view(i, j) = matrix[i * K + j];
            }
        }

        int const err = KokkosBatched::SerialLU<KokkosBatched::Algo::SolveLU::Unblocked>::invoke(
                reduced.allocation_kokkos_view());
        if (err != 0) {
            return 0.;
        }

        double det = 1.;
        for (std::size_t i = 0; i < K; ++i) {
            det *= reduced_view(i, i);
        }
        return det;
    }
}

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

template <DualStrategy Strategy, std::size_t N>
KOKKOS_FUNCTION double dual_volume_factor(std::size_t const simplex_dim)
{
    if constexpr (Strategy == DualStrategy::Circumcentric) {
        return 1.;
    } else {
        return 1. / static_cast<double>(misc::binomial_coefficient(N, simplex_dim));
    }
}

} // namespace detail

template <std::size_t N, class MetricType, class PositionType, class BatchElem>
struct SimplexVolume
{
    template <std::size_t K>
    KOKKOS_FUNCTION static double run(
            MetricType metric,
            PositionType position,
            BatchElem elem,
            std::array<std::size_t, K> const& ids)
    {
        double const det = detail::determinant<K, typename MetricType::memory_space>(
                tensor::GramMatrix<MetricType, PositionType, BatchElem, N>::template value<
                        K>(metric, position, elem, ids));
        return Kokkos::sqrt(Kokkos::abs(det));
    }
};

template <
        DualStrategy Strategy,
        std::size_t N,
        class MetricType,
        class PositionType,
        class BatchElem>
struct DualSimplexVolume
{
    template <std::size_t K>
    KOKKOS_FUNCTION static double run(
            MetricType metric,
            PositionType position,
            BatchElem elem,
            std::array<std::size_t, K> const& ids)
    {
        return detail::dual_volume_factor<Strategy, N>(K)
               * SimplexVolume<N, MetricType, PositionType, BatchElem>::
                       run(metric, position, elem, detail::complement<N>(ids));
    }
};

} // namespace exterior

} // namespace sil
