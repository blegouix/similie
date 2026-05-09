// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: MIT

#pragma once

#include <array>

#include <ddc/ddc.hpp>

#include <similie/misc/factorial.hpp>
#include <similie/misc/macros.hpp>
#include <similie/misc/specialization.hpp>
#include <similie/misc/type_seq_conversion.hpp>
#include <similie/tensor/antisymmetric_tensor.hpp>
#include <similie/tensor/character.hpp>
#include <similie/tensor/full_tensor.hpp>
#include <similie/tensor/metric.hpp>

#include "volume.hpp"

namespace sil {

namespace exterior {

template <
        misc::Specialization<ddc::detail::TypeSeq> Indices1,
        misc::Specialization<ddc::detail::TypeSeq> Indices2>
using hodge_star_domain_t
        = ddc::detail::convert_type_seq_to_discrete_domain_t<ddc::type_seq_merge_t<
                ddc::detail::TypeSeq<
                        misc::convert_type_seq_to_t<tensor::TensorFullIndex, Indices1>>,
                std::conditional_t<
                        (ddc::type_seq_size_v<Indices2> == 0),
                        ddc::detail::TypeSeq<>,
                        ddc::detail::TypeSeq<misc::convert_type_seq_to_t<
                                tensor::TensorAntisymmetricIndex,
                                Indices2>>>>>;

namespace detail {

template <class Indices1, class Indices2>
struct AmbientIndex;

template <class HeadIndex, class... TailIndex, class Indices2>
struct AmbientIndex<ddc::detail::TypeSeq<HeadIndex, TailIndex...>, Indices2>
{
    using type = HeadIndex;
};

template <class HeadIndex, class... TailIndex>
struct AmbientIndex<ddc::detail::TypeSeq<>, ddc::detail::TypeSeq<HeadIndex, TailIndex...>>
{
    using type = HeadIndex;
};

template <std::size_t N, std::size_t M>
KOKKOS_FUNCTION bool has_unique_ids(std::array<std::size_t, M> const& ids)
{
    for (std::size_t i = 0; i < M; ++i) {
        if (ids[i] >= N) {
            return false;
        }
        for (std::size_t j = i + 1; j < M; ++j) {
            if (ids[i] == ids[j]) {
                return false;
            }
        }
    }
    return true;
}

template <std::size_t N, std::size_t M1, std::size_t M2>
KOKKOS_FUNCTION bool is_complete_permutation(
        std::array<std::size_t, M1> const& source_ids,
        std::array<std::size_t, M2> const& target_ids)
{
    std::array<int, N> counts {};
    for (std::size_t id : source_ids) {
        if (id >= N) {
            return false;
        }
        counts[id]++;
    }
    for (std::size_t id : target_ids) {
        if (id >= N) {
            return false;
        }
        counts[id]++;
    }
    for (int count : counts) {
        if (count != 1) {
            return false;
        }
    }
    return true;
}

template <std::size_t N, std::size_t M1, std::size_t M2>
KOKKOS_FUNCTION int permutation_sign(
        std::array<std::size_t, M1> const& source_ids,
        std::array<std::size_t, M2> const& target_ids)
{
    std::array<std::size_t, N> permutation {};
    for (std::size_t i = 0; i < M1; ++i) {
        permutation[i] = source_ids[i];
    }
    for (std::size_t i = 0; i < M2; ++i) {
        permutation[M1 + i] = target_ids[i];
    }

    bool odd = false;
    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = i + 1; j < N; ++j) {
            odd = (permutation[i] > permutation[j]) != odd;
        }
    }
    return odd ? -1 : 1;
}

template <std::size_t N, std::size_t M>
KOKKOS_FUNCTION std::array<std::size_t, N - M> complement_ids(std::array<std::size_t, M> const& ids)
{
    std::array<std::size_t, N - M> complement {};
    std::size_t complement_id = 0;
    for (std::size_t i = 0; i < N; ++i) {
        bool found = false;
        for (std::size_t id : ids) {
            found = found || (id == i);
        }
        if (!found) {
            complement[complement_id++] = i;
        }
    }
    return complement;
}

struct HodgeStarMatrixIndex
{
};

struct HodgeStarPrimeMatrixIndex
{
};

template <misc::Specialization<ddc::detail::TypeSeq> Indices>
struct HodgeStarTargetSize
{
    static constexpr std::size_t value
            = misc::convert_type_seq_to_t<tensor::TensorAntisymmetricIndex, Indices>::access_size();
};

template <>
struct HodgeStarTargetSize<ddc::detail::TypeSeq<>>
{
    static constexpr std::size_t value = 1;
};

template <std::size_t N, class MemorySpace>
KOKKOS_FUNCTION bool invert_matrix(
        std::array<double, N * N>& inverse_alloc,
        double& determinant,
        std::array<double, N * N> const& matrix_alloc)
{
    std::array<double, N * N> matrix = matrix_alloc;
    inverse_alloc.fill(0.);
    for (std::size_t i = 0; i < N; ++i) {
        inverse_alloc[i * N + i] = 1.;
    }

    determinant = 1.;
    int sign = 1;

    for (std::size_t i = 0; i < N; ++i) {
        std::size_t pivot = i;
        double pivot_abs = Kokkos::abs(matrix[i * N + i]);
        for (std::size_t j = i + 1; j < N; ++j) {
            double const candidate_abs = Kokkos::abs(matrix[j * N + i]);
            if (candidate_abs > pivot_abs) {
                pivot = j;
                pivot_abs = candidate_abs;
            }
        }

        if (pivot_abs == 0.) {
            determinant = 0.;
            inverse_alloc.fill(0.);
            return false;
        }

        if (pivot != i) {
            sign *= -1;
            for (std::size_t j = 0; j < N; ++j) {
                Kokkos::kokkos_swap(matrix[i * N + j], matrix[pivot * N + j]);
                Kokkos::kokkos_swap(inverse_alloc[i * N + j], inverse_alloc[pivot * N + j]);
            }
        }

        double const diagonal = matrix[i * N + i];
        determinant *= diagonal;

        for (std::size_t j = 0; j < N; ++j) {
            matrix[i * N + j] /= diagonal;
            inverse_alloc[i * N + j] /= diagonal;
        }

        for (std::size_t row = 0; row < N; ++row) {
            if (row == i) {
                continue;
            }
            double const factor = matrix[row * N + i];
            for (std::size_t col = 0; col < N; ++col) {
                matrix[row * N + col] -= factor * matrix[i * N + col];
                inverse_alloc[row * N + col] -= factor * inverse_alloc[i * N + col];
            }
        }
    }

    determinant *= static_cast<double>(sign);
    return true;
}

template <std::size_t N, std::size_t K, class MemorySpace>
KOKKOS_FUNCTION double submatrix_determinant(
        std::array<double, N * N> const& matrix,
        std::array<std::size_t, K> const& row_ids,
        std::array<std::size_t, K> const& col_ids)
{
    if constexpr (K == 0) {
        return 1.;
    } else {
        std::array<double, K * K> submatrix_alloc {};
        for (std::size_t i = 0; i < K; ++i) {
            for (std::size_t j = 0; j < K; ++j) {
                submatrix_alloc[i * K + j] = matrix[row_ids[i] * N + col_ids[j]];
            }
        }

        std::array<double, K * K> inverse_alloc {};
        double determinant = 0.;
        if (!invert_matrix<K, MemorySpace>(inverse_alloc, determinant, submatrix_alloc)) {
            return 0.;
        }
        return determinant;
    }
}

} // namespace detail

template <
        misc::Specialization<ddc::detail::TypeSeq> Indices1,
        misc::Specialization<ddc::detail::TypeSeq> Indices2,
        class MetricType,
        class BatchElem>
struct ContinuousHodgeStar;

template <
        CellComplex Complex,
        misc::Specialization<ddc::detail::TypeSeq> Indices1,
        misc::Specialization<ddc::detail::TypeSeq> Indices2,
        class MetricType,
        class PositionType,
        class BatchElem>
struct DiscreteHodgeStar
{
    template <
            misc::Specialization<tensor::Tensor> HodgeTensorType,
            misc::Specialization<tensor::Tensor> FormTensorType>
    KOKKOS_FUNCTION static void run(
            HodgeTensorType hodge_tensor,
            FormTensorType form_tensor,
            MetricType metric,
            PositionType position,
            BatchElem elem)
    {
        [[maybe_unused]] sil::tensor::tensor_accessor_for_domain_t<
                hodge_star_domain_t<Indices1, Indices2>> hodge_star_accessor;
        static constexpr std::size_t HODGE_STAR_SIZE
                = misc::convert_type_seq_to_t<tensor::TensorFullIndex, Indices1>::access_size()
                  * detail::HodgeStarTargetSize<Indices2>::value;
        std::array<double, HODGE_STAR_SIZE> hodge_star_alloc {};
        ddc::ChunkSpan<
                double,
                hodge_star_domain_t<Indices1, Indices2>,
                Kokkos::layout_right,
                typename HodgeTensorType::memory_space>
                hodge_star_span(hodge_star_alloc.data(), hodge_star_accessor.domain());
        sil::tensor::Tensor hodge_star(hodge_star_span);
        ddc::device_for_each(hodge_star.domain(), [&](auto it) {
            hodge_star.mem(it)
                    = value(metric, position, elem, hodge_star.canonical_natural_element(it));
        });
        sil::tensor::tensor_prod(hodge_tensor, hodge_star, form_tensor);
    }

    KOKKOS_FUNCTION static double value(
            MetricType metric,
            PositionType position,
            BatchElem elem,
            auto natural_elem)
    {
        using AmbientIndex = typename detail::AmbientIndex<Indices1, Indices2>::type;
        constexpr std::size_t N = AmbientIndex::size();
        using SourceElem = misc::convert_type_seq_to_t<ddc::DiscreteElement, Indices1>;
        using TargetElem = misc::convert_type_seq_to_t<ddc::DiscreteElement, Indices2>;
        std::array const source_ids = ddc::detail::array(SourceElem(natural_elem));
        std::array const target_ids = ddc::detail::array(TargetElem(natural_elem));

        if (!detail::has_unique_ids<N>(source_ids) || !detail::has_unique_ids<N>(target_ids)
            || !detail::is_complete_permutation<N>(source_ids, target_ids)) {
            return 0.;
        }
        double const primal_volume
                = SimplexVolume<CellComplex::Primal, N, MetricType, PositionType, BatchElem>::
                        run(metric, position, elem, source_ids);
        if (primal_volume == 0.) {
            return 0.;
        }

        return static_cast<double>(detail::permutation_sign<N>(source_ids, target_ids))
               * DualSimplexVolume<Complex, N, MetricType, PositionType, BatchElem>::template run<
                       ddc::type_seq_size_v<Indices1>>(metric, position, elem, source_ids)
               / (primal_volume * misc::factorial(ddc::type_seq_size_v<Indices1>));
    }
};

template <
        misc::Specialization<ddc::detail::TypeSeq> Indices1,
        misc::Specialization<ddc::detail::TypeSeq> Indices2,
        class MetricType,
        class BatchElem>
struct ContinuousHodgeStar
{
    template <
            misc::Specialization<tensor::Tensor> HodgeTensorType,
            misc::Specialization<tensor::Tensor> FormTensorType>
    KOKKOS_FUNCTION static void run(
            HodgeTensorType hodge_tensor,
            FormTensorType form_tensor,
            MetricType metric,
            BatchElem elem)
    {
        [[maybe_unused]] sil::tensor::tensor_accessor_for_domain_t<
                hodge_star_domain_t<Indices1, Indices2>> hodge_star_accessor;
        static constexpr std::size_t HODGE_STAR_SIZE
                = misc::convert_type_seq_to_t<tensor::TensorFullIndex, Indices1>::access_size()
                  * detail::HodgeStarTargetSize<Indices2>::value;
        std::array<double, HODGE_STAR_SIZE> hodge_star_alloc {};
        ddc::ChunkSpan<
                double,
                hodge_star_domain_t<Indices1, Indices2>,
                Kokkos::layout_right,
                typename HodgeTensorType::memory_space>
                hodge_star_span(hodge_star_alloc.data(), hodge_star_accessor.domain());
        sil::tensor::Tensor hodge_star(hodge_star_span);
        ddc::device_for_each(hodge_star.domain(), [&](auto it) {
            hodge_star.mem(it) = value(metric, elem, hodge_star.canonical_natural_element(it));
        });
        sil::tensor::tensor_prod(hodge_tensor, hodge_star, form_tensor);
    }

    KOKKOS_FUNCTION static double value(MetricType metric, BatchElem elem, auto natural_elem)
    {
        using AmbientIndex = typename detail::AmbientIndex<Indices1, Indices2>::type;
        constexpr std::size_t N = AmbientIndex::size();
        constexpr std::size_t K = ddc::type_seq_size_v<Indices1>;
        using SourceElem = misc::convert_type_seq_to_t<ddc::DiscreteElement, Indices1>;
        using TargetElem = misc::convert_type_seq_to_t<ddc::DiscreteElement, Indices2>;
        std::array const source_ids = ddc::detail::array(SourceElem(natural_elem));
        std::array const target_ids = ddc::detail::array(TargetElem(natural_elem));

        if (!detail::has_unique_ids<N>(source_ids) || !detail::has_unique_ids<N>(target_ids)
            || !detail::is_complete_permutation<N>(source_ids, target_ids)) {
            return 0.;
        }

        std::array<std::size_t, K> const complement = detail::complement_ids<N>(target_ids);
        using MetricIndex = ddc::
                type_seq_element_t<0, ddc::to_type_seq_t<typename MetricType::indices_domain_t>>;
        using MetricIndex1 = tensor::metric_index_1<MetricIndex>;
        using MetricIndex2 = tensor::metric_index_2<MetricIndex>;
        std::array<double, N * N> metric_alloc {};
        std::array<double, N * N> inverse_metric_alloc {};
        for (std::size_t i = 0; i < N; ++i) {
            for (std::size_t j = 0; j < N; ++j) {
                metric_alloc[i * N + j] = metric.get(metric.access_element(
                        elem,
                        ddc::DiscreteElement<MetricIndex1, MetricIndex2>(i, j)));
            }
        }

        double determinant = 0.;
        if (!detail::invert_matrix<
                    N,
                    typename MetricType::
                            memory_space>(inverse_metric_alloc, determinant, metric_alloc)) {
            return 0.;
        }

        return Kokkos::sqrt(Kokkos::abs(determinant))
               * static_cast<double>(detail::permutation_sign<N>(complement, target_ids))
               * detail::submatrix_determinant<
                       N,
                       K,
                       typename MetricType::
                               memory_space>(inverse_metric_alloc, source_ids, complement)
               / misc::factorial(K);
    }
};

template <
        misc::Specialization<ddc::detail::TypeSeq> Indices1,
        misc::Specialization<ddc::detail::TypeSeq> Indices2,
        CellComplex Complex = CellComplex::CircumcentricDual,
        misc::Specialization<tensor::Tensor> HodgeStarType,
        misc::Specialization<tensor::Tensor> MetricType,
        misc::Specialization<tensor::Tensor> PositionType,
        class ExecSpace>
HodgeStarType fill_discrete_hodge_star(
        ExecSpace const& exec_space,
        HodgeStarType hodge_star,
        MetricType metric,
        PositionType position)
{
    static_assert(
            ddc::type_seq_size_v<ddc::to_type_seq_t<typename PositionType::indices_domain_t>> == 1);

    SIMILIE_DEBUG_LOG("similie_compute_hodge_star");
    ddc::parallel_for_each(
            "similie_compute_hodge_star",
            exec_space,
            hodge_star.non_indices_domain(),
            KOKKOS_LAMBDA(
                    typename HodgeStarType::non_indices_domain_t::discrete_element_type elem) {
                ddc::device_for_each(hodge_star.accessor().domain(), [&](auto mem_elem) {
                    hodge_star.mem(typename HodgeStarType::discrete_element_type(elem, mem_elem))
                            = DiscreteHodgeStar<
                                    Complex,
                                    Indices1,
                                    Indices2,
                                    MetricType,
                                    PositionType,
                                    typename HodgeStarType::non_indices_domain_t::
                                            discrete_element_type>::
                                    value(metric,
                                          position,
                                          elem,
                                          hodge_star.accessor().canonical_natural_element(
                                                  mem_elem));
                });
            });
    return hodge_star;
}

template <
        misc::Specialization<ddc::detail::TypeSeq> Indices1,
        misc::Specialization<ddc::detail::TypeSeq> Indices2,
        misc::Specialization<tensor::Tensor> HodgeStarType,
        misc::Specialization<tensor::Tensor> MetricType,
        class ExecSpace>
HodgeStarType fill_continuous_hodge_star(
        ExecSpace const& exec_space,
        HodgeStarType hodge_star,
        MetricType metric)
{
    SIMILIE_DEBUG_LOG("similie_compute_continuous_hodge_star");
    ddc::parallel_for_each(
            "similie_compute_continuous_hodge_star",
            exec_space,
            hodge_star.non_indices_domain(),
            KOKKOS_LAMBDA(
                    typename HodgeStarType::non_indices_domain_t::discrete_element_type elem) {
                ddc::device_for_each(hodge_star.accessor().domain(), [&](auto mem_elem) {
                    hodge_star.mem(typename HodgeStarType::discrete_element_type(elem, mem_elem))
                            = ContinuousHodgeStar<
                                    Indices1,
                                    Indices2,
                                    MetricType,
                                    typename HodgeStarType::non_indices_domain_t::
                                            discrete_element_type>::
                                    value(metric,
                                          elem,
                                          hodge_star.accessor().canonical_natural_element(
                                                  mem_elem));
                });
            });
    return hodge_star;
}

} // namespace exterior

} // namespace sil
