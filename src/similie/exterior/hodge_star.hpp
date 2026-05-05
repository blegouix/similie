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
#include <similie/tensor/gram_matrix.hpp>

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
struct AmbientDimension;

template <class HeadIndex, class... TailIndex, class Indices2>
struct AmbientDimension<ddc::detail::TypeSeq<HeadIndex, TailIndex...>, Indices2>
{
    static constexpr std::size_t value = HeadIndex::size();
};

template <class HeadIndex, class... TailIndex>
struct AmbientDimension<ddc::detail::TypeSeq<>, ddc::detail::TypeSeq<HeadIndex, TailIndex...>>
{
    static constexpr std::size_t value = HeadIndex::size();
};

template <class TypeSeq>
struct ExtractIds;

template <class... Indices>
struct ExtractIds<ddc::detail::TypeSeq<Indices...>>
{
    template <class NaturalElem>
    KOKKOS_FUNCTION static std::array<std::size_t, sizeof...(Indices)> run(NaturalElem natural_elem)
    {
        return std::array<std::size_t, sizeof...(Indices)> {
                static_cast<std::size_t>(natural_elem.template uid<Indices>())...};
    }
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
KOKKOS_FUNCTION std::array<bool, N> active_dimensions(std::array<std::size_t, M> const& ids)
{
    std::array<bool, N> active_dims {};
    for (std::size_t id : ids) {
        active_dims[id] = true;
    }
    return active_dims;
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

template <std::size_t N>
KOKKOS_FUNCTION constexpr std::size_t flat_index(std::size_t const i, std::size_t const j)
{
    return i * N + j;
}

template <std::size_t N>
KOKKOS_FUNCTION std::array<double, N * N> identity_matrix()
{
    std::array<double, N * N> matrix {};
    for (std::size_t i = 0; i < N; ++i) {
        matrix[flat_index<N>(i, i)] = 1.;
    }
    return matrix;
}

template <std::size_t N>
struct MatrixInversionResult
{
    double determinant;
    std::array<double, N * N> inverse;
    bool invertible;
};

template <std::size_t N>
KOKKOS_FUNCTION MatrixInversionResult<N> invert_matrix(std::array<double, N * N> matrix)
{
    std::array<double, N * N> inverse = identity_matrix<N>();
    double determinant = 1.;
    int sign = 1;

    for (std::size_t i = 0; i < N; ++i) {
        std::size_t pivot = i;
        double pivot_abs = Kokkos::abs(matrix[flat_index<N>(i, i)]);
        for (std::size_t j = i + 1; j < N; ++j) {
            double const candidate_abs = Kokkos::abs(matrix[flat_index<N>(j, i)]);
            if (candidate_abs > pivot_abs) {
                pivot = j;
                pivot_abs = candidate_abs;
            }
        }

        if (pivot_abs == 0.) {
            return MatrixInversionResult<N> {0., std::array<double, N * N> {}, false};
        }

        if (pivot != i) {
            sign *= -1;
            for (std::size_t j = 0; j < N; ++j) {
                Kokkos::kokkos_swap(matrix[flat_index<N>(i, j)], matrix[flat_index<N>(pivot, j)]);
                Kokkos::kokkos_swap(inverse[flat_index<N>(i, j)], inverse[flat_index<N>(pivot, j)]);
            }
        }

        double const diagonal = matrix[flat_index<N>(i, i)];
        determinant *= diagonal;

        for (std::size_t j = 0; j < N; ++j) {
            matrix[flat_index<N>(i, j)] /= diagonal;
            inverse[flat_index<N>(i, j)] /= diagonal;
        }

        for (std::size_t row = 0; row < N; ++row) {
            if (row == i) {
                continue;
            }
            double const factor = matrix[flat_index<N>(row, i)];
            for (std::size_t col = 0; col < N; ++col) {
                matrix[flat_index<N>(row, col)] -= factor * matrix[flat_index<N>(i, col)];
                inverse[flat_index<N>(row, col)] -= factor * inverse[flat_index<N>(i, col)];
            }
        }
    }

    return MatrixInversionResult<N> {static_cast<double>(sign) * determinant, inverse, true};
}

template <std::size_t N>
KOKKOS_FUNCTION double determinant(std::array<double, N * N> matrix)
{
    return invert_matrix<N>(matrix).determinant;
}

template <std::size_t N, std::size_t K>
KOKKOS_FUNCTION double submatrix_determinant(
        std::array<double, N * N> const& matrix,
        std::array<std::size_t, K> const& row_ids,
        std::array<std::size_t, K> const& col_ids)
{
    if constexpr (K == 0) {
        return 1.;
    } else {
        std::array<double, K * K> submatrix {};
        for (std::size_t i = 0; i < K; ++i) {
            for (std::size_t j = 0; j < K; ++j) {
                submatrix[flat_index<K>(i, j)] = matrix[flat_index<N>(row_ids[i], col_ids[j])];
            }
        }
        return determinant<K>(submatrix);
    }
}

template <
        DualStrategy Strategy,
        class Indices1,
        class Indices2,
        class MetricType,
        class PositionType,
        class BatchElem>
struct DiscreteHodgeStar
{
    KOKKOS_FUNCTION static double value(
            MetricType metric,
            PositionType position,
            BatchElem elem,
            auto natural_elem)
    {
        constexpr std::size_t N = AmbientDimension<Indices1, Indices2>::value;
        std::array const source_ids = ExtractIds<Indices1>::run(natural_elem);
        std::array const target_ids = ExtractIds<Indices2>::run(natural_elem);

        if (!has_unique_ids<N>(source_ids) || !has_unique_ids<N>(target_ids)
            || !is_complete_permutation<N>(source_ids, target_ids)) {
            return 0.;
        }

        std::array<bool, N> const active_dims = active_dimensions<N>(source_ids);
        double const primal_volume = SimplexVolume<N, MetricType, PositionType, BatchElem>::
                value(metric, position, elem, active_dims);
        if (primal_volume == 0.) {
            return 0.;
        }

        return static_cast<double>(permutation_sign<N>(source_ids, target_ids))
               * DualSimplexVolume<Strategy, N, MetricType, PositionType, BatchElem>::
                       value(metric, position, elem, active_dims)
               / (primal_volume * misc::factorial(ddc::type_seq_size_v<Indices1>));
    }
};

template <class Indices1, class Indices2, class MetricType, class BatchElem>
struct ContinuousHodgeStar
{
    KOKKOS_FUNCTION static double value(MetricType metric, BatchElem elem, auto natural_elem)
    {
        constexpr std::size_t N = AmbientDimension<Indices1, Indices2>::value;
        constexpr std::size_t K = ddc::type_seq_size_v<Indices1>;
        std::array const source_ids = ExtractIds<Indices1>::run(natural_elem);
        std::array const target_ids = ExtractIds<Indices2>::run(natural_elem);

        if (!has_unique_ids<N>(source_ids) || !has_unique_ids<N>(target_ids)
            || !is_complete_permutation<N>(source_ids, target_ids)) {
            return 0.;
        }

        std::array<std::size_t, K> const complement = complement_ids<N>(target_ids);
        std::array<double, N * N> const local_metric_values
                = tensor::detail::local_metric<N>(metric, elem);
        MatrixInversionResult<N> const inverse_metric = invert_matrix<N>(local_metric_values);
        if (!inverse_metric.invertible) {
            return 0.;
        }

        return Kokkos::sqrt(Kokkos::abs(inverse_metric.determinant))
               * static_cast<double>(permutation_sign<N>(complement, target_ids))
               * submatrix_determinant<N, K>(inverse_metric.inverse, source_ids, complement)
               / misc::factorial(K);
    }
};

} // namespace detail

template <
        DualStrategy Strategy,
        misc::Specialization<ddc::detail::TypeSeq> Indices1,
        misc::Specialization<ddc::detail::TypeSeq> Indices2,
        class MetricType,
        class PositionType,
        class BatchElem>
struct DiscreteHodgeStar
{
    KOKKOS_FUNCTION static double value(
            MetricType metric,
            PositionType position,
            BatchElem elem,
            auto natural_elem)
    {
        return detail::DiscreteHodgeStar<
                Strategy,
                Indices1,
                Indices2,
                MetricType,
                PositionType,
                BatchElem>::value(metric, position, elem, natural_elem);
    }

    template <
            misc::Specialization<tensor::Tensor> OutTensorType,
            misc::Specialization<tensor::Tensor> InTensorType,
            class ExecSpace>
    static OutTensorType run(
            ExecSpace const& exec_space,
            OutTensorType out_tensor,
            InTensorType in_tensor,
            MetricType metric,
            PositionType position)
    {
        [[maybe_unused]] sil::tensor::tensor_accessor_for_domain_t<
                hodge_star_domain_t<Indices1, Indices2>> hodge_star_accessor;
        ddc::cartesian_prod_t<
                typename MetricType::non_indices_domain_t,
                hodge_star_domain_t<Indices1, Indices2>>
                hodge_star_dom(metric.non_indices_domain(), hodge_star_accessor.domain());
        ddc::Chunk hodge_star_alloc(
                hodge_star_dom,
                ddc::KokkosAllocator<double, typename ExecSpace::memory_space>());
        sil::tensor::Tensor hodge_star(hodge_star_alloc);
        fill_discrete_hodge_star<
                Indices1,
                Indices2,
                Strategy>(exec_space, hodge_star, metric, position);

        ddc::parallel_for_each(
                exec_space,
                out_tensor.non_indices_domain(),
                KOKKOS_LAMBDA(
                        typename OutTensorType::non_indices_domain_t::discrete_element_type elem) {
                    sil::tensor::tensor_prod(out_tensor[elem], hodge_star[elem], in_tensor[elem]);
                });
        return out_tensor;
    }
};

template <
        misc::Specialization<ddc::detail::TypeSeq> Indices1,
        misc::Specialization<ddc::detail::TypeSeq> Indices2,
        class MetricType,
        class BatchElem>
struct ContinuousHodgeStar
{
    KOKKOS_FUNCTION static double value(MetricType metric, BatchElem elem, auto natural_elem)
    {
        return detail::ContinuousHodgeStar<Indices1, Indices2, MetricType, BatchElem>::
                value(metric, elem, natural_elem);
    }
    template <
            misc::Specialization<tensor::Tensor> OutTensorType,
            misc::Specialization<tensor::Tensor> InTensorType,
            class ExecSpace>
    static OutTensorType run(
            ExecSpace const& exec_space,
            OutTensorType out_tensor,
            InTensorType in_tensor,
            MetricType metric)
    {
        [[maybe_unused]] sil::tensor::tensor_accessor_for_domain_t<
                hodge_star_domain_t<Indices1, Indices2>> hodge_star_accessor;
        ddc::cartesian_prod_t<
                typename MetricType::non_indices_domain_t,
                hodge_star_domain_t<Indices1, Indices2>>
                hodge_star_dom(metric.non_indices_domain(), hodge_star_accessor.domain());
        ddc::Chunk hodge_star_alloc(
                hodge_star_dom,
                ddc::KokkosAllocator<double, typename ExecSpace::memory_space>());
        sil::tensor::Tensor hodge_star(hodge_star_alloc);
        fill_continuous_hodge_star<Indices1, Indices2>(exec_space, hodge_star, metric);

        ddc::parallel_for_each(
                exec_space,
                out_tensor.non_indices_domain(),
                KOKKOS_LAMBDA(
                        typename OutTensorType::non_indices_domain_t::discrete_element_type elem) {
                    sil::tensor::tensor_prod(out_tensor[elem], hodge_star[elem], in_tensor[elem]);
                });
        return out_tensor;
    }
};

template <
        misc::Specialization<ddc::detail::TypeSeq> Indices1,
        misc::Specialization<ddc::detail::TypeSeq> Indices2,
        DualStrategy Strategy = DualStrategy::Circumcentric,
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
            hodge_star.domain(),
            KOKKOS_LAMBDA(
                    typename HodgeStarType::discrete_domain_type::discrete_element_type elem) {
                hodge_star.mem(elem) = DiscreteHodgeStar<
                        Strategy,
                        Indices1,
                        Indices2,
                        MetricType,
                        PositionType,
                        typename HodgeStarType::non_indices_domain_t::discrete_element_type>::
                        value(metric,
                              position,
                              typename HodgeStarType::non_indices_domain_t::discrete_element_type(
                                      elem),
                              hodge_star.canonical_natural_element(elem));
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
            hodge_star.domain(),
            KOKKOS_LAMBDA(
                    typename HodgeStarType::discrete_domain_type::discrete_element_type elem) {
                hodge_star.mem(elem) = ContinuousHodgeStar<
                        Indices1,
                        Indices2,
                        MetricType,
                        typename HodgeStarType::non_indices_domain_t::discrete_element_type>::
                        value(metric,
                              typename HodgeStarType::non_indices_domain_t::discrete_element_type(
                                      elem),
                              hodge_star.canonical_natural_element(elem));
            });
    return hodge_star;
}

} // namespace exterior

} // namespace sil
