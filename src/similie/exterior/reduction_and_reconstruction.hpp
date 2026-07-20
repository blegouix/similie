// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

#include <array>

#include <ddc/ddc.hpp>

#include <similie/misc/clamp_to_domain.hpp>
#include <similie/misc/factorial.hpp>
#include <similie/misc/small_matrix.hpp>
#include <similie/misc/specialization.hpp>
#include <similie/misc/type_seq_conversion.hpp>
#include <similie/tensor/antisymmetric_tensor.hpp>
#include <similie/tensor/character.hpp>
#include <similie/tensor/full_tensor.hpp>
#include <similie/tensor/prime.hpp>
#include <similie/tensor/tensor_impl.hpp>

#include "volume.hpp"


namespace sil {

namespace exterior {

namespace detail {

template <class ElemType>
struct FlatNaturalElemRank;

template <class... Tags>
struct FlatNaturalElemRank<ddc::DiscreteElement<Tags...>>
{
    static constexpr std::size_t value = (FlatNaturalElemRank<Tags>::value + ... + 0);
};

template <class Tag>
struct FlatNaturalElemRank
{
    static constexpr std::size_t value = 1;
};

template <class ElemType, std::size_t N>
KOKKOS_FUNCTION void copy_flat_natural_elem_ids(
        std::array<std::size_t, N>& ids,
        std::size_t& offset,
        ElemType const& elem)
{
    auto const entries = ddc::detail::array(elem);
    if constexpr (std::tuple_size_v<decltype(entries)> == 0) {
        return;
    } else {
        using EntryType = std::remove_cvref_t<decltype(entries[0])>;
        if constexpr (std::is_same_v<EntryType, std::size_t>) {
            for (auto const& entry : entries) {
                ids[offset++] = entry;
            }
        } else {
            for (auto const& entry : entries) {
                copy_flat_natural_elem_ids(ids, offset, entry);
            }
        }
    }
}

template <class ElemType>
KOKKOS_FUNCTION auto flat_natural_elem_ids(ElemType const& elem)
{
    std::array<std::size_t, FlatNaturalElemRank<ElemType>::value> ids {};
    std::size_t offset = 0;
    copy_flat_natural_elem_ids(ids, offset, elem);
    return ids;
}

template <class ElemType, std::size_t N>
KOKKOS_FUNCTION void assign_flat_natural_elem_ids(
        ElemType& elem,
        std::array<std::size_t, N> const& ids,
        std::size_t& offset)
{
    auto entries = ddc::detail::array(elem);
    if constexpr (std::tuple_size_v<decltype(entries)> == 0) {
        return;
    } else {
        using EntryType = std::remove_cvref_t<decltype(entries[0])>;
        if constexpr (std::is_same_v<EntryType, std::size_t>) {
            for (auto& entry : entries) {
                entry = ids[offset++];
            }
            ddc::detail::array(elem) = entries;
        } else {
            for (auto& entry : entries) {
                assign_flat_natural_elem_ids(entry, ids, offset);
            }
            ddc::detail::array(elem) = entries;
        }
    }
}

template <class ElemType, std::size_t N>
KOKKOS_FUNCTION ElemType natural_elem_from_flat_ids(std::array<std::size_t, N> const& ids)
{
    static_assert(
            FlatNaturalElemRank<ElemType>::value == N,
            "Flat natural element ids must match the element rank.");
    ElemType elem;
    std::size_t offset = 0;
    assign_flat_natural_elem_ids(elem, ids, offset);
    return elem;
}

template <misc::Specialization<ddc::detail::TypeSeq> Indices>
struct ReductionTargetIndex;

template <>
struct ReductionTargetIndex<ddc::detail::TypeSeq<>>
{
    using type = tensor::Covariant<tensor::ScalarIndex>;
};

template <tensor::TensorNatIndex Index>
struct ReductionTargetIndex<ddc::detail::TypeSeq<Index>>
{
    using type = Index;
};

template <tensor::TensorNatIndex... Index>
    requires(sizeof...(Index) > 1)
struct ReductionTargetIndex<ddc::detail::TypeSeq<Index...>>
{
    using type = tensor::TensorAntisymmetricIndex<Index...>;
};

template <std::size_t N>
KOKKOS_FUNCTION bool has_unique_reduction_ids(std::array<std::size_t, N> const& ids)
{
    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = i + 1; j < N; ++j) {
            if (ids[i] == ids[j]) {
                return false;
            }
        }
    }
    return true;
}

template <std::size_t N>
KOKKOS_FUNCTION bool have_same_reduction_ids(
        std::array<std::size_t, N> const& lhs,
        std::array<std::size_t, N> const& rhs)
{
    for (std::size_t i = 0; i < N; ++i) {
        if (lhs[i] != rhs[i]) {
            return false;
        }
    }
    return true;
}

template <std::size_t K>
KOKKOS_FUNCTION double reduction_determinant(std::array<double, K * K> const& matrix)
{
    if constexpr (K == 0) {
        return 1.;
    } else if constexpr (K == 1) {
        return matrix[0];
    } else {
        double det = 0.;
        for (std::size_t col = 0; col < K; ++col) {
            std::array<double, (K - 1) * (K - 1)> minor {};
            for (std::size_t row = 1; row < K; ++row) {
                std::size_t minor_col = 0;
                for (std::size_t source_col = 0; source_col < K; ++source_col) {
                    if (source_col == col) {
                        continue;
                    }
                    minor[(row - 1) * (K - 1) + minor_col] = matrix[row * K + source_col];
                    ++minor_col;
                }
            }
            double const sign = (col % 2 == 0) ? 1. : -1.;
            det += sign * matrix[col] * reduction_determinant<K - 1>(minor);
        }
        return det;
    }
}

template <std::size_t N, std::size_t K>
KOKKOS_FUNCTION std::array<std::size_t, K> combination_from_rank(std::size_t rank)
{
    std::array<std::size_t, K> ids {};
    if constexpr (K == 0) {
        return ids;
    } else {
        std::size_t next_candidate = 0;
        for (std::size_t i = 0; i < K; ++i) {
            for (std::size_t candidate = next_candidate; candidate < N; ++candidate) {
                std::size_t const remaining = K - i - 1;
                std::size_t const remaining_space = N - candidate - 1;
                std::size_t const count
                        = (remaining == 0) ? 1
                                           : misc::binomial_coefficient(remaining_space, remaining);
                if (rank < count) {
                    ids[i] = candidate;
                    next_candidate = candidate + 1;
                    break;
                }
                rank -= count;
            }
        }
        return ids;
    }
}

template <std::size_t N, std::size_t K>
KOKKOS_FUNCTION std::size_t combination_rank(std::array<std::size_t, K> const& ids)
{
    if constexpr (K == 0) {
        return 0;
    } else {
        std::size_t rank = 0;
        std::size_t next_candidate = 0;
        for (std::size_t i = 0; i < K; ++i) {
            for (std::size_t candidate = next_candidate; candidate < ids[i]; ++candidate) {
                std::size_t const remaining = K - i - 1;
                std::size_t const remaining_space = N - candidate - 1;
                rank += (remaining == 0) ? 1
                                         : misc::binomial_coefficient(remaining_space, remaining);
            }
            next_candidate = ids[i] + 1;
        }
        return rank;
    }
}

template <std::size_t N, std::size_t M>
KOKKOS_FUNCTION bool hodge_has_unique_ids(std::array<std::size_t, M> const& ids)
{
    if constexpr (M > 0) {
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
    }
    return true;
}

template <std::size_t N, std::size_t M1, std::size_t M2>
KOKKOS_FUNCTION bool hodge_is_complete_permutation(
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
KOKKOS_FUNCTION int hodge_permutation_sign(
        std::array<std::size_t, M1> const& source_ids,
        std::array<std::size_t, M2> const& target_ids)
{
    std::array<std::size_t, N> permutation {};
    for (std::size_t i = 0; i < M1; ++i) {
        permutation[i] = source_ids[i];
    }
    if constexpr (M2 > 0) {
        for (std::size_t i = 0; i < M2; ++i) {
            permutation[M1 + i] = target_ids[i];
        }
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
KOKKOS_FUNCTION std::array<std::size_t, N - M> hodge_complement_ids(
        std::array<std::size_t, M> const& ids)
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

template <class PositionIndex, class PositionType, class BatchElem, std::size_t K>
KOKKOS_FUNCTION double primal_reconstruction_diagonal(
        PositionType position,
        BatchElem elem,
        std::array<std::size_t, K> const& ids)
{
    if constexpr (K == 0) {
        return 1.;
    } else {
        std::array<double, K * K> jacobian_matrix {};
        for (std::size_t row = 0; row < K; ++row) {
            for (std::size_t col = 0; col < K; ++col) {
                std::array<double, PositionIndex::size()> const edge
                        = sil::exterior::detail::edge_vector<
                                PositionIndex>(position, elem, ids[col]);
                jacobian_matrix[row * K + col] = edge[ids[row]];
            }
        }
        double const jacobian = reduction_determinant<K>(jacobian_matrix);
        if (Kokkos::abs(jacobian) < 1e-14) {
            return 0.;
        }
        return 1. / jacobian;
    }
}

template <std::size_t N, std::size_t K, class MetricType, class BatchElem>
KOKKOS_FUNCTION double continuous_hodge_value_from_ids(
        MetricType metric,
        BatchElem elem,
        std::array<std::size_t, K> const& source_ids,
        std::array<std::size_t, N - K> const& target_ids)
{
    if (!hodge_has_unique_ids<N>(source_ids) || !hodge_has_unique_ids<N>(target_ids)
        || !hodge_is_complete_permutation<N>(source_ids, target_ids)) {
        return 0.;
    }

    std::array<std::size_t, K> const complement = hodge_complement_ids<N>(target_ids);
    using MetricIndex
            = ddc::type_seq_element_t<0, ddc::to_type_seq_t<typename MetricType::indices_domain_t>>;
    using MetricIndex1 = tensor::metric_index_1<MetricIndex>;
    using MetricIndex2 = tensor::metric_index_2<MetricIndex>;
    std::array<double, N * N> metric_alloc {};
    std::array<double, N * N> determinant_metric_alloc {};
    std::array<double, N * N> inverse_metric_alloc {};
    std::array<double, N * N> workspace_alloc {};
    std::array<double, K * K> submatrix_alloc {};
    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = 0; j < N; ++j) {
            metric_alloc[i * N + j] = metric.get(metric.access_element(
                    elem,
                    ddc::DiscreteElement<MetricIndex1, MetricIndex2>(i, j)));
        }
    }
    determinant_metric_alloc = metric_alloc;

    auto determinant_metric_view = misc::math::matrix_view<
            double,
            typename MetricType::memory_space>(determinant_metric_alloc.data(), N, N);
    auto metric_view = misc::math::
            matrix_view<double, typename MetricType::memory_space>(metric_alloc.data(), N, N);
    double const determinant = misc::math::determinant(determinant_metric_view);
    auto inverse_metric_view = misc::math::matrix_view<
            double,
            typename MetricType::memory_space>(inverse_metric_alloc.data(), N, N);
    auto workspace = misc::math::
            vector_view<double, typename MetricType::memory_space>(workspace_alloc.data(), N * N);
    if (!misc::math::invert(inverse_metric_view, metric_view, workspace)) {
        return 0.;
    }

    return Kokkos::sqrt(Kokkos::abs(determinant))
           * static_cast<double>(hodge_permutation_sign<N>(complement, target_ids))
           * misc::math::submatrix_determinant<
                   K>(inverse_metric_view, source_ids, complement, submatrix_alloc)
           / misc::factorial(K);
}

template <
        CellComplex Complex,
        std::size_t N,
        std::size_t K,
        class MetricType,
        class PositionType,
        class BatchElem>
KOKKOS_FUNCTION double legacy_discrete_hodge_value_from_ids(
        MetricType metric,
        PositionType position,
        BatchElem elem,
        std::array<std::size_t, K> const& source_ids,
        std::array<std::size_t, N - K> const& target_ids)
{
    if (!hodge_has_unique_ids<N>(source_ids) || !hodge_has_unique_ids<N>(target_ids)
        || !hodge_is_complete_permutation<N>(source_ids, target_ids)) {
        return 0.;
    }
    double const primal_volume
            = SimplexVolume<CellComplex::Primal, N, MetricType, PositionType, BatchElem>::
                    template run<K>(metric, position, elem, source_ids);
    if (primal_volume == 0.) {
        return 0.;
    }

    return static_cast<double>(hodge_permutation_sign<N>(source_ids, target_ids))
           * DualSimplexVolume<Complex, N, MetricType, PositionType, BatchElem>::template run<
                   K>(metric, position, elem, source_ids)
           / (primal_volume * misc::factorial(K));
}

template <class ReductionNaturalElemType, std::size_t N1, std::size_t N2>
KOKKOS_FUNCTION ReductionNaturalElemType merge_reduction_natural_elems(
        std::array<std::size_t, N1> const& first_ids,
        std::array<std::size_t, N2> const& second_ids)
{
    std::array<std::size_t, N1 + N2> reduction_ids {};
    for (std::size_t i = 0; i < N1; ++i) {
        reduction_ids[i] = first_ids[i];
    }
    for (std::size_t i = 0; i < N2; ++i) {
        reduction_ids[N1 + i] = second_ids[i];
    }
    return natural_elem_from_flat_ids<ReductionNaturalElemType>(reduction_ids);
}

} // namespace detail

template <misc::Specialization<ddc::detail::TypeSeq> Indices>
using reduction_index_t = typename detail::ReductionTargetIndex<Indices>::type;

template <misc::Specialization<ddc::detail::TypeSeq> Indices>
using reduction_domain_t = ddc::detail::convert_type_seq_to_discrete_domain_t<ddc::type_seq_merge_t<
        ddc::detail::TypeSeq<misc::convert_type_seq_to_t<
                tensor::TensorFullIndex,
                tensor::primes<tensor::upper_t<Indices>>>>,
        ddc::detail::TypeSeq<reduction_index_t<Indices>>>>;

template <misc::Specialization<ddc::detail::TypeSeq> Indices>
using reconstruction_domain_t
        = ddc::detail::convert_type_seq_to_discrete_domain_t<ddc::type_seq_merge_t<
                ddc::detail::TypeSeq<reduction_index_t<Indices>>,
                ddc::detail::TypeSeq<reduction_index_t<tensor::primes<Indices>>>>>;

template <
        misc::Specialization<ddc::detail::TypeSeq> Indices,
        misc::Specialization<tensor::Tensor> PositionType,
        class BatchElem,
        CellComplex Complex = CellComplex::Primal>
struct Reduction
{
    using source_index_type = misc::convert_type_seq_to_t<
            tensor::TensorFullIndex,
            tensor::primes<tensor::upper_t<Indices>>>;
    using target_index_type = reduction_index_t<Indices>;
    using source_natural_elem_type =
            typename tensor::natural_domain_t<source_index_type>::discrete_element_type;
    using target_natural_elem_type =
            typename tensor::natural_domain_t<target_index_type>::discrete_element_type;
    using position_index_type = ddc::
            type_seq_element_t<0, ddc::to_type_seq_t<typename PositionType::indices_domain_t>>;

    template <
            misc::Specialization<tensor::Tensor> ReductionTensorType,
            misc::Specialization<tensor::Tensor> FormTensorType>
    KOKKOS_FUNCTION static void run(
            ReductionTensorType reduced_tensor,
            FormTensorType form_tensor,
            PositionType position,
            BatchElem elem)
    {
        using reduction_accessor_t
                = sil::tensor::tensor_accessor_for_domain_t<reduction_domain_t<Indices>>;
        using reduction_natural_elem_type =
                typename reduction_accessor_t::natural_domain_t::discrete_element_type;
        using form_natural_elem_type =
                typename FormTensorType::accessor_t::natural_domain_t::discrete_element_type;
        if constexpr (target_index_type::rank() == 0) {
            reduced_tensor.mem(ddc::DiscreteElement<target_index_type>(0))
                    = form_tensor.get(form_tensor.access_element(form_natural_elem_type()));
        } else {
            [[maybe_unused]] sil::tensor::TensorAccessor<source_index_type> source_accessor;
            ddc::device_for_each(reduced_tensor.domain(), [&](auto target_mem_elem) {
                auto const target_natural_elem
                        = reduced_tensor.canonical_natural_element(target_mem_elem);
                double reduced_value = 0.;
                ddc::device_for_each(
                        source_accessor.natural_domain(),
                        [&](auto source_natural_elem) {
                            reduction_natural_elem_type reduction_natural_elem;
                            auto const source_ids = ddc::detail::array(source_natural_elem);
                            auto const target_ids = ddc::detail::array(target_natural_elem);
                            reduction_natural_elem = detail::merge_reduction_natural_elems<
                                    reduction_natural_elem_type>(source_ids, target_ids);

                            form_natural_elem_type form_natural_elem;
                            ddc::detail::array(form_natural_elem)
                                    = ddc::detail::array(source_natural_elem);

                            reduced_value += value(position, elem, reduction_natural_elem)
                                             * form_tensor.get(
                                                     form_tensor.access_element(form_natural_elem));
                        });
                reduced_tensor.mem(target_mem_elem) = reduced_value;
            });
        }
    }

    template <
            misc::Specialization<tensor::Tensor> ReductionTensorType,
            misc::Specialization<tensor::Tensor> FormTensorType,
            misc::Specialization<tensor::Tensor> MetricType>
    KOKKOS_FUNCTION static void run(
            ReductionTensorType reduced_tensor,
            FormTensorType form_tensor,
            MetricType metric,
            PositionType position,
            BatchElem elem)
    {
        if constexpr (Complex == CellComplex::Primal) {
            run(reduced_tensor, form_tensor, position, elem);
        } else {
            static_assert(
                    Complex == CellComplex::CircumcentricDual,
                    "Metric-aware reduction is only implemented for the circumcentric dual "
                    "complex.");

            constexpr std::size_t N = position_index_type::size();
            constexpr std::size_t Q = target_index_type::rank();
            constexpr std::size_t K = N - Q;
            constexpr std::size_t NBASIS = misc::binomial_coefficient(N, Q);
            static_assert(
                    NBASIS == misc::binomial_coefficient(N, K),
                    "Complementary basis sizes must match.");

            std::array<double, NBASIS * NBASIS> coeff_from_primal_alloc {};
            std::array<double, NBASIS * NBASIS> coeff_from_primal_inverse_alloc {};
            std::array<double, NBASIS * NBASIS> old_hodge_alloc {};
            std::array<double, NBASIS * NBASIS> dual_reduction_alloc {};

            for (std::size_t primal_id = 0; primal_id < NBASIS; ++primal_id) {
                std::array<std::size_t, K> const primal_ids
                        = detail::combination_from_rank<N, K>(primal_id);
                double const reconstruction_diag = detail::primal_reconstruction_diagonal<
                        position_index_type>(position, elem, primal_ids);
                for (std::size_t source_id = 0; source_id < NBASIS; ++source_id) {
                    std::array<std::size_t, Q> const source_ids
                            = detail::combination_from_rank<N, Q>(source_id);
                    coeff_from_primal_alloc[source_id * NBASIS + primal_id]
                            = detail::continuous_hodge_value_from_ids<
                                      N,
                                      K>(metric, elem, primal_ids, source_ids)
                              * reconstruction_diag;
                }
                for (std::size_t target_id = 0; target_id < NBASIS; ++target_id) {
                    std::array<std::size_t, Q> const target_ids
                            = detail::combination_from_rank<N, Q>(target_id);
                    old_hodge_alloc[target_id * NBASIS + primal_id]
                            = detail::legacy_discrete_hodge_value_from_ids<
                                    Complex,
                                    N,
                                    K>(metric, position, elem, primal_ids, target_ids);
                }
            }

            std::array<double, NBASIS * NBASIS> workspace_alloc {};
            auto coeff_from_primal_view = misc::math::matrix_view<
                    double,
                    typename MetricType::
                            memory_space>(coeff_from_primal_alloc.data(), NBASIS, NBASIS);
            auto coeff_from_primal_inverse_view = misc::math::matrix_view<
                    double,
                    typename MetricType::
                            memory_space>(coeff_from_primal_inverse_alloc.data(), NBASIS, NBASIS);
            auto workspace = misc::math::vector_view<
                    double,
                    typename MetricType::memory_space>(workspace_alloc.data(), NBASIS * NBASIS);
            bool const invertible = misc::math::
                    invert(coeff_from_primal_inverse_view, coeff_from_primal_view, workspace);

            ddc::device_for_each(reduced_tensor.domain(), [&](auto target_mem_elem) {
                auto const target_natural_elem
                        = reduced_tensor.canonical_natural_element(target_mem_elem);
                std::size_t const target_id = [&]() {
                    if constexpr (Q == 0) {
                        return std::size_t(0);
                    } else {
                        std::array<std::size_t, Q> const target_ids
                                = ddc::detail::array(target_natural_elem);
                        return detail::combination_rank<N, Q>(target_ids);
                    }
                }();

                double reduced_value = 0.;
                ddc::device_for_each(form_tensor.domain(), [&](auto source_mem_elem) {
                    auto const source_natural_elem
                            = form_tensor.canonical_natural_element(source_mem_elem);
                    std::size_t const source_id = [&]() {
                        if constexpr (Q == 0) {
                            return std::size_t(0);
                        } else {
                            std::array<std::size_t, Q> const source_ids
                                    = ddc::detail::array(source_natural_elem);
                            return detail::combination_rank<N, Q>(source_ids);
                        }
                    }();
                    double dual_reduction_value = 0.;
                    if (invertible) {
                        for (std::size_t primal_id = 0; primal_id < NBASIS; ++primal_id) {
                            dual_reduction_value += old_hodge_alloc[target_id * NBASIS + primal_id]
                                                    * coeff_from_primal_inverse_alloc
                                                            [primal_id * NBASIS + source_id];
                        }
                    }

                    using form_natural_elem_type = typename FormTensorType::accessor_t::
                            natural_domain_t::discrete_element_type;
                    form_natural_elem_type form_natural_elem;
                    ddc::detail::array(form_natural_elem) = ddc::detail::array(source_natural_elem);
                    reduced_value += dual_reduction_value * form_tensor.mem(source_mem_elem);
                });
                reduced_tensor.mem(target_mem_elem) = reduced_value;
            });
        }
    }

    KOKKOS_FUNCTION static double value(PositionType position, BatchElem elem, auto natural_elem)
    {
        constexpr std::size_t K = target_index_type::rank();
        if constexpr (K == 0) {
            return 1.;
        } else {
            std::array const source_ids
                    = ddc::detail::array(source_natural_elem_type(natural_elem));
            std::array const target_ids
                    = ddc::detail::array(target_natural_elem_type(natural_elem));
            if (!detail::has_unique_reduction_ids<K>(source_ids)
                || !detail::has_unique_reduction_ids<K>(target_ids)) {
                return 0.;
            }

            std::array<double, K * K> jacobian_matrix {};
            for (std::size_t row = 0; row < K; ++row) {
                for (std::size_t col = 0; col < K; ++col) {
                    std::array<double, position_index_type::size()> const edge
                            = sil::exterior::detail::edge_vector<
                                    position_index_type>(position, elem, target_ids[col]);
                    jacobian_matrix[row * K + col] = edge[source_ids[row]];
                }
            }
            return detail::reduction_determinant<K>(jacobian_matrix)
                   * detail::complex_volume_factor<Complex, source_index_type::size()>(K)
                   / misc::factorial(K);
        }
    }

    template <misc::Specialization<tensor::Tensor> MetricType>
    KOKKOS_FUNCTION static double value(
            MetricType metric,
            PositionType position,
            BatchElem elem,
            auto natural_elem)
    {
        if constexpr (Complex == CellComplex::Primal) {
            return value(position, elem, natural_elem);
        } else {
            [[maybe_unused]] sil::tensor::TensorAccessor<source_index_type> source_accessor;
            std::array<double, source_index_type::access_size()> source_alloc {};
            ddc::ChunkSpan<
                    double,
                    ddc::DiscreteDomain<source_index_type>,
                    Kokkos::layout_right,
                    typename MetricType::memory_space>
                    source_span(source_alloc.data(), source_accessor.domain());
            sil::tensor::Tensor source_tensor(source_span);
            ddc::device_for_each(source_tensor.domain(), [&](auto source_mem_elem) {
                source_tensor.mem(source_mem_elem) = 0.;
            });

            [[maybe_unused]] sil::tensor::TensorAccessor<target_index_type> target_accessor;
            std::array<double, target_index_type::access_size()> target_alloc {};
            ddc::ChunkSpan<
                    double,
                    ddc::DiscreteDomain<target_index_type>,
                    Kokkos::layout_right,
                    typename MetricType::memory_space>
                    target_span(target_alloc.data(), target_accessor.domain());
            sil::tensor::Tensor target_tensor(target_span);
            ddc::device_for_each(target_tensor.domain(), [&](auto target_mem_elem) {
                target_tensor.mem(target_mem_elem) = 0.;
            });

            source_natural_elem_type source_natural_elem;
            ddc::detail::array(source_natural_elem)
                    = ddc::detail::array(source_natural_elem_type(natural_elem));
            if constexpr (source_index_type::rank() > 1) {
                if (!detail::has_unique_reduction_ids(ddc::detail::array(source_natural_elem))) {
                    return 0.;
                }
            }
            source_tensor(source_tensor.accessor().access_element(source_natural_elem)) = 1.;

            run(target_tensor, source_tensor, metric, position, elem);

            if constexpr (target_index_type::rank() == 0) {
                return target_tensor.get(
                        target_tensor.accessor().access_element(target_natural_elem_type()));
            } else {
                target_natural_elem_type target_natural_elem;
                ddc::detail::array(target_natural_elem)
                        = ddc::detail::array(target_natural_elem_type(natural_elem));
                return target_tensor.get(
                        target_tensor.accessor().access_element(target_natural_elem));
            }
        }
    }
};

template <
        misc::Specialization<ddc::detail::TypeSeq> Indices,
        misc::Specialization<tensor::Tensor> ReductionTensorType,
        misc::Specialization<tensor::Tensor> PositionType,
        CellComplex Complex,
        class BatchElem>
struct FillReductionOperatorMem
{
    ReductionTensorType m_reduction_tensor;
    PositionType m_position;
    BatchElem m_elem;

    template <class MemElem>
    KOKKOS_FUNCTION void operator()(MemElem mem_elem) const
    {
        m_reduction_tensor.mem(
                typename ReductionTensorType::discrete_element_type(m_elem, mem_elem))
                = Reduction<Indices, PositionType, BatchElem, Complex>::
                        value(m_position,
                              m_elem,
                              m_reduction_tensor.accessor().canonical_natural_element(mem_elem));
    }
};

template <
        misc::Specialization<ddc::detail::TypeSeq> Indices,
        misc::Specialization<tensor::Tensor> ReductionTensorType,
        misc::Specialization<tensor::Tensor> PositionType,
        class ExecSpace,
        CellComplex Complex = CellComplex::Primal>
ReductionTensorType fill_reduction_operator(
        ExecSpace const& exec_space,
        ReductionTensorType reduction_tensor,
        PositionType position)
{
    SIMILIE_DEBUG_LOG("similie_compute_reduction_operator");
    ddc::parallel_for_each(
            "similie_compute_reduction_operator",
            exec_space,
            reduction_tensor.non_indices_domain(),
            KOKKOS_LAMBDA(
                    typename ReductionTensorType::non_indices_domain_t::discrete_element_type
                            elem) {
                ddc::device_for_each(
                        reduction_tensor.accessor().domain(),
                        FillReductionOperatorMem<
                                Indices,
                                ReductionTensorType,
                                PositionType,
                                Complex,
                                typename ReductionTensorType::non_indices_domain_t::
                                        discrete_element_type> {reduction_tensor, position, elem});
            });
    return reduction_tensor;
}

template <
        misc::Specialization<ddc::detail::TypeSeq> Indices,
        misc::Specialization<tensor::Tensor> PositionType,
        class BatchElem,
        CellComplex Complex = CellComplex::Primal>
struct Reconstruction
{
    using source_index_type = reduction_index_t<Indices>;
    using target_index_type = reduction_index_t<tensor::primes<Indices>>;
    using source_natural_elem_type =
            typename tensor::natural_domain_t<source_index_type>::discrete_element_type;
    using target_natural_elem_type =
            typename tensor::natural_domain_t<target_index_type>::discrete_element_type;

    template <
            misc::Specialization<tensor::Tensor> ReconstructedTensorType,
            misc::Specialization<tensor::Tensor> CochainTensorType>
    KOKKOS_FUNCTION static void run(
            ReconstructedTensorType reconstructed_tensor,
            CochainTensorType cochain_tensor,
            PositionType position,
            BatchElem elem)
    {
        using reconstruction_accessor_t
                = sil::tensor::tensor_accessor_for_domain_t<reconstruction_domain_t<Indices>>;
        using reconstruction_natural_elem_type =
                typename reconstruction_accessor_t::natural_domain_t::discrete_element_type;
        using cochain_natural_elem_type =
                typename CochainTensorType::accessor_t::natural_domain_t::discrete_element_type;
        if constexpr (source_index_type::rank() == 0) {
            reconstructed_tensor.mem(ddc::DiscreteElement<source_index_type>(0))
                    = cochain_tensor.get(
                            cochain_tensor.access_element(cochain_natural_elem_type()));
        } else {
            [[maybe_unused]] sil::tensor::TensorAccessor<source_index_type> source_accessor;
            ddc::device_for_each(reconstructed_tensor.domain(), [&](auto target_mem_elem) {
                auto const target_natural_elem
                        = reconstructed_tensor.canonical_natural_element(target_mem_elem);
                double reconstructed_value = 0.;
                ddc::device_for_each(source_accessor.domain(), [&](auto source_mem_elem) {
                    auto const source_natural_elem
                            = source_accessor.canonical_natural_element(source_mem_elem);
                    reconstruction_natural_elem_type reconstruction_natural_elem;
                    reconstruction_natural_elem = detail::merge_reduction_natural_elems<
                            reconstruction_natural_elem_type>(
                            ddc::detail::array(source_natural_elem),
                            ddc::detail::array(target_natural_elem));

                    cochain_natural_elem_type cochain_natural_elem;
                    ddc::detail::array(cochain_natural_elem)
                            = ddc::detail::array(source_natural_elem);

                    reconstructed_value += value(position, elem, reconstruction_natural_elem)
                                           * cochain_tensor.get(cochain_tensor.access_element(
                                                   cochain_natural_elem));
                });
                reconstructed_tensor.mem(target_mem_elem) = reconstructed_value;
            });
        }
    }

    KOKKOS_FUNCTION static double value(PositionType position, BatchElem elem, auto natural_elem)
    {
        static_assert(
                Complex != CellComplex::BarycentricDual,
                "Reconstruction is not implemented for the barycentric dual complex.");
        constexpr std::size_t K = source_index_type::rank();
        if constexpr (K == 0) {
            return 1.;
        } else {
            std::array const source_ids
                    = ddc::detail::array(source_natural_elem_type(natural_elem));
            std::array const target_ids
                    = ddc::detail::array(target_natural_elem_type(natural_elem));
            using reduction_accessor_t
                    = sil::tensor::tensor_accessor_for_domain_t<reduction_domain_t<Indices>>;
            using reduction_natural_elem_type =
                    typename reduction_accessor_t::natural_domain_t::discrete_element_type;
            constexpr std::size_t matrix_size = source_index_type::mem_size();
            using memory_space = typename PositionType::memory_space;
            [[maybe_unused]] tensor::TensorAccessor<source_index_type> source_accessor;
            [[maybe_unused]] tensor::TensorAccessor<target_index_type> target_accessor;

            std::array<double, matrix_size * matrix_size> reduction_alloc {};
            std::array<double, matrix_size * matrix_size> inverse_alloc {};
            std::array<double, matrix_size * matrix_size> workspace_alloc {};
            auto reduction_matrix = misc::math::matrix_view<double, memory_space>(
                    reduction_alloc.data(),
                    matrix_size,
                    matrix_size);
            auto inverse_matrix = misc::math::matrix_view<double, memory_space>(
                    inverse_alloc.data(),
                    matrix_size,
                    matrix_size);
            auto workspace = misc::math::vector_view<double, memory_space>(
                    workspace_alloc.data(),
                    matrix_size * matrix_size);

            ddc::device_for_each(source_accessor.domain(), [&](auto source_mem_elem) {
                auto const source_natural_elem
                        = source_accessor.canonical_natural_element(source_mem_elem);
                ddc::device_for_each(target_accessor.domain(), [&](auto target_mem_elem) {
                    auto const target_natural_elem
                            = target_accessor.canonical_natural_element(target_mem_elem);
                    reduction_natural_elem_type reduction_natural_elem
                            = detail::merge_reduction_natural_elems<reduction_natural_elem_type>(
                                    ddc::detail::array(source_natural_elem),
                                    ddc::detail::array(target_natural_elem));
                    reduction_matrix(
                            source_mem_elem.template uid<source_index_type>(),
                            target_mem_elem.template uid<target_index_type>())
                            = Reduction<Indices, PositionType, BatchElem, Complex>::
                                    value(position, elem, reduction_natural_elem);
                });
            });

            bool const success = misc::math::invert(inverse_matrix, reduction_matrix, workspace);
            if (!success) {
                return 0.;
            }
            if (!detail::has_unique_reduction_ids<K>(source_ids)
                || !detail::has_unique_reduction_ids<K>(target_ids)) {
                return 0.;
            }
            std::size_t const target_mem_id = target_index_type::access_id_to_mem_id(
                    target_accessor.access_element(target_natural_elem_type(natural_elem))
                            .template uid<target_index_type>());
            std::size_t const source_mem_id = source_index_type::access_id_to_mem_id(
                    source_accessor.access_element(source_natural_elem_type(natural_elem))
                            .template uid<source_index_type>());
            return inverse_matrix(target_mem_id, source_mem_id)
                   / misc::factorial(K);
        }
    }
};

template <
        misc::Specialization<ddc::detail::TypeSeq> Indices,
        misc::Specialization<tensor::Tensor> ReconstructionTensorType,
        misc::Specialization<tensor::Tensor> PositionType,
        CellComplex Complex,
        class BatchElem>
struct FillReconstructionOperatorMem
{
    ReconstructionTensorType m_reconstruction_tensor;
    PositionType m_position;
    BatchElem m_elem;

    template <class MemElem>
    KOKKOS_FUNCTION void operator()(MemElem mem_elem) const
    {
        m_reconstruction_tensor.mem(
                typename ReconstructionTensorType::discrete_element_type(m_elem, mem_elem))
                = Reconstruction<Indices, PositionType, BatchElem, Complex>::
                        value(m_position,
                              m_elem,
                              m_reconstruction_tensor.accessor().canonical_natural_element(
                                      mem_elem));
    }
};

template <
        misc::Specialization<ddc::detail::TypeSeq> Indices,
        misc::Specialization<tensor::Tensor> ReconstructionTensorType,
        misc::Specialization<tensor::Tensor> PositionType,
        class ExecSpace,
        CellComplex Complex = CellComplex::Primal>
ReconstructionTensorType fill_reconstruction_operator(
        ExecSpace const& exec_space,
        ReconstructionTensorType reconstruction_tensor,
        PositionType position)
{
    SIMILIE_DEBUG_LOG("similie_compute_reconstruction_operator");
    ddc::parallel_for_each(
            "similie_compute_reconstruction_operator",
            exec_space,
            reconstruction_tensor.non_indices_domain(),
            KOKKOS_LAMBDA(
                    typename ReconstructionTensorType::non_indices_domain_t::discrete_element_type
                            elem) {
                ddc::device_for_each(
                        reconstruction_tensor.accessor().domain(),
                        FillReconstructionOperatorMem<
                                Indices,
                                ReconstructionTensorType,
                                PositionType,
                                Complex,
                                typename ReconstructionTensorType::non_indices_domain_t::
                                        discrete_element_type> {
                                reconstruction_tensor,
                                position,
                                elem});
            });
    return reconstruction_tensor;
}

} // namespace exterior

} // namespace sil
