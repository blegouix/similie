// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: MIT

#pragma once

#include <array>

#include <ddc/ddc.hpp>

#include <similie/misc/factorial.hpp>
#include <similie/misc/macros.hpp>
#include <similie/misc/small_matrix.hpp>
#include <similie/misc/specialization.hpp>
#include <similie/misc/type_seq_conversion.hpp>
#include <similie/tensor/antisymmetric_tensor.hpp>
#include <similie/tensor/character.hpp>
#include <similie/tensor/full_tensor.hpp>
#include <similie/tensor/metric.hpp>

#include "reduction_and_reconstruction.hpp"
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
        using InputIndex = ddc::type_seq_element_t<
                0,
                ddc::to_type_seq_t<typename FormTensorType::indices_domain_t>>;
        using OutputIndex = ddc::type_seq_element_t<
                0,
                ddc::to_type_seq_t<typename HodgeTensorType::indices_domain_t>>;
        using InputIndexSeq
                = ddc::to_type_seq_t<typename FormTensorType::accessor_t::natural_domain_t>;
        using OutputIndexSeq
                = ddc::to_type_seq_t<typename HodgeTensorType::accessor_t::natural_domain_t>;

        [[maybe_unused]] tensor::TensorAccessor<InputIndex> reconstructed_accessor;
        std::array<double, InputIndex::access_size()> reconstructed_alloc {};
        ddc::ChunkSpan<
                double,
                ddc::DiscreteDomain<InputIndex>,
                Kokkos::layout_right,
                typename FormTensorType::memory_space>
                reconstructed_span(reconstructed_alloc.data(), reconstructed_accessor.domain());
        sil::tensor::Tensor reconstructed_form(reconstructed_span);

        [[maybe_unused]] tensor::TensorAccessor<OutputIndex> continuous_output_accessor;
        std::array<double, OutputIndex::access_size()> continuous_output_alloc {};
        ddc::ChunkSpan<
                double,
                ddc::DiscreteDomain<OutputIndex>,
                Kokkos::layout_right,
                typename HodgeTensorType::memory_space>
                continuous_output_span(
                        continuous_output_alloc.data(),
                        continuous_output_accessor.domain());
        sil::tensor::Tensor continuous_output(continuous_output_span);

        Reconstruction<InputIndexSeq, PositionType, BatchElem, CellComplex::Primal>::
                run(reconstructed_form, form_tensor, position, elem);
        ContinuousHodgeStar<Indices1, Indices2, MetricType, BatchElem>::
                run(continuous_output, reconstructed_form, metric, elem);
        Reduction<OutputIndexSeq, PositionType, BatchElem, Complex>::
                run(hodge_tensor, continuous_output, metric, position, elem);
    }

    KOKKOS_FUNCTION static double value(
            MetricType metric,
            PositionType position,
            BatchElem elem,
            auto natural_elem)
    {
        using SourceIndex = tensor::lower_t<reduction_index_t<Indices1>>;
        using TargetIndex = reduction_index_t<Indices2>;
        using SourceNaturalElem =
                typename tensor::natural_domain_t<SourceIndex>::discrete_element_type;
        using TargetNaturalElem =
                typename tensor::natural_domain_t<TargetIndex>::discrete_element_type;
        using SourceFullElem = misc::convert_type_seq_to_t<ddc::DiscreteElement, Indices1>;
        using TargetFullElem = misc::convert_type_seq_to_t<ddc::DiscreteElement, Indices2>;

        [[maybe_unused]] tensor::TensorAccessor<SourceIndex> source_accessor;
        std::array<double, SourceIndex::access_size()> source_alloc {};
        ddc::ChunkSpan<
                double,
                ddc::DiscreteDomain<SourceIndex>,
                Kokkos::layout_right,
                typename MetricType::memory_space>
                source_span(source_alloc.data(), source_accessor.domain());
        sil::tensor::Tensor source_tensor(source_span);
        ddc::device_for_each(source_tensor.domain(), [&](auto source_mem_elem) {
            source_tensor.mem(source_mem_elem) = 0.;
        });

        [[maybe_unused]] tensor::TensorAccessor<TargetIndex> target_accessor;
        std::array<double, TargetIndex::access_size()> target_alloc {};
        ddc::ChunkSpan<
                double,
                ddc::DiscreteDomain<TargetIndex>,
                Kokkos::layout_right,
                typename MetricType::memory_space>
                target_span(target_alloc.data(), target_accessor.domain());
        sil::tensor::Tensor target_tensor(target_span);
        ddc::device_for_each(target_tensor.domain(), [&](auto target_mem_elem) {
            target_tensor.mem(target_mem_elem) = 0.;
        });

        std::array source_ids = ddc::detail::array(SourceFullElem(natural_elem));
        if constexpr (ddc::type_seq_size_v<Indices1> > 1) {
            if (!detail::has_unique_ids<SourceIndex::size()>(source_ids)) {
                return 0.;
            }
        }
        bool odd = false;
        for (std::size_t i = 0; i < source_ids.size(); ++i) {
            for (std::size_t j = i + 1; j < source_ids.size(); ++j) {
                odd = (source_ids[i] > source_ids[j]) != odd;
            }
        }
        std::array canonical_source_ids(source_ids);
        if constexpr (source_ids.size() > 1) {
            for (std::size_t i = 0; i < canonical_source_ids.size(); ++i) {
                for (std::size_t j = i + 1; j < canonical_source_ids.size(); ++j) {
                    if (canonical_source_ids[j] < canonical_source_ids[i]) {
                        Kokkos::kokkos_swap(canonical_source_ids[i], canonical_source_ids[j]);
                    }
                }
            }
        }

        SourceNaturalElem source_natural_elem;
        ddc::detail::array(source_natural_elem) = canonical_source_ids;
        source_tensor(source_tensor.accessor().access_element(source_natural_elem)) = 1.;

        run(target_tensor, source_tensor, metric, position, elem);

        double const source_factor
                = (odd ? -1. : 1.) / misc::factorial(ddc::type_seq_size_v<Indices1>);
        if constexpr (ddc::type_seq_size_v<Indices2> == 0) {
            return source_factor
                   * target_tensor(target_tensor.accessor().access_element(TargetNaturalElem()));
        } else {
            TargetNaturalElem target_natural_elem;
            ddc::detail::array(target_natural_elem)
                    = ddc::detail::array(TargetFullElem(natural_elem));
            return source_factor
                   * target_tensor(target_tensor.accessor().access_element(target_natural_elem));
        }
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
        constexpr std::size_t N = ddc::type_seq_size_v<Indices1> + ddc::type_seq_size_v<Indices2>;
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
        auto workspace = misc::math::vector_view<
                double,
                typename MetricType::memory_space>(workspace_alloc.data(), N * N);
        if (!misc::math::invert(inverse_metric_view, metric_view, workspace)) {
            return 0.;
        }

        return Kokkos::sqrt(Kokkos::abs(determinant))
               * static_cast<double>(detail::permutation_sign<N>(complement, target_ids))
               * misc::math::submatrix_determinant<
                       K>(inverse_metric_view, source_ids, complement, submatrix_alloc)
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
