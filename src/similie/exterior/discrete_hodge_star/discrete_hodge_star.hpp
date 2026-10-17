// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

#include <similie/exterior/hodge_star_detail.hpp>

namespace sil {
namespace exterior {

template <
        CellComplex Complex,
        misc::Specialization<ddc::TypeSeq> Indices1,
        misc::Specialization<ddc::TypeSeq> Indices2,
        class MetricType,
        class PositionType,
        class BatchElem>
struct DiscreteHodgeStar
{
    using staged_storage_type = detail::LocalOperatorStorage<
            DiscreteHodgeStar,
            tensor::lower_t<reduction_index_t<Indices1>>,
            reduction_index_t<Indices2>,
            typename MetricType::non_indices_domain_t,
            typename MetricType::memory_space,
            typename MetricType::memory_space::execution_space,
            tensor::TensorFullIndex,
            misc::convert_type_seq_to_t<ddc::DiscreteElement, Indices1>,
            misc::convert_type_seq_to_t<ddc::DiscreteElement, Indices2>>;

    /** Pair primal one-cochains using the full cell metric and orientation.
     * \important This operator and documentation is fully AI-generated.
     */
    KOKKOS_FUNCTION static auto one_form_pairing(
            MetricType metric,
            PositionType position,
            BatchElem elem)
    {
        using PositionIndex = ddc::
                type_seq_element_t<0, ddc::to_type_seq_t<typename PositionType::indices_domain_t>>;
        using MetricIndex = ddc::
                type_seq_element_t<0, ddc::to_type_seq_t<typename MetricType::indices_domain_t>>;
        constexpr std::size_t N = PositionIndex::size();
        std::array<double, N * N> jacobian {}, gram {}, inverse {}, workspace {};
        for (std::size_t axis = 0; axis < N; ++axis) {
            std::array<double, N> const edge
                    = detail::edge_vector<PositionIndex>(position, elem, axis);
            for (std::size_t physical = 0; physical < N; ++physical)
                jacobian[physical * N + axis] = edge[physical];
        }
        for (std::size_t i = 0; i < N; ++i)
            for (std::size_t j = 0; j < N; ++j)
                for (std::size_t a = 0; a < N; ++a)
                    for (std::size_t b = 0; b < N; ++b)
                        gram[i * N + j]
                                += jacobian[a * N + i] * jacobian[b * N + j]
                                   * metric.get(metric.access_element(
                                           elem,
                                           ddc::DiscreteElement<
                                                   tensor::metric_index_1<MetricIndex>,
                                                   tensor::metric_index_2<MetricIndex>>(a, b)));
        auto gram_view = misc::math::
                matrix_view<double, typename MetricType::memory_space>(gram.data(), N, N);
        auto inverse_view = misc::math::
                matrix_view<double, typename MetricType::memory_space>(inverse.data(), N, N);
        if (!misc::math::
                    invert(inverse_view,
                           gram_view,
                           misc::math::vector_view<
                                   double,
                                   typename MetricType::memory_space>(workspace.data(), N * N)))
            return inverse;
        double const orientation = detail::reduction_determinant<N>(jacobian);
        double const volume = (orientation < 0.0 ? -1.0 : 1.0)
                              * Kokkos::sqrt(Kokkos::abs(detail::reduction_determinant<N>(gram)));
        for (double& coefficient : inverse)
            coefficient *= volume;
        return inverse;
    }

    template <
            misc::Specialization<tensor::Tensor> HodgeTensorType,
            misc::Specialization<tensor::Tensor> FormTensorType>
    KOKKOS_FUNCTION static void operator()(
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
        std::array<double, InputIndex::mem_size()> reconstructed_alloc {};
        ddc::ChunkSpan<
                double,
                ddc::DiscreteDomain<InputIndex>,
                Kokkos::layout_right,
                typename FormTensorType::memory_space>
                reconstructed_span(reconstructed_alloc.data(), reconstructed_accessor.domain());
        sil::tensor::Tensor reconstructed_form(reconstructed_span);

        [[maybe_unused]] tensor::TensorAccessor<OutputIndex> continuous_output_accessor;
        std::array<double, OutputIndex::mem_size()> continuous_output_alloc {};
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
        operator()(reconstructed_form, form_tensor, position, elem);
        ContinuousHodgeStar<Indices1, Indices2, MetricType, BatchElem>::
        operator()(continuous_output, reconstructed_form, metric, elem);
        Reduction<OutputIndexSeq, PositionType, BatchElem, Complex>::
        operator()(hodge_tensor, continuous_output, metric, position, elem);
    }

    KOKKOS_FUNCTION static detail::LocalOperatorStencil<
            tensor::lower_t<reduction_index_t<Indices1>>,
            reduction_index_t<Indices2>>
    value(MetricType metric, PositionType position, BatchElem element)
    {
        using Source = tensor::lower_t<reduction_index_t<Indices1>>;
        using Target = reduction_index_t<Indices2>;
        detail::LocalOperatorStencil<Source, Target> stencil;
        auto unit = detail::make_stencil<typename MetricType::memory_space, Source>(
                ddc::DiscreteElement<>());
        auto output = detail::make_stencil<typename MetricType::memory_space, Target>(
                ddc::DiscreteElement<>());
        for (std::size_t source = 0; source < Source::mem_size(); ++source) {
            unit.mem(ddc::DiscreteElement<Source>(source)) = 1.0;
            operator()(
                    tensor::Tensor(output.span_view()),
                    tensor::Tensor(unit.span_view()),
                    metric,
                    position,
                    element);
            for (std::size_t target = 0; target < Target::mem_size(); ++target)
                stencil.weights[target * Source::mem_size() + source]
                        = output.mem(ddc::DiscreteElement<Target>(target));
            unit.mem(ddc::DiscreteElement<Source>(source)) = 0.0;
        }
        return stencil;
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
        std::array<double, SourceIndex::mem_size()> source_alloc {};
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
        std::array<double, TargetIndex::mem_size()> target_alloc {};
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
        for (auto i = source_ids.begin(); i < source_ids.end(); ++i) {
            for (auto j = i + 1; j < source_ids.end(); ++j) {
                odd = (*i > *j) != odd;
            }
        }
        std::array canonical_source_ids(source_ids);
        if constexpr (source_ids.size() > 1) {
            for (auto i = canonical_source_ids.begin(); i < canonical_source_ids.end(); ++i) {
                for (auto j = i + 1; j < canonical_source_ids.end(); ++j) {
                    if (*j < *i) {
                        Kokkos::kokkos_swap(*i, *j);
                    }
                }
            }
        }

        SourceNaturalElem source_natural_elem;
        ddc::detail::array(source_natural_elem) = canonical_source_ids;
        source_tensor(source_tensor.accessor().access_element(source_natural_elem)) = 1.;

        operator()(target_tensor, source_tensor, metric, position, elem);

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
                   * target_tensor.get(
                           target_tensor.accessor().access_element(target_natural_elem));
        }
    }
};

static_assert(DiscreteHodgeStarConcept<
              DiscreteHodgeStar<
                      CellComplex::Primal,
                      ddc::TypeSeq<detail::DirectionConcept>,
                      ddc::TypeSeq<>,
                      detail::FieldConcept<detail::MetricIndexConcept>,
                      detail::FieldConcept<detail::PositionIndexConcept>,
                      ddc::DiscreteElement<detail::GridConcept>>,
              detail::FieldConcept<detail::ScalarConcept>,
              detail::FieldConcept<detail::DirectionConcept>,
              detail::FieldConcept<detail::MetricIndexConcept>,
              detail::FieldConcept<detail::PositionIndexConcept>,
              ddc::DiscreteElement<detail::GridConcept>,
              ddc::DiscreteElement<detail::DirectionConcept>>);


} // namespace exterior
} // namespace sil

#include <similie/exterior/continuous_hodge_star/continuous_hodge_star.hpp>
