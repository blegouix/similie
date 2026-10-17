// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#pragma once

#include <similie/exterior/continuous_hodge_star/continuous_hodge_star.hpp>
#include <similie/exterior/continuous_hodge_star/staged_continuous_hodge_star.hpp>
#include <similie/exterior/reduction_and_reconstruction/staged_reduction.hpp>

namespace sil::exterior {
namespace detail {
template <class Position, class Index>
using LocalOperatorField = tensor::Tensor<
        double,
        misc::cartesian_prod_t<typename Position::non_indices_domain_t, ddc::DiscreteDomain<Index>>,
        Kokkos::layout_right,
        typename Position::memory_space>;

template <
        class Indices,
        class Position,
        class Element,
        CellComplex Complex,
        template <class...> class Coefficients = tensor::TensorFullIndex>
using CachedReconstruction = StagedReconstruction<
        Indices,
        LocalOperatorField<Position, reduction_index_t<tensor::primes<Indices>>>,
        LocalOperatorField<Position, reduction_index_t<Indices>>,
        Position,
        typename Position::memory_space::execution_space,
        Complex,
        Coefficients>;

template <
        class Indices,
        class Position,
        class Element,
        CellComplex Complex,
        template <class...> class Coefficients = tensor::TensorFullIndex>
using CachedReduction = StagedReduction<
        Indices,
        LocalOperatorField<Position, reduction_index_t<Indices>>,
        LocalOperatorField<Position, reduction_index_t<tensor::primes<tensor::upper_t<Indices>>>>,
        Position,
        typename Position::memory_space::execution_space,
        Complex,
        Coefficients>;

template <class Operator, class Exec, class Metric, class Position>
Operator make_composed_operator(Exec const& exec, Metric metric, Position position)
{
    if constexpr (std::is_constructible_v<Operator, Exec const&, Metric, Position>)
        return Operator(exec, metric, position);
    else if constexpr (std::is_constructible_v<Operator, Exec const&, Metric>)
        return Operator(exec, metric);
    else if constexpr (std::is_constructible_v<
                               Operator,
                               Exec const&,
                               Position,
                               typename Metric::non_indices_domain_t>)
        return Operator(exec, position, metric.non_indices_domain());
    else
        return Operator {};
}
} // namespace detail

/** Compose reconstruction, continuous Hodge star and reduction policies.
 * \important This operator and documentation is fully AI-generated.
 * Each policy can evaluate geometry at application or cache its coefficients.
 * The pointwise interface matches DiscreteHodgeStar, so the composition can be
 * used directly by Laplacian. Rebuild cached policies when their geometry changes.
 */
template <
        CellComplex Complex,
        class Indices1,
        class Indices2,
        class MetricType,
        class PositionType,
        class BatchElem,
        template <class, class, class, CellComplex> class ReconstructionPolicy = Reconstruction,
        template <class, class, class, CellComplex> class ReductionPolicy = Reduction,
        template <class, class, class, class> class HodgePolicy = ContinuousHodgeStar>
class ComposedDiscreteHodgeStar
{
    using Source = tensor::lower_t<reduction_index_t<Indices1>>;
    using Target = reduction_index_t<Indices2>;
    using ReconstructionOperator = ReconstructionPolicy<
            tensor::lower_t<Indices1>,
            PositionType,
            BatchElem,
            CellComplex::Primal>;
    using ReductionOperator = ReductionPolicy<Indices2, PositionType, BatchElem, Complex>;
    using HodgeOperator = HodgePolicy<Indices1, Indices2, MetricType, BatchElem>;
    ReconstructionOperator m_reconstruction;
    ReductionOperator m_reduction;
    HodgeOperator m_hodge;

public:
    template <class Exec>
    ComposedDiscreteHodgeStar(Exec const& exec, MetricType metric, PositionType position)
        : m_reconstruction(
                  detail::make_composed_operator<ReconstructionOperator>(exec, metric, position))
        , m_reduction(detail::make_composed_operator<ReductionOperator>(exec, metric, position))
        , m_hodge(detail::make_composed_operator<HodgeOperator>(exec, metric, position))
    {
    }

    template <class Output, class Input>
    KOKKOS_FUNCTION void operator()(
            Output output,
            Input input,
            MetricType metric,
            PositionType position,
            BatchElem element) const
    {
        auto reconstructed = detail::make_stencil<typename MetricType::memory_space, Source>(
                ddc::DiscreteElement<>());
        auto transformed = detail::make_stencil<typename MetricType::memory_space, Target>(
                ddc::DiscreteElement<>());
        m_reconstruction(tensor::Tensor(reconstructed.span_view()), input, position, element);
        m_hodge(tensor::Tensor(transformed.span_view()),
                tensor::Tensor(reconstructed.span_view()),
                metric,
                element);
        m_reduction(output, tensor::Tensor(transformed.span_view()), metric, position, element);
    }

    KOKKOS_FUNCTION detail::LocalOperatorStencil<Source, Target> value(
            MetricType metric,
            PositionType position,
            BatchElem element) const
    {
        detail::LocalOperatorStencil<Source, Target> result;
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
                result.weights[target * Source::mem_size() + source]
                        = output.mem(ddc::DiscreteElement<Target>(target));
            unit.mem(ddc::DiscreteElement<Source>(source)) = 0.0;
        }
        return result;
    }

    template <class NaturalElement>
    KOKKOS_FUNCTION double value(
            MetricType metric,
            PositionType position,
            BatchElem element,
            NaturalElement natural) const
    {
        using SourceElement = misc::convert_type_seq_to_t<ddc::DiscreteElement, Indices1>;
        using TargetElement = misc::convert_type_seq_to_t<ddc::DiscreteElement, Indices2>;
        return detail::local_stencil_coefficient<Source, Target>(
                value(metric, position, element),
                ddc::detail::array(SourceElement(natural)),
                ddc::detail::array(TargetElement(natural)));
    }
};

static_assert(DiscreteHodgeStarConcept<
              ComposedDiscreteHodgeStar<
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

} // namespace sil::exterior
