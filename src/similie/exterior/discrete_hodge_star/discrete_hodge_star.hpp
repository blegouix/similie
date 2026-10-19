// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#pragma once

#include <similie/exterior/continuous_hodge_star/continuous_hodge_star.hpp>

namespace sil::exterior {
namespace detail {
template <class Operator, class Exec, class Metric, class Position>
Operator make_composed_operator(Exec const& exec, Metric const& metric, Position const& position)
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
        template <class, class, class, CellComplex> class ReconstructionPolicy,
        template <class, class, class, CellComplex> class ReductionPolicy,
        template <class, class, class, class> class HodgePolicy>
struct DiscreteHodgeStar
{
private:
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


    KOKKOS_FUNCTION DiscreteHodgeStar() = default;

    template <class Exec>
    DiscreteHodgeStar(Exec const& exec, MetricType const& metric, PositionType const& position)
        : m_reconstruction(
                  detail::make_composed_operator<ReconstructionOperator>(exec, metric, position))
        , m_reduction(detail::make_composed_operator<ReductionOperator>(exec, metric, position))
        , m_hodge(detail::make_composed_operator<HodgeOperator>(exec, metric, position))
    {
    }

    template <class Output, class Input>
    KOKKOS_FUNCTION void operator()(
            Output const& output,
            Input const& input,
            MetricType const& metric,
            PositionType const& position,
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
            MetricType const& metric,
            PositionType const& position,
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
            MetricType const& metric,
            PositionType const& position,
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

namespace detail {
template <
        CellComplex Complex,
        misc::Specialization<ddc::TypeSeq> Indices1,
        misc::Specialization<ddc::TypeSeq> Indices2,
        misc::Specialization<tensor::Tensor> HodgeStarType,
        misc::Specialization<tensor::Tensor> MetricType,
        misc::Specialization<tensor::Tensor> PositionType,
        class BatchElem>
struct FillDiscreteHodgeStarMem
{
    HodgeStarType m_hodge_star;
    MetricType m_metric;
    PositionType m_position;
    BatchElem m_elem;

    template <class MemElem>
    KOKKOS_FUNCTION void operator()(MemElem mem_elem) const
    {
        m_hodge_star.mem(typename HodgeStarType::discrete_element_type(m_elem, mem_elem))
                = DiscreteHodgeStar<
                          Complex,
                          Indices1,
                          Indices2,
                          MetricType,
                          PositionType,
                          BatchElem> {}
                          .value(m_metric,
                                 m_position,
                                 m_elem,
                                 m_hodge_star.accessor().canonical_natural_element(mem_elem));
    }
};

} // namespace detail

} // namespace sil::exterior
