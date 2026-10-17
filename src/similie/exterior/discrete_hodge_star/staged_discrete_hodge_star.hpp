// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#pragma once

#include <similie/exterior/continuous_hodge_star/continuous_hodge_star.hpp>

namespace sil::exterior {

/** Discrete Hodge star with geometry coefficients fixed at construction.
 * \important This operator and documentation is fully AI-generated.
 * The pointwise interface matches DiscreteHodgeStar. The metric and position
 * arguments of that interface are ignored: rebuild when geometry changes.
 * Whole-field application uses the execution space supplied at construction.
 */
template <
        CellComplex Complex,
        misc::Specialization<ddc::TypeSeq> Indices1,
        misc::Specialization<ddc::TypeSeq> Indices2,
        class MetricType,
        class PositionType,
        class BatchElem>
class StagedHodgeStar
{
    using Memory = typename MetricType::memory_space;
    using Domain = typename MetricType::non_indices_domain_t;
    using Source = tensor::lower_t<reduction_index_t<Indices1>>;
    using Target = reduction_index_t<Indices2>;
    using Operator
            = DiscreteHodgeStar<Complex, Indices1, Indices2, MetricType, PositionType, BatchElem>;
    Kokkos::View<typename Memory::execution_space, Kokkos::HostSpace> m_exec;
    detail::LocalStencilStorage<Source, Target, Domain, Memory> m_stencils;

public:
    template <class Exec>
    StagedHodgeStar(Exec const& exec, MetricType metric, PositionType position)
        : m_exec("staged_hodge_execution")
    {
        m_exec() = exec;
        m_stencils = detail::cache_local_stencils<Source, Target, Memory>(
                exec,
                metric.non_indices_domain(),
                detail::HodgeStencilValue<Operator, MetricType, PositionType> {metric, position});
    }

    KOKKOS_FUNCTION detail::LocalOperatorStencil<Source, Target> value(
            MetricType,
            PositionType,
            BatchElem element) const
    {
        detail::LocalOperatorStencil<Source, Target> result;
        std::size_t const node = detail::stencil_node(m_stencils.domain, element);
        for (std::size_t entry = 0; entry < result.weights.size(); ++entry)
            result.weights[entry] = m_stencils.weights(node, entry);
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

    template <class Output, class Input>
    KOKKOS_FUNCTION void operator()(
            Output output,
            Input input,
            MetricType,
            PositionType,
            BatchElem element) const
    {
        std::size_t const node = detail::stencil_node(m_stencils.domain, element);
        detail::apply_local_stencil<Source, Target>(output, input, [&](std::size_t entry) {
            return m_stencils.weights(node, entry);
        });
    }

    template <class Output, class Input>
    Output operator()(Output output, Input input) const
    {
        auto const weights = m_stencils.weights;
        Domain const domain = m_stencils.domain;
        ddc::parallel_for_each(
                "similie_apply_staged_hodge_star",
                m_exec(),
                output.non_indices_domain(),
                KOKKOS_LAMBDA(BatchElem element) {
                    std::size_t const node = detail::stencil_node(domain, element);
                    detail::apply_local_stencil<
                            Source,
                            Target>(output[element], input[element], [&](std::size_t entry) {
                        return weights(node, entry);
                    });
                });
        return output;
    }
};

static_assert(DiscreteHodgeStarConcept<
              StagedHodgeStar<
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

static_assert(StagedOperatorConcept<
              StagedHodgeStar<
                      CellComplex::Primal,
                      ddc::TypeSeq<detail::DirectionConcept>,
                      ddc::TypeSeq<>,
                      detail::FieldConcept<detail::MetricIndexConcept>,
                      detail::FieldConcept<detail::PositionIndexConcept>,
                      ddc::DiscreteElement<detail::GridConcept>>,
              detail::FieldConcept<detail::ScalarConcept>,
              detail::FieldConcept<detail::DirectionConcept>>);

} // namespace sil::exterior
