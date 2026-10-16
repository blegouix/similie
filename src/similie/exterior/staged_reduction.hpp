// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#pragma once

#include "reduction_and_reconstruction.hpp"

namespace sil::exterior {
namespace detail {
template <class Operator, class Metric, class Position>
struct ReductionMetricStencilValue
{
    Metric metric;
    Position position;
    template <class Element>
    KOKKOS_FUNCTION auto operator()(Element element) const
    {
        return Operator::value(metric, position, element);
    }
};
} // namespace detail

/** Cache reduction in the canonical form-component basis.
 * \important This operator and documentation is fully AI-generated.
 * Coefficients selects TensorFullIndex or TensorDiagonalIndex. The diagonal
 * choice is a geometric contract, checked by assertions only in Debug builds.
 * Rebuild when positions change, or when the metric changes for dual reduction.
 */
template <
        misc::Specialization<ddc::TypeSeq> Indices,
        misc::Specialization<tensor::Tensor> ReducedTensorType,
        misc::Specialization<tensor::Tensor> FormTensorType,
        misc::Specialization<tensor::Tensor> PositionType,
        class ExecSpace,
        CellComplex Complex = CellComplex::Primal,
        template <class...> class Coefficients = tensor::TensorFullIndex>
class StagedReduction
{
    using Source = reduction_index_t<tensor::primes<tensor::upper_t<Indices>>>;
    using Target = reduction_index_t<Indices>;
    using Domain = typename ReducedTensorType::non_indices_domain_t;
    using Memory = typename ReducedTensorType::memory_space;
    using Operator
            = Reduction<Indices, PositionType, typename Domain::discrete_element_type, Complex>;
    Kokkos::View<ExecSpace, Kokkos::HostSpace> m_exec;
    detail::LocalStencilStorage<Source, Target, Domain, Memory, Coefficients> m_stencils;

public:
    StagedReduction(
            ExecSpace const& exec,
            ReducedTensorType output,
            FormTensorType,
            PositionType position)
        : StagedReduction(exec, position, output.non_indices_domain())
    {
    }

    template <class Metric>
    StagedReduction(
            ExecSpace const& exec,
            ReducedTensorType output,
            FormTensorType,
            Metric metric,
            PositionType position)
        : StagedReduction(exec, metric, position, output.non_indices_domain())
    {
    }

    StagedReduction(ExecSpace const& exec, PositionType position, Domain domain)
        : m_exec("staged_reduction_execution")
    {
        m_exec() = exec;
        m_stencils = detail::cache_local_stencils<Source, Target, Memory, Coefficients>(
                exec,
                domain,
                detail::ReconstructionStencilValue<Operator, PositionType> {position});
    }

    template <class Metric>
    StagedReduction(ExecSpace const& exec, Metric metric, PositionType position, Domain domain)
        : m_exec("staged_reduction_execution")
    {
        m_exec() = exec;
        m_stencils = detail::cache_local_stencils<Source, Target, Memory, Coefficients>(
                exec,
                domain,
                detail::ReductionMetricStencilValue<
                        Operator,
                        Metric,
                        PositionType> {metric, position});
    }

    template <class Output, class Input>
    KOKKOS_FUNCTION void operator()(
            Output output,
            Input input,
            PositionType,
            typename Domain::discrete_element_type element) const
    {
        std::size_t const node = detail::stencil_node(m_stencils.domain, element);
        detail::apply_local_stencil<
                Source,
                Target,
                Coefficients>(output, input, [&](std::size_t entry) {
            return m_stencils.weights(node, entry);
        });
    }

    template <class Output, class Input, class Metric>
    KOKKOS_FUNCTION void operator()(
            Output output,
            Input input,
            Metric,
            PositionType position,
            typename Domain::discrete_element_type element) const
    {
        operator()(output, input, position, element);
    }

    ReducedTensorType operator()(ReducedTensorType output, FormTensorType input) const
    {
        auto const weights = m_stencils.weights;
        Domain const domain = m_stencils.domain;
        ddc::parallel_for_each(
                "similie_apply_staged_reduction",
                m_exec(),
                output.non_indices_domain(),
                KOKKOS_LAMBDA(typename Domain::discrete_element_type element) {
                    std::size_t const node = detail::stencil_node(domain, element);
                    detail::apply_local_stencil<
                            Source,
                            Target,
                            Coefficients>(output[element], input[element], [&](std::size_t entry) {
                        return weights(node, entry);
                    });
                });
        return output;
    }
};

static_assert(StagedOperatorConcept<
              StagedReduction<
                      ddc::TypeSeq<detail::DirectionConcept>,
                      detail::FieldConcept<detail::DirectionConcept>,
                      detail::FieldConcept<detail::DirectionConcept>,
                      detail::FieldConcept<detail::PositionIndexConcept>,
                      Kokkos::DefaultHostExecutionSpace>,
              detail::FieldConcept<detail::DirectionConcept>>);


static_assert(LocalOperatorConcept<
              StagedReduction<
                      ddc::TypeSeq<detail::DirectionConcept>,
                      detail::FieldConcept<detail::DirectionConcept>,
                      detail::FieldConcept<detail::DirectionConcept>,
                      detail::FieldConcept<detail::PositionIndexConcept>,
                      Kokkos::DefaultHostExecutionSpace>,
              detail::FieldConcept<detail::DirectionConcept>,
              detail::FieldConcept<detail::DirectionConcept>,
              detail::FieldConcept<detail::PositionIndexConcept>,
              ddc::DiscreteElement<detail::GridConcept>>);
static_assert(LocalOperatorConcept<
              StagedReduction<
                      ddc::TypeSeq<detail::DirectionConcept>,
                      detail::FieldConcept<detail::DirectionConcept>,
                      detail::FieldConcept<detail::DirectionConcept>,
                      detail::FieldConcept<detail::PositionIndexConcept>,
                      Kokkos::DefaultHostExecutionSpace>,
              detail::FieldConcept<detail::DirectionConcept>,
              detail::FieldConcept<detail::DirectionConcept>,
              detail::FieldConcept<detail::MetricIndexConcept>,
              detail::FieldConcept<detail::PositionIndexConcept>,
              ddc::DiscreteElement<detail::GridConcept>>);

template <
        class Indices,
        CellComplex Complex = CellComplex::Primal,
        template <class...> class Coefficients = tensor::TensorFullIndex,
        class Exec,
        class Output,
        class Input,
        class Position>
StagedReduction<Indices, Output, Input, Position, Exec, Complex, Coefficients>
make_staged_reduction(Exec const& exec, Output output, Input input, Position position)
{
    return StagedReduction<
            Indices,
            Output,
            Input,
            Position,
            Exec,
            Complex,
            Coefficients>(exec, output, input, position);
}

} // namespace sil::exterior
