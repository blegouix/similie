// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#pragma once

#include <similie/exterior/laplacian/laplacian.hpp>

namespace sil::exterior {
/** Cache the complete cochain Laplacian as fixed stencils for repeated application.
 * \important This operator and documentation is fully AI-generated.
 * Construction stores the fixed-size stencil returned by Laplacian::value(),
 * retaining every coefficient in the standard layout. Application is a
 * single stencil kernel. Geometry and linear or affine boundary rules
 * are fixed at construction; rebuild the object when either changes. Nonlinear
 * boundary rules are not supported. The fixed support has Manhattan radius two
 * (25 spatial points in three dimensions); closures that sample outside this
 * support are rejected during construction. Input and output domains must remain equal
 * to the construction domains, and the allocations must not overlap.
 * Each row is evaluated using differences from the cochain at its center.
 * The response to each constant component is cached separately. This equivalent
 * form preserves constant null modes without cancellation of stencil weights.
 */
template <
        tensor::TensorIndex MetricIndex,
        tensor::TensorNatIndex Index,
        tensor::TensorIndex Cochain,
        misc::Specialization<tensor::Tensor> Tensor,
        misc::Specialization<tensor::Tensor> Metric,
        misc::Specialization<tensor::Tensor> Position,
        class Exec>
class StagedLaplacian
{
    using Memory = typename Tensor::memory_space;
    using Domain = typename Tensor::non_indices_domain_t;
    using Operator = Laplacian<
            MetricIndex,
            Index,
            Cochain,
            Tensor,
            Metric,
            Position,
            Exec,
            DiscreteHodgeStar>;
    Exec m_exec;
    Domain m_input_domain, m_output_domain;
    detail::StencilStorage<Index, Cochain, Cochain, 2, Domain, Memory> m_stencils;

public:
    template <
            class Primal = tensor::ClampExtrapolationRule,
            class Dual = tensor::ZeroExtrapolationRule>
    StagedLaplacian(
            Exec const& exec,
            Tensor output,
            Tensor input,
            Metric metric,
            Position position,
            Primal primal = {},
            Dual dual = {})
        : m_exec(exec)
        , m_input_domain(input.non_indices_domain())
        , m_output_domain(output.non_indices_domain())
    {
        if (m_input_domain != m_output_domain)
            throw std::invalid_argument(
                    "StagedLaplacian requires equal input/output spatial domains");
        m_stencils = detail::cache_stencils<
                Operator,
                Index,
                Cochain,
                Cochain,
                2>(exec, input, Operator::geometry(exec, metric, position), primal, dual);
    }

    Tensor operator()(Tensor output, Tensor input) const
    {
        if (input.non_indices_domain() != m_input_domain
            || output.non_indices_domain() != m_output_domain)
            throw std::invalid_argument("StagedLaplacian domain changed");
        return detail::
                apply_stencils<Index, Cochain, Cochain, 2>(m_exec, output, input, m_stencils);
    }
};

static_assert(StagedOperatorConcept<
              StagedLaplacian<
                      detail::MetricIndexConcept,
                      detail::DirectionConcept,
                      detail::ScalarConcept,
                      detail::FieldConcept<detail::ScalarConcept>,
                      detail::FieldConcept<detail::MetricIndexConcept>,
                      detail::FieldConcept<detail::PositionIndexConcept>,
                      Kokkos::DefaultHostExecutionSpace>,
              detail::FieldConcept<detail::ScalarConcept>>);

template <
        tensor::TensorIndex MetricIndex,
        tensor::TensorNatIndex Index,
        tensor::TensorIndex Cochain,
        class Exec,
        class Tensor,
        class Metric,
        class Position,
        class Primal = tensor::ClampExtrapolationRule,
        class Dual = tensor::ZeroExtrapolationRule>
StagedLaplacian<MetricIndex, Index, Cochain, Tensor, Metric, Position, Exec> make_staged_laplacian(
        Exec const& exec,
        Tensor output,
        Tensor input,
        Metric metric,
        Position position,
        Primal primal = {},
        Dual dual = {})
{
    return {exec, output, input, metric, position, primal, dual};
}

} // namespace sil::exterior
