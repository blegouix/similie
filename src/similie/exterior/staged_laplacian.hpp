// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#pragma once

#include "laplacian.hpp"

namespace sil::exterior {
/** Cache the complete cochain Laplacian as fixed stencils for repeated application.
 * \important This operator and documentation is fully AI-generated.
 * Construction stores the fixed-size stencil returned by Laplacian::value(),
 * using the compile-time stencil_nonzero_indices() pattern. Application is a
 * single stencil kernel. Geometry and linear or affine boundary rules
 * are fixed at construction; rebuild the object when either changes. Nonlinear
 * boundary rules are not supported. The fixed support has Manhattan radius two
 * (25 spatial points in three dimensions); closures that sample outside this
 * support are rejected during construction. Input and output domains must remain equal
 * to the construction domains, and the allocations must not overlap.
 * GeneralStencil permits every supported geometry/closure coupling.
 * CartesianPeriodicStencil checks uniform orthogonal coordinates and requires
 * closures declaring is_periodic; it retains only the componentwise nearest
 * neighbors. No numerical threshold is used to choose stored entries.
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
        class Exec,
        class Pattern = GeneralStencil>
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
            DiscreteHodgeStar,
            Pattern>;
    Exec m_exec;
    Domain m_input_domain, m_output_domain;
    detail::StencilStorage<Operator, Index, Cochain, Cochain, 2, Domain, Memory> m_stencils;

public:
    KOKKOS_FUNCTION static constexpr auto stencil_nonzero_indices()
    {
        return Operator::stencil_nonzero_indices();
    }

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
        if constexpr (std::is_same_v<Pattern, CartesianPeriodicStencil>) {
            if constexpr (
                    !detail::periodic_stencil_rule<Primal>()
                    || !detail::periodic_stencil_rule<Dual>())
                throw std::invalid_argument(
                        "CartesianPeriodicStencil requires periodic primal and dual closures");
            detail::validate_cartesian_stencil_geometry<Index, MetricIndex>(exec, input, position);
        }
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
        return detail::apply_stencils<
                Operator,
                Index,
                Cochain,
                Cochain,
                2>(m_exec, output, input, m_stencils);
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
        class Pattern = GeneralStencil,
        class Exec,
        class Tensor,
        class Metric,
        class Position,
        class Primal = tensor::ClampExtrapolationRule,
        class Dual = tensor::ZeroExtrapolationRule>
StagedLaplacian<MetricIndex, Index, Cochain, Tensor, Metric, Position, Exec, Pattern>
make_staged_laplacian(
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
