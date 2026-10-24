// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#pragma once

#include <similie/exterior/stencil.hpp>

namespace sil::exterior::detail {

template <
        class Operator,
        class Index,
        class Source,
        class Target,
        std::size_t Radius,
        class Tensor,
        class Exec = typename Tensor::memory_space::execution_space,
        bool EqualDomains = false>
class SpatialOperatorStorage
{
    using Domain = typename Tensor::non_indices_domain_t;
    StencilStorage<Index, Source, Target, Radius, Domain, typename Tensor::memory_space> m_stencils;

public:
    using execution_space = Exec;
    template <class OtherExec>
    using rebind = SpatialOperatorStorage<
            Operator,
            Index,
            Source,
            Target,
            Radius,
            Tensor,
            OtherExec,
            EqualDomains>;

    template <class Metric, class Position, class Rules = tensor::ZeroExtrapolationRule>
        requires requires(
                Exec const& exec,
                Tensor const& input,
                Metric const& metric,
                Position const& position,
                typename Domain::discrete_element_type element,
                Rules const& rules) {
            Operator::value(Operator::geometry(exec, metric, position), input, element, rules);
        }
    static SpatialOperatorStorage build(
            Exec const& exec,
            Tensor const& input,
            Metric const& metric,
            Position const& position,
            Rules const& rules = {})
    {
        SpatialOperatorStorage result;
        result.m_stencils = cache_stencils<
                Operator,
                Index,
                Source,
                Target,
                Radius>(exec, input, Operator::geometry(exec, metric, position), rules);
        return result;
    }

    template <
            class Output,
            class Metric,
            class Position,
            class Primal = tensor::ClampExtrapolationRule,
            class Dual = tensor::ZeroExtrapolationRule>
        requires requires(
                Exec const& exec,
                Tensor const& input,
                Metric const& metric,
                Position const& position,
                typename Domain::discrete_element_type element,
                Primal const& primal,
                Dual const& dual) {
            Operator::
                    value(Operator::geometry(exec, metric, position), input, element, primal, dual);
        }
    static SpatialOperatorStorage build(
            Exec const& exec,
            Output const& output,
            Tensor const& input,
            Metric const& metric,
            Position const& position,
            Primal const& primal = {},
            Dual const& dual = {})
    {
        if constexpr (EqualDomains) {
            if (input.non_indices_domain() != output.non_indices_domain())
                throw std::invalid_argument(
                        "Staged operator requires equal input/output spatial domains");
        }
        SpatialOperatorStorage result;
        result.m_stencils = cache_stencils<
                Operator,
                Index,
                Source,
                Target,
                Radius>(exec, input, Operator::geometry(exec, metric, position), primal, dual);
        return result;
    }

    template <misc::Specialization<tensor::Tensor> Output>
    Output apply(Exec const& exec, Output const& output, Tensor const& input) const
    {
        if constexpr (EqualDomains) {
            if (output.non_indices_domain() != m_stencils.domain)
                throw std::invalid_argument("Staged operator output domain changed");
        }
        return apply_stencils<Index, Source, Target, Radius>(exec, output, input, m_stencils);
    }
};

} // namespace sil::exterior::detail
