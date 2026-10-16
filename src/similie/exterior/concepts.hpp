// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#pragma once

#include <concepts>

#include <similie/tensor/character.hpp>
#include <similie/tensor/metric.hpp>

#include "local_chain.hpp"

namespace sil::exterior::detail {

using GridConcept = tensor::detail::GridConcept;
using DirectionConcept
        = tensor::Covariant<tensor::TensorNaturalIndex<tensor::detail::DimensionConcept>>;
using ScalarConcept = tensor::Covariant<tensor::ScalarIndex>;
using MetricIndexConcept = tensor::TensorIdentityIndex<
        tensor::Covariant<tensor::MetricIndex1<tensor::detail::DimensionConcept>>,
        tensor::Covariant<tensor::MetricIndex2<tensor::detail::DimensionConcept>>>;
using PositionIndexConcept
        = tensor::Contravariant<tensor::TensorNaturalIndex<tensor::detail::DimensionConcept>>;

template <class Index>
using FieldConcept = tensor::Tensor<
        double,
        ddc::DiscreteDomain<GridConcept, Index>,
        Kokkos::layout_right,
        Kokkos::HostSpace>;

struct EvaluatorConcept
{
    template <class Element, class Component>
    KOKKOS_FUNCTION double operator()(Element element, Component component) const;
};

} // namespace sil::exterior::detail


namespace sil::exterior {

template <
        class Operator,
        class Output,
        class Evaluator,
        class Chain,
        class LowerChain,
        class Element,
        class NaturalElement>
concept CoboundaryConcept = requires(
        Output output,
        Evaluator evaluator,
        Chain chain,
        LowerChain lower_chain,
        Element element,
        NaturalElement natural_element) {
    Operator::value(chain, lower_chain, element, natural_element);
    { Operator::operator()(output, evaluator, chain, lower_chain, element) } -> std::same_as<void>;
};

template <
        class Operator,
        class Output,
        class Input,
        class Metric,
        class Position,
        class Element,
        class NaturalElement>
concept DiscreteHodgeStarConcept = requires(
        Operator operation,
        Output output,
        Input input,
        Metric metric,
        Position position,
        Element element,
        NaturalElement natural_element) {
    { operation.value(metric, position, element, natural_element) } -> std::same_as<double>;
    { operation(output, input, metric, position, element) } -> std::same_as<void>;
};

template <
        class Operator,
        class Output,
        class Input,
        class Metric,
        class Element,
        class NaturalElement>
concept ContinuousHodgeStarConcept = requires(
        Output output,
        Input input,
        Metric metric,
        Element element,
        NaturalElement natural_element) {
    { Operator::value(metric, element, natural_element) } -> std::same_as<double>;
    { Operator::operator()(output, input, metric, element) } -> std::same_as<void>;
};

template <class Operator, class Output, class Input = Output>
concept StagedOperatorConcept = requires(Operator& operation, Output output, Input input) {
    { operation(output, input) } -> std::same_as<Output>;
};

} // namespace sil::exterior
