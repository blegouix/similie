// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#pragma once

#include <concepts>

#include <similie/tensor/metric.hpp>
#include <similie/tensor/tensor_indices/character.hpp>

#include "cochain.hpp"
#include "form.hpp"
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

template <class Index>
using CochainFieldConcept = Cochain<FieldConcept<Index>>;

template <class Index>
using FormFieldConcept = Form<FieldConcept<Index>>;

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
concept CoboundaryConcept
        = CochainType<Output>
          && requires(
                  Output const& output,
                  Evaluator const& evaluator,
                  Chain const& chain,
                  LowerChain const& lower_chain,
                  Element element,
                  NaturalElement natural_element) {
                 Operator::value(chain, lower_chain, element, natural_element);
                 {
                     Operator::operator()(output, evaluator, chain, lower_chain, element)
                 } -> std::same_as<void>;
             };

template <
        class Operator,
        class Output,
        class Input,
        class Metric,
        class Position,
        class Element,
        class NaturalElement>
concept DiscreteHodgeStarConcept
        = CochainType<Output> && CochainType<Input>
          && requires(
                  Operator operation,
                  Output const& output,
                  Input const& input,
                  Metric const& metric,
                  Position const& position,
                  Element element,
                  NaturalElement natural_element) {
                 operation.value(metric, position, element);
                 {
                     operation.value(metric, position, element, natural_element)
                 } -> std::same_as<double>;
                 { operation(output, input, metric, position, element) } -> std::same_as<void>;
             };

template <
        class Operator,
        class Output,
        class Input,
        class Metric,
        class Element,
        class NaturalElement>
concept ContinuousHodgeStarConcept = FormType<Output> && FormType<Input>
                                     && requires(
                                             Operator operation,
                                             Output const& output,
                                             Input const& input,
                                             Metric const& metric,
                                             Element element,
                                             NaturalElement natural_element) {
                                            requires Operator::input_is_form;
                                            requires Operator::output_is_form;
                                            operation.value(metric, element);
                                            {
                                                operation.value(metric, element, natural_element)
                                            } -> std::same_as<double>;
                                            {
                                                operation(output, input, metric, element)
                                            } -> std::same_as<void>;
                                        };

template <class T>
concept ExteriorFieldType = FormType<T> || CochainType<T>;

namespace detail {
template <class Operator>
constexpr bool input_is_form = [] {
    if constexpr (requires { Operator::input_is_form; })
        return Operator::input_is_form;
    else
        return false;
}();
template <class Operator>
constexpr bool output_is_form = [] {
    if constexpr (requires { Operator::output_is_form; })
        return Operator::output_is_form;
    else
        return false;
}();
} // namespace detail

template <class Operator, class Input>
concept OperatorInputType = (detail::input_is_form<Operator> && FormType<Input>)
                            || (!detail::input_is_form<Operator> && CochainType<Input>);
template <class Operator, class Output>
concept OperatorOutputType = (detail::output_is_form<Operator> && FormType<Output>)
                             || (!detail::output_is_form<Operator> && CochainType<Output>);

template <class Operator, class Output, class Input, class... Arguments>
concept LocalOperatorConcept = requires(
        Operator const& operation,
        Output const& output,
        Input const& input,
        Arguments... arguments) {
    { operation(output, input, arguments...) } -> std::same_as<void>;
};

template <class Operator, class Output, class Input = Output>
concept StagedOperatorConcept
        = requires(Operator const& operation, Output const& output, Input const& input) {
              { operation(output, input) } -> std::same_as<Output>;
          };

template <class Storage, class Output, class Input = Output>
concept StagingStorageConcept = requires(
        Storage const& storage,
        typename Storage::execution_space const& exec,
        Output const& output,
        Input const& input) {
    { storage.apply(exec, output, input) } -> std::same_as<Output>;
};

template <class Operator, class Output, class Input = Output>
concept StageableOperatorConcept
        = OperatorInputType<Operator, Input> && OperatorOutputType<Operator, Output> && requires {
              typename Operator::staged_storage_type;
              requires StagingStorageConcept<typename Operator::staged_storage_type, Output, Input>;
          };

} // namespace sil::exterior
