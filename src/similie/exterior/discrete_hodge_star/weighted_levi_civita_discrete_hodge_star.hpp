// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#pragma once

#include <similie/exterior/discrete_hodge_star/discrete_hodge_star.hpp>
#include <similie/tensor/tensor_indices/weighted_levi_civita_tensor.hpp>

namespace sil::exterior {
namespace detail {

template <class SourceIndices, class TargetIndices>
using WeightedHodgeIndex = tensor::TensorWeightedLeviCivitaIndex<
        misc::convert_type_seq_to_t<tensor::TensorAntisymmetricIndex, SourceIndices>,
        misc::convert_type_seq_to_t<tensor::TensorAntisymmetricIndex, TargetIndices>>;

template <class SourceIndices, class TargetIndices>
KOKKOS_FUNCTION std::array<std::size_t, 2> weighted_hodge_components(std::size_t entry)
{
    using Index = WeightedHodgeIndex<SourceIndices, TargetIndices>;
    using Target = misc::convert_type_seq_to_t<tensor::TensorAntisymmetricIndex, TargetIndices>;
    std::array<std::size_t, Target::rank()> target {};
    std::array const ids = Index::mem_id_to_canonical_natural_ids(entry);
    for (std::size_t i = 0; i < target.size(); ++i)
        target[i] = ids[ddc::type_seq_size_v<SourceIndices> + i];
    return {entry, Target::mem_id(target)};
}

template <class SourceIndices, class TargetIndices, class Output, class Input>
KOKKOS_FUNCTION void apply_weighted_hodge(
        Output const& output,
        Input const& input,
        std::array<double, WeightedHodgeIndex<SourceIndices, TargetIndices>::mem_size()> const&
                weights)
{
    using Index = WeightedHodgeIndex<SourceIndices, TargetIndices>;
    tensor::Tensor coefficients(
            ddc::ChunkSpan<
                    double const,
                    ddc::DiscreteDomain<Index>,
                    Kokkos::layout_right,
                    typename Input::
                            memory_space>(weights.data(), tensor::TensorAccessor<Index>::domain()));
    if constexpr (ddc::type_seq_size_v<SourceIndices> == 0) {
        tensor::Tensor scalar(
                ddc::ChunkSpan<
                        typename Input::element_type,
                        ddc::DiscreteDomain<>,
                        Kokkos::layout_right,
                        typename Input::
                                memory_space>(input.data_handle(), ddc::DiscreteDomain<>()));
        tensor::tensor_prod(output, coefficients, scalar);
    } else {
        tensor::tensor_prod(output, coefficients, input);
    }
    ddc::device_for_each(output.domain(), [&](auto component) {
        output.mem(component) /= misc::factorial(ddc::type_seq_size_v<SourceIndices>);
    });
}

template <class Operator, class SourceIndices, class TargetIndices, class Metric>
class WeightedHodgeStorage
{
    using Index = WeightedHodgeIndex<SourceIndices, TargetIndices>;
    using Source = tensor::lower_t<reduction_index_t<SourceIndices>>;
    using Target = reduction_index_t<TargetIndices>;
    using Domain = typename Metric::non_indices_domain_t;
    using Element = typename Domain::discrete_element_type;
    Domain m_domain;
    Kokkos::View<double**, Kokkos::LayoutLeft, typename Metric::memory_space> m_weights;

    KOKKOS_FUNCTION std::array<double, Index::mem_size()> weights(Element element) const
    {
        std::array<double, Index::mem_size()> result {};
        std::size_t const node = stencil_node(m_domain, element);
        for (std::size_t entry = 0; entry < result.size(); ++entry)
            result[entry] = m_weights(node, entry);
        return result;
    }

public:
    using execution_space = typename Metric::memory_space::execution_space;

    template <class Position>
    static WeightedHodgeStorage build(
            execution_space const& exec,
            Metric const& metric,
            Position const& position)
    {
        WeightedHodgeStorage result;
        result.m_domain = metric.non_indices_domain();
        result.m_weights
                = Kokkos::View<double**, Kokkos::LayoutLeft, typename Metric::memory_space>(
                        "weighted_levi_civita_hodge_coefficients",
                        result.m_domain.size(),
                        Index::mem_size());
        WeightedHodgeStorage const storage = result;
        ddc::parallel_for_each(
                "cache_weighted_levi_civita_hodge",
                exec,
                result.m_domain,
                KOKKOS_LAMBDA(Element element) {
                    auto const stencil = Operator::value(metric, position, element);
                    std::size_t const node = stencil_node(storage.m_domain, element);
                    for (std::size_t entry = 0; entry < Index::mem_size(); ++entry) {
                        std::array const components
                                = weighted_hodge_components<SourceIndices, TargetIndices>(entry);
                        storage.m_weights(node, entry)
                                = stencil.weights
                                          [components[1] * Source::mem_size() + components[0]];
                    }
                });
        return result;
    }

    template <class Output, class Input, class Position>
    KOKKOS_FUNCTION void apply_pointwise(
            Output const& output,
            Input const& input,
            Metric const&,
            Position const&,
            Element element) const
    {
        apply_weighted_hodge<SourceIndices, TargetIndices>(output, input, weights(element));
    }
    template <class Output, class Input>
    Output apply(execution_space const& exec, Output const& output, Input const& input) const
    {
        WeightedHodgeStorage const storage = *this;
        ddc::parallel_for_each(
                "apply_weighted_levi_civita_hodge",
                exec,
                output.non_indices_domain(),
                KOKKOS_LAMBDA(Element element) {
                    apply_weighted_hodge<SourceIndices, TargetIndices>(
                            output[element],
                            input[element],
                            storage.weights(element));
                });
        return output;
    }
    template <class Position>
    KOKKOS_FUNCTION LocalOperatorStencil<Source, Target> value(
            Metric const&,
            Position const&,
            Element element) const
    {
        LocalOperatorStencil<Source, Target> result;
        std::array const coefficients = weights(element);
        for (std::size_t entry = 0; entry < coefficients.size(); ++entry) {
            std::array const components
                    = weighted_hodge_components<SourceIndices, TargetIndices>(entry);
            result.weights[components[1] * Source::mem_size() + components[0]]
                    = coefficients[entry];
        }
        return result;
    }
    template <class Position, class Natural>
    KOKKOS_FUNCTION double value(
            Metric const& metric,
            Position const& position,
            Element element,
            Natural natural) const
    {
        using SourceElement = misc::convert_type_seq_to_t<ddc::DiscreteElement, SourceIndices>;
        using TargetElement = misc::convert_type_seq_to_t<ddc::DiscreteElement, TargetIndices>;
        return local_stencil_coefficient<Source, Target>(
                value(metric, position, element),
                ddc::detail::array(SourceElement(natural)),
                ddc::detail::array(TargetElement(natural)));
    }
};
} // namespace detail

/** Discrete Hodge star for an identity metric and coordinate-aligned orthogonal cells.
 * \important This operator and documentation is fully AI-generated.
 * Store one signed coefficient per complementary pair of canonical components.
 * Anisotropic cell sizes are allowed. Diagonality is checked in Debug builds.
 */
template <
        CellComplex Complex,
        class Indices1,
        class Indices2,
        class MetricType,
        class PositionType,
        class BatchElem>
class WeightedLeviCivitaDiscreteHodgeStar
{
    using Source = tensor::lower_t<reduction_index_t<Indices1>>;
    using Target = reduction_index_t<Indices2>;
    using Index = detail::WeightedHodgeIndex<Indices1, Indices2>;
    using MetricIndex
            = ddc::type_seq_element_t<0, ddc::to_type_seq_t<typename MetricType::indices_domain_t>>;
    static_assert(misc::Specialization<MetricIndex, tensor::TensorIdentityIndex>);
    static_assert(tensor::metric_index_1<MetricIndex>::size() == Index::rank());

    KOKKOS_FUNCTION static std::array<double, Index::mem_size()> weights(
            MetricType const& metric,
            PositionType const& position,
            BatchElem element)
    {
        auto const reconstruction = Reconstruction<
                tensor::lower_t<Indices1>,
                PositionType,
                BatchElem,
                CellComplex::Primal,
                tensor::TensorDiagonalIndex>::value(position, element);
        auto const reduction = Reduction<
                Indices2,
                PositionType,
                BatchElem,
                Complex,
                tensor::TensorDiagonalIndex>::value(metric, position, element);
        std::array<double, Index::mem_size()> result {};
        for (std::size_t entry = 0; entry < result.size(); ++entry) {
            std::array const components
                    = detail::weighted_hodge_components<Indices1, Indices2>(entry);
            result[entry]
                    = reconstruction.weights[components[0] * Source::mem_size() + components[0]]
                      * reduction.weights[components[1] * Target::mem_size() + components[1]]
                      * misc::permutation_parity(Index::mem_id_to_canonical_natural_ids(entry));
        }
        return result;
    }

public:
    using staged_storage_type = detail::WeightedHodgeStorage<
            WeightedLeviCivitaDiscreteHodgeStar,
            Indices1,
            Indices2,
            MetricType>;

    template <class Output, class Input>
    KOKKOS_FUNCTION static void operator()(
            Output const& output,
            Input const& input,
            MetricType const& metric,
            PositionType const& position,
            BatchElem element)
    {
        detail::apply_weighted_hodge<
                Indices1,
                Indices2>(output, input, weights(metric, position, element));
    }
    KOKKOS_FUNCTION static detail::LocalOperatorStencil<Source, Target> value(
            MetricType const& metric,
            PositionType const& position,
            BatchElem element)
    {
        detail::LocalOperatorStencil<Source, Target> result;
        std::array const coefficients = weights(metric, position, element);
        for (std::size_t entry = 0; entry < coefficients.size(); ++entry) {
            std::array const components
                    = detail::weighted_hodge_components<Indices1, Indices2>(entry);
            result.weights[components[1] * Source::mem_size() + components[0]]
                    = coefficients[entry];
        }
        return result;
    }
    template <class Natural>
    KOKKOS_FUNCTION static double value(
            MetricType const& metric,
            PositionType const& position,
            BatchElem element,
            Natural natural)
    {
        using SourceElement = misc::convert_type_seq_to_t<ddc::DiscreteElement, Indices1>;
        using TargetElement = misc::convert_type_seq_to_t<ddc::DiscreteElement, Indices2>;
        return detail::local_stencil_coefficient<Source, Target>(
                value(metric, position, element),
                ddc::detail::array(SourceElement(natural)),
                ddc::detail::array(TargetElement(natural)));
    }
};

template <
        CellComplex Complex,
        class Indices1,
        class Indices2,
        class Metric,
        class Position,
        class Element>
using StagedWeightedLeviCivitaDiscreteHodgeStar
        = StagedOperator<WeightedLeviCivitaDiscreteHodgeStar<
                Complex,
                Indices1,
                Indices2,
                Metric,
                Position,
                Element>>;

} // namespace sil::exterior
