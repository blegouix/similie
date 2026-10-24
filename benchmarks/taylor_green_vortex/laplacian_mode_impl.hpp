// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#pragma once

#include <memory>
#include <stdexcept>
#include <type_traits>

#include <similie/exterior/continuous_hodge_star/levi_civita_continuous_hodge_star.hpp>
#include <similie/exterior/discrete_hodge_star/weighted_levi_civita_discrete_hodge_star.hpp>
#include <similie/exterior/exterior.hpp>
#include <similie/tensor/periodic_extrapolation_rule.hpp>

#include "laplacian_mode.hpp"

namespace taylor_green_vortex::detail {

template <bool Diagonal>
struct CoefficientStructure
{
    template <class... Indices>
    using Index = std::conditional_t<
            Diagonal,
            sil::tensor::TensorDiagonalIndex<Indices...>,
            sil::tensor::TensorFullIndex<Indices...>>;
};

template <int Mode>
struct HodgePolicies
{
    // Specialized discrete Hodge modes use the same policies as mode 31.
    // Mode 32 retains the general policies for the plain staged operator.
    static constexpr int coefficient_mode
            = Mode >= 34 && Mode <= 38 ? 31 / 2 : (Mode == 32 ? 0 : Mode / 2);
    static constexpr int reduction_mode = coefficient_mode % 4;
    static constexpr int reconstruction_mode = coefficient_mode / 4;

    template <class Indices, class Position, class Element, sil::exterior::CellComplex Complex>
    using Reconstruction = std::conditional_t<
            reconstruction_mode == 0 || reconstruction_mode == 3,
            sil::exterior::Reconstruction<
                    Indices,
                    Position,
                    Element,
                    Complex,
                    CoefficientStructure<reconstruction_mode == 3>::template Index>,
            sil::exterior::StagedOperator<sil::exterior::Reconstruction<
                    Indices,
                    Position,
                    Element,
                    Complex,
                    CoefficientStructure<reconstruction_mode == 2>::template Index>>>;

    template <class Indices, class Position, class Element, sil::exterior::CellComplex Complex>
    using Reduction = std::conditional_t<
            reduction_mode == 0 || reduction_mode == 3,
            sil::exterior::Reduction<
                    Indices,
                    Position,
                    Element,
                    Complex,
                    CoefficientStructure<reduction_mode == 3>::template Index>,
            sil::exterior::StagedOperator<sil::exterior::Reduction<
                    Indices,
                    Position,
                    Element,
                    Complex,
                    CoefficientStructure<reduction_mode == 2>::template Index>>>;

    template <class Indices1, class Indices2, class Metric, class Element>
    using Continuous = std::conditional_t<
            Mode == 34 || Mode == 35 || Mode == 36 || Mode == 38 || Mode % 2 == 1,
            sil::exterior::LeviCivitaContinuousHodgeStar<Indices1, Indices2, Metric, Element>,
            sil::exterior::ContinuousHodgeStar<Indices1, Indices2, Metric, Element>>;

    template <
            sil::exterior::CellComplex Complex,
            class Source,
            class Target,
            class Metric,
            class Position,
            class Element>
    using Discrete = std::conditional_t<
            Mode == 34 || Mode == 36 || Mode == 37,
            sil::exterior::WeightedLeviCivitaDiscreteHodgeStar<
                    Complex,
                    Source,
                    Target,
                    Metric,
                    Position,
                    Element>,
            std::conditional_t<
                    Mode == 35,
                    sil::exterior::StagedOperator<
                            sil::exterior::WeightedLeviCivitaDiscreteHodgeStar<
                                    Complex,
                                    Source,
                                    Target,
                                    Metric,
                                    Position,
                                    Element>>,
                    std::conditional_t<
                            Mode == 32 || Mode == 38,
                            sil::exterior::StagedOperator<sil::exterior::DiscreteHodgeStar<
                                    Complex,
                                    Source,
                                    Target,
                                    Metric,
                                    Position,
                                    Element,
                                    Reconstruction,
                                    Reduction,
                                    Continuous>>,
                            sil::exterior::DiscreteHodgeStar<
                                    Complex,
                                    Source,
                                    Target,
                                    Metric,
                                    Position,
                                    Element,
                                    Reconstruction,
                                    Reduction,
                                    Continuous>>>>;
};

template <int Mode, class Tensor>
class SelectedLaplacian final : public LaplacianOperator<Tensor>
{
    using Form = ddc::type_seq_element_t<0, ddc::to_type_seq_t<typename Tensor::indices_domain_t>>;
    using DirectionIndex
            = std::conditional_t<Form::rank() == 0, Vector, sil::tensor::Covariant<CurlDirection>>;
    using StagedLaplacian = sil::exterior::StagedOperator<sil::exterior::Laplacian<
            MetricIndex,
            DirectionIndex,
            Form,
            Tensor,
            MetricTensor,
            PositionTensor,
            Kokkos::DefaultExecutionSpace>>;
    using ConfiguredLaplacian = sil::exterior::Laplacian<
            MetricIndex,
            DirectionIndex,
            Form,
            Tensor,
            MetricTensor,
            PositionTensor,
            Kokkos::DefaultExecutionSpace,
            HodgePolicies<Mode>::template Discrete>;
    using StagedConfiguredLaplacian = sil::exterior::StagedOperator<ConfiguredLaplacian>;
    using Operation = std::conditional_t<
            Mode == 33,
            StagedLaplacian,
            std::conditional_t<
                    Mode == 36 || Mode == 37,
                    StagedConfiguredLaplacian,
                    ConfiguredLaplacian>>;
    Operation m_operation;

    static Operation make_operation(
            Tensor const& output,
            Tensor const& input,
            MetricTensor const& metric,
            PositionTensor const& position)
    {
        if constexpr (Mode == 33 || Mode == 36 || Mode == 37)
            return Operation(
                    Kokkos::DefaultExecutionSpace(),
                    output,
                    input,
                    metric,
                    position,
                    sil::tensor::PeriodicExtrapolationRule {},
                    sil::tensor::PeriodicExtrapolationRule {});
        else
            return Operation(Kokkos::DefaultExecutionSpace(), output, input, metric, position);
    }

public:
    SelectedLaplacian(
            Tensor const& output,
            Tensor const& input,
            MetricTensor const& metric,
            PositionTensor const& position)
        : m_operation(make_operation(output, input, metric, position))
    {
    }
    void apply(Tensor const& output, Tensor const& input) override
    {
        if constexpr (Mode == 33 || Mode == 36 || Mode == 37)
            m_operation(output, input);
        else
            m_operation(
                    output,
                    input,
                    sil::tensor::PeriodicExtrapolationRule {},
                    sil::tensor::PeriodicExtrapolationRule {});
    }
};

template <int Mode, class Tensor>
std::unique_ptr<LaplacianOperator<Tensor>> make_laplacian_mode(
        Tensor const& output,
        Tensor const& input,
        MetricTensor const& metric,
        PositionTensor const& position)
{
    return std::make_unique<SelectedLaplacian<Mode, Tensor>>(output, input, metric, position);
}

template <int First, int Last, class Tensor>
std::unique_ptr<LaplacianOperator<Tensor>> make_laplacian_mode_in_range(
        int selected,
        Tensor const& output,
        Tensor const& input,
        MetricTensor const& metric,
        PositionTensor const& position)
{
    if (selected == First)
        return make_laplacian_mode<First>(output, input, metric, position);
    if constexpr (First < Last)
        return make_laplacian_mode_in_range<
                First + 1,
                Last>(selected, output, input, metric, position);
    throw std::invalid_argument("Unknown Taylor-Green Laplacian staging mode");
}

} // namespace taylor_green_vortex::detail
