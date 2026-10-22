// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#pragma once

#include <memory>
#include <stdexcept>

#include <similie/exterior/continuous_hodge_star/levi_civita_continuous_hodge_star.hpp>
#include <similie/exterior/discrete_hodge_star/weighted_levi_civita_discrete_hodge_star.hpp>
#include <similie/exterior/exterior.hpp>

#include "configuration.hpp"
#include "grid_types.hpp"
#include "staged_codifferential_laplacian.hpp"

namespace taylor_green_vortex {
// Host dispatch selects a compiled operator type once, before solver iterations.
template <class Tensor>
class LaplacianOperator
{
public:
    virtual ~LaplacianOperator() = default;
    virtual void apply(Tensor const& output, Tensor const& input) = 0;
};

namespace detail {
struct Periodic
{
    template <class Tensor, class Element>
    KOKKOS_FUNCTION auto operator()(Tensor const& field, Element element) const
    {
        SpatialDomain const domain(field.non_indices_domain());
        SpatialDomain::discrete_element_type wrapped;
        for (std::size_t axis = 0; axis < 3; ++axis) {
            std::ptrdiff_t const front = ddc::detail::array(domain.front())[axis];
            std::ptrdiff_t const count = ddc::detail::array(domain.extents())[axis];
            std::ptrdiff_t const index = std::ptrdiff_t(ddc::detail::array(element)[axis]) - front;
            ddc::detail::array(wrapped)[axis] = front + (index % count + count) % count;
        }
        return sil::tensor::detail::pointwise_tensor<Tensor>(
                [&](auto component) { return field.mem(wrapped, component); });
    }
};

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
    // Weighted discrete Hodge modes are orthogonal to the reduction,
    // reconstruction, and continuous Hodge choices. Keep their benchmark
    // composition aligned with the optimized mode (31).
    static constexpr int coefficient_mode
            = Mode == 34 || Mode == 35 || Mode == 36 ? 31 / 2 : Mode / 2;
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
            Mode == 34 || Mode == 35 || Mode == 36 || Mode % 2 == 1,
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
            Mode == 34 || Mode == 36,
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
                            Mode == 32,
                            sil::exterior::StagedOperator<sil::exterior::DiscreteHodgeStar<
                                    Complex,
                                    Source,
                                    Target,
                                    Metric,
                                    Position,
                                    Element>>,
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
            Mode == 37,
            StagedCodifferentialLaplacian<Tensor, Periodic>,
            std::conditional_t<
                    Mode == 33,
                    StagedLaplacian,
                    std::conditional_t<
                            Mode == 36,
                            StagedConfiguredLaplacian,
                            ConfiguredLaplacian>>>;
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
                    Periodic {},
                    Periodic {});
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
            m_operation(output, input, Periodic {}, Periodic {});
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

} // namespace detail

namespace detail {
template <int Mode = 0, class Tensor>
std::unique_ptr<LaplacianOperator<Tensor>> dispatch(
        int selected,
        Tensor const& output,
        Tensor const& input,
        MetricTensor const& metric,
        PositionTensor const& position)
{
    if (selected == Mode)
        return detail::make_laplacian_mode<Mode>(output, input, metric, position);
    if constexpr (Mode < 37)
        return dispatch<Mode + 1>(selected, output, input, metric, position);
    throw std::invalid_argument("Unknown Taylor-Green Laplacian staging mode");
}
} // namespace detail

inline std::unique_ptr<LaplacianOperator<ScalarTensor>> make_scalar_laplacian(
        Configuration const& selected,
        ScalarTensor const& output,
        ScalarTensor const& input,
        MetricTensor const& metric,
        PositionTensor const& position)
{
    return detail::dispatch(selected.laplacian_mode(), output, input, metric, position);
}
inline std::unique_ptr<LaplacianOperator<TwoFormTensor>> make_two_form_laplacian(
        Configuration const& selected,
        TwoFormTensor const& output,
        TwoFormTensor const& input,
        MetricTensor const& metric,
        PositionTensor const& position)
{
    return detail::dispatch(selected.laplacian_mode(), output, input, metric, position);
}
} // namespace taylor_green_vortex
