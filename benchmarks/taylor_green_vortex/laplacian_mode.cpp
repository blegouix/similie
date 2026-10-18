// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#include <similie/exterior/discrete_hodge_star/staged_discrete_hodge_star.hpp>
#include <similie/exterior/exterior.hpp>
#include <similie/exterior/laplacian/staged_laplacian.hpp>

#include "laplacian_modes.hpp"

namespace taylor_green_vortex::detail {
namespace {
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
    template <class Indices, class Position, class Element, sil::exterior::CellComplex Complex>
    using Reconstruction = std::conditional_t<
            (Mode / 3) % 3 == 0,
            sil::exterior::Reconstruction<Indices, Position, Element, Complex>,
            sil::exterior::detail::CachedReconstruction<
                    Indices,
                    Position,
                    Element,
                    Complex,
                    CoefficientStructure<(Mode / 3) % 3 == 2>::template Index>>;

    template <class Indices, class Position, class Element, sil::exterior::CellComplex Complex>
    using Reduction = std::conditional_t<
            Mode % 3 == 0,
            sil::exterior::Reduction<Indices, Position, Element, Complex>,
            sil::exterior::detail::CachedReduction<
                    Indices,
                    Position,
                    Element,
                    Complex,
                    CoefficientStructure<Mode % 3 == 2>::template Index>>;

    template <
            sil::exterior::CellComplex Complex,
            class Source,
            class Target,
            class Metric,
            class Position,
            class Element>
    using Discrete = std::conditional_t<
            Mode == 9,
            sil::exterior::StagedHodgeStar<Complex, Source, Target, Metric, Position, Element>,
            sil::exterior::ComposedDiscreteHodgeStar<
                    Complex,
                    Source,
                    Target,
                    Metric,
                    Position,
                    Element,
                    Reconstruction,
                    Reduction,
                    sil::exterior::LeviCivitaContinuousHodgeStar>>;
};

template <int Mode, class Tensor>
class SelectedLaplacian final : public LaplacianOperator<Tensor>
{
    using Form = ddc::type_seq_element_t<0, ddc::to_type_seq_t<typename Tensor::indices_domain_t>>;
    using DirectionIndex
            = std::conditional_t<Form::rank() == 0, Vector, sil::tensor::Covariant<CurlDirection>>;
    using Operation = std::conditional_t<
            Mode == 10,
            sil::exterior::StagedLaplacian<
                    MetricIndex,
                    DirectionIndex,
                    Form,
                    Tensor,
                    MetricTensor,
                    PositionTensor,
                    Kokkos::DefaultExecutionSpace>,
            sil::exterior::Laplacian<
                    MetricIndex,
                    DirectionIndex,
                    Form,
                    Tensor,
                    MetricTensor,
                    PositionTensor,
                    Kokkos::DefaultExecutionSpace,
                    HodgePolicies<Mode>::template Discrete>>;
    Operation m_operation;

    static Operation make_operation(
            Tensor const& output,
            Tensor const& input,
            MetricTensor const& metric,
            PositionTensor const& position)
    {
        if constexpr (Mode == 10)
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
        if constexpr (Mode == 10)
            m_operation(output, input);
        else
            m_operation(output, input, Periodic {}, Periodic {});
    }
};
} // namespace

template <int Mode, class Tensor>
std::unique_ptr<LaplacianOperator<Tensor>> make_laplacian_mode(
        Tensor const& output,
        Tensor const& input,
        MetricTensor const& metric,
        PositionTensor const& position)
{
    return std::make_unique<SelectedLaplacian<Mode, Tensor>>(output, input, metric, position);
}

template std::unique_ptr<LaplacianOperator<ScalarTensor>> make_laplacian_mode<TGV_LAPLACIAN_MODE>(
        ScalarTensor const&,
        ScalarTensor const&,
        MetricTensor const&,
        PositionTensor const&);
template std::unique_ptr<LaplacianOperator<TwoFormTensor>> make_laplacian_mode<TGV_LAPLACIAN_MODE>(
        TwoFormTensor const&,
        TwoFormTensor const&,
        MetricTensor const&,
        PositionTensor const&);
} // namespace taylor_green_vortex::detail
