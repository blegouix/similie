// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#pragma once

#include <stdexcept>

#include <similie/exterior/covariant_derivative/covariant_derivative.hpp>
#include <similie/exterior/quadrature/quadrature.hpp>
#include <similie/exterior/reduction_and_reconstruction/reduction_and_reconstruction.hpp>
#include <similie/exterior/volume.hpp>

namespace sil::exterior {

/** Reconstruct derivatives of nodal, flat-bundle-valued zero-cochains at the
 * vertices of one cubical cell. FibreRank is independent of spatial dimension.
 * \important This operator and documentation is fully AI-generated.
 * The canonical covariant coboundary supplies incidence; Reconstruction maps
 * each one-cochain into the physical frame. Coefficients are cached for every
 * nodal basis function. This evaluator supports vertex sampling only; it makes
 * no assumption about the quadrature rule used by a variational assembler.
 * measures contains the positive Euclidean volume Jacobian at each vertex,
 * without quadrature weights. Components are packed fibre first, then axis.
 */
template <std::size_t FibreRank, class... SpatialIndex>
class VertexReconstructedDerivative
{
    template <class Dimension>
    struct Grid
    {
        using continuous_dimension_type = Dimension;
    };
    static constexpr std::size_t dimension = sizeof...(SpatialIndex);
    static constexpr std::size_t vertex_count = std::size_t(1) << dimension;
    static constexpr std::size_t basis_count = vertex_count * FibreRank;
    using Element = ddc::DiscreteElement<Grid<SpatialIndex>...>;

    [[nodiscard]] KOKKOS_FUNCTION static Element element(std::size_t vertex)
    {
        Element result;
        for (std::size_t axis = 0; axis < dimension; ++axis)
            ddc::detail::array(result)[axis] = (vertex >> axis) & 1;
        return result;
    }

    struct NodalBasis
    {
        std::size_t input;

        [[nodiscard]] KOKKOS_FUNCTION std::array<double, FibreRank> operator()(
                std::size_t,
                std::size_t node) const
        {
            std::array<double, FibreRank> value {};
            if (node == input / FibreRank)
                value[input % FibreRank] = 1.0;
            return value;
        }
    };

    template <class MemorySpace>
    KOKKOS_FUNCTION void initialize(
            std::array<std::array<double, dimension>, vertex_count> const& positions)
    {
        using PositionIndex = tensor::Contravariant<tensor::TensorNaturalIndex<SpatialIndex...>>;
        using GradientIndex = tensor::Covariant<tensor::TensorNaturalIndex<SpatialIndex...>>;
        using MetricIndex = tensor::TensorIdentityIndex<
                tensor::Covariant<tensor::MetricIndex1<SpatialIndex...>>,
                tensor::Covariant<tensor::MetricIndex2<SpatialIndex...>>>;
        using Domain = ddc::DiscreteDomain<Grid<SpatialIndex>...>;
        Domain const
                nodes(Element(),
                      ddc::DiscreteVector<Grid<SpatialIndex>...>(
                              (static_cast<void>(sizeof(SpatialIndex)), 2)...));
        std::array<double, vertex_count * dimension> position_storage {};
        tensor::Tensor position(
                ddc::ChunkSpan<
                        double,
                        ddc::DiscreteDomain<Grid<SpatialIndex>..., PositionIndex>,
                        Kokkos::layout_right,
                        MemorySpace>(
                        position_storage.data(),
                        ddc::DiscreteDomain<Grid<SpatialIndex>..., PositionIndex>(
                                nodes,
                                tensor::TensorAccessor<PositionIndex>::domain())));
        for (std::size_t vertex = 0; vertex < vertex_count; ++vertex)
            for (std::size_t axis = 0; axis < dimension; ++axis)
                position.mem(element(vertex), ddc::DiscreteElement<PositionIndex>(axis))
                        = positions[vertex][axis];
        tensor::Tensor metric(
                ddc::ChunkSpan<
                        double,
                        ddc::DiscreteDomain<MetricIndex>,
                        Kokkos::layout_right,
                        MemorySpace>(nullptr, tensor::TensorAccessor<MetricIndex>::domain()));
        for (std::size_t vertex = 0; vertex < vertex_count; ++vertex) {
            // Geometry and the reconstruction must use the same corner frame.
            std::array<double, dimension * dimension> jacobian {};
            std::array<std::size_t, dimension> directions {};
            for (std::size_t axis = 0; axis < dimension; ++axis) {
                directions[axis] = axis;
                for (std::size_t physical = 0; physical < dimension; ++physical)
                    jacobian[physical * dimension + axis]
                            = ((vertex >> axis) & 1 ? -1.0 : 1.0)
                              * (positions[vertex ^ (std::size_t(1) << axis)][physical]
                                 - positions[vertex][physical]);
            }
            if (!(detail::reduction_determinant<dimension>(jacobian) > 0.0)) {
                KOKKOS_IF_ON_HOST((
                        throw std::runtime_error(
                                "vertex reconstruction requires an unfolded, nondegenerate cell");))
                KOKKOS_IF_ON_DEVICE((
                        Kokkos::abort(
                                "vertex reconstruction requires an unfolded, nondegenerate cell");))
            }
            measures[vertex] = SimplexVolume<
                    CellComplex::Primal,
                    dimension,
                    decltype(metric),
                    decltype(position),
                    Element>::template
            operator()<dimension>(metric, position, element(vertex), directions);
            auto const reconstruction
                    = Reconstruction<ddc::TypeSeq<GradientIndex>, decltype(position), Element>::
                            value(position, element(vertex));
            for (std::size_t input = 0; input < basis_count; ++input) {
                std::array<std::array<double, FibreRank>, dimension> differential {};
                for (std::size_t axis = 0; axis < dimension; ++axis)
                    differential[axis] = CovariantDerivative<SpatialIndex...>::
                            template cochain_value<0, FibreRank>(
                                    {axis},
                                    vertex & ~(std::size_t(1) << axis),
                                    NodalBasis {input});
                for (std::size_t fibre = 0; fibre < FibreRank; ++fibre)
                    for (std::size_t physical = 0; physical < dimension; ++physical)
                        for (std::size_t axis = 0; axis < dimension; ++axis)
                            gradients[vertex][input][fibre * dimension + physical]
                                    += reconstruction.weights[physical * dimension + axis]
                                       * differential[axis][fibre];
            }
        }
    }

public:
    static_assert(dimension > 0 && FibreRank > 0);
    std::array<double, vertex_count> measures {};
    std::array<std::array<std::array<double, FibreRank * dimension>, basis_count>, vertex_count>
            gradients {};

    KOKKOS_FUNCTION explicit VertexReconstructedDerivative(
            std::array<std::array<double, dimension>, vertex_count> const& positions)
    {
        KOKKOS_IF_ON_HOST((initialize<Kokkos::HostSpace>(positions);))
        KOKKOS_IF_ON_DEVICE(
                (initialize<typename Kokkos::DefaultExecutionSpace::memory_space>(positions);))
    }

    [[nodiscard]] KOKKOS_FUNCTION std::array<double, FibreRank * dimension> gradient(
            std::array<std::array<double, FibreRank>, vertex_count> const& nodal) const
    {
        auto const measure = [&](auto const&, std::size_t sample) { return measures[sample]; };
        double const volume = quadrature(
                VertexQuadrature<dimension>(),
                [](auto const&, std::size_t) { return 1.0; },
                measure);
        std::array<double, FibreRank * dimension> result {};
        for (std::size_t component = 0; component < result.size(); ++component)
            result[component]
                    = quadrature(
                              VertexQuadrature<dimension>(),
                              [&](auto const&, std::size_t sample) {
                                  double value = 0.0;
                                  for (std::size_t input = 0; input < basis_count; ++input)
                                      value += gradients[sample][input][component]
                                               * nodal[input / FibreRank][input % FibreRank];
                                  return value;
                              },
                              measure)
                      / volume;
        return result;
    }
};

} // namespace sil::exterior
