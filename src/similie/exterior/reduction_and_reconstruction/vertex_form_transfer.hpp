// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#pragma once

#include <array>
#include <cassert>
#include <cstddef>

#include <similie/exterior/quadrature/quadrature.hpp>
#include <similie/exterior/reduction_and_reconstruction/reduction_and_reconstruction.hpp>
#include <similie/tensor/scalar_extrapolation_rules.hpp>

namespace sil::exterior {
namespace detail {

/** A single cell's geometry, retaining both endpoints of each logical edge. */
template <class Position, class Element>
struct FormTransferCell;

template <class Position, class... Grid>
struct FormTransferCell<Position, ddc::DiscreteElement<Grid...>>
{
    using Element = ddc::DiscreteElement<Grid...>;
    using Domain = ddc::DiscreteDomain<Grid...>;
    using PositionIndex
            = ddc::type_seq_element_t<0, ddc::to_type_seq_t<typename Position::indices_domain_t>>;
    static constexpr std::size_t dimension = Domain::rank();
    static constexpr std::size_t vertices = std::size_t(1) << dimension;
    using GeometryDomain = ddc::DiscreteDomain<Grid..., PositionIndex>;
    using Geometry = tensor::
            Tensor<double, GeometryDomain, Kokkos::layout_right, typename Position::memory_space>;

    std::array<double, vertices * PositionIndex::size()> storage {};
    Geometry geometry;

    KOKKOS_FUNCTION static Element corner(std::size_t vertex)
    {
        Element result;
        for (std::size_t axis = 0; axis < dimension; ++axis)
            ddc::detail::array(result)[axis] = (vertex >> axis) & 1;
        return result;
    }

    KOKKOS_FUNCTION FormTransferCell(Position const& position, Element const& base)
        : geometry(
                  ddc::ChunkSpan<
                          double,
                          GeometryDomain,
                          Kokkos::layout_right,
                          typename Position::memory_space>(
                          storage.data(),
                          GeometryDomain(
                                  Domain(Element(),
                                         ddc::DiscreteVector<Grid...>(
                                                 (static_cast<void>(sizeof(Grid)), 2)...)),
                                  tensor::TensorAccessor<PositionIndex>::domain())))
    {
        static_assert(PositionIndex::size() == dimension);
        for (std::size_t vertex = 0; vertex < vertices; ++vertex) {
            Element node = base;
            for (std::size_t axis = 0; axis < dimension; ++axis)
                ddc::detail::array(node)[axis] += (vertex >> axis) & 1;
            assert(misc::domain_contains(position.non_indices_domain(), node));
            for (std::size_t component = 0; component < dimension; ++component)
                geometry.mem(corner(vertex), ddc::DiscreteElement<PositionIndex>(component))
                        = position.get(position.access_element(
                                node,
                                ddc::DiscreteElement<PositionIndex>(component)));
        }
    }
};

} // namespace detail

/** Project a nodally sampled differential form onto primal cochains.
 * Each k-cell uses tensor-product vertex quadrature of the geometric pullback.
 * Reduction evaluates that pullback in the cell's local geometry at each
 * quadrature vertex, so physical and logical axes may be oblique and cell
 * sizes may vary. The geometry is the multilinear map of the cell vertices.
 * This rule is exact for constant forms on affine cells and second-order for
 * smooth data on a smoothly mapped, refined grid. It is separate from the
 * cell-local Reduction API, whose input is piecewise constant.
 * Position must cover the full forward cell. Input/output must not alias;
 * the extrapolation rule applies to field samples, never to geometry.
 * \important This operator and documentation is fully AI-generated.
 */
template <misc::Specialization<ddc::TypeSeq> Indices>
struct VertexFormProjection
{
    template <
            class Output,
            class Input,
            class Position,
            class Element,
            class Extrapolation = tensor::ClampExtrapolationRule>
    KOKKOS_FUNCTION static void operator()(
            Output const& output,
            Input const& input,
            Position const& position,
            Element const& element,
            Extrapolation const& extrapolation = {})
    {
        using Index = reduction_index_t<Indices>;
        if constexpr (Index::rank() == 0) {
            auto const sample = extrapolation(input, element);
            output.mem(output.domain().front()) = sample.get(sample.domain().front());
        } else {
            using Cell = detail::FormTransferCell<Position, Element>;
            using LocalTensor = tensor::Tensor<
                    double,
                    ddc::DiscreteDomain<Index>,
                    Kokkos::layout_right,
                    typename Position::memory_space>;
            Cell cell(position, element);
            std::array<double, Index::mem_size()> storage {};
            LocalTensor const reduced(
                    ddc::ChunkSpan<
                            double,
                            ddc::DiscreteDomain<Index>,
                            Kokkos::layout_right,
                            typename Position::memory_space>(
                            storage.data(),
                            tensor::TensorAccessor<Index>::domain()));
            ddc::device_for_each(output.domain(), [&](auto component) {
                auto const natural = output.accessor().canonical_natural_element(component);
                std::array const axes = detail::flat_natural_elem_ids(natural);
                double value = 0;
                VertexQuadrature<Index::rank()> const rule;
                for (std::size_t sample = 0; sample < rule.size; ++sample) {
                    std::size_t vertex = 0;
                    Element node = element;
                    for (std::size_t axis = 0; axis < axes.size(); ++axis) {
                        std::size_t const offset = (sample >> axis) & 1;
                        vertex |= offset << axes[axis];
                        ddc::detail::array(node)[axes[axis]] += offset;
                    }
                    auto const coefficients = extrapolation(input, node);
                    Reduction<Indices, typename Cell::Geometry, Element>::
                    operator()(reduced, coefficients, cell.geometry, Cell::corner(vertex));
                    value += rule.weight(sample) * reduced.get(reduced.access_element(natural));
                }
                output.mem(component) = value;
            });
        }
    }
};

/** Reconstruct a primal cochain as a differential form at a mesh vertex.
 * In each incident cell, the cochain on every face through the vertex supplies
 * the corresponding reference-form component. Reconstruction maps that whole
 * form to physical coordinates using that cell's corner geometry. The mean
 * of these traces gives a common physical observation point for all components.
 * This is a vertex observation operator, not the inverse of projection and
 * not a replacement for the local Reconstruction in a DEC/Hodge operator.
 * Geometry must include all incident cells (a one-cell halo for periodic grids).
 * Field extrapolation is independent of geometry; input/output must not alias.
 * \important This operator and documentation is fully AI-generated.
 */
template <misc::Specialization<ddc::TypeSeq> Indices>
struct VertexFormReconstruction
{
    template <
            class Output,
            class Input,
            class Position,
            class Element,
            class Extrapolation = tensor::ClampExtrapolationRule>
    KOKKOS_FUNCTION static void operator()(
            Output const& output,
            Input const& input,
            Position const& position,
            Element const& element,
            Extrapolation const& extrapolation = {})
    {
        using Index = reduction_index_t<Indices>;
        if constexpr (Index::rank() == 0) {
            auto const sample = extrapolation(input, element);
            output.mem(output.domain().front()) = sample.get(sample.domain().front());
        } else {
            using Cell = detail::FormTransferCell<Position, Element>;
            using LocalTensor = tensor::Tensor<
                    double,
                    ddc::DiscreteDomain<Index>,
                    Kokkos::layout_right,
                    typename Position::memory_space>;
            std::array<double, Index::mem_size()> cochain_storage {}, coefficient_storage {};
            LocalTensor const cochain(
                    ddc::ChunkSpan<
                            double,
                            ddc::DiscreteDomain<Index>,
                            Kokkos::layout_right,
                            typename Position::memory_space>(
                            cochain_storage.data(),
                            tensor::TensorAccessor<Index>::domain()));
            LocalTensor const coefficients(
                    ddc::ChunkSpan<
                            double,
                            ddc::DiscreteDomain<Index>,
                            Kokkos::layout_right,
                            typename Position::memory_space>(
                            coefficient_storage.data(),
                            tensor::TensorAccessor<Index>::domain()));
            ddc::device_for_each(output.domain(), [&](auto component) {
                output.mem(component) = 0;
            });
            for (std::size_t vertex = 0; vertex < Cell::vertices; ++vertex) {
                Element base = element;
                for (std::size_t axis = 0; axis < Cell::dimension; ++axis)
                    ddc::detail::array(base)[axis] -= (vertex >> axis) & 1;
                Cell cell(position, base);
                ddc::device_for_each(cochain.domain(), [&](auto component) {
                    auto const natural = cochain.accessor().canonical_natural_element(component);
                    std::array const axes = detail::flat_natural_elem_ids(natural);
                    Element node = element;
                    for (std::size_t axis : axes)
                        ddc::detail::array(node)[axis] -= (vertex >> axis) & 1;
                    auto const sample = extrapolation(input, node);
                    cochain.mem(component) = sample.get(sample.access_element(natural));
                });
                Reconstruction<Indices, typename Cell::Geometry, Element>::
                operator()(coefficients, cochain, cell.geometry, Cell::corner(vertex));
                ddc::device_for_each(output.domain(), [&](auto component) {
                    auto const natural = output.accessor().canonical_natural_element(component);
                    output.mem(component) += coefficients.get(coefficients.access_element(natural))
                                             / Cell::vertices;
                });
            }
        }
    }
};

} // namespace sil::exterior
