// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED
#pragma once

#include <similie/exterior/interior_product/interior_product.hpp>
#include <similie/exterior/reduction_and_reconstruction/vertex_form_transfer.hpp>
#include <similie/exterior/staged_operator/staged_operator.hpp>
#include <similie/tensor/periodic_extrapolation_rule.hpp>

namespace sil::exterior {
namespace detail {
// Each source k-face contributes through its 2^k vertices. Retain those
// coefficients rather than expanding the product of the two spatial supports.
template <class Index, class Domain, class Memory>
struct VertexTransferFactors
{
    static constexpr std::size_t corners = std::size_t(1) << Index::rank();
    using Element = typename Domain::discrete_element_type;
    using Local = tensor::Tensor<double, ddc::DiscreteDomain<Index>, Kokkos::layout_right, Memory>;
    Kokkos::View<
            double * [Index::mem_size() * Index::mem_size() * corners],
            Kokkos::LayoutLeft,
            Memory>
            weights;

    KOKKOS_FUNCTION static Element sample(
            Element node,
            std::size_t component,
            std::size_t corner,
            int sign)
    {
        tensor::TensorAccessor<Index> accessor;
        auto const natural
                = accessor.canonical_natural_element(ddc::DiscreteElement<Index>(component));
        auto const axes = flat_natural_elem_ids(natural);
        for (std::size_t axis = 0; axis < axes.size(); ++axis)
            ddc::detail::array(node)[axes[axis]] += sign * int((corner >> axis) & 1);
        return node;
    }

    struct UnitSample
    {
        Element node;
        std::size_t component;
        template <class Field, class Point>
        KOKKOS_FUNCTION auto operator()(Field const&, Point point) const
        {
            return tensor::detail::pointwise_tensor<ddc::DiscreteDomain<Index>, Memory>(
                    [&](auto c) {
                        return point == node && c.template uid<Index>() == component ? 1.0 : 0.0;
                    });
        }
    };

    template <bool Projection, class Exec, class Field, class Position>
    static VertexTransferFactors build(
            Exec const& exec,
            Field const& field,
            Position const& position)
    {
        VertexTransferFactors result;
        Domain const domain = field.non_indices_domain();
        result.weights = decltype(weights)(
                Kokkos::view_alloc(Kokkos::WithoutInitializing, "staged_vertex_transfer_factors"),
                domain.size());
        auto const weights = result.weights;
        using Indices = ddc::to_type_seq_t<tensor::natural_domain_t<Index>>;
        Kokkos::parallel_for(
                "similie_cache_vertex_transfer_factors",
                Kokkos::RangePolicy<Exec>(exec, 0, domain.size()),
                KOKKOS_LAMBDA(std::size_t node) {
                    (void)weights;
                    (void)field;
                    (void)position;
                    Element const element = stencil_element(domain, node);
                    std::array<double, Index::mem_size()> buffer {};
                    Local local(
                            ddc::ChunkSpan<
                                    double,
                                    ddc::DiscreteDomain<Index>,
                                    Kokkos::layout_right,
                                    Memory>(
                                    buffer.data(),
                                    tensor::TensorAccessor<Index>::domain()));
                    for (std::size_t source = 0; source < Index::mem_size(); ++source)
                        for (std::size_t corner = 0; corner < corners; ++corner) {
                            // Projection support follows the output face; reconstruction
                            // support follows the source face. Probe the canonical operators.
                            if constexpr (!Projection) {
                                VertexFormReconstruction<Indices>::operator()(
                                        Form(local),
                                        field,
                                        position,
                                        element,
                                        UnitSample {sample(element, source, corner, -1), source});
                                for (std::size_t target = 0; target < Index::mem_size(); ++target)
                                    weights(node,
                                            (target * Index::mem_size() + source) * corners
                                                    + corner)
                                            = local.mem(ddc::DiscreteElement<Index>(target));
                            } else {
                                for (std::size_t target = 0; target < Index::mem_size(); ++target) {
                                    VertexFormProjection<Indices>::operator()(
                                            Cochain(local),
                                            field,
                                            position,
                                            element,
                                            UnitSample {
                                                    sample(element, target, corner, 1),
                                                    source});
                                    weights(node,
                                            (target * Index::mem_size() + source) * corners
                                                    + corner)
                                            = local.mem(ddc::DiscreteElement<Index>(target));
                                }
                            }
                        }
                });
        return result;
    }

    template <class Output, class Field>
    KOKKOS_FUNCTION void reconstruct(
            Output const& output,
            Field const& field,
            Domain const& domain,
            Element element) const
    {
        std::size_t const node = stencil_node(domain, element);
        for (std::size_t target = 0; target < Index::mem_size(); ++target) {
            double value = 0;
            for (std::size_t source = 0; source < Index::mem_size(); ++source)
                for (std::size_t corner = 0; corner < corners; ++corner) {
                    auto const point = tensor::PeriodicExtrapolationRule {}(
                            field,
                            sample(element, source, corner, -1));
                    value += weights(node, (target * Index::mem_size() + source) * corners + corner)
                             * point.mem(ddc::DiscreteElement<Index>(source));
                }
            output.mem(ddc::DiscreteElement<Index>(target)) = value;
        }
    }
};
} // namespace detail

/** Periodic vertex reconstruction, bilinear form operator, and projection.
 * Coefficients come from the canonical vertex transfers. Application keeps
 * their factorization and evaluates the supplied form operator locally.
 * Geometry and metric must remain unchanged; output must not alias an input.
 * \important This operator and documentation is fully AI-generated.
 */
template <
        class Operator,
        class First,
        class Second,
        class Target,
        class Domain,
        class Memory,
        class InverseMetric>
struct VertexFormBilinearComposition
{
    struct Storage
    {
        using execution_space = typename Memory::execution_space;
        Domain domain;
        InverseMetric inverse_metric;
        detail::VertexTransferFactors<First, Domain, Memory> first_factors;
        detail::VertexTransferFactors<Second, Domain, Memory> second_factors;
        detail::VertexTransferFactors<Target, Domain, Memory> projection;

        template <class Output, class A, class B, class Position>
        static Storage build(
                execution_space const& exec,
                Output const& output,
                A const& a,
                B const& b,
                Position const& position,
                InverseMetric const& inverse_metric)
        {
            if (a.non_indices_domain() != b.non_indices_domain())
                throw std::invalid_argument("Vertex composition requires equal input domains");
            return {a.non_indices_domain(),
                    inverse_metric,
                    detail::VertexTransferFactors<First, Domain, Memory>::template build<
                            false>(exec, a, position),
                    detail::VertexTransferFactors<Second, Domain, Memory>::template build<
                            false>(exec, b, position),
                    detail::VertexTransferFactors<Target, Domain, Memory>::template build<
                            true>(exec, Form(output.coefficients()), position)};
        }

        template <class Output, class A, class B>
        Output apply(execution_space const& exec, Output const& output, A const& a, B const& b)
                const
        {
            if (a.non_indices_domain() != domain || b.non_indices_domain() != domain
                || output.non_indices_domain() != domain)
                throw std::invalid_argument("Staged vertex composition domain changed");
            Storage const storage = *this;
            Kokkos::parallel_for(
                    "similie_interior_product",
                    Kokkos::RangePolicy<execution_space>(exec, 0, domain.size()),
                    KOKKOS_LAMBDA(std::size_t node) {
                        auto const element = detail::stencil_element(storage.domain, node);
                        std::array<double, First::mem_size()> av {};
                        std::array<double, Second::mem_size()> bv {};
                        std::array<double, Target::mem_size()> ov {};
                        typename detail::VertexTransferFactors<First, Domain, Memory>::Local first(
                                ddc::ChunkSpan<
                                        double,
                                        ddc::DiscreteDomain<First>,
                                        Kokkos::layout_right,
                                        Memory>(
                                        av.data(),
                                        tensor::TensorAccessor<First>::domain()));
                        typename detail::VertexTransferFactors<Second, Domain, Memory>::Local
                                second(ddc::ChunkSpan<
                                        double,
                                        ddc::DiscreteDomain<Second>,
                                        Kokkos::layout_right,
                                        Memory>(
                                        bv.data(),
                                        tensor::TensorAccessor<Second>::domain()));
                        typename detail::VertexTransferFactors<Target, Domain, Memory>::Local
                                transformed(
                                        ddc::ChunkSpan<
                                                double,
                                                ddc::DiscreteDomain<Target>,
                                                Kokkos::layout_right,
                                                Memory>(
                                                ov.data(),
                                                tensor::TensorAccessor<Target>::domain()));
                        for (std::size_t target = 0; target < Target::mem_size(); ++target) {
                            double value = 0;
                            for (std::size_t corner = 0;
                                 corner < decltype(storage.projection)::corners;
                                 ++corner) {
                                auto const sample = decltype(storage.projection)::
                                        sample(element, target, corner, 1);
                                // Apply the same periodic observation rule as the field adapter.
                                auto wrapped = sample;
                                for (std::size_t axis = 0; axis < Domain::rank(); ++axis) {
                                    std::ptrdiff_t const front
                                            = ddc::detail::array(storage.domain.front())[axis];
                                    std::ptrdiff_t const count
                                            = ddc::detail::array(storage.domain.extents())[axis];
                                    std::ptrdiff_t const relative
                                            = ddc::detail::array(sample)[axis] - front;
                                    ddc::detail::array(wrapped)[axis]
                                            = front + (relative % count + count) % count;
                                }
                                storage.first_factors
                                        .reconstruct(first, a, storage.domain, wrapped);
                                storage.second_factors
                                        .reconstruct(second, b, storage.domain, wrapped);
                                Operator {}(
                                        Form(transformed),
                                        Form(first),
                                        Form(second),
                                        storage.inverse_metric,
                                        wrapped);
                                for (std::size_t source = 0; source < Target::mem_size(); ++source)
                                    value += storage.projection.weights(
                                                     node,
                                                     (target * Target::mem_size() + source)
                                                                     * decltype(storage.projection)::
                                                                             corners
                                                             + corner)
                                             * transformed.mem(
                                                     ddc::DiscreteElement<Target>(source));
                            }
                            output.mem(element, ddc::DiscreteElement<Target>(target)) = value;
                        }
                    });
            return output;
        }
    };
    using staged_storage_type = Storage;
};
} // namespace sil::exterior
