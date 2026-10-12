// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#pragma once

#include <cmath>
#include <numbers>
#include <stdexcept>

#include <ddc/ddc.hpp>

#include <similie/exterior/coboundary.hpp>
#include <similie/tensor/scalar_extrapolation_rules.hpp>

namespace sil::physics::fluid {
struct X
{
};
struct Y
{
};
struct Z
{
};
struct GridX
{
    using continuous_dimension_type = X;
};
struct GridY
{
    using continuous_dimension_type = Y;
};
struct GridZ
{
    using continuous_dimension_type = Z;
};
using SpatialDomain = ddc::DiscreteDomain<GridX, GridY, GridZ>;
using Scalar = tensor::Covariant<tensor::ScalarIndex>;
using Field = Kokkos::View<double**, Kokkos::LayoutRight>;

/** Periodic Cartesian cochains, with one component slab per scalar cochain.
 * On a uniform Euclidean grid the Hodge weights reduce to powers of h.
 * d/h is the forward gradient; -d^T/h is the backward divergence.
 * Their average is the centered derivative and -d^T d/h^2 the Laplacian.
 * No interpolation basis or high-order reconstruction is used.
 * \important This operator and documentation is fully AI-generated.
 */
class CartesianDEC
{
    SpatialDomain m_domain;
    Field m_forward;
    Field m_backward;

    struct Periodic
    {
        template <class Tensor, class Element>
        KOKKOS_FUNCTION auto operator()(Tensor const& field, Element element) const
        {
            SpatialDomain const domain(field.non_indices_domain());
            std::ptrdiff_t const n = domain.extents().template get<GridX>();
            ddc::DiscreteElement<GridX, GridY, GridZ> const
                    wrapped((static_cast<std::ptrdiff_t>(element.template uid<GridX>()) % n + n)
                                    % n,
                            (static_cast<std::ptrdiff_t>(element.template uid<GridY>()) % n + n)
                                    % n,
                            (static_cast<std::ptrdiff_t>(element.template uid<GridZ>()) % n + n)
                                    % n);
            return tensor::detail::pointwise_tensor<Tensor>(
                    [&](auto component) { return field.mem(wrapped, component); });
        }
    };

    template <class Axis>
    void incidence(Field input, int component, Field output, int out_component, bool transpose)
    {
        using Direction = tensor::Covariant<tensor::TensorNaturalIndex<Axis>>;
        [[maybe_unused]] tensor::TensorAccessor<Scalar> scalar_accessor;
        [[maybe_unused]] tensor::TensorAccessor<Direction> direction_accessor;
        ddc::DiscreteDomain<GridX, GridY, GridZ, Scalar> const
                in_domain(m_domain, scalar_accessor.domain());
        ddc::DiscreteDomain<GridX, GridY, GridZ, Direction> const
                out_domain(m_domain, direction_accessor.domain());
        tensor::Tensor in(
                ddc::ChunkSpan<
                        double,
                        ddc::DiscreteDomain<GridX, GridY, GridZ, Scalar>,
                        Kokkos::layout_right,
                        Field::memory_space>(input.data() + component * cells, in_domain));
        tensor::Tensor out(
                ddc::ChunkSpan<
                        double,
                        ddc::DiscreteDomain<GridX, GridY, GridZ, Direction>,
                        Kokkos::layout_right,
                        Field::memory_space>(output.data() + out_component * cells, out_domain));
        if (transpose)
            exterior::transposed_coboundary<
                    Direction,
                    Scalar>(Kokkos::DefaultExecutionSpace(), out, in, Periodic {});
        else
            exterior::
                    deriv<Direction, Scalar>(Kokkos::DefaultExecutionSpace(), out, in, Periodic {});
    }

public:
    int const n;
    int const cells;
    double const h;
    explicit CartesianDEC(int nodes)
        : m_domain(
                  ddc::DiscreteElement<GridX, GridY, GridZ>(0, 0, 0),
                  ddc::DiscreteVector<GridX, GridY, GridZ>(nodes, nodes, nodes))
        , m_forward("dec_forward", 1, nodes * nodes * nodes)
        , m_backward("dec_backward", 1, nodes * nodes * nodes)
        , n(nodes)
        , cells(nodes * nodes * nodes)
        , h(2.0 * std::numbers::pi / nodes)
    {
    }

    KOKKOS_FUNCTION static int neighbor(int index, int axis, int offset, int nodes)
    {
        int const stride = axis == 0 ? nodes * nodes : (axis == 1 ? nodes : 1);
        int const coordinate = (index / stride) % nodes;
        return index + (((coordinate + offset + nodes) % nodes) - coordinate) * stride;
    }
    void difference(
            Field input,
            int component,
            Field output,
            int out_component,
            int axis,
            bool transpose = false)
    {
        if (axis == 0)
            incidence<X>(input, component, output, out_component, transpose);
        else if (axis == 1)
            incidence<Y>(input, component, output, out_component, transpose);
        else
            incidence<Z>(input, component, output, out_component, transpose);
    }
    void derivative(Field input, int component, Field output, int out_component, int axis)
    {
        difference(input, component, m_forward, 0, axis);
        difference(input, component, m_backward, 0, axis, true);
        Field const forward = m_forward, backward = m_backward;
        double const scale = 0.5 / h;
        Kokkos::parallel_for(
                "dec_centered_derivative",
                cells,
                KOKKOS_LAMBDA(int i) {
                    output(out_component, i) = scale * (forward(0, i) - backward(0, i));
                });
    }
    void laplacian(Field input, int component, Field output, int out_component)
    {
        double const scale = 1.0 / (h * h);
        for (int axis = 0; axis < 3; ++axis) {
            difference(input, component, m_forward, 0, axis);
            difference(m_forward, 0, m_backward, 0, axis, true);
            Field const backward = m_backward;
            Kokkos::parallel_for(
                    "dec_hodge_laplacian",
                    cells,
                    KOKKOS_LAMBDA(int i) {
                        if (axis == 0)
                            output(out_component, i) = -scale * backward(0, i);
                        else
                            output(out_component, i) -= scale * backward(0, i);
                    });
        }
    }
};
} // namespace sil::physics::fluid
