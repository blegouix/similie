// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#include <cmath>
#include <numbers>

#include <gtest/gtest.h>
#include <similie/tensor/periodic_extrapolation_rule.hpp>

#include "local_staging_fixture.hpp"

namespace {
using namespace local_staging_test;

template <class InputIndex, class OutputIndex>
void constant_contraction()
{
    Domain const
            mesh(ddc::DiscreteElement<GridX, GridY, GridZ>(0, 0, 0),
                 ddc::DiscreteVector<GridX, GridY, GridZ>(4, 4, 4));
    std::array<double, 3> const spacing {0.3, 0.5, 0.7};
    Allocation<One> velocity_alloc = allocate<One>(mesh);
    Allocation<InputIndex> form_alloc = allocate<InputIndex>(mesh);
    Allocation<OutputIndex> output_alloc = allocate<OutputIndex>(mesh);
    Field<One> const velocity(velocity_alloc.span_view());
    Field<InputIndex> const form(form_alloc.span_view());
    Field<OutputIndex> const output(output_alloc.span_view());
    ddc::parallel_for_each(
            Exec(),
            velocity.domain(),
            KOKKOS_LAMBDA(FullDomain<One>::discrete_element_type element) {
                int const axis = element.template uid<One>();
                velocity.mem(element) = spacing[axis] * (axis + 1);
            });
    ddc::parallel_for_each(
            Exec(),
            form.domain(),
            KOKKOS_LAMBDA(typename FullDomain<InputIndex>::discrete_element_type element) {
                std::array<std::size_t, InputIndex::rank()> const axes
                        = sil::exterior::detail::flat_natural_elem_ids(
                                form.accessor().canonical_natural_element(
                                        ddc::DiscreteElement<InputIndex>(element)));
                double volume = 1;
                for (std::size_t axis : axes)
                    volume *= spacing[axis];
                form.mem(element) = volume;
            });
    sil::exterior::InteriorProduct<One, InputIndex> {}(
            Exec(),
            output,
            velocity,
            form,
            spacing,
            sil::tensor::PeriodicExtrapolationRule {});
    double error = 0;
    error = ddc::parallel_transform_reduce(
            Exec(),
            output.domain(),
            0.0,
            ddc::reducer::sum<double>(),
            KOKKOS_LAMBDA(typename FullDomain<OutputIndex>::discrete_element_type element) {
                auto const natural_element = output.accessor().canonical_natural_element(
                        ddc::DiscreteElement<OutputIndex>(element));
                std::array<std::size_t, OutputIndex::rank()> axes {};
                if constexpr (OutputIndex::rank() > 0)
                    axes = sil::exterior::detail::flat_natural_elem_ids(natural_element);
                double volume = 1;
                for (std::size_t axis : axes)
                    volume *= spacing[axis];
                double expected = 0;
                for (std::size_t b = 0; b < 3; ++b) {
                    bool repeated = false;
                    int sign = 1;
                    for (std::size_t axis : axes) {
                        repeated = repeated || axis == b;
                        if (axis < b)
                            sign *= -1;
                    }
                    if (!repeated)
                        expected += sign * double(b + 1) * volume;
                }
                return std::pow(output.mem(element) - expected, 2);
            });
    EXPECT_LT(error, 1e-25);
}

void skew_pairing()
{
    Domain const
            mesh(ddc::DiscreteElement<GridX, GridY, GridZ>(0, 0, 0),
                 ddc::DiscreteVector<GridX, GridY, GridZ>(8, 8, 8));
    std::array<double, 3> const spacing {0.3, 0.5, 0.7};
    Allocation<One> velocity_alloc = allocate<One>(mesh), output_alloc = allocate<One>(mesh);
    Allocation<Two> form_alloc = allocate<Two>(mesh);
    Field<One> const velocity(velocity_alloc.span_view()), output(output_alloc.span_view());
    Field<Two> const form(form_alloc.span_view());
    fill_form(velocity);
    fill_form(form, 1.2);
    sil::exterior::InteriorProduct<One, Two> {}(
            Exec(),
            output,
            velocity,
            form,
            spacing,
            sil::tensor::PeriodicExtrapolationRule {});
    double pairing = 0;
    pairing = ddc::parallel_transform_reduce(
            Exec(),
            output.domain(),
            0.0,
            ddc::reducer::sum<double>(),
            KOKKOS_LAMBDA(FullDomain<One>::discrete_element_type element) {
                double const h = spacing[element.template uid<One>()];
                return velocity.mem(element) * output.mem(element) / (h * h);
            });
    EXPECT_NEAR(pairing, 0, 1e-10);
}

using ShearTwo = sil::tensor::TensorAntisymmetricIndex<sil::tensor::Covariant<Other>, One>;

double shear_error(int n, double stream)
{
    Domain const
            mesh(ddc::DiscreteElement<GridX, GridY, GridZ>(0, 0, 0),
                 ddc::DiscreteVector<GridX, GridY, GridZ>(n, 2, 2));
    double const h = 2 * std::numbers::pi / n;
    Allocation<One> velocity_alloc = allocate<One>(mesh), transport_alloc = allocate<One>(mesh);
    Allocation<ShearTwo> form_alloc = allocate<ShearTwo>(mesh),
                         derivative_alloc = allocate<ShearTwo>(mesh);
    Field<One> const velocity(velocity_alloc.span_view()), transport(transport_alloc.span_view());
    Field<ShearTwo> const form(form_alloc.span_view()), derivative(derivative_alloc.span_view());
    ddc::parallel_for_each(
            Exec(),
            velocity.domain(),
            KOKKOS_LAMBDA(FullDomain<One>::discrete_element_type element) {
                velocity.mem(element) = element.template uid<One>() == 0 ? stream * h : 0;
            });
    ddc::parallel_for_each(
            Exec(),
            form.domain(),
            KOKKOS_LAMBDA(FullDomain<ShearTwo>::discrete_element_type element) {
                form.mem(element) = element.template uid<ShearTwo>() == 0
                                            ? h * h * std::sin(h * element.template uid<GridX>())
                                            : 0;
            });
    sil::exterior::InteriorProduct<One, ShearTwo> {}(
            Exec(),
            transport,
            velocity,
            form,
            std::array<double, 3> {h, h, h},
            sil::tensor::PeriodicExtrapolationRule {});
    sil::exterior::deriv<
            sil::tensor::Covariant<Other>,
            One>(Exec(), derivative, transport, sil::tensor::PeriodicExtrapolationRule {});
    double error = 0;
    error = ddc::parallel_transform_reduce(
            Exec(),
            derivative.domain(),
            0.0,
            ddc::reducer::sum<double>(),
            KOKKOS_LAMBDA(FullDomain<ShearTwo>::discrete_element_type element) {
                double const expected
                        = element.template uid<ShearTwo>() == 0
                                  ? stream * std::cos(h * element.template uid<GridX>())
                                  : 0;
                return std::pow(derivative.mem(element) / (h * h) - expected, 2);
            });
    // No advective energy growth: centered advection is skew for either sign.
    double production = 0;
    production = ddc::parallel_transform_reduce(
            Exec(),
            derivative.domain(),
            0.0,
            ddc::reducer::sum<double>(),
            KOKKOS_LAMBDA(FullDomain<ShearTwo>::discrete_element_type element) {
                return form.mem(element) * derivative.mem(element);
            });
    EXPECT_NEAR(production, 0, 1e-12);
    return std::sqrt(error / mesh.size());
}

TEST(InteriorProduct, ConstantOneForm)
{
    constant_contraction<One, Scalar>();
}
TEST(InteriorProduct, ConstantTwoForm)
{
    constant_contraction<Two, One>();
}
TEST(InteriorProduct, ConstantThreeForm)
{
    constant_contraction<Three, Two>();
}
TEST(InteriorProduct, SkewPairingAtPeriodicSeam)
{
    skew_pairing();
}
TEST(InteriorProduct, AdvectedShearConvergesForBothDirections)
{
    for (double stream : {-1.0, 1.0}) {
        double const coarse = shear_error(16, stream), fine = shear_error(32, stream);
        EXPECT_GT(coarse, 0);
        EXPECT_LT(fine, 0.26 * coarse);
    }
}
} // namespace
