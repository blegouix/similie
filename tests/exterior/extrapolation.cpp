// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#include <ddc/ddc.hpp>

#include <gtest/gtest.h>
#include <similie/exterior/exterior.hpp>
#include <similie/tensor/identity_tensor.hpp>

namespace {
struct X
{
};
struct Grid
{
    using continuous_dimension_type = X;
};
using Scalar = sil::tensor::Covariant<sil::tensor::ScalarIndex>;
using Vector = sil::tensor::Covariant<sil::tensor::TensorNaturalIndex<X>>;
using Position = sil::tensor::Contravariant<sil::tensor::TensorNaturalIndex<X>>;
using Metric = sil::tensor::TensorIdentityIndex<
        sil::tensor::Covariant<sil::tensor::MetricIndex1<X>>,
        sil::tensor::Covariant<sil::tensor::MetricIndex2<X>>>;

struct ShiftedTraceMap
{
    template <class TensorType>
    KOKKOS_FUNCTION double operator()(
            TensorType field,
            ddc::DiscreteElement<Grid> elem,
            ddc::DiscreteElement<Scalar> component) const
    {
        return field.mem(elem - ddc::DiscreteVector<Grid>(5), component);
    }
};

void test_callable_rules_and_connected_derivative()
{
    ddc::DiscreteDomain<Grid> const
            grid(ddc::DiscreteElement<Grid>(10), ddc::DiscreteVector<Grid>(5));
    [[maybe_unused]] sil::tensor::TensorAccessor<Scalar> scalar_accessor;
    [[maybe_unused]] sil::tensor::TensorAccessor<Vector> vector_accessor;
    ddc::DiscreteDomain<Grid, Scalar> const scalar_domain(grid, scalar_accessor.domain());
    ddc::DiscreteDomain<Grid, Vector> const vector_domain(grid, vector_accessor.domain());
    ddc::Chunk potential_alloc(scalar_domain, ddc::DeviceAllocator<double>());
    ddc::Chunk neighbor_alloc(scalar_domain, ddc::DeviceAllocator<double>());
    ddc::Chunk derivative_alloc(vector_domain, ddc::DeviceAllocator<double>());
    sil::tensor::Tensor potential(potential_alloc);
    sil::tensor::Tensor neighbor(neighbor_alloc);
    sil::tensor::Tensor derivative(derivative_alloc);
    ddc::parallel_for_each(
            Kokkos::DefaultExecutionSpace(),
            grid,
            KOKKOS_LAMBDA(ddc::DiscreteElement<Grid> elem) {
                double const x = static_cast<double>(elem.uid<Grid>() - 10);
                potential.mem(elem, ddc::DiscreteElement<Scalar>(0)) = x * x;
                neighbor.mem(elem, ddc::DiscreteElement<Scalar>(0)) = (x + 5) * (x + 5);
            });
    // A neighbor can have a different logical origin. The map owns the seam
    // convention, while the operator only invokes the sampling policy.
    sil::exterior::ConnectedScalarExtrapolationRule const
            connected {neighbor, ShiftedTraceMap {}, 2.0};
    sil::exterior::deriv<
            Vector,
            Scalar>(Kokkos::DefaultExecutionSpace(), derivative, potential, connected);
    auto host_alloc = ddc::create_mirror_view_and_copy(
            Kokkos::DefaultHostExecutionSpace(),
            derivative_alloc.span_view());
    sil::tensor::Tensor host(host_alloc);
    EXPECT_DOUBLE_EQ(host.mem(ddc::DiscreteElement<Grid, Vector>(14, 0)), 11.0);
    EXPECT_DOUBLE_EQ(host.mem(ddc::DiscreteElement<Grid, Vector>(12, 0)), 5.0);

    sil::exterior::deriv<Vector, Scalar>(
            Kokkos::DefaultExecutionSpace(),
            derivative,
            potential,
            sil::exterior::NaturalScalarExtrapolationRule {});
    ddc::parallel_deepcopy(host, derivative);
    EXPECT_DOUBLE_EQ(host.mem(ddc::DiscreteElement<Grid, Vector>(14, 0)), 7.0);
    sil::exterior::deriv<Vector, Scalar>(
            Kokkos::DefaultExecutionSpace(),
            derivative,
            potential,
            sil::exterior::PrescribedScalarExtrapolationRule {30.0});
    ddc::parallel_deepcopy(host, derivative);
    EXPECT_DOUBLE_EQ(host.mem(ddc::DiscreteElement<Grid, Vector>(14, 0)), 14.0);
    sil::exterior::deriv<Vector, Scalar>(Kokkos::DefaultExecutionSpace(), derivative, potential);
    ddc::parallel_deepcopy(host, derivative);
    EXPECT_DOUBLE_EQ(host.mem(ddc::DiscreteElement<Grid, Vector>(14, 0)), 0.0);

    sil::exterior::transposed_coboundary<Vector, Scalar>(
            Kokkos::DefaultExecutionSpace(),
            derivative,
            potential,
            sil::exterior::NormalScalarFluxExtrapolationRule {3.0});
    ddc::parallel_deepcopy(host, derivative);
    EXPECT_DOUBLE_EQ(host.mem(ddc::DiscreteElement<Grid, Vector>(10, 0)), -3.0);
}

void test_laplacian_agrees_with_composition()
{
    ddc::DiscreteDomain<Grid> const
            grid(ddc::DiscreteElement<Grid>(10), ddc::DiscreteVector<Grid>(7));
    [[maybe_unused]] sil::tensor::TensorAccessor<Scalar> scalar_accessor;
    [[maybe_unused]] sil::tensor::TensorAccessor<Vector> vector_accessor;
    [[maybe_unused]] sil::tensor::TensorAccessor<Position> position_accessor;
    [[maybe_unused]] sil::tensor::TensorAccessor<Metric> metric_accessor;
    ddc::DiscreteDomain<Grid, Scalar> const scalar_domain(grid, scalar_accessor.domain());
    ddc::DiscreteDomain<Grid, Vector> const vector_domain(grid, vector_accessor.domain());
    ddc::DiscreteDomain<Grid, Position> const position_domain(grid, position_accessor.domain());
    ddc::DiscreteDomain<Grid, Metric> const metric_domain(grid, metric_accessor.domain());
    ddc::Chunk potential_alloc(scalar_domain, ddc::DeviceAllocator<double>());
    ddc::Chunk result_alloc(scalar_domain, ddc::DeviceAllocator<double>());
    ddc::Chunk composed_alloc(scalar_domain, ddc::DeviceAllocator<double>());
    ddc::Chunk derivative_alloc(vector_domain, ddc::DeviceAllocator<double>());
    ddc::Chunk position_alloc(position_domain, ddc::DeviceAllocator<double>());
    ddc::Chunk metric_alloc(metric_domain, ddc::DeviceAllocator<double>());
    sil::tensor::Tensor potential(potential_alloc);
    sil::tensor::Tensor result(result_alloc);
    sil::tensor::Tensor composed(composed_alloc);
    sil::tensor::Tensor derivative(derivative_alloc);
    sil::tensor::Tensor position(position_alloc);
    sil::tensor::Tensor metric(metric_alloc);
    ddc::parallel_for_each(
            Kokkos::DefaultExecutionSpace(),
            grid,
            KOKKOS_LAMBDA(ddc::DiscreteElement<Grid> elem) {
                double const x = static_cast<double>(elem.uid<Grid>() - 10);
                potential.mem(elem, ddc::DiscreteElement<Scalar>(0)) = x * x;
                position.mem(elem, ddc::DiscreteElement<Position>(0)) = x;
            });
    auto laplacian = sil::exterior::make_staged_laplacian<
            Metric,
            Vector,
            Scalar>(Kokkos::DefaultExecutionSpace(), result, potential, metric, position);
    sil::exterior::NaturalScalarExtrapolationRule const primal;
    sil::exterior::NormalScalarFluxExtrapolationRule const dual {3.0};
    laplacian(result, potential, primal, dual);
    sil::exterior::
            deriv<Vector, Scalar>(Kokkos::DefaultExecutionSpace(), derivative, potential, primal);
    sil::exterior::codifferential<
            Metric,
            Vector,
            Vector>(Kokkos::DefaultExecutionSpace(), composed, derivative, metric, position, dual);
    auto result_host = ddc::create_mirror_view_and_copy(
            Kokkos::DefaultHostExecutionSpace(),
            result_alloc.span_view());
    auto composed_host = ddc::create_mirror_view_and_copy(
            Kokkos::DefaultHostExecutionSpace(),
            composed_alloc.span_view());
    ddc::host_for_each(scalar_domain, [&](ddc::DiscreteElement<Grid, Scalar> elem) {
        EXPECT_NEAR(result_host(elem), composed_host(elem), 1.0e-12);
    });
    // Reusing a staged operator with another closure must not retain the old policy.
    laplacian(result, potential);
    auto default_host = ddc::create_mirror_view_and_copy(
            Kokkos::DefaultHostExecutionSpace(),
            result_alloc.span_view());
    EXPECT_NE(
            default_host(ddc::DiscreteElement<Grid, Scalar>(10, 0)),
            result_host(ddc::DiscreteElement<Grid, Scalar>(10, 0)));
}
TEST(Extrapolation, NaturalCornersAndSingletonDimensions)
{
    struct Y
    {
    };
    struct GridY
    {
        using continuous_dimension_type = Y;
    };
    [[maybe_unused]] sil::tensor::TensorAccessor<Scalar> accessor;
    ddc::DiscreteDomain<Grid, GridY> const
            grid(ddc::DiscreteElement<Grid, GridY>(0, 0), ddc::DiscreteVector<Grid, GridY>(3, 2));
    ddc::DiscreteDomain<Grid, GridY, Scalar> const domain(grid, accessor.domain());
    ddc::Chunk allocation(domain, ddc::HostAllocator<double>());
    sil::tensor::Tensor field(allocation);
    ddc::host_for_each(grid, [&](ddc::DiscreteElement<Grid, GridY> elem) {
        field.mem(elem, ddc::DiscreteElement<Scalar>(0))
                = 1.0 + 2.0 * elem.uid<Grid>() + 3.0 * elem.uid<GridY>();
    });
    sil::exterior::NaturalScalarExtrapolationRule const rule;
    EXPECT_DOUBLE_EQ(
            rule(field,
                 grid.front() - ddc::DiscreteVector<Grid, GridY>(1, 1),
                 ddc::DiscreteElement<Scalar>(0)),
            -4.0);
    EXPECT_DOUBLE_EQ(
            rule(field,
                 grid.back() + ddc::DiscreteVector<Grid, GridY>(1, 1),
                 ddc::DiscreteElement<Scalar>(0)),
            13.0);
    ddc::DiscreteDomain<Grid, GridY> const singleton_grid
            = grid.take_first(ddc::DiscreteVector<Grid, GridY>(3, 1));
    ddc::DiscreteDomain<Grid, GridY, Scalar> const
            singleton_domain(singleton_grid, accessor.domain());
    ddc::Chunk singleton_allocation(singleton_domain, ddc::HostAllocator<double>());
    sil::tensor::Tensor singleton(singleton_allocation);
    ddc::parallel_fill(singleton, 4.0);
    EXPECT_DOUBLE_EQ(
            rule(singleton,
                 grid.front() - ddc::DiscreteVector<Grid, GridY>(1, 1),
                 ddc::DiscreteElement<Scalar>(0)),
            4.0);
}

TEST(Extrapolation, CallableRulesAndConnectedDerivative)
{
    test_callable_rules_and_connected_derivative();
}

TEST(Extrapolation, LaplacianAgreesWithComposition)
{
    test_laplacian_agrees_with_composition();
}
} // namespace
