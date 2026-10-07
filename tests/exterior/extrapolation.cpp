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

struct ConstantExteriorRule
{
    double value;

    template <class TensorType, class Element, class Component>
    KOKKOS_FUNCTION double operator()(TensorType, Element, Component) const
    {
        return value;
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
    sil::exterior::ExtrapolationRules const connected_rules(
            std::pair {sil::exterior::ZeroCochainExtrapolationRule {}, connected});
    sil::exterior::deriv<
            Vector,
            Scalar>(Kokkos::DefaultExecutionSpace(), derivative, potential, connected_rules);
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
            ConstantExteriorRule {30.0});
    ddc::parallel_deepcopy(host, derivative);
    EXPECT_DOUBLE_EQ(host.mem(ddc::DiscreteElement<Grid, Vector>(14, 0)), 14.0);
    // Automatic broadcasting invokes this rule only for exterior samples.
    EXPECT_DOUBLE_EQ(host.mem(ddc::DiscreteElement<Grid, Vector>(12, 0)), 5.0);
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
    sil::exterior::ExtrapolationRules const primal(
            std::
                    pair {sil::exterior::ZeroCochainExtrapolationRule {},
                          sil::exterior::NaturalScalarExtrapolationRule {}});
    sil::exterior::ExtrapolationRules const dual(
            std::
                    pair {sil::exterior::NormalScalarFluxExtrapolationRule {3.0},
                          sil::exterior::ZeroCochainExtrapolationRule {}});
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
template <std::size_t Dimension>
struct Axis
{
};

template <std::size_t Dimension>
struct GridDimension
{
    using continuous_dimension_type = Axis<Dimension>;
};

struct Sample
{
};

template <std::size_t... Dimension>
void test_per_boundary_rules(std::index_sequence<Dimension...>)
{
    [[maybe_unused]] sil::tensor::TensorAccessor<Scalar> accessor;
    ddc::DiscreteDomain<GridDimension<Dimension>...> const
            grid(ddc::DiscreteElement<GridDimension<Dimension>...>((Dimension * 0)...),
                 ddc::DiscreteVector<GridDimension<Dimension>...>((Dimension * 0 + 3)...));
    ddc::DiscreteDomain<GridDimension<Dimension>..., Scalar> const domain(grid, accessor.domain());
    ddc::Chunk field_alloc(domain, ddc::DeviceAllocator<double>());
    sil::tensor::Tensor field(field_alloc);
    ddc::parallel_fill(field, 42.0);
    sil::exterior::ExtrapolationRules const rules(
            std::
                    pair {sil::exterior::PrescribedScalarExtrapolationRule {2.0 * Dimension + 1.0},
                          sil::exterior::PrescribedScalarExtrapolationRule {
                                  2.0 * Dimension + 2.0}}...);
    auto const uniform_rules = sil::exterior::make_extrapolation_rules<sizeof...(Dimension)>(
            sil::exterior::PrescribedScalarExtrapolationRule {17.5});
    decltype(uniform_rules) implicit_rules
            = sil::exterior::PrescribedScalarExtrapolationRule {17.5};
    auto const preserved_rules
            = sil::exterior::make_extrapolation_rules<sizeof...(Dimension)>(rules);
    static_assert(std::is_same_v<decltype(preserved_rules), decltype(rules)>);
    ddc::DiscreteDomain<Sample> const
            samples(ddc::DiscreteElement<Sample>(0),
                    ddc::DiscreteVector<Sample>(2 * sizeof...(Dimension) + 9));
    ddc::Chunk results(samples, ddc::DeviceAllocator<double>());
    ddc::Chunk uniform_results(samples, ddc::DeviceAllocator<double>());
    ddc::Chunk implicit_results(samples, ddc::DeviceAllocator<double>());
    ddc::ChunkSpan const result = results.span_view();
    ddc::ChunkSpan const uniform_result = uniform_results.span_view();
    ddc::ChunkSpan const implicit_result = implicit_results.span_view();
    ddc::parallel_for_each(
            Kokkos::DefaultExecutionSpace(),
            samples,
            KOKKOS_LAMBDA(ddc::DiscreteElement<Sample> sample) {
                std::size_t const id = sample.uid<Sample>();
                ddc::DiscreteElement<GridDimension<Dimension>...> elem = grid.front();
                if (id < 2 * sizeof...(Dimension)) {
                    std::size_t const dim = id / 2;
                    if (id % 2 == 0)
                        --ddc::detail::array(elem)[dim];
                    else
                        ddc::detail::array(elem)[dim] = ddc::detail::array(grid.back())[dim] + 1;
                } else if (id == 2 * sizeof...(Dimension)) {
                    elem = grid.front()
                           - ddc::DiscreteVector<GridDimension<Dimension>...>(
                                   (Dimension * 0 + 1)...);
                } else if (id == 2 * sizeof...(Dimension) + 1) {
                    elem = grid.back()
                           + ddc::DiscreteVector<GridDimension<Dimension>...>(
                                   (Dimension * 0 + 1)...);
                } else if (id == 2 * sizeof...(Dimension) + 2) {
                    elem = grid.back();
                } else if (id == 2 * sizeof...(Dimension) + 4) {
                    for (std::size_t dim = 0; dim < sizeof...(Dimension); ++dim) {
                        if (dim % 2 == 0)
                            --ddc::detail::array(elem)[dim];
                        else
                            ddc::detail::array(elem)[dim]
                                    = ddc::detail::array(grid.back())[dim] + 1;
                    }
                } else if (id == 2 * sizeof...(Dimension) + 5) {
                    if constexpr (sizeof...(Dimension) > 1)
                        --ddc::detail::array(elem)[0];
                    ddc::detail::array(elem)[sizeof...(Dimension) - 1]
                            = ddc::detail::array(grid.back())[sizeof...(Dimension) - 1] + 1;
                }
                if (id == 2 * sizeof...(Dimension) + 6) {
                    elem = grid.front()
                           - ddc::DiscreteVector<GridDimension<Dimension>...>((Dimension + 1)...);
                } else if (id == 2 * sizeof...(Dimension) + 7) {
                    elem = grid.back()
                           + ddc::DiscreteVector<GridDimension<Dimension>...>((Dimension + 1)...);
                } else if (id == 2 * sizeof...(Dimension) + 8) {
                    if constexpr (sizeof...(Dimension) > 1)
                        --ddc::detail::array(elem)[0];
                    ddc::detail::array(elem)[sizeof...(Dimension) - 1]
                            = ddc::detail::array(grid.back())[sizeof...(Dimension) - 1] + 3;
                }
                result(sample) = preserved_rules(field, elem, ddc::DiscreteElement<Scalar>(0));
                uniform_result(sample)
                        = uniform_rules(field, elem, ddc::DiscreteElement<Scalar>(0));
                implicit_result(sample)
                        = implicit_rules(field, elem, ddc::DiscreteElement<Scalar>(0));
                if constexpr (sizeof...(Dimension) == 2) {
                    if (id == 2 * sizeof...(Dimension) + 3) {
                        // Pair order follows the tensor domain, not these reversed tags.
                        ddc::DiscreteElement<GridDimension<1>, GridDimension<0>> const
                                reversed(3, 1);
                        result(sample)
                                = preserved_rules(field, reversed, ddc::DiscreteElement<Scalar>(0));
                    }
                }
            });
    auto host = ddc::
            create_mirror_view_and_copy(Kokkos::DefaultHostExecutionSpace(), results.span_view());
    auto uniform_host = ddc::create_mirror_view_and_copy(
            Kokkos::DefaultHostExecutionSpace(),
            uniform_results.span_view());
    auto implicit_host = ddc::create_mirror_view_and_copy(
            Kokkos::DefaultHostExecutionSpace(),
            implicit_results.span_view());
    for (std::size_t id = 0; id < 2 * sizeof...(Dimension) + 9; ++id) {
        double const expected
                = (id == 2 * sizeof...(Dimension) + 2 || id == 2 * sizeof...(Dimension) + 3) ? 42.0
                                                                                             : 17.5;
        EXPECT_DOUBLE_EQ(uniform_host(ddc::DiscreteElement<Sample>(id)), expected);
        EXPECT_DOUBLE_EQ(implicit_host(ddc::DiscreteElement<Sample>(id)), expected);
    }
    for (std::size_t id = 0; id < 2 * sizeof...(Dimension); ++id)
        EXPECT_DOUBLE_EQ(host(ddc::DiscreteElement<Sample>(id)), static_cast<double>(id + 1));
    EXPECT_DOUBLE_EQ(
            host(ddc::DiscreteElement<Sample>(2 * sizeof...(Dimension))),
            static_cast<double>(sizeof...(Dimension)));
    EXPECT_DOUBLE_EQ(
            host(ddc::DiscreteElement<Sample>(2 * sizeof...(Dimension) + 1)),
            static_cast<double>(sizeof...(Dimension) + 1));
    EXPECT_DOUBLE_EQ(host(ddc::DiscreteElement<Sample>(2 * sizeof...(Dimension) + 2)), 42.0);
    EXPECT_DOUBLE_EQ(
            host(ddc::DiscreteElement<Sample>(2 * sizeof...(Dimension) + 3)),
            sizeof...(Dimension) == 2 ? 4.0 : 42.0);
    EXPECT_DOUBLE_EQ(
            host(ddc::DiscreteElement<Sample>(2 * sizeof...(Dimension) + 4)),
            ((0.0 + ... + (2.0 * Dimension + (Dimension % 2 == 0 ? 1.0 : 2.0)))
             / sizeof...(Dimension)));
    EXPECT_DOUBLE_EQ(
            host(ddc::DiscreteElement<Sample>(2 * sizeof...(Dimension) + 5)),
            sizeof...(Dimension) > 1 ? sizeof...(Dimension) + 0.5 : 2.0);
    EXPECT_DOUBLE_EQ(
            host(ddc::DiscreteElement<Sample>(2 * sizeof...(Dimension) + 6)),
            ((0.0 + ... + ((Dimension + 1) * (2.0 * Dimension + 1.0)))
             / (0.0 + ... + (Dimension + 1))));
    EXPECT_DOUBLE_EQ(
            host(ddc::DiscreteElement<Sample>(2 * sizeof...(Dimension) + 7)),
            ((0.0 + ... + ((Dimension + 1) * (2.0 * Dimension + 2.0)))
             / (0.0 + ... + (Dimension + 1))));
    EXPECT_DOUBLE_EQ(
            host(ddc::DiscreteElement<Sample>(2 * sizeof...(Dimension) + 8)),
            sizeof...(Dimension) > 1 ? (1.0 + 6.0 * sizeof...(Dimension)) / 4.0 : 2.0);
}

void test_corner_blending_limits()
{
    [[maybe_unused]] sil::tensor::TensorAccessor<Scalar> accessor;
    ddc::DiscreteDomain<GridDimension<0>, GridDimension<1>> const
            grid(ddc::DiscreteElement<GridDimension<0>, GridDimension<1>>(0, 0),
                 ddc::DiscreteVector<GridDimension<0>, GridDimension<1>>(3, 3));
    ddc::DiscreteDomain<GridDimension<0>, GridDimension<1>, Scalar> const
            domain(grid, accessor.domain());
    ddc::Chunk field_alloc(domain, ddc::DeviceAllocator<double>());
    sil::tensor::Tensor field(field_alloc);
    ddc::parallel_fill(field, 42.0);
    sil::exterior::ExtrapolationRules const
            rules(std::
                          pair {sil::exterior::PrescribedScalarExtrapolationRule {1.0},
                                sil::exterior::PrescribedScalarExtrapolationRule {2.0}},
                  std::
                          pair {sil::exterior::PrescribedScalarExtrapolationRule {3.0},
                                sil::exterior::PrescribedScalarExtrapolationRule {4.0}});
    Kokkos::Array<int, 7> const left_distances {0, 1, 1, 1, 3, 1, 1000000};
    Kokkos::Array<int, 7> const top_distances {1, 0, 1, 3, 1, 1000000, 1};
    ddc::DiscreteDomain<Sample> const
            samples(ddc::DiscreteElement<Sample>(0), ddc::DiscreteVector<Sample>(7));
    ddc::Chunk allocation(samples, ddc::DeviceAllocator<double>());
    ddc::ChunkSpan const result = allocation.span_view();
    ddc::parallel_for_each(
            Kokkos::DefaultExecutionSpace(),
            samples,
            KOKKOS_LAMBDA(ddc::DiscreteElement<Sample> sample) {
                std::size_t const id = sample.uid<Sample>();
                ddc::DiscreteElement<GridDimension<0>, GridDimension<1>> const elem
                        = ddc::DiscreteElement<GridDimension<0>, GridDimension<1>>(0, 2)
                          + ddc::DiscreteVector<
                                  GridDimension<0>,
                                  GridDimension<1>>(-left_distances[id], top_distances[id]);
                result(sample) = rules(field, elem, ddc::DiscreteElement<Scalar>(0));
            });
    auto host = ddc::create_mirror_view_and_copy(
            Kokkos::DefaultHostExecutionSpace(),
            allocation.span_view());
    EXPECT_DOUBLE_EQ(host(ddc::DiscreteElement<Sample>(0)), 4.0);
    EXPECT_DOUBLE_EQ(host(ddc::DiscreteElement<Sample>(1)), 1.0);
    EXPECT_DOUBLE_EQ(host(ddc::DiscreteElement<Sample>(2)), 2.5);
    EXPECT_DOUBLE_EQ(host(ddc::DiscreteElement<Sample>(3)), 3.25);
    EXPECT_DOUBLE_EQ(host(ddc::DiscreteElement<Sample>(4)), 1.75);
    EXPECT_NEAR(host(ddc::DiscreteElement<Sample>(5)), 4.0, 4.0e-6);
    EXPECT_NEAR(host(ddc::DiscreteElement<Sample>(6)), 1.0, 4.0e-6);
}

TEST(Extrapolation, CornerBlendingLimits)
{
    test_corner_blending_limits();
}

TEST(Extrapolation, PerBoundaryRules1D)
{
    test_per_boundary_rules(std::make_index_sequence<1>());
}

TEST(Extrapolation, PerBoundaryRules2D)
{
    test_per_boundary_rules(std::make_index_sequence<2>());
}

TEST(Extrapolation, PerBoundaryRules4D)
{
    test_per_boundary_rules(std::make_index_sequence<4>());
}

TEST(Extrapolation, EmptyRulesForScalarDomain)
{
    [[maybe_unused]] sil::tensor::TensorAccessor<Scalar> accessor;
    ddc::Chunk allocation(accessor.domain(), ddc::HostAllocator<double>());
    sil::tensor::Tensor field(allocation);
    field.mem(ddc::DiscreteElement<Scalar>(0)) = 42.0;
    auto const rules = sil::exterior::make_extrapolation_rules<0>(
            sil::exterior::PrescribedScalarExtrapolationRule {17.5});
    static_assert(
            std::is_same_v<std::remove_cv_t<decltype(rules)>, sil::exterior::ExtrapolationRules<>>);
    EXPECT_DOUBLE_EQ(rules(field, ddc::DiscreteElement<>(), ddc::DiscreteElement<Scalar>(0)), 42.0);
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
