// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#include <cmath>

#include <gtest/gtest.h>

#include "configuration.hpp"
#include "staged_codifferential_laplacian.hpp"

namespace {
struct Periodic
{
    template <class Tensor, class Element>
    KOKKOS_FUNCTION auto operator()(Tensor const& field, Element element) const
    {
        typename Tensor::non_indices_domain_t const domain(field.non_indices_domain());
        typename Tensor::non_indices_domain_t::discrete_element_type wrapped;
        for (std::size_t axis = 0; axis < 3; ++axis) {
            std::ptrdiff_t const front = ddc::detail::array(domain.front())[axis];
            std::ptrdiff_t const size = ddc::detail::array(domain.extents())[axis];
            std::ptrdiff_t const offset = ddc::detail::array(element)[axis] - front;
            ddc::detail::array(wrapped)[axis] = front + (offset % size + size) % size;
        }
        return sil::tensor::detail::pointwise_tensor<Tensor>(
                [&](auto component) { return field.mem(wrapped, component); });
    }
};

template <class Form>
void compare()
{
    using namespace taylor_green_vortex;
    using Tensor = detail::FormField<Form>;
    using Index
            = std::conditional_t<Form::rank() == 0, Vector, sil::tensor::Covariant<CurlDirection>>;
    SpatialDomain const
            domain(ddc::DiscreteElement<GridX, GridY, GridZ>(2, 2, 2),
                   ddc::DiscreteVector<GridX, GridY, GridZ>(4, 4, 4));
    SpatialDomain const
            halo(ddc::DiscreteElement<GridX, GridY, GridZ>(0, 0, 0),
                 ddc::DiscreteVector<GridX, GridY, GridZ>(8, 8, 8));
    ddc::Chunk positions(
            PositionTensor::discrete_domain_type(
                    halo,
                    sil::tensor::TensorAccessor<PositionIndex>::domain()),
            ddc::KokkosAllocator<double, Field::memory_space>());
    ddc::Chunk
            metrics(MetricTensor::discrete_domain_type(
                            domain,
                            sil::tensor::TensorAccessor<MetricIndex>::domain()),
                    ddc::KokkosAllocator<double, Field::memory_space>());
    typename Tensor::discrete_domain_type const
            fields(domain, sil::tensor::TensorAccessor<Form>::domain());
    ddc::Chunk input_alloc(fields, ddc::KokkosAllocator<double, Field::memory_space>());
    ddc::Chunk expected_alloc(fields, ddc::KokkosAllocator<double, Field::memory_space>());
    ddc::Chunk actual_alloc(fields, ddc::KokkosAllocator<double, Field::memory_space>());
    Tensor const input(input_alloc.span_view()), expected(expected_alloc.span_view()),
            actual(actual_alloc.span_view());
    PositionTensor const position(positions.span_view());
    MetricTensor const metric(metrics.span_view());
    ddc::parallel_fill(Kokkos::DefaultExecutionSpace(), metrics, 1.0);
    ddc::parallel_for_each(
            Kokkos::DefaultExecutionSpace(),
            halo,
            KOKKOS_LAMBDA(SpatialDomain::discrete_element_type element) {
                for (int axis = 0; axis < 3; ++axis) {
                    double const coordinate = ddc::detail::array(element)[axis];
                    position.mem(element, ddc::DiscreteElement<PositionIndex>(axis))
                            = (0.7 + 0.1 * axis) * coordinate + 0.003 * coordinate * coordinate;
                }
            });
    sil::exterior::Laplacian<
            MetricIndex,
            Index,
            Form,
            Tensor,
            MetricTensor,
            PositionTensor,
            Kokkos::DefaultExecutionSpace,
            sil::exterior::WeightedLeviCivitaDiscreteHodgeStar>
            direct(Kokkos::DefaultExecutionSpace(), expected, input, metric, position);
    detail::StagedCodifferentialLaplacian<Tensor, Periodic>
            staged(Kokkos::DefaultExecutionSpace(),
                   actual,
                   input,
                   metric,
                   position,
                   Periodic {},
                   Periodic {});
    for (int trial = 0; trial < 2; ++trial) {
        ddc::parallel_for_each(
                Kokkos::DefaultExecutionSpace(),
                fields,
                KOKKOS_LAMBDA(typename Tensor::discrete_element_type element) {
                    input.mem(element) = trial == 0 ? 1.0
                                                    : Kokkos::sin(
                                                              0.7 * element.template uid<GridX>()
                                                              + 0.4 * element.template uid<GridY>()
                                                              + 0.3 * element.template uid<GridZ>()
                                                              + element.template uid<Form>());
                });
        direct(expected, input, Periodic {}, Periodic {});
        staged(actual, input);
        Kokkos::View<double*, Field::memory_space, Kokkos::MemoryUnmanaged> const
                expected_view(expected.data_handle(), expected.size()),
                actual_view(actual.data_handle(), actual.size());
        Kokkos::View<double*, Kokkos::HostSpace> const expected_host
                = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), expected_view);
        Kokkos::View<double*, Kokkos::HostSpace> const actual_host
                = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), actual_view);
        for (std::size_t i = 0; i < actual.size(); ++i)
            EXPECT_NEAR(
                    actual_host(i),
                    expected_host(i),
                    2.e-11 * (1.0 + std::abs(expected_host(i))));
    }
}

TEST(TgvStagedCodifferential, ScalarLaplacian)
{
    compare<taylor_green_vortex::Scalar>();
}

TEST(TgvStagedCodifferential, TwoFormLaplacian)
{
    compare<taylor_green_vortex::TwoForm>();
}

TEST(TgvStagedCodifferential, ConfigurationAndFlagNames)
{
    taylor_green_vortex::Configuration configuration;
    EXPECT_FALSE(configuration.consume("--levi-civita-discrete-hodge-star"));
    EXPECT_TRUE(configuration.consume("--weighted-levi-civita-discrete-hodge-star"));
    EXPECT_TRUE(configuration.consume("--diagonal-reduction"));
    EXPECT_TRUE(configuration.consume("--levi-civita-continuous-hodge-star"));
    EXPECT_TRUE(configuration.consume("--diagonal-reconstruction"));
    EXPECT_TRUE(configuration.consume("--staged-codifferential"));
    EXPECT_NO_THROW(configuration.validate());
    EXPECT_EQ(configuration.laplacian_mode(), 37);
    EXPECT_EQ(configuration.staging_name(), "codifferential");
    EXPECT_EQ(
            configuration.strategy_flags(),
            "--staged-codifferential+--weighted-levi-civita-discrete-hodge-star+--diagonal-"
            "reduction+--levi-civita-continuous-hodge-star+--diagonal-reconstruction");
    configuration.staged_codifferential = false;
    EXPECT_TRUE(configuration.consume("--staged-laplacian"));
    EXPECT_EQ(configuration.laplacian_mode(), 36);
}
} // namespace
