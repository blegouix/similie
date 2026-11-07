// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#include <array>

#include <gtest/gtest.h>
#include <similie/physics/magnetostatics/magnetic_form_response.hpp>

namespace {
struct X
{
};
struct Y
{
};
struct Z
{
};

using MetricIndex = sil::tensor::TensorSymmetricIndex<
        sil::tensor::Covariant<sil::tensor::MetricIndex1<X, Y, Z>>,
        sil::tensor::Covariant<sil::tensor::MetricIndex2<X, Y, Z>>>;
using Metric = sil::tensor::
        Tensor<double, ddc::DiscreteDomain<MetricIndex>, Kokkos::layout_right, Kokkos::HostSpace>;
using Response = similie::physics::magnetostatics::MagneticFormResponse<X, Y, Z>;

struct MagneticFieldCheck
{
    Kokkos::View<double*> errors;
    Response response;

    KOKKOS_FUNCTION void operator()(int) const
    {
        sil::exterior::Form<std::array<double, 3>> const induction {8.0, -3.0, 6.0};
        sil::exterior::Form<std::array<double, 3>> const field = response.field(induction, 0.5);
        std::array<double, 3> const expected {13.0 / 24.0, -7.0 / 24.0, 2.0};
        double error = 0.0;
        for (std::size_t component = 0; component < expected.size(); ++component)
            error += Kokkos::abs(field[component] - expected[component]);
        errors(0) = error;
    }
};

TEST(MagneticForms, NonDiagonalMetricMapsTwoFormToOneForm)
{
    std::array<double, MetricIndex::mem_size()> storage {};
    Metric metric(
            ddc::ChunkSpan<
                    double,
                    ddc::DiscreteDomain<MetricIndex>,
                    Kokkos::layout_right,
                    Kokkos::HostSpace>(
                    storage.data(),
                    sil::tensor::TensorAccessor<MetricIndex>::domain()));
    metric(metric.template access_element<X, X>()) = 4.0;
    metric(metric.template access_element<X, Y>()) = 2.0;
    metric(metric.template access_element<Y, Y>()) = 10.0;
    metric(metric.template access_element<Z, Z>()) = 16.0;

    Response const response(metric, ddc::DiscreteElement<>());
    Kokkos::View<double*> errors("magnetic_form_errors", 1);
    Kokkos::parallel_for("magnetic_form_check", 1, MagneticFieldCheck {errors, response});
    auto const host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), errors);
    EXPECT_LT(host(0), 1.e-12);
}
} // namespace
