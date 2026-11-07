// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#include <gtest/gtest.h>
#include <similie/exterior/quadrature/quadrature.hpp>
#include <similie/exterior/reduction_and_reconstruction/vertex_reconstructed_derivative.hpp>

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

struct QuadratureChecks
{
    Kokkos::View<double*> errors;

    KOKKOS_FUNCTION void operator()(int) const
    {
        auto const unit_measure = [](auto const&, std::size_t) { return 1.0; };
        double const vertex = sil::exterior::quadrature(
                sil::exterior::VertexQuadrature<3>(),
                [](auto const& point, std::size_t) { return point[0] * point[1] * point[2]; },
                unit_measure);
        // A quadratic reconstruction squared needs degree-four integration.
        // The evaluator is independent of the quadrature rule and geometry.
        auto const reconstructed_square = [](auto const& point, std::size_t) {
            double const value = point[0] * point[0] * point[1];
            return value * value;
        };
        double const gauss = sil::exterior::quadrature(
                sil::exterior::GaussLegendreQuadrature<2, 3>(),
                reconstructed_square,
                [](auto const&, std::size_t) { return 6.0; });
        errors(0) = Kokkos::abs(vertex - 0.125) + Kokkos::abs(gauss - 6.0 / 15.0);

        double const midpoint = sil::exterior::quadrature(
                sil::exterior::GaussLegendreQuadrature<2, 1>(),
                [](auto const& point, std::size_t) { return point[0] * point[1]; },
                unit_measure);
        double const cubic = sil::exterior::quadrature(
                sil::exterior::GaussLegendreQuadrature<2, 2>(),
                [](auto const& point, std::size_t) {
                    double const value = point[0] * point[1];
                    return value * value * value;
                },
                unit_measure);
        errors(0) += Kokkos::abs(midpoint - 0.25) + Kokkos::abs(cubic - 1.0 / 16.0);

        // A scalar fibre in 3D exercises dimension and fibre-rank independence.
        std::array<std::array<double, 3>, 8> positions {};
        std::array<std::array<double, 1>, 8> nodal {};
        for (std::size_t vertex = 0; vertex < 8; ++vertex) {
            double const u = vertex & 1, v = (vertex >> 1) & 1, w = (vertex >> 2) & 1;
            positions[vertex] = {2.0 * u + 0.3 * v, 3.0 * v + 0.2 * w, 4.0 * w};
            nodal[vertex][0] = 2.0 * positions[vertex][0] - 3.0 * positions[vertex][1]
                               + 0.5 * positions[vertex][2] + 7.0;
        }
        sil::exterior::VertexReconstructedDerivative<1, X, Y, Z> const reconstruction(positions);
        std::array<double, 3> const gradient = reconstruction.gradient(nodal);
        errors(1) = Kokkos::abs(gradient[0] - 2.0) + Kokkos::abs(gradient[1] + 3.0)
                    + Kokkos::abs(gradient[2] - 0.5);
        double const volume = sil::exterior::quadrature(
                sil::exterior::VertexQuadrature<3>(),
                [](auto const&, std::size_t) { return 1.0; },
                [&](auto const&, std::size_t sample) { return reconstruction.measures[sample]; });
        errors(1) += Kokkos::abs(volume - 24.0);
    }
};

TEST(Quadrature, IndependentEvaluatorsAndThreeDimensionalReconstruction)
{
    Kokkos::View<double*> errors("quadrature_errors", 2);
    Kokkos::parallel_for("quadrature_checks", 1, QuadratureChecks {errors});
    auto const host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), errors);
    EXPECT_LT(host(0), 1.e-13);
    EXPECT_LT(host(1), 1.e-12);
}
TEST(Quadrature, RejectsFoldedAndDegenerateReconstructionCells)
{
    using Reconstruction = sil::exterior::VertexReconstructedDerivative<1, X, Y>;
    std::array<std::array<double, 2>, 4> positions {
            {{0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}, {-1.0, 1.0}}};
    EXPECT_THROW((Reconstruction(positions)), std::runtime_error);
    positions = {{{0.0, 0.0}, {0.0, 0.0}, {0.0, 1.0}, {1.0, 1.0}}};
    EXPECT_THROW((Reconstruction(positions)), std::runtime_error);
}

} // namespace
