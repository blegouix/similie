// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#include <gtest/gtest.h>
#include <similie/exterior/covariant_derivative.hpp>

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

template <class... Tags>
void affine_test()
{
    constexpr std::size_t n = sizeof...(Tags);
    std::array<std::array<double, n>, std::size_t(1) << n> positions {};
    for (std::size_t v = 0; v < positions.size(); ++v) {
        for (std::size_t d = 0; d < n; ++d) {
            positions[v][d] = 1.2 + 2 * ((v >> d) & 1);
            for (std::size_t j = 0; j < n; ++j)
                if (j != d)
                    positions[v][d] += 0.13 * ((v >> j) & 1);
            // Non-affine mapped cells as well as skew Jacobians.
            if constexpr (n > 1)
                positions[v][d] += 0.07 * (d + 1) * (v & 1) * ((v >> 1) & 1);
        }
    }
    std::array<double, n> point {};
    point.fill(0.37);
    sil::exterior::CubicalReconstruction<n> const reconstruction(positions, point);
    ASSERT_TRUE(reconstruction.valid());
    // Rank 2 is intentional, even in 1D and 3D: arbitrary vector bundles.
    auto const stencil
            = sil::exterior::CovariantDerivative<Tags...>::template cell_stencil<2>(reconstruction);
    for (std::size_t a = 0; a < 2; ++a) {
        for (std::size_t d = 0; d < n; ++d) {
            double derivative = 0, constant = 0;
            for (std::size_t v = 0; v < positions.size(); ++v) {
                for (std::size_t b = 0; b < 2; ++b) {
                    double field = 3 + b;
                    for (std::size_t j = 0; j < n; ++j)
                        field += (b + 1.0) * (j + 2.0) * positions[v][j];
                    derivative += stencil[v][d][a][b] * field;
                    constant += stencil[v][d][a][b] * (b + 1.0);
                }
            }
            EXPECT_NEAR(derivative, (a + 1.0) * (d + 2.0), 2e-14);
            EXPECT_NEAR(constant, 0, 2e-15);
        }
    }
}

struct FrameTransport
{
    KOKKOS_FUNCTION auto operator()(std::size_t to, std::size_t from) const
    {
        double const angle = 0.3 * (double(to) - double(from));
        return std::array<std::array<double, 2>, 2> {
                {{Kokkos::cos(angle), -Kokkos::sin(angle)},
                 {Kokkos::sin(angle), Kokkos::cos(angle)}}};
    }
};

struct ExecutionSpaceEvaluation
{
    Kokkos::View<double*> output;

    KOKKOS_FUNCTION void operator()(int) const
    {
        std::array<std::array<double, 1>, 2> const positions {{{0}, {2}}};
        auto const stencil = sil::exterior::CovariantDerivative<X>::cell_stencil<1>(
                sil::exterior::CubicalReconstruction<1>(positions, {0.3}));
        output(0) = stencil[1][0][0][0];
    }
};
} // namespace

TEST(CovariantDerivative, Affine1D)
{
    affine_test<X>();
}
TEST(CovariantDerivative, Affine2D)
{
    affine_test<X, Y>();
}
TEST(CovariantDerivative, Affine3D)
{
    affine_test<X, Y, Z>();
}

TEST(CovariantDerivative, ScalarCommutingReconstructionAndCrossMode)
{
    std::array<std::array<double, 2>, 4> const positions {{{0, 0}, {1, 0}, {0, 1}, {1, 1}}};
    double energy = 0;
    for (int q = 0; q < 4; ++q) {
        double const x = 0.5 + (q % 2 ? 1 : -1) / std::sqrt(12.0);
        double const y = 0.5 + (q / 2 ? 1 : -1) / std::sqrt(12.0);
        sil::exterior::CubicalReconstruction<2> const reconstruction(positions, {x, y});
        auto const stencil
                = sil::exterior::CovariantDerivative<X, Y>::cell_stencil<1>(reconstruction);
        // u = x*y is the fourth mode; its derivative is (y,x).
        EXPECT_NEAR(stencil[3][0][0][0], y, 1e-15);
        EXPECT_NEAR(stencil[3][1][0][0], x, 1e-15);
        EXPECT_NEAR(reconstruction.basis<0>({}, 3, {}), x * y, 1e-15);
        energy += 0.25 * (x * x + y * y);
    }
    EXPECT_NEAR(energy, 2.0 / 3, 1e-15);
}

TEST(CovariantDerivative, CubicalStokesAndFlatSquareZero)
{
    using D = sil::exterior::CovariantDerivative<X, Y, Z>;
    auto field
            = [](std::size_t, std::size_t v) { return std::array<double, 1> {std::sin(0.7 * v)}; };
    auto edge = [&](std::size_t mask, std::size_t v) {
        std::size_t axis = 0;
        while (!(mask & (std::size_t(1) << axis)))
            ++axis;
        return D::cochain_value<0, 1>({axis}, v, field);
    };
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = i + 1; j < 3; ++j) {
            EXPECT_NEAR((D::cochain_value<1, 1>({i, j}, 0, edge)[0]), 0, 2e-15);
        }
    auto arbitrary_edges = [](std::size_t mask, std::size_t v) {
        return std::array<double, 1> {std::cos(mask + 0.2 * v)};
    };
    auto face = [&](std::size_t mask, std::size_t v) {
        std::array<std::size_t, 2> axes {};
        int j = 0;
        for (std::size_t i = 0; i < 3; ++i)
            if (mask & (std::size_t(1) << i))
                axes[j++] = i;
        return D::cochain_value<1, 1>(axes, v, arbitrary_edges);
    };
    EXPECT_NEAR((D::cochain_value<2, 1>({0, 1, 2}, 0, face)[0]), 0, 2e-15);
    auto linear = [](std::size_t mask, std::size_t v) {
        // omega = x dy, integral around the unit xy face equals 1.
        return std::array<double, 1> {mask == 2 ? double(v & 1) : 0};
    };
    EXPECT_DOUBLE_EQ((D::cochain_value<1, 1>({0, 1}, 0, linear)[0]), 1);
}

TEST(CovariantDerivative, OrientedFaceValuesForTwoComponentBundle)
{
    auto sampler = [](std::size_t mask, std::size_t vertex) {
        return std::array<double, 2> {double(mask * vertex * vertex), double((mask + 1) * vertex)};
    };
    // Boundary of xyz: +yz at x=1, -xz at y=1, +xy at z=1,
    // with the opposite signs on the three lower faces.
    auto const volume = sil::exterior::CovariantDerivative<X, Y, Z>::
            cochain_value<2, 2>({0, 1, 2}, 0, sampler);
    EXPECT_DOUBLE_EQ(volume[0], 6 - 5 * 4 + 3 * 16);
    EXPECT_DOUBLE_EQ(volume[1], 7 - 6 * 2 + 4 * 4);
    // A yz face based at x=1 also exercises nonzero base vertices.
    auto const face
            = sil::exterior::CovariantDerivative<X, Y, Z>::cochain_value<1, 2>({1, 2}, 1, sampler);
    EXPECT_DOUBLE_EQ(face[0], 4 * (9 - 1) - 2 * (25 - 1));
    EXPECT_DOUBLE_EQ(face[1], 5 * (3 - 1) - 3 * (5 - 1));
}

TEST(CovariantDerivative, ChangeOfFrameAndParallelSection)
{
    std::array<std::array<double, 2>, 4> const positions {{{0, 0}, {2, 0}, {0, 3}, {2, 3}}};
    auto const stencil = sil::exterior::CovariantDerivative<X, Y>::cell_stencil<
            2>(sil::exterior::CubicalReconstruction<2>(positions, {0.31, 0.62}), FrameTransport {});
    for (std::size_t a = 0; a < 2; ++a)
        for (std::size_t d = 0; d < 2; ++d) {
            double affine = 0, parallel = 0;
            for (std::size_t v = 0; v < 4; ++v) {
                auto const frame = FrameTransport {}(v, 0);
                for (std::size_t b = 0; b < 2; ++b)
                    for (std::size_t c = 0; c < 2; ++c) {
                        affine += stencil[v][d][a][b] * frame[b][c] * (c + 1)
                                  * (positions[v][0] + 2 * positions[v][1]);
                        parallel += stencil[v][d][a][b] * frame[b][c] * (c + 1);
                    }
            }
            EXPECT_NEAR(affine, (a + 1) * (d + 1), 1e-14);
            EXPECT_NEAR(parallel, 0, 1e-15);
        }
}

TEST(CovariantDerivative, OneFormReconstructionCommutesOnMappedCell)
{
    std::array<std::array<double, 2>, 4> const positions {{{0, 0}, {2, 0.2}, {0.4, 3}, {2.5, 3.4}}};
    auto edge = [](std::size_t mask, std::size_t vertex) {
        return std::array<double, 1> {std::sin(double(mask + 2 * vertex))};
    };
    auto reconstruct = [&](double xi, double eta) {
        sil::exterior::CubicalReconstruction<2> const map(positions, {xi, eta});
        std::array<double, 2> result {};
        for (std::size_t axis = 0; axis < 2; ++axis) {
            for (std::size_t vertex = 0; vertex < 4; ++vertex) {
                if (vertex & (std::size_t(1) << axis))
                    continue;
                for (std::size_t i = 0; i < 2; ++i) {
                    result[i] += map.basis<1>({axis}, vertex, {i})
                                 * edge(std::size_t(1) << axis, vertex)[0];
                }
            }
        }
        return result;
    };
    double const xi = 0.31, eta = 0.62, h = 1e-5;
    auto const xp = reconstruct(xi + h, eta), xm = reconstruct(xi - h, eta);
    auto const yp = reconstruct(xi, eta + h), ym = reconstruct(xi, eta - h);
    double const j00 = 2 + 0.1 * eta, j01 = 0.4 + 0.1 * xi;
    double const j10 = 0.2 + 0.2 * eta, j11 = 3 + 0.2 * xi;
    double const curl = (j11 * (xp[1] - xm[1]) - j10 * (yp[1] - ym[1]) + j01 * (xp[0] - xm[0])
                         - j00 * (yp[0] - ym[0]))
                        / (2 * h * (j00 * j11 - j01 * j10));
    double const cochain
            = sil::exterior::CovariantDerivative<X, Y>::cochain_value<1, 1>({0, 1}, 0, edge)[0];
    sil::exterior::CubicalReconstruction<2> const map(positions, {xi, eta});
    EXPECT_NEAR(curl, (cochain * map.basis<2>({0, 1}, 0, {0, 1})), 1e-10);
}

TEST(CovariantDerivative, CurvatureIsNotForcedToZero)
{
    using D = sil::exterior::CovariantDerivative<X, Y>;
    auto transport = [](std::size_t to, std::size_t from) {
        return std::array<std::array<double, 1>, 1> {
                {{to == 1 && from == 3 ? 2.0 : (to == 3 && from == 1 ? 0.5 : 1.0)}}};
    };
    auto section = [](std::size_t, std::size_t) { return std::array<double, 1> {1}; };
    auto edge = [&](std::size_t mask, std::size_t v) {
        return D::cochain_value<0, 1>({mask == 1 ? 0UL : 1UL}, v, section, transport);
    };
    EXPECT_DOUBLE_EQ((D::cochain_value<1, 1>({0, 1}, 0, edge, transport)[0]), 1);
}

TEST(CovariantDerivative, NonzeroConnectionConverges)
{
    // Smooth scalar connection A = 0.7 dx: covariant derivative of u=1
    // and exterior covariant derivative of omega=dy both have coefficient 0.7.
    double previous_error = 1;
    for (double h : {0.1, 0.05, 0.025, 0.0125}) {
        auto transport = [h](std::size_t to, std::size_t from) {
            return std::array<std::array<double, 1>, 1> {
                    {{std::exp(0.7 * h * (double(from & 1) - double(to & 1)))}}};
        };
        std::array<std::array<double, 2>, 4> const positions {{{0, 0}, {h, 0}, {0, h}, {h, h}}};
        auto const stencil = sil::exterior::CovariantDerivative<X, Y>::cell_stencil<
                1>(sil::exterior::CubicalReconstruction<2>(positions, {0.5, 0.5}), transport);
        double value = 0;
        for (std::size_t v = 0; v < 4; ++v)
            value += stencil[v][0][0][0];
        auto omega = [h](std::size_t mask, std::size_t) {
            return std::array<double, 1> {mask == 2 ? h : 0};
        };
        double const curl = sil::exterior::CovariantDerivative<X, Y>::
                                    cochain_value<1, 1>({0, 1}, 0, omega, transport)[0]
                            / (h * h);
        EXPECT_NEAR(value, curl, 2e-14);
        EXPECT_LT(std::abs(value - 0.7), 0.52 * previous_error);
        previous_error = std::abs(value - 0.7);
    }
}

TEST(CovariantDerivative, FormOrientationAndDegenerateGeometry)
{
    std::array<std::array<double, 2>, 4> positions {{{0, 0}, {2, 0}, {0, 3}, {2, 3}}};
    sil::exterior::CubicalReconstruction<2> const reconstruction(positions, {0.2, 0.7});
    EXPECT_NEAR((reconstruction.basis<2>({0, 1}, 0, {0, 1})), 1.0 / 6, 1e-15);
    EXPECT_NEAR((reconstruction.basis<2>({0, 1}, 0, {1, 0})), -1.0 / 6, 1e-15);
    std::swap(positions[1], positions[3]);
    EXPECT_THROW(
            sil::exterior::CubicalReconstruction<2>::check_orientation(positions),
            std::runtime_error);
    positions.fill({0, 0});
    EXPECT_FALSE((sil::exterior::CubicalReconstruction<2>(positions, {0.5, 0.5}).valid()));
}

TEST(CovariantDerivative, ExecutionSpaceEvaluation)
{
    Kokkos::View<double*> output("covariant_derivative", 1);
    Kokkos::parallel_for("covariant_derivative", 1, ExecutionSpaceEvaluation {output});
    auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), output);
    EXPECT_DOUBLE_EQ(host(0), 0.5);
}
