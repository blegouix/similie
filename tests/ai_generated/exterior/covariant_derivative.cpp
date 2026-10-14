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
} // namespace

TEST(CovariantDerivative, CochainStokesAndFlatSquareZero)
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
