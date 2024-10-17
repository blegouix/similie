// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: GPL-3.0

#include <ddc/ddc.hpp>

#include <gtest/gtest.h>

#include "mesher.hpp"

// 1D test
static constexpr std::size_t s_degree = 3;

struct X
{
    static constexpr bool PERIODIC = false;
};

using MesherX = sil::mesher::detail::Mesher1D<s_degree, X>;

struct BSplinesX : MesherX::bsplines_type
{
};

struct DDimX : MesherX::discrete_dimension_type
{
};

TEST(Mesher, 1D)
{
    MesherX mesher;
    ddc::DiscreteDomain<DDimX> mesh_x = mesher.template mesh<DDimX, BSplinesX>(
            ddc::Coordinate<X>(0.),
            ddc::Coordinate<X>(1.),
            ddc::DiscreteVector<DDimX>(10));

    EXPECT_TRUE(
            mesh_x
            == ddc::DiscreteDomain<
                    DDimX>(ddc::DiscreteElement<DDimX>(0), ddc::DiscreteVector<DDimX>(13)));
    EXPECT_TRUE(ddc::coordinate(mesh_x.front()) == 0.);
}
