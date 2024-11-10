// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: GPL-3.0-or-later

#include <ddc/ddc.hpp>

#include <gtest/gtest.h>

#include "mesher.hpp"

static constexpr std::size_t s_degree = 3;

struct X
{
    static constexpr bool PERIODIC = false;
};

struct Y
{
    static constexpr bool PERIODIC = false;
};

using MesherXY = sil::mesher::Mesher<s_degree, X, Y>;

struct BSplinesX : MesherXY::template bsplines_type<X>
{
};

struct DDimX : MesherXY::template discrete_dimension_type<X>
{
};

struct BSplinesY : MesherXY::template bsplines_type<Y>
{
};

struct DDimY : MesherXY::template discrete_dimension_type<Y>
{
};


TEST(Mesher, 2D)
{
    MesherXY mesher;
    ddc::Coordinate<X, Y> lower_bounds(0., 0.);
    ddc::Coordinate<X, Y> upper_bounds(1., 1.);
    ddc::DiscreteVector<DDimX, DDimY> nb_cells(10, 10);
    ddc::DiscreteDomain<DDimX, DDimY> mesh_xy = mesher.template mesh<
            ddc::detail::TypeSeq<DDimX, DDimY>,
            ddc::detail::TypeSeq<BSplinesX, BSplinesY>>(lower_bounds, upper_bounds, nb_cells);

    EXPECT_TRUE(
            (mesh_xy
             == ddc::DiscreteDomain<DDimX, DDimY>(
                     ddc::DiscreteElement<DDimX, DDimY>(0, 0),
                     ddc::DiscreteVector<DDimX, DDimY>(13, 13))));
    EXPECT_TRUE((ddc::coordinate(mesh_xy.front()) == ddc::Coordinate<X, Y>(0., 0.)));
    EXPECT_TRUE((ddc::coordinate(mesh_xy.back()) == ddc::Coordinate<X, Y>(1., 1.)));
}
