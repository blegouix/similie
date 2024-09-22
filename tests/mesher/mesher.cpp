// SPDX-License-Identifier: GPL-3.0

#include <ddc/ddc.hpp>

#include <gtest/gtest.h>

#include "mesher.hpp"

static constexpr std::size_t s_degree_x = 3;

struct X
{
    static constexpr bool PERIODIC = true;
};

/*
using MesherX = Mesher1D<s_degree_x, X>;

struct BSplinesX : MesherX::bsplines_type
{
};

struct DDimX : MesherX::discrete_dimension_type
{
};
*/

struct Y
{
    static constexpr bool PERIODIC = true;
};

using MesherXY = Mesher<s_degree_x, X, Y>;

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

/*
TEST(Mesher, 1D)
{
    MesherX mesher;
    ddc::DiscreteDomain<DDimX> dom_x = mesher.template mesh<DDimX, BSplinesX>(0., 1., 10);
}
*/

TEST(Mesher, 2D)
{
    MesherXY mesher;
    std::array<double, 2> lower_bounds({-1., 0.});
    std::array<double, 2> upper_bounds({1., 2.});
    std::array<std::size_t, 2> nb_cells({10, 20});
    ddc::DiscreteDomain<DDimX, DDimY> dom_xy = mesher.template mesh<
            ddc::detail::TypeSeq<DDimX, DDimY>,
            ddc::detail::TypeSeq<BSplinesX, BSplinesY>>(lower_bounds, upper_bounds, nb_cells);

    std::cout << ddc::coordinate(dom_xy.front()) << "\n";
    std::cout << ddc::coordinate(dom_xy.back()) << "\n";
    std::cout << dom_xy.extents() << "\n";
}
