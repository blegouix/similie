// SPDX-License-Identifier: GPL-3.0

#include <ddc/ddc.hpp>

#include <gtest/gtest.h>

#include "mesher.hpp"

// 2D test
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
    std::array<double, 2> lower_bounds({0., 0.});
    std::array<double, 2> upper_bounds({1., 1.});
    std::array<std::size_t, 2> nb_cells({10, 10});
    ddc::DiscreteDomain<DDimX, DDimY> mesh_xy = mesher.template mesh<
            ddc::detail::TypeSeq<DDimX, DDimY>,
            ddc::detail::TypeSeq<BSplinesX, BSplinesY>>(lower_bounds, upper_bounds, nb_cells);

    std::cout << ddc::coordinate(mesh_xy.front()) << "\n";
    std::cout << ddc::coordinate(mesh_xy.back()) << "\n";
    std::cout << mesh_xy.extents() << "\n";
}
