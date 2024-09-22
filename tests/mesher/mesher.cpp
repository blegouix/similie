// SPDX-License-Identifier: GPL-3.0

#include <ddc/ddc.hpp>

#include <gtest/gtest.h>

#include "mesher.hpp"

static constexpr std::size_t s_degree_x = 3;

struct X
{
    static constexpr bool PERIODIC = false;
};

using MesherX = Mesher<X, s_degree_x>;

struct BSplinesX : MesherX::bsplines_type
{
};

struct DDimX : MesherX::discrete_dimension_type
{
};

TEST(Mesher, 1D)
{
    MesherX mesher;
    ddc::DiscreteDomain<DDimX> dom_x = mesher.template mesh<DDimX, BSplinesX>(0., 1., 10);
    printf("end of test");
}
