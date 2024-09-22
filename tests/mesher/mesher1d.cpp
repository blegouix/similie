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
    ddc::DiscreteDomain<DDimX> dom_x = mesher.template mesh<DDimX, BSplinesX>(0., 1., 10);
}
