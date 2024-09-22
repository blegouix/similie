// SPDX-License-Identifier: GPL-3.0

#include <ddc/ddc.hpp>

#include <gtest/gtest.h>

#include "mesher.hpp"

static constexpr std::size_t s_degree_x = 3;

struct X
{
    static constexpr bool PERIODIC = false;
};

struct BSplinesX : ddc::UniformBSplines<X, s_degree_x>
{
};

TEST(Mesher, Init)
{
    Mesher<BSplinesX> mesher();
    printf("end of test");
}
