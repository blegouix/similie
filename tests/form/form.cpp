// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: GPL-3.0-or-later

#include <cmath>

#include <ddc/ddc.hpp>

#include <gtest/gtest.h>

#include "form.hpp"

struct T
{
};

struct X
{
};

struct Y
{
};

struct Z
{
};

struct DDimT : ddc::UniformPointSampling<T>
{
};

struct DDimX : ddc::UniformPointSampling<X>
{
};

struct DDimY : ddc::UniformPointSampling<Y>
{
};

struct DDimZ : ddc::UniformPointSampling<Z>
{
};

TEST(Chain, Optimization)
{
    sil::form::Chain chain
            = sil::form::Chain(
                      sil::form::
                              Simplex(ddc::DiscreteElement<DDimT, DDimX, DDimY, DDimZ> {0, 0, 0, 0},
                                      ddc::DiscreteVector<DDimX, DDimY> {1, 1}))
              + sil::form::
                      Simplex(ddc::DiscreteElement<DDimT, DDimX, DDimY, DDimZ> {0, 1, 1, 0},
                              ddc::DiscreteVector<DDimX, DDimY> {1, -1})
              + sil::form::
                      Simplex(ddc::DiscreteElement<DDimT, DDimX, DDimY, DDimZ> {0, 0, 1, 0},
                              ddc::DiscreteVector<DDimX, DDimY> {1, -1})
              - sil::form::
                      Simplex(ddc::DiscreteElement<DDimT, DDimX, DDimY, DDimZ> {0, 0, 0, 0},
                              ddc::DiscreteVector<DDimX, DDimY> {1, 1})
              - sil::form::Chain(
                      sil::form::
                              Simplex(ddc::DiscreteElement<DDimT, DDimX, DDimY, DDimZ> {0, 0, 0, 1},
                                      ddc::DiscreteVector<DDimX, DDimY> {1, 1}));
    chain.optimize();
    EXPECT_TRUE(
            chain
            == sil::form::
                    Chain(sil::form::
                                  Simplex(ddc::DiscreteElement<
                                                  DDimT,
                                                  DDimX,
                                                  DDimY,
                                                  DDimZ> {0, 1, 0, 0},
                                          ddc::DiscreteVector<DDimX, DDimY> {1, 1}),
                          sil::form::
                                  Simplex(ddc::DiscreteElement<
                                                  DDimT,
                                                  DDimX,
                                                  DDimY,
                                                  DDimZ> {0, 0, 0, 0},
                                          ddc::DiscreteVector<DDimX, DDimY> {1, 1}),
                          sil::form::
                                  Simplex(ddc::DiscreteElement<
                                                  DDimT,
                                                  DDimX,
                                                  DDimY,
                                                  DDimZ> {0, 0, 0, 1},
                                          ddc::DiscreteVector<DDimX, DDimY> {1, 1},
                                          true)));
}
