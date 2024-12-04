// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: GPL-3.0-or-later

#include <cmath>

#include <ddc/ddc.hpp>

#include <gtest/gtest.h>

#include "exterior.hpp"

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
    sil::exterior::Chain chain = sil::exterior::Chain(
            sil::exterior::
                    Simplex(ddc::DiscreteElement<DDimT, DDimX, DDimY, DDimZ> {0, 0, 0, 0},
                            ddc::DiscreteVector<DDimX, DDimY> {1, 1}));
    chain = chain
            + sil::exterior::
                    Simplex(ddc::DiscreteElement<DDimT, DDimX, DDimY, DDimZ> {0, 1, 1, 0},
                            ddc::DiscreteVector<DDimX, DDimY> {1, -1})
            + sil::exterior::
                    Simplex(ddc::DiscreteElement<DDimT, DDimX, DDimY, DDimZ> {1, 0, 1, 0},
                            ddc::DiscreteVector<DDimX, DDimY> {1, -1})
            - sil::exterior::
                    Simplex(ddc::DiscreteElement<DDimT, DDimX, DDimY, DDimZ> {0, 0, 0, 0},
                            ddc::DiscreteVector<DDimX, DDimY> {1, 1})
            - sil::exterior::Chain(
                    sil::exterior::
                            Simplex(ddc::DiscreteElement<DDimT, DDimX, DDimY, DDimZ> {0, 0, 0, 1},
                                    ddc::DiscreteVector<DDimX, DDimY> {1, 1}));
    chain.optimize();
    EXPECT_TRUE(
            chain
            == sil::exterior::
                    Chain(sil::exterior::
                                  Simplex(ddc::DiscreteElement<
                                                  DDimT,
                                                  DDimX,
                                                  DDimY,
                                                  DDimZ> {0, 1, 0, 0},
                                          ddc::DiscreteVector<DDimX, DDimY> {1, 1},
                                          true),
                          sil::exterior::
                                  Simplex(ddc::DiscreteElement<
                                                  DDimT,
                                                  DDimX,
                                                  DDimY,
                                                  DDimZ> {1, 0, 0, 0},
                                          ddc::DiscreteVector<DDimX, DDimY> {1, 1},
                                          true),
                          sil::exterior::
                                  Simplex(ddc::DiscreteElement<
                                                  DDimT,
                                                  DDimX,
                                                  DDimY,
                                                  DDimZ> {0, 0, 0, 1},
                                          ddc::DiscreteVector<DDimX, DDimY> {1, 1},
                                          true)));
}

TEST(Boundary, 1Simplex)
{
    sil::exterior::Simplex
            simplex(ddc::DiscreteElement<DDimT, DDimX, DDimY, DDimZ> {0, 0, 0, 0},
                    ddc::DiscreteVector<DDimX> {1});
    sil::exterior::Chain chain = sil::exterior::boundary(simplex);
    EXPECT_TRUE(
            chain
            == sil::exterior::
                    Chain(sil::exterior::
                                  Simplex(ddc::DiscreteElement<
                                                  DDimT,
                                                  DDimX,
                                                  DDimY,
                                                  DDimZ> {0, 0, 0, 0},
                                          ddc::DiscreteVector<> {},
                                          true),
                          sil::exterior::
                                  Simplex(ddc::DiscreteElement<
                                                  DDimT,
                                                  DDimX,
                                                  DDimY,
                                                  DDimZ> {0, 1, 0, 0},
                                          ddc::DiscreteVector<> {})));
}

TEST(Boundary, 2Simplex)
{
    sil::exterior::Simplex
            simplex(ddc::DiscreteElement<DDimT, DDimX, DDimY, DDimZ> {0, 0, 0, 0},
                    ddc::DiscreteVector<DDimT, DDimX> {1, 1});
    sil::exterior::Chain chain = sil::exterior::boundary(simplex);
    EXPECT_TRUE(
            chain
            == sil::exterior::
                    Chain(sil::exterior::
                                  Simplex(ddc::DiscreteElement<
                                                  DDimT,
                                                  DDimX,
                                                  DDimY,
                                                  DDimZ> {0, 1, 0, 0},
                                          ddc::DiscreteVector<DDimX> {-1}),
                          sil::exterior::
                                  Simplex(ddc::DiscreteElement<
                                                  DDimT,
                                                  DDimX,
                                                  DDimY,
                                                  DDimZ> {0, 0, 0, 0},
                                          ddc::DiscreteVector<DDimT> {1}),
                          sil::exterior::
                                  Simplex(ddc::DiscreteElement<
                                                  DDimT,
                                                  DDimX,
                                                  DDimY,
                                                  DDimZ> {1, 0, 0, 0},
                                          ddc::DiscreteVector<DDimX> {1}),
                          sil::exterior::
                                  Simplex(ddc::DiscreteElement<
                                                  DDimT,
                                                  DDimX,
                                                  DDimY,
                                                  DDimZ> {1, 1, 0, 0},
                                          ddc::DiscreteVector<DDimT> {-1})));

    sil::exterior::Simplex simplex2(
            ddc::DiscreteElement<DDimT, DDimX, DDimY, DDimZ> {0, 0, 0, 0},
            ddc::DiscreteVector<DDimX, DDimZ> {1, 1});
    sil::exterior::Chain chain2 = sil::exterior::boundary(simplex2);
    EXPECT_TRUE(
            chain2
            == sil::exterior::
                    Chain(sil::exterior::
                                  Simplex(ddc::DiscreteElement<
                                                  DDimT,
                                                  DDimX,
                                                  DDimY,
                                                  DDimZ> {0, 0, 0, 1},
                                          ddc::DiscreteVector<DDimZ> {-1}),
                          sil::exterior::
                                  Simplex(ddc::DiscreteElement<
                                                  DDimT,
                                                  DDimX,
                                                  DDimY,
                                                  DDimZ> {0, 0, 0, 0},
                                          ddc::DiscreteVector<DDimX> {1}),
                          sil::exterior::
                                  Simplex(ddc::DiscreteElement<
                                                  DDimT,
                                                  DDimX,
                                                  DDimY,
                                                  DDimZ> {0, 1, 0, 0},
                                          ddc::DiscreteVector<DDimZ> {1}),
                          sil::exterior::
                                  Simplex(ddc::DiscreteElement<
                                                  DDimT,
                                                  DDimX,
                                                  DDimY,
                                                  DDimZ> {0, 1, 0, 1},
                                          ddc::DiscreteVector<DDimX> {-1})));
}

TEST(Boundary, 3Simplex)
{
    sil::exterior::Simplex
            simplex(ddc::DiscreteElement<DDimT, DDimX, DDimY, DDimZ> {0, 0, 0, 0},
                    ddc::DiscreteVector<DDimT, DDimY, DDimZ> {1, 1, 1});
    sil::exterior::Chain chain = sil::exterior::boundary(simplex);
    EXPECT_TRUE(
            chain
            == sil::exterior::
                    Chain(sil::exterior::
                                  Simplex(ddc::DiscreteElement<
                                                  DDimT,
                                                  DDimX,
                                                  DDimY,
                                                  DDimZ> {0, 0, 0, 0},
                                          ddc::DiscreteVector<DDimY, DDimZ> {1, 1},
                                          true),
                          sil::exterior::
                                  Simplex(ddc::DiscreteElement<
                                                  DDimT,
                                                  DDimX,
                                                  DDimY,
                                                  DDimZ> {0, 0, 0, 0},
                                          ddc::DiscreteVector<DDimT, DDimZ> {1, 1}),
                          sil::exterior::
                                  Simplex(ddc::DiscreteElement<
                                                  DDimT,
                                                  DDimX,
                                                  DDimY,
                                                  DDimZ> {0, 0, 0, 0},
                                          ddc::DiscreteVector<DDimT, DDimY> {1, 1},
                                          true),
                          sil::exterior::
                                  Simplex(ddc::DiscreteElement<
                                                  DDimT,
                                                  DDimX,
                                                  DDimY,
                                                  DDimZ> {1, 0, 1, 1},
                                          ddc::DiscreteVector<DDimY, DDimZ> {-1, -1}),
                          sil::exterior::
                                  Simplex(ddc::DiscreteElement<
                                                  DDimT,
                                                  DDimX,
                                                  DDimY,
                                                  DDimZ> {1, 0, 1, 1},
                                          ddc::DiscreteVector<DDimT, DDimZ> {-1, -1},
                                          true),
                          sil::exterior::
                                  Simplex(ddc::DiscreteElement<
                                                  DDimT,
                                                  DDimX,
                                                  DDimY,
                                                  DDimZ> {1, 0, 1, 1},
                                          ddc::DiscreteVector<DDimT, DDimY> {-1, -1})));
    sil::exterior::Simplex simplex2(
            ddc::DiscreteElement<DDimT, DDimX, DDimY, DDimZ> {0, 0, 0, 0},
            ddc::DiscreteVector<DDimX, DDimY, DDimZ> {1, 1, 1});
    sil::exterior::Chain chain2 = sil::exterior::boundary(simplex2);
    EXPECT_TRUE(
            chain2
            == sil::exterior::
                    Chain(sil::exterior::
                                  Simplex(ddc::DiscreteElement<
                                                  DDimT,
                                                  DDimX,
                                                  DDimY,
                                                  DDimZ> {0, 0, 0, 0},
                                          ddc::DiscreteVector<DDimY, DDimZ> {1, 1},
                                          true),
                          sil::exterior::
                                  Simplex(ddc::DiscreteElement<
                                                  DDimT,
                                                  DDimX,
                                                  DDimY,
                                                  DDimZ> {0, 0, 0, 0},
                                          ddc::DiscreteVector<DDimX, DDimZ> {1, 1}),
                          sil::exterior::
                                  Simplex(ddc::DiscreteElement<
                                                  DDimT,
                                                  DDimX,
                                                  DDimY,
                                                  DDimZ> {0, 0, 0, 0},
                                          ddc::DiscreteVector<DDimX, DDimY> {1, 1},
                                          true),
                          sil::exterior::
                                  Simplex(ddc::DiscreteElement<
                                                  DDimT,
                                                  DDimX,
                                                  DDimY,
                                                  DDimZ> {0, 1, 1, 1},
                                          ddc::DiscreteVector<DDimY, DDimZ> {-1, -1}),
                          sil::exterior::
                                  Simplex(ddc::DiscreteElement<
                                                  DDimT,
                                                  DDimX,
                                                  DDimY,
                                                  DDimZ> {0, 1, 1, 1},
                                          ddc::DiscreteVector<DDimX, DDimZ> {-1, -1},
                                          true),
                          sil::exterior::
                                  Simplex(ddc::DiscreteElement<
                                                  DDimT,
                                                  DDimX,
                                                  DDimY,
                                                  DDimZ> {0, 1, 1, 1},
                                          ddc::DiscreteVector<DDimX, DDimY> {-1, -1})));
}

TEST(Boundary, Chain)
{
    sil::exterior::Chain chain = sil::exterior::
            Chain(sil::exterior::
                          Simplex(ddc::DiscreteElement<DDimT, DDimX, DDimY, DDimZ> {0, 0, 0, 0},
                                  ddc::DiscreteVector<DDimX, DDimY> {1, 1}),
                  sil::exterior::
                          Simplex(ddc::DiscreteElement<DDimT, DDimX, DDimY, DDimZ> {0, 1, 0, 0},
                                  ddc::DiscreteVector<DDimX, DDimY> {1, 1}));
    sil::exterior::Chain boundary_chain = sil::exterior::boundary(chain);
    EXPECT_TRUE(
            boundary_chain
            == sil::exterior::
                    Chain(sil::exterior::
                                  Simplex(ddc::DiscreteElement<
                                                  DDimT,
                                                  DDimX,
                                                  DDimY,
                                                  DDimZ> {0, 0, 1, 0},
                                          ddc::DiscreteVector<DDimY> {-1}),
                          sil::exterior::
                                  Simplex(ddc::DiscreteElement<
                                                  DDimT,
                                                  DDimX,
                                                  DDimY,
                                                  DDimZ> {0, 0, 0, 0},
                                          ddc::DiscreteVector<DDimX> {1}),
                          sil::exterior::
                                  Simplex(ddc::DiscreteElement<
                                                  DDimT,
                                                  DDimX,
                                                  DDimY,
                                                  DDimZ> {0, 1, 1, 0},
                                          ddc::DiscreteVector<DDimX> {-1}),
                          sil::exterior::
                                  Simplex(ddc::DiscreteElement<
                                                  DDimT,
                                                  DDimX,
                                                  DDimY,
                                                  DDimZ> {0, 1, 0, 0},
                                          ddc::DiscreteVector<DDimX> {1}),
                          sil::exterior::
                                  Simplex(ddc::DiscreteElement<
                                                  DDimT,
                                                  DDimX,
                                                  DDimY,
                                                  DDimZ> {0, 2, 0, 0},
                                          ddc::DiscreteVector<DDimY> {1}),
                          sil::exterior::
                                  Simplex(ddc::DiscreteElement<
                                                  DDimT,
                                                  DDimX,
                                                  DDimY,
                                                  DDimZ> {0, 2, 1, 0},
                                          ddc::DiscreteVector<DDimX> {-1})));
}

TEST(Boundary, PoincarreLemma2)
{
    sil::exterior::Simplex simplex = sil::exterior::
            Simplex(ddc::DiscreteElement<DDimT, DDimX, DDimY, DDimZ> {0, 0, 0, 0},
                    ddc::DiscreteVector<DDimX, DDimY> {1, 1});
    sil::exterior::Chain boundary_chain = sil::exterior::boundary(simplex);
    sil::exterior::Chain boundary_chain2 = sil::exterior::boundary(boundary_chain);
    auto empty_chain
            = sil::exterior::Chain<sil::exterior::Simplex<0, DDimT, DDimX, DDimY, DDimZ>> {};
    EXPECT_TRUE(boundary_chain2 == empty_chain);
}

TEST(Boundary, PoincarreLemma3)
{
    sil::exterior::Simplex simplex = sil::exterior::
            Simplex(ddc::DiscreteElement<DDimT, DDimX, DDimY, DDimZ> {0, 0, 0, 0},
                    ddc::DiscreteVector<DDimX, DDimY, DDimZ> {1, 1, 1});
    sil::exterior::Chain boundary_chain = sil::exterior::boundary(simplex);
    sil::exterior::Chain boundary_chain2 = sil::exterior::boundary(boundary_chain);
    auto empty_chain
            = sil::exterior::Chain<sil::exterior::Simplex<1, DDimT, DDimX, DDimY, DDimZ>> {};
    EXPECT_TRUE(boundary_chain2 == empty_chain);
}

TEST(Boundary, PoincarreLemma4)
{
    sil::exterior::Simplex simplex = sil::exterior::
            Simplex(ddc::DiscreteElement<DDimT, DDimX, DDimY, DDimZ> {0, 0, 0, 0},
                    ddc::DiscreteVector<DDimT, DDimX, DDimY, DDimZ> {1, 1, 1, 1});
    sil::exterior::Chain boundary_chain = sil::exterior::boundary(simplex);
    sil::exterior::Chain boundary_chain2 = sil::exterior::boundary(boundary_chain);
    auto empty_chain
            = sil::exterior::Chain<sil::exterior::Simplex<2, DDimT, DDimX, DDimY, DDimZ>> {};
    EXPECT_TRUE(boundary_chain2 == empty_chain);
}

TEST(Form, Alias)
{
    sil::exterior::Chain chain(
            sil::exterior::
                    Simplex(ddc::DiscreteElement<DDimT, DDimX, DDimY, DDimZ> {0, 1, 0, 0},
                            ddc::DiscreteVector<DDimX, DDimY> {1, 1}));
    // Unfortunately CTAD cannot deduce template arguments
    sil::exterior::Form<typename decltype(chain)::simplex_type> cosimplex(chain[0], 0.);
    sil::exterior::Form<decltype(chain)> cochain(chain, 0.);
}

TEST(Cochain, Test)
{
    sil::exterior::Chain
            chain(sil::exterior::
                          Simplex(ddc::DiscreteElement<DDimT, DDimX, DDimY, DDimZ> {0, 1, 0, 0},
                                  ddc::DiscreteVector<DDimX, DDimY> {1, 1}),
                  sil::exterior::
                          Simplex(ddc::DiscreteElement<DDimT, DDimX, DDimY, DDimZ> {0, 0, 0, 0},
                                  ddc::DiscreteVector<DDimX, DDimY> {1, 1}),
                  sil::exterior::
                          Simplex(ddc::DiscreteElement<DDimT, DDimX, DDimY, DDimZ> {0, 0, 0, 1},
                                  ddc::DiscreteVector<DDimX, DDimY> {1, 1},
                                  true));
    sil::exterior::Cochain cochain(chain, 0., 1., 2.);
    EXPECT_EQ(cochain.integrate(), -1.);
}

TEST(Coboundary, Test)
{
    sil::exterior::Simplex
            simplex(ddc::DiscreteElement<DDimT, DDimX, DDimY, DDimZ> {0, 1, 0, 0},
                    ddc::DiscreteVector<DDimX, DDimY> {1, 1});
    sil::exterior::Chain simplex_boundary = boundary(simplex);
    sil::exterior::Cochain cochain_boundary(simplex_boundary, 5., 8., 3., 2.);
    sil::exterior::Cosimplex cosimplex = sil::exterior::coboundary(cochain_boundary);
    EXPECT_EQ(cosimplex.simplex(), simplex);
    EXPECT_EQ(cosimplex.value(), 4.);
}

TEST(LocalChain, Test)
{
    sil::exterior::LocalChain
            chain(sil::exterior::
                          Simplex(ddc::DiscreteElement<DDimT, DDimX, DDimY, DDimZ> {0, 0, 0, 0},
                                  ddc::DiscreteVector<DDimX> {1}),
                  sil::exterior::
                          Simplex(ddc::DiscreteElement<DDimT, DDimX, DDimY, DDimZ> {0, 0, 0, 0},
                                  ddc::DiscreteVector<DDimY> {1}));
    chain = chain
            + sil::exterior::
                    Simplex(ddc::DiscreteElement<DDimT, DDimX, DDimY, DDimZ> {0, 0, 0, 0},
                            ddc::DiscreteVector<DDimT> {1});
    chain = chain
            + sil::exterior::LocalChain(
                    sil::exterior::
                            Simplex(ddc::DiscreteElement<DDimT, DDimX, DDimY, DDimZ> {0, 0, 0, 0},
                                    ddc::DiscreteVector<DDimZ> {1}));
    EXPECT_TRUE(
            chain
            == sil::exterior::LocalChain(
                    sil::exterior::
                            Simplex(ddc::DiscreteElement<DDimT, DDimX, DDimY, DDimZ> {0, 0, 0, 0},
                                    ddc::DiscreteVector<DDimX> {1}),
                    sil::exterior::
                            Simplex(ddc::DiscreteElement<DDimT, DDimX, DDimY, DDimZ> {0, 0, 0, 0},
                                    ddc::DiscreteVector<DDimY> {1}),
                    sil::exterior::
                            Simplex(ddc::DiscreteElement<DDimT, DDimX, DDimY, DDimZ> {0, 0, 0, 0},
                                    ddc::DiscreteVector<DDimT> {1}),
                    sil::exterior::
                            Simplex(ddc::DiscreteElement<DDimT, DDimX, DDimY, DDimZ> {0, 0, 0, 0},
                                    ddc::DiscreteVector<DDimZ> {1})));
}

TEST(LocalCochain, Test)
{
    sil::exterior::LocalChain
            chain(sil::exterior::
                          Simplex(ddc::DiscreteElement<DDimT, DDimX, DDimY, DDimZ> {0, 0, 0, 0},
                                  ddc::DiscreteVector<DDimX> {1}),
                  sil::exterior::
                          Simplex(ddc::DiscreteElement<DDimT, DDimX, DDimY, DDimZ> {0, 0, 0, 0},
                                  ddc::DiscreteVector<DDimY> {1}));
    sil::exterior::Cochain cochain(chain, 1., 2.);
    EXPECT_EQ(cochain.integrate(), 3.);
}
