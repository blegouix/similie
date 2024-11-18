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
    sil::exterior::Chain chain
            = sil::exterior::Chain(
                      sil::exterior::
                              Simplex(ddc::DiscreteElement<DDimT, DDimX, DDimY, DDimZ> {0, 0, 0, 0},
                                      ddc::DiscreteVector<DDimX, DDimY> {1, 1}))
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
    EXPECT_EQ(cosimplex(), 4.);
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

struct Mu : sil::tensor::TensorNaturalIndex<X, Y>
{
};

struct Nu : sil::tensor::TensorNaturalIndex<X, Y>
{
};

using Form1Index = sil::tensor::TensorAntisymmetricIndex<Nu>;
using Form2Index = sil::tensor::TensorAntisymmetricIndex<Mu, Nu>;

TEST(ExteriorDerivative, Rotational)
{
    sil::tensor::TensorAccessor<Form1Index> tensor_accessor;
    ddc::DiscreteDomain<Form1Index, DDimX, DDimY>
            dom(tensor_accessor.mem_domain(),
                ddc::DiscreteDomain(
                        ddc::DiscreteElement<DDimX, DDimY> {0, 0},
                        ddc::DiscreteVector<DDimX, DDimY> {3, 3}));
    ddc::Chunk tensor_alloc(dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor<
            double,
            ddc::DiscreteDomain<Form1Index, DDimX, DDimY>,
            Kokkos::layout_right,
            Kokkos::DefaultHostExecutionSpace::memory_space>
            tensor(tensor_alloc);
    ddc::parallel_for_each(
            Kokkos::DefaultHostExecutionSpace(),
            tensor.non_indices_domain(),
            [&](auto elem) {
                tensor(tensor_accessor.access_element<X>(), elem) = 1.;
                tensor(tensor_accessor.access_element<Y>(), elem) = 2.;
            });
    tensor(tensor_accessor.access_element<X>(), ddc::DiscreteElement<DDimX, DDimY> {1, 1}) = 4.;
    tensor(tensor_accessor.access_element<Y>(), ddc::DiscreteElement<DDimX, DDimY> {1, 1}) = 3.;

    sil::tensor::TensorAccessor<Form2Index> derivative_accessor;
    ddc::DiscreteDomain<Form2Index, DDimX, DDimY> derivative_dom(
            derivative_accessor.mem_domain(),
            ddc::DiscreteDomain(
                    ddc::DiscreteElement<DDimX, DDimY> {0, 0},
                    ddc::DiscreteVector<DDimX, DDimY> {2, 2}));
    ddc::Chunk derivative_alloc(derivative_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor<
            double,
            ddc::DiscreteDomain<Form2Index, DDimX, DDimY>,
            Kokkos::layout_right,
            Kokkos::DefaultHostExecutionSpace::memory_space>
            derivative(derivative_alloc);
    ddc::parallel_for_each(
            Kokkos::DefaultHostExecutionSpace(),
            derivative.non_indices_domain(),
            [&](auto elem) { derivative(derivative_accessor.access_element<X, Y>(), elem) = 1.; });
    sil::exterior::deriv<Form1Index, Mu>(derivative, tensor);
    EXPECT_EQ(
            derivative(
                    ddc::DiscreteElement<DDimX, DDimY> {0, 0},
                    derivative_accessor.access_element<X, Y>()),
            0.);
    EXPECT_EQ(
            derivative(
                    ddc::DiscreteElement<DDimX, DDimY> {1, 0},
                    derivative_accessor.access_element<X, Y>()),
            -3.);
    EXPECT_EQ(
            derivative(
                    ddc::DiscreteElement<DDimX, DDimY> {0, 1},
                    derivative_accessor.access_element<X, Y>()),
            1.);
    EXPECT_EQ(
            derivative(
                    ddc::DiscreteElement<DDimX, DDimY> {1, 1},
                    derivative_accessor.access_element<X, Y>()),
            2.);
}
