// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: MIT

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
    sil::exterior::Chain chain = sil::exterior::
            Chain(Kokkos::View<
                          sil::exterior::Simplex<2, DDimT, DDimX, DDimY, DDimZ>*,
                          Kokkos::LayoutRight,
                          Kokkos::HostSpace>("", 5),
                  sil::exterior::
                          Simplex(ddc::DiscreteElement<DDimT, DDimX, DDimY, DDimZ> {0, 0, 0, 0},
                                  ddc::DiscreteVector<DDimX, DDimY> {1, 1}));
    sil::exterior::Chain
            chain2(Kokkos::View<
                           sil::exterior::Simplex<2, DDimT, DDimX, DDimY, DDimZ>*,
                           Kokkos::LayoutRight,
                           Kokkos::HostSpace>("", 1),
                   1);
    chain2[0] = sil::exterior::
            Simplex(ddc::DiscreteElement<DDimT, DDimX, DDimY, DDimZ> {0, 0, 0, 1},
                    ddc::DiscreteVector<DDimX, DDimY> {1, 1});
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
            - chain2;
    chain.optimize();
    chain.resize();
    EXPECT_TRUE(
            chain
            == sil::exterior::
                    Chain(Kokkos::View<
                                  sil::exterior::Simplex<2, DDimT, DDimX, DDimY, DDimZ>*,
                                  Kokkos::LayoutRight,
                                  Kokkos::HostSpace>("", 3),
                          sil::exterior::
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
    sil::exterior::Chain chain = sil::exterior::boundary(
            Kokkos::View<
                    sil::exterior::Simplex<0, DDimT, DDimX, DDimY, DDimZ>*,
                    Kokkos::LayoutRight,
                    Kokkos::HostSpace>("", 2),
            simplex);
    EXPECT_TRUE(
            chain
            == sil::exterior::
                    Chain(Kokkos::View<
                                  sil::exterior::Simplex<0, DDimT, DDimX, DDimY, DDimZ>*,
                                  Kokkos::LayoutRight,
                                  Kokkos::HostSpace>("", 2),
                          sil::exterior::
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
    sil::exterior::Chain chain = sil::exterior::boundary(
            Kokkos::View<
                    sil::exterior::Simplex<1, DDimT, DDimX, DDimY, DDimZ>*,
                    Kokkos::LayoutRight,
                    Kokkos::HostSpace>("", 4),
            simplex);
    EXPECT_TRUE(
            chain
            == sil::exterior::
                    Chain(Kokkos::View<
                                  sil::exterior::Simplex<1, DDimT, DDimX, DDimY, DDimZ>*,
                                  Kokkos::LayoutRight,
                                  Kokkos::HostSpace>("", 4),
                          sil::exterior::
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
    sil::exterior::Chain chain2 = sil::exterior::boundary(
            Kokkos::View<
                    sil::exterior::Simplex<1, DDimT, DDimX, DDimY, DDimZ>*,
                    Kokkos::LayoutRight,
                    Kokkos::HostSpace>("", 4),
            simplex2);
    EXPECT_TRUE(
            chain2
            == sil::exterior::
                    Chain(Kokkos::View<
                                  sil::exterior::Simplex<1, DDimT, DDimX, DDimY, DDimZ>*,
                                  Kokkos::LayoutRight,
                                  Kokkos::HostSpace>("", 4),
                          sil::exterior::
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
    sil::exterior::Chain chain = sil::exterior::boundary(
            Kokkos::View<
                    sil::exterior::Simplex<2, DDimT, DDimX, DDimY, DDimZ>*,
                    Kokkos::LayoutRight,
                    Kokkos::HostSpace>("", 6),
            simplex);
    EXPECT_TRUE(
            chain
            == sil::exterior::
                    Chain(Kokkos::View<
                                  sil::exterior::Simplex<2, DDimT, DDimX, DDimY, DDimZ>*,
                                  Kokkos::LayoutRight,
                                  Kokkos::HostSpace>("", 6),
                          sil::exterior::
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
    sil::exterior::Chain chain2 = sil::exterior::boundary(
            Kokkos::View<
                    sil::exterior::Simplex<2, DDimT, DDimX, DDimY, DDimZ>*,
                    Kokkos::LayoutRight,
                    Kokkos::HostSpace>("", 6),
            simplex2);
    EXPECT_TRUE(
            chain2
            == sil::exterior::
                    Chain(Kokkos::View<
                                  sil::exterior::Simplex<2, DDimT, DDimX, DDimY, DDimZ>*,
                                  Kokkos::LayoutRight,
                                  Kokkos::HostSpace>("", 6),
                          sil::exterior::
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
            Chain(Kokkos::View<
                          sil::exterior::Simplex<2, DDimT, DDimX, DDimY, DDimZ>*,
                          Kokkos::LayoutRight,
                          Kokkos::HostSpace>("", 2),
                  sil::exterior::
                          Simplex(ddc::DiscreteElement<DDimT, DDimX, DDimY, DDimZ> {0, 0, 0, 0},
                                  ddc::DiscreteVector<DDimX, DDimY> {1, 1}),
                  sil::exterior::
                          Simplex(ddc::DiscreteElement<DDimT, DDimX, DDimY, DDimZ> {0, 1, 0, 0},
                                  ddc::DiscreteVector<DDimX, DDimY> {1, 1}));
    sil::exterior::Chain boundary_chain = sil::exterior::boundary(
            Kokkos::View<
                    sil::exterior::Simplex<1, DDimT, DDimX, DDimY, DDimZ>*,
                    Kokkos::LayoutRight,
                    Kokkos::HostSpace>("", 8),
            chain);
    EXPECT_TRUE(
            boundary_chain
            == sil::exterior::
                    Chain(Kokkos::View<
                                  sil::exterior::Simplex<1, DDimT, DDimX, DDimY, DDimZ>*,
                                  Kokkos::LayoutRight,
                                  Kokkos::HostSpace>("", 8),
                          sil::exterior::
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
    sil::exterior::Chain boundary_chain = sil::exterior::boundary(
            Kokkos::View<
                    sil::exterior::Simplex<1, DDimT, DDimX, DDimY, DDimZ>*,
                    Kokkos::LayoutRight,
                    Kokkos::HostSpace>("", 4),
            simplex);
    sil::exterior::Chain boundary_chain2 = sil::exterior::boundary(
            Kokkos::View<
                    sil::exterior::Simplex<0, DDimT, DDimX, DDimY, DDimZ>*,
                    Kokkos::LayoutRight,
                    Kokkos::HostSpace>("", 8),
            boundary_chain);
    auto empty_chain
            = sil::exterior::Chain<sil::exterior::Simplex<0, DDimT, DDimX, DDimY, DDimZ>> {};
    EXPECT_TRUE(boundary_chain2 == empty_chain);
}

TEST(Boundary, PoincarreLemma3)
{
    sil::exterior::Simplex simplex = sil::exterior::
            Simplex(ddc::DiscreteElement<DDimT, DDimX, DDimY, DDimZ> {0, 0, 0, 0},
                    ddc::DiscreteVector<DDimX, DDimY, DDimZ> {1, 1, 1});
    sil::exterior::Chain boundary_chain = sil::exterior::boundary(
            Kokkos::View<
                    sil::exterior::Simplex<2, DDimT, DDimX, DDimY, DDimZ>*,
                    Kokkos::LayoutRight,
                    Kokkos::HostSpace>("", 6),
            simplex);
    sil::exterior::Chain boundary_chain2 = sil::exterior::boundary(
            Kokkos::View<
                    sil::exterior::Simplex<1, DDimT, DDimX, DDimY, DDimZ>*,
                    Kokkos::LayoutRight,
                    Kokkos::HostSpace>("", 24),
            boundary_chain);
    auto empty_chain
            = sil::exterior::Chain<sil::exterior::Simplex<1, DDimT, DDimX, DDimY, DDimZ>> {};
    EXPECT_TRUE(boundary_chain2 == empty_chain);
}

TEST(Boundary, PoincarreLemma4)
{
    sil::exterior::Simplex simplex = sil::exterior::
            Simplex(ddc::DiscreteElement<DDimT, DDimX, DDimY, DDimZ> {0, 0, 0, 0},
                    ddc::DiscreteVector<DDimT, DDimX, DDimY, DDimZ> {1, 1, 1, 1});
    sil::exterior::Chain boundary_chain = sil::exterior::boundary(
            Kokkos::View<
                    sil::exterior::Simplex<3, DDimT, DDimX, DDimY, DDimZ>*,
                    Kokkos::LayoutRight,
                    Kokkos::HostSpace>("", 8),
            simplex);
    sil::exterior::Chain boundary_chain2 = sil::exterior::boundary(
            Kokkos::View<
                    sil::exterior::Simplex<2, DDimT, DDimX, DDimY, DDimZ>*,
                    Kokkos::LayoutRight,
                    Kokkos::HostSpace>("", 64),
            boundary_chain);
    auto empty_chain
            = sil::exterior::Chain<sil::exterior::Simplex<2, DDimT, DDimX, DDimY, DDimZ>> {};
    EXPECT_TRUE(boundary_chain2 == empty_chain);
}

TEST(Form, Alias)
{
    sil::exterior::Chain
            chain(Kokkos::View<
                          sil::exterior::Simplex<2, DDimT, DDimX, DDimY, DDimZ>*,
                          Kokkos::LayoutRight,
                          Kokkos::HostSpace>("", 1),
                  sil::exterior::
                          Simplex(ddc::DiscreteElement<DDimT, DDimX, DDimY, DDimZ> {0, 1, 0, 0},
                                  ddc::DiscreteVector<DDimX, DDimY> {1, 1}));
    // Unfortunately CTAD cannot deduce template arguments
    sil::exterior::Form<typename decltype(chain)::simplex_type> cosimplex(chain[0], 0.);
    sil::exterior::Form<decltype(chain)>
            cochain(chain,
                    Kokkos::View<double*, Kokkos::LayoutRight, Kokkos::HostSpace>("", 1),
                    0.);
}

TEST(Form, TensorFormDeriv)
{
    using DummyIndex = sil::tensor::Covariant<sil::tensor::ScalarIndex>;
    using XDualizer = sil::mesher::HalfShiftDualizer<X>;
    using YDualizer = sil::mesher::HalfShiftDualizer<Y>;
    using DDimXDual = sil::mesher::dual_discrete_dimension_t<XDualizer, DDimX>;
    using DDimYDual = sil::mesher::dual_discrete_dimension_t<YDualizer, DDimY>;

    auto const x_dom = ddc::init_discrete_space<DDimX>(DDimX::init<DDimX>(
            ddc::Coordinate<X>(0.),
            ddc::Coordinate<X>(1.),
            ddc::DiscreteVector<DDimX>(6)));
    auto const y_dom = ddc::init_discrete_space<DDimY>(DDimY::init<DDimY>(
            ddc::Coordinate<Y>(0.),
            ddc::Coordinate<Y>(1.),
            ddc::DiscreteVector<DDimY>(6)));
    ddc::DiscreteDomain<DDimX, DDimY> const mesh(x_dom, y_dom);
    XDualizer const x_dualizer;
    YDualizer const y_dualizer;
    ddc::DiscreteDomain<DDimXDual, DDimY> const x_face_dom = x_dualizer(mesh);
    ddc::DiscreteDomain<DDimX, DDimYDual> const y_face_dom = y_dualizer(mesh);

    [[maybe_unused]] sil::tensor::TensorAccessor<DummyIndex> scalar_accessor;
    ddc::Chunk scalar_alloc(
            ddc::DiscreteDomain<DDimX, DDimY, DummyIndex>(mesh, scalar_accessor.domain()),
            ddc::HostAllocator<double>());
    sil::tensor::Tensor scalar(scalar_alloc);
    ddc::parallel_for_each(
            Kokkos::DefaultHostExecutionSpace(),
            mesh,
            KOKKOS_LAMBDA(ddc::DiscreteElement<DDimX, DDimY> elem) {
                double const x = ddc::coordinate(ddc::DiscreteElement<DDimX>(elem));
                double const y = ddc::coordinate(ddc::DiscreteElement<DDimY>(elem));
                scalar(elem, ddc::DiscreteElement<DummyIndex>(0)) = x * x + y;
            });

    ddc::Chunk grad_x_alloc(
            ddc::DiscreteDomain<DDimXDual, DDimY, DummyIndex>(
                    x_face_dom,
                    scalar_accessor.domain()),
            ddc::HostAllocator<double>());
    sil::tensor::Tensor grad_x(grad_x_alloc);
    ddc::Chunk grad_y_alloc(
            ddc::DiscreteDomain<DDimX, DDimYDual, DummyIndex>(
                    y_face_dom,
                    scalar_accessor.domain()),
            ddc::HostAllocator<double>());
    sil::tensor::Tensor grad_y(grad_y_alloc);
    auto grad_form = sil::exterior::make_tensor_form(
            sil::exterior::component<X>(grad_x),
            sil::exterior::component<Y>(grad_y));

    sil::exterior::deriv(Kokkos::DefaultHostExecutionSpace(), grad_form, scalar);

    ddc::DiscreteElement<DDimXDual, DDimY> const x_face_elem
            = x_face_dom.front() + ddc::DiscreteVector<DDimXDual, DDimY>(2, 2);
    ddc::DiscreteElement<DDimX, DDimYDual> const y_face_elem
            = y_face_dom.front() + ddc::DiscreteVector<DDimX, DDimYDual>(2, 2);
    EXPECT_NEAR(grad_x(x_face_elem, ddc::DiscreteElement<DummyIndex>(0)), 1.0, 1e-12);
    EXPECT_NEAR(grad_y(y_face_elem, ddc::DiscreteElement<DummyIndex>(0)), 1.0, 1e-12);
}

TEST(Form, TensorFormCodifferential)
{
    struct DDimX2 : ddc::UniformPointSampling<X>
    {
    };
    struct DDimY2 : ddc::UniformPointSampling<Y>
    {
    };
    using MetricIndex = sil::tensor::TensorIdentityIndex<
            sil::tensor::Covariant<sil::tensor::MetricIndex1<X, Y>>,
            sil::tensor::Covariant<sil::tensor::MetricIndex2<X, Y>>>;
    using InverseMetricIndex = sil::tensor::upper_t<MetricIndex>;
    using DummyIndex = sil::tensor::Covariant<sil::tensor::ScalarIndex>;
    using XDualizer = sil::mesher::HalfShiftDualizer<X>;
    using YDualizer = sil::mesher::HalfShiftDualizer<Y>;
    using DDimXDual = sil::mesher::dual_discrete_dimension_t<XDualizer, DDimX2>;
    using DDimYDual = sil::mesher::dual_discrete_dimension_t<YDualizer, DDimY2>;

    auto const x_dom = ddc::init_discrete_space<DDimX2>(DDimX2::init<DDimX2>(
            ddc::Coordinate<X>(0.),
            ddc::Coordinate<X>(1.),
            ddc::DiscreteVector<DDimX2>(6)));
    auto const y_dom = ddc::init_discrete_space<DDimY2>(DDimY2::init<DDimY2>(
            ddc::Coordinate<Y>(0.),
            ddc::Coordinate<Y>(1.),
            ddc::DiscreteVector<DDimY2>(6)));
    ddc::DiscreteDomain<DDimX2, DDimY2> const mesh(x_dom, y_dom);
    XDualizer const x_dualizer;
    YDualizer const y_dualizer;
    ddc::DiscreteDomain<DDimXDual, DDimY2> const x_face_dom = x_dualizer(mesh);
    ddc::DiscreteDomain<DDimX2, DDimYDual> const y_face_dom = y_dualizer(mesh);

    [[maybe_unused]] sil::tensor::TensorAccessor<DummyIndex> scalar_accessor;
    ddc::Chunk grad_x_alloc(
            ddc::DiscreteDomain<DDimXDual, DDimY2, DummyIndex>(
                    x_face_dom,
                    scalar_accessor.domain()),
            ddc::HostAllocator<double>());
    sil::tensor::Tensor grad_x(grad_x_alloc);
    ddc::Chunk grad_y_alloc(
            ddc::DiscreteDomain<DDimX2, DDimYDual, DummyIndex>(
                    y_face_dom,
                    scalar_accessor.domain()),
            ddc::HostAllocator<double>());
    sil::tensor::Tensor grad_y(grad_y_alloc);

    ddc::parallel_for_each(
            Kokkos::DefaultHostExecutionSpace(),
            grad_x.domain(),
            KOKKOS_LAMBDA(ddc::DiscreteElement<DDimXDual, DDimY2, DummyIndex> elem) {
                double const x = ddc::coordinate(ddc::DiscreteElement<DDimXDual>(elem));
                grad_x(elem) = 2. * x;
            });
    ddc::parallel_for_each(
            Kokkos::DefaultHostExecutionSpace(),
            grad_y.domain(),
            KOKKOS_LAMBDA(ddc::DiscreteElement<DDimX2, DDimYDual, DummyIndex> elem) {
                grad_y(elem) = 1.;
            });

    auto form = sil::exterior::make_tensor_form(
            sil::exterior::component<X>(grad_x),
            sil::exterior::component<Y>(grad_y));

    [[maybe_unused]] sil::tensor::TensorAccessor<InverseMetricIndex> inv_metric_accessor;
    ddc::Chunk inv_metric_alloc(
            ddc::DiscreteDomain<DDimX2, DDimY2, InverseMetricIndex>(
                    mesh,
                    inv_metric_accessor.domain()),
            ddc::HostAllocator<double>());
    sil::tensor::Tensor inv_metric(inv_metric_alloc);

    ddc::Chunk div_alloc(
            ddc::DiscreteDomain<DDimX2, DDimY2, DummyIndex>(mesh, scalar_accessor.domain()),
            ddc::HostAllocator<double>());
    sil::tensor::Tensor div(div_alloc);
    ddc::parallel_for_each(
            Kokkos::DefaultHostExecutionSpace(),
            div.domain(),
            KOKKOS_LAMBDA(ddc::DiscreteElement<DDimX2, DDimY2, DummyIndex> elem) { div(elem) = 0.; });

    sil::exterior::codifferential<MetricIndex>(
            Kokkos::DefaultHostExecutionSpace(),
            div,
            form,
            inv_metric);

    ddc::DiscreteElement<DDimX2, DDimY2> const center
            = mesh.front() + ddc::DiscreteVector<DDimX2, DDimY2>(2, 2);
    EXPECT_NEAR(div(center, ddc::DiscreteElement<DummyIndex>(0)), 2.0, 1e-12);
}

TEST(Cochain, Test)
{
    sil::exterior::Chain
            chain(Kokkos::View<
                          sil::exterior::Simplex<2, DDimT, DDimX, DDimY, DDimZ>*,
                          Kokkos::LayoutRight,
                          Kokkos::HostSpace>("", 3),
                  sil::exterior::
                          Simplex(ddc::DiscreteElement<DDimT, DDimX, DDimY, DDimZ> {0, 1, 0, 0},
                                  ddc::DiscreteVector<DDimX, DDimY> {1, 1}),
                  sil::exterior::
                          Simplex(ddc::DiscreteElement<DDimT, DDimX, DDimY, DDimZ> {0, 0, 0, 0},
                                  ddc::DiscreteVector<DDimX, DDimY> {1, 1}),
                  sil::exterior::
                          Simplex(ddc::DiscreteElement<DDimT, DDimX, DDimY, DDimZ> {0, 0, 0, 1},
                                  ddc::DiscreteVector<DDimX, DDimY> {1, 1},
                                  true));
    sil::exterior::Cochain
            cochain(chain,
                    Kokkos::View<double*, Kokkos::LayoutRight, Kokkos::HostSpace>("", 3),
                    0.,
                    1.,
                    2.);
    EXPECT_EQ(cochain.integrate(), -1.);
}

TEST(Coboundary, Test)
{
    sil::exterior::Simplex
            simplex(ddc::DiscreteElement<DDimT, DDimX, DDimY, DDimZ> {0, 1, 0, 0},
                    ddc::DiscreteVector<DDimX, DDimY> {1, 1});
    sil::exterior::Chain simplex_boundary = boundary(
            Kokkos::View<
                    sil::exterior::Simplex<1, DDimT, DDimX, DDimY, DDimZ>*,
                    Kokkos::LayoutRight,
                    Kokkos::HostSpace>("", 4),
            simplex);
    sil::exterior::Cochain cochain_boundary(
            simplex_boundary,
            Kokkos::View<double*, Kokkos::LayoutRight, Kokkos::HostSpace>("", 4),
            5.,
            8.,
            3.,
            2.);
    sil::exterior::Cosimplex cosimplex = sil::exterior::coboundary(cochain_boundary);
    EXPECT_EQ(cosimplex.simplex(), simplex);
    EXPECT_EQ(cosimplex.value(), 4.);
}

TEST(LocalChain, Test)
{
    sil::exterior::LocalChain
            chain(Kokkos::View<
                          ddc::DiscreteVector<DDimT, DDimX, DDimY, DDimZ>*,
                          Kokkos::LayoutRight,
                          Kokkos::HostSpace>("", 4),
                  sil::exterior::
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
                    Kokkos::View<
                            ddc::DiscreteVector<DDimT, DDimX, DDimY, DDimZ>*,
                            Kokkos::LayoutRight,
                            Kokkos::HostSpace>("", 1),
                    sil::exterior::
                            Simplex(ddc::DiscreteElement<DDimT, DDimX, DDimY, DDimZ> {0, 0, 0, 0},
                                    ddc::DiscreteVector<DDimZ> {1}));
    EXPECT_TRUE(
            chain
            == sil::exterior::LocalChain(
                    Kokkos::View<
                            ddc::DiscreteVector<DDimT, DDimX, DDimY, DDimZ>*,
                            Kokkos::LayoutRight,
                            Kokkos::HostSpace>("", 4),
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
            chain(Kokkos::View<
                          ddc::DiscreteVector<DDimT, DDimX, DDimY, DDimZ>*,
                          Kokkos::LayoutRight,
                          Kokkos::HostSpace>("", 2),
                  sil::exterior::
                          Simplex(ddc::DiscreteElement<DDimT, DDimX, DDimY, DDimZ> {0, 0, 0, 0},
                                  ddc::DiscreteVector<DDimX> {1}),
                  sil::exterior::
                          Simplex(ddc::DiscreteElement<DDimT, DDimX, DDimY, DDimZ> {0, 0, 0, 0},
                                  ddc::DiscreteVector<DDimY> {1}));
    sil::exterior::Cochain
            cochain(chain,
                    Kokkos::View<double*, Kokkos::LayoutRight, Kokkos::HostSpace>("", 2),
                    1.,
                    2.);
    EXPECT_EQ(cochain.integrate(), 3.);
}
