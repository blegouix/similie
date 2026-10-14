// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

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

TEST(Coboundary, NativeStencilOrientation)
{
    using Direction = sil::tensor::Covariant<sil::tensor::TensorNaturalIndex<X, Y>>;
    struct CochainIndex : sil::tensor::TensorNaturalIndex<X, Y>
    {
    };
    using Cochain = sil::tensor::Covariant<CochainIndex>;
    using Output = sil::exterior::coboundary_index_t<Direction, Cochain>;
    auto const chain = sil::exterior::tangent_basis<2, ddc::DiscreteDomain<DDimX, DDimY>>(
            ddc::DiscreteElement<DDimX, DDimY>(4, 4));
    auto const lower = sil::exterior::tangent_basis<1, ddc::DiscreteDomain<DDimX, DDimY>>(
            ddc::DiscreteElement<DDimX, DDimY>(4, 4));
    auto verify = [&](auto operation) {
        auto const positive
                = operation
                          .value(chain,
                                 lower,
                                 ddc::DiscreteElement<DDimX, DDimY>(4, 4),
                                 ddc::DiscreteElement<Output>(Output::access_id({0, 1})));
        auto const negative
                = operation
                          .value(chain,
                                 lower,
                                 ddc::DiscreteElement<DDimX, DDimY>(4, 4),
                                 ddc::DiscreteElement<Output>(Output::access_id({1, 0})));
        auto const zero = operation
                                  .value(chain,
                                         lower,
                                         ddc::DiscreteElement<DDimX, DDimY>(4, 4),
                                         ddc::DiscreteElement<Output>(Output::access_id({0, 0})));
        double norm = 0.0;
        ddc::device_for_each(positive.domain(), [&](auto entry) {
            EXPECT_DOUBLE_EQ(negative.mem(entry), -positive.mem(entry));
            EXPECT_DOUBLE_EQ(zero.mem(entry), 0.0);
            norm += std::abs(positive.mem(entry));
        });
        EXPECT_DOUBLE_EQ(norm, 4.0);
        auto sample = [](auto point, auto component) {
            return std::sin(
                    0.17 * point.template uid<DDimX>() + 0.23 * point.template uid<DDimY>()
                    + 0.31 * component.template uid<Cochain>());
        };
        auto result = sil::exterior::detail::make_stencil<Kokkos::HostSpace, Output>(
                ddc::DiscreteElement<>());
        operation(
                sil::tensor::Tensor(result.span_view()),
                sample,
                chain,
                lower,
                ddc::DiscreteElement<DDimX, DDimY>(4, 4));
        double contracted = 0.0;
        ddc::device_for_each(positive.domain(), [&](auto entry) {
            contracted += positive.mem(entry)
                          * sample(
                                  ddc::DiscreteElement<DDimX, DDimY>(entry),
                                  ddc::DiscreteElement<Cochain>(entry));
        });
        EXPECT_NEAR(contracted, result.mem(ddc::DiscreteElement<Output>(0)), 1.e-14);
    };
    verify(sil::exterior::Coboundary<Direction, Cochain> {});
    verify(sil::exterior::TransposedCoboundary<Direction, Cochain> {});
}
