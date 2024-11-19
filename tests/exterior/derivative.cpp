// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: GPL-3.0-or-later

#include <cmath>

#include <ddc/ddc.hpp>

#include <gtest/gtest.h>

#include "exterior.hpp"
#include "filled_struct.hpp"
#include "type_seq_conversion.hpp"

template <std::size_t N, bool CoalescentIndexing, class InIndex, class OutIndex, class... DDim>
static auto test_derivative()
{
    sil::tensor::TensorAccessor<InIndex> tensor_accessor;
    if constexpr (CoalescentIndexing) {
        ddc::DiscreteDomain<InIndex, DDim...>
                dom(tensor_accessor.mem_domain(),
                    ddc::DiscreteDomain(
                            sil::misc::filled_struct<ddc::DiscreteElement<DDim...>>(0),
                            sil::misc::filled_struct<ddc::DiscreteVector<DDim...>>(N)));
        ddc::Chunk tensor_alloc(dom, ddc::HostAllocator<double>());
        sil::tensor::Tensor<
                double,
                ddc::DiscreteDomain<InIndex, DDim...>,
                Kokkos::layout_right,
                Kokkos::DefaultHostExecutionSpace::memory_space>
                tensor(tensor_alloc);
        for (std::size_t i = 0; i < InIndex::mem_size(); ++i) {
            ddc::parallel_for_each(
                    Kokkos::DefaultHostExecutionSpace(),
                    tensor.non_indices_domain(),
                    [&](auto elem) {
                        tensor.mem(elem, ddc::DiscreteElement<InIndex>(i)) = i + 1.;
                    });
            tensor
                    .mem(sil::misc::filled_struct<ddc::DiscreteElement<DDim...>>(1),
                         ddc::DiscreteElement<InIndex>(i))
                    = i + 2.;
        }
        sil::tensor::TensorAccessor<OutIndex> derivative_accessor;
        ddc::DiscreteDomain<OutIndex, DDim...> derivative_dom(
                derivative_accessor.mem_domain(),
                ddc::DiscreteDomain(
                        sil::misc::filled_struct<ddc::DiscreteElement<DDim...>>(0),
                        sil::misc::filled_struct<ddc::DiscreteVector<DDim...>>(N - 1)));
        ddc::Chunk derivative_alloc(derivative_dom, ddc::HostAllocator<double>());
        sil::tensor::Tensor<
                double,
                ddc::DiscreteDomain<OutIndex, DDim...>,
                Kokkos::layout_right,
                Kokkos::DefaultHostExecutionSpace::memory_space>
                derivative(derivative_alloc);
        sil::exterior::deriv<
                InIndex,
                ddc::type_seq_element_t<
                        0,
                        ddc::type_seq_remove_t<
                                sil::misc::to_type_seq_t<OutIndex>,
                                sil::misc::to_type_seq_t<InIndex>>>>(derivative, tensor);
        return std::make_pair(std::move(derivative_alloc), derivative);
    } else {
        ddc::DiscreteDomain<DDim..., InIndex>
                dom(ddc::DiscreteDomain(
                            sil::misc::filled_struct<ddc::DiscreteElement<DDim...>>(0),
                            sil::misc::filled_struct<ddc::DiscreteVector<DDim...>>(N)),
                    tensor_accessor.mem_domain());
        ddc::Chunk tensor_alloc(dom, ddc::HostAllocator<double>());
        sil::tensor::Tensor<
                double,
                ddc::DiscreteDomain<DDim..., InIndex>,
                Kokkos::layout_right,
                Kokkos::DefaultHostExecutionSpace::memory_space>
                tensor(tensor_alloc);
        for (std::size_t i = 0; i < InIndex::mem_size(); ++i) {
            ddc::parallel_for_each(
                    Kokkos::DefaultHostExecutionSpace(),
                    tensor.non_indices_domain(),
                    [&](auto elem) {
                        tensor.mem(elem, ddc::DiscreteElement<InIndex>(i)) = i + 1.;
                    });
            tensor
                    .mem(sil::misc::filled_struct<ddc::DiscreteElement<DDim...>>(1),
                         ddc::DiscreteElement<InIndex>(i))
                    = i + 2.;
        }
        sil::tensor::TensorAccessor<OutIndex> derivative_accessor;
        ddc::DiscreteDomain<DDim..., OutIndex> derivative_dom(
                ddc::DiscreteDomain(
                        sil::misc::filled_struct<ddc::DiscreteElement<DDim...>>(0),
                        sil::misc::filled_struct<ddc::DiscreteVector<DDim...>>(N - 1)),
                derivative_accessor.mem_domain());
        ddc::Chunk derivative_alloc(derivative_dom, ddc::HostAllocator<double>());
        sil::tensor::Tensor<
                double,
                ddc::DiscreteDomain<DDim..., OutIndex>,
                Kokkos::layout_right,
                Kokkos::DefaultHostExecutionSpace::memory_space>
                derivative(derivative_alloc);
        sil::exterior::deriv<
                InIndex,
                ddc::type_seq_element_t<
                        0,
                        ddc::type_seq_remove_t<
                                sil::misc::to_type_seq_t<OutIndex>,
                                sil::misc::to_type_seq_t<InIndex>>>>(derivative, tensor);
        return std::make_pair(std::move(derivative_alloc), derivative);
    }
}

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

struct Mu1 : sil::tensor::TensorNaturalIndex<X>
{
};

TEST(ExteriorDerivative, 1DGradient)
{
    auto [alloc, derivative] = test_derivative<
            3,
            false,
            sil::tensor::TensorAntisymmetricIndex<>,
            sil::tensor::TensorAntisymmetricIndex<Mu1>,
            DDimX>();
    EXPECT_EQ(
            derivative(ddc::DiscreteElement<DDimX> {0}, derivative.accessor().access_element<X>()),
            1.);
    EXPECT_EQ(
            derivative(ddc::DiscreteElement<DDimX> {1}, derivative.accessor().access_element<X>()),
            -1.);

    auto [alloc2, derivative2] = test_derivative<
            3,
            true,
            sil::tensor::TensorAntisymmetricIndex<>,
            sil::tensor::TensorAntisymmetricIndex<Mu1>,
            DDimX>();
    EXPECT_EQ(
            derivative2(
                    ddc::DiscreteElement<DDimX> {0},
                    derivative2.accessor().access_element<X>()),
            1.);
    EXPECT_EQ(
            derivative2(
                    ddc::DiscreteElement<DDimX> {1},
                    derivative2.accessor().access_element<X>()),
            -1.);
}

struct Mu2 : sil::tensor::TensorNaturalIndex<X, Y>
{
};

TEST(ExteriorDerivative, 2DGradient)
{
    auto [alloc, derivative] = test_derivative<
            3,
            false,
            sil::tensor::TensorAntisymmetricIndex<>,
            sil::tensor::TensorAntisymmetricIndex<Mu2>,
            DDimX,
            DDimY>();
    EXPECT_EQ(
            derivative(
                    ddc::DiscreteElement<DDimX, DDimY> {0, 0},
                    derivative.accessor().access_element<X>()),
            0.);
    EXPECT_EQ(
            derivative(
                    ddc::DiscreteElement<DDimX, DDimY> {0, 0},
                    derivative.accessor().access_element<Y>()),
            0.);
    EXPECT_EQ(
            derivative(
                    ddc::DiscreteElement<DDimX, DDimY> {0, 1},
                    derivative.accessor().access_element<X>()),
            1.);
    EXPECT_EQ(
            derivative(
                    ddc::DiscreteElement<DDimX, DDimY> {0, 1},
                    derivative.accessor().access_element<Y>()),
            0.);
    EXPECT_EQ(
            derivative(
                    ddc::DiscreteElement<DDimX, DDimY> {1, 0},
                    derivative.accessor().access_element<X>()),
            0.);
    EXPECT_EQ(
            derivative(
                    ddc::DiscreteElement<DDimX, DDimY> {1, 0},
                    derivative.accessor().access_element<Y>()),
            1.);
    EXPECT_EQ(
            derivative(
                    ddc::DiscreteElement<DDimX, DDimY> {1, 1},
                    derivative.accessor().access_element<X>()),
            -1.);
    EXPECT_EQ(
            derivative(
                    ddc::DiscreteElement<DDimX, DDimY> {1, 1},
                    derivative.accessor().access_element<Y>()),
            -1.);

    auto [alloc2, derivative2] = test_derivative<
            3,
            true,
            sil::tensor::TensorAntisymmetricIndex<>,
            sil::tensor::TensorAntisymmetricIndex<Mu2>,
            DDimX,
            DDimY>();
    EXPECT_EQ(
            derivative2(
                    ddc::DiscreteElement<DDimX, DDimY> {0, 0},
                    derivative2.accessor().access_element<X>()),
            0.);
    EXPECT_EQ(
            derivative2(
                    ddc::DiscreteElement<DDimX, DDimY> {0, 0},
                    derivative2.accessor().access_element<Y>()),
            0.);
    EXPECT_EQ(
            derivative2(
                    ddc::DiscreteElement<DDimX, DDimY> {0, 1},
                    derivative2.accessor().access_element<X>()),
            1.);
    EXPECT_EQ(
            derivative2(
                    ddc::DiscreteElement<DDimX, DDimY> {0, 1},
                    derivative2.accessor().access_element<Y>()),
            0.);
    EXPECT_EQ(
            derivative2(
                    ddc::DiscreteElement<DDimX, DDimY> {1, 0},
                    derivative2.accessor().access_element<X>()),
            0.);
    EXPECT_EQ(
            derivative2(
                    ddc::DiscreteElement<DDimX, DDimY> {1, 0},
                    derivative2.accessor().access_element<Y>()),
            1.);
    EXPECT_EQ(
            derivative2(
                    ddc::DiscreteElement<DDimX, DDimY> {1, 1},
                    derivative2.accessor().access_element<X>()),
            -1.);
    EXPECT_EQ(
            derivative2(
                    ddc::DiscreteElement<DDimX, DDimY> {1, 1},
                    derivative2.accessor().access_element<Y>()),
            -1.);
}

struct Nu2 : sil::tensor::TensorNaturalIndex<X, Y>
{
};

TEST(ExteriorDerivative, 2DRotational)
{
    auto [alloc, derivative] = test_derivative<
            3,
            false,
            sil::tensor::TensorAntisymmetricIndex<Mu2>,
            sil::tensor::TensorAntisymmetricIndex<Nu2, Mu2>,
            DDimX,
            DDimY>();
    EXPECT_EQ(
            derivative(
                    ddc::DiscreteElement<DDimX, DDimY> {0, 0},
                    derivative.accessor().access_element<X, Y>()),
            0.);
    EXPECT_EQ(
            derivative(
                    ddc::DiscreteElement<DDimX, DDimY> {1, 0},
                    derivative.accessor().access_element<X, Y>()),
            -1.);
    EXPECT_EQ(
            derivative(
                    ddc::DiscreteElement<DDimX, DDimY> {0, 1},
                    derivative.accessor().access_element<X, Y>()),
            1.);
    EXPECT_EQ(
            derivative(
                    ddc::DiscreteElement<DDimX, DDimY> {1, 1},
                    derivative.accessor().access_element<X, Y>()),
            0.);

    auto [alloc2, derivative2] = test_derivative<
            3,
            true,
            sil::tensor::TensorAntisymmetricIndex<Mu2>,
            sil::tensor::TensorAntisymmetricIndex<Nu2, Mu2>,
            DDimX,
            DDimY>();
    EXPECT_EQ(
            derivative2(
                    ddc::DiscreteElement<DDimX, DDimY> {0, 0},
                    derivative2.accessor().access_element<X, Y>()),
            0.);
    EXPECT_EQ(
            derivative2(
                    ddc::DiscreteElement<DDimX, DDimY> {1, 0},
                    derivative2.accessor().access_element<X, Y>()),
            -1.);
    EXPECT_EQ(
            derivative2(
                    ddc::DiscreteElement<DDimX, DDimY> {0, 1},
                    derivative2.accessor().access_element<X, Y>()),
            1.);
    EXPECT_EQ(
            derivative2(
                    ddc::DiscreteElement<DDimX, DDimY> {1, 1},
                    derivative2.accessor().access_element<X, Y>()),
            0.);
}

struct Mu3 : sil::tensor::TensorNaturalIndex<X, Y, Z>
{
};

TEST(ExteriorDerivative, 3DGradient)
{
    auto [alloc, derivative] = test_derivative<
            3,
            false,
            sil::tensor::TensorAntisymmetricIndex<>,
            sil::tensor::TensorAntisymmetricIndex<Mu3>,
            DDimX,
            DDimY,
            DDimZ>();
    EXPECT_EQ(
            derivative(
                    ddc::DiscreteElement<DDimX, DDimY, DDimZ> {0, 0, 0},
                    derivative.accessor().access_element<X>()),
            0.);
    EXPECT_EQ(
            derivative(
                    ddc::DiscreteElement<DDimX, DDimY, DDimZ> {0, 1, 1},
                    derivative.accessor().access_element<X>()),
            1.);
    EXPECT_EQ(
            derivative(
                    ddc::DiscreteElement<DDimX, DDimY, DDimZ> {1, 0, 1},
                    derivative.accessor().access_element<Y>()),
            1.);
    EXPECT_EQ(
            derivative(
                    ddc::DiscreteElement<DDimX, DDimY, DDimZ> {1, 1, 0},
                    derivative.accessor().access_element<Z>()),
            1.);
    EXPECT_EQ(
            derivative(
                    ddc::DiscreteElement<DDimX, DDimY, DDimZ> {1, 1, 1},
                    derivative.accessor().access_element<X>()),
            -1.);
    EXPECT_EQ(
            derivative(
                    ddc::DiscreteElement<DDimX, DDimY, DDimZ> {1, 1, 1},
                    derivative.accessor().access_element<Y>()),
            -1.);
    EXPECT_EQ(
            derivative(
                    ddc::DiscreteElement<DDimX, DDimY, DDimZ> {1, 1, 1},
                    derivative.accessor().access_element<Z>()),
            -1.);

    auto [alloc2, derivative2] = test_derivative<
            3,
            true,
            sil::tensor::TensorAntisymmetricIndex<>,
            sil::tensor::TensorAntisymmetricIndex<Mu3>,
            DDimX,
            DDimY,
            DDimZ>();
    EXPECT_EQ(
            derivative2(
                    ddc::DiscreteElement<DDimX, DDimY, DDimZ> {0, 0, 0},
                    derivative2.accessor().access_element<X>()),
            0.);
    EXPECT_EQ(
            derivative2(
                    ddc::DiscreteElement<DDimX, DDimY, DDimZ> {0, 1, 1},
                    derivative2.accessor().access_element<X>()),
            1.);
    EXPECT_EQ(
            derivative2(
                    ddc::DiscreteElement<DDimX, DDimY, DDimZ> {1, 0, 1},
                    derivative2.accessor().access_element<Y>()),
            1.);
    EXPECT_EQ(
            derivative2(
                    ddc::DiscreteElement<DDimX, DDimY, DDimZ> {1, 1, 0},
                    derivative2.accessor().access_element<Z>()),
            1.);
    EXPECT_EQ(
            derivative2(
                    ddc::DiscreteElement<DDimX, DDimY, DDimZ> {1, 1, 1},
                    derivative2.accessor().access_element<X>()),
            -1.);
    EXPECT_EQ(
            derivative2(
                    ddc::DiscreteElement<DDimX, DDimY, DDimZ> {1, 1, 1},
                    derivative2.accessor().access_element<Y>()),
            -1.);
    EXPECT_EQ(
            derivative2(
                    ddc::DiscreteElement<DDimX, DDimY, DDimZ> {1, 1, 1},
                    derivative2.accessor().access_element<Z>()),
            -1.);
}
