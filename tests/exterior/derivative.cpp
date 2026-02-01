// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: MIT

#include <cmath>

#include <ddc/ddc.hpp>

#include <gtest/gtest.h>
#include <similie/tensor/symmetric_tensor.hpp>

#include "exterior.hpp"
#include "filled_struct.hpp"
#include "type_seq_conversion.hpp"

template <std::size_t N, bool CoalescentIndexing, class InIndex, class OutIndex, class... DDim>
static auto test_derivative()
{
    [[maybe_unused]] sil::tensor::TensorAccessor<InIndex> tensor_accessor;
    if constexpr (CoalescentIndexing) {
        ddc::DiscreteDomain<InIndex, DDim...>
                dom(tensor_accessor.domain(),
                    ddc::DiscreteDomain(
                            sil::misc::filled_struct<ddc::DiscreteElement<DDim...>>(0),
                            sil::misc::filled_struct<ddc::DiscreteVector<DDim...>>(N)));
        ddc::Chunk tensor_alloc(dom, ddc::HostAllocator<double>());
        sil::tensor::Tensor tensor(tensor_alloc);
        for (std::size_t i = 0; i < InIndex::mem_size(); ++i) {
            ddc::host_for_each(tensor.non_indices_domain(), [&](auto elem) {
                tensor.mem(elem, ddc::DiscreteElement<InIndex>(i)) = i + 1.;
            });
            tensor
                    .mem(sil::misc::filled_struct<ddc::DiscreteElement<DDim...>>(1),
                         ddc::DiscreteElement<InIndex>(i))
                    = i + 2.;
        }
        [[maybe_unused]] sil::tensor::TensorAccessor<OutIndex> derivative_accessor;
        ddc::DiscreteDomain<OutIndex, DDim...> derivative_dom(
                derivative_accessor.domain(),
                ddc::DiscreteDomain(
                        sil::misc::filled_struct<ddc::DiscreteElement<DDim...>>(0),
                        sil::misc::filled_struct<ddc::DiscreteVector<DDim...>>(
                                OutIndex::rank() == 1 ? N : N - 1)));
        ddc::Chunk derivative_alloc(derivative_dom, ddc::HostAllocator<double>());
        sil::tensor::Tensor derivative(derivative_alloc);
        if constexpr (sil::tensor::TensorNatIndex<OutIndex>) {
            sil::exterior::deriv<
                    OutIndex,
                    InIndex>(Kokkos::DefaultHostExecutionSpace(), derivative, tensor);
        } else {
            sil::exterior::deriv<
                    ddc::type_seq_element_t<
                            0,
                            ddc::type_seq_remove_t<
                                    sil::misc::to_type_seq_t<OutIndex>,
                                    sil::misc::to_type_seq_t<InIndex>>>,
                    InIndex>(Kokkos::DefaultHostExecutionSpace(), derivative, tensor);
        }
        return std::make_pair(std::move(derivative_alloc), derivative);
    } else {
        ddc::DiscreteDomain<DDim..., InIndex>
                dom(ddc::DiscreteDomain(
                            sil::misc::filled_struct<ddc::DiscreteElement<DDim...>>(0),
                            sil::misc::filled_struct<ddc::DiscreteVector<DDim...>>(N)),
                    tensor_accessor.domain());
        ddc::Chunk tensor_alloc(dom, ddc::HostAllocator<double>());
        sil::tensor::Tensor tensor(tensor_alloc);
        for (std::size_t i = 0; i < InIndex::mem_size(); ++i) {
            ddc::host_for_each(tensor.non_indices_domain(), [&](auto elem) {
                tensor.mem(elem, ddc::DiscreteElement<InIndex>(i)) = i + 1.;
            });
            tensor
                    .mem(sil::misc::filled_struct<ddc::DiscreteElement<DDim...>>(1),
                         ddc::DiscreteElement<InIndex>(i))
                    = i + 2.;
        }
        [[maybe_unused]] sil::tensor::TensorAccessor<OutIndex> derivative_accessor;
        ddc::DiscreteDomain<DDim..., OutIndex> derivative_dom(
                ddc::DiscreteDomain(
                        sil::misc::filled_struct<ddc::DiscreteElement<DDim...>>(0),
                        sil::misc::filled_struct<ddc::DiscreteVector<DDim...>>(
                                OutIndex::rank() == 1 ? N : N - 1)),
                derivative_accessor.domain());
        ddc::Chunk derivative_alloc(derivative_dom, ddc::HostAllocator<double>());
        sil::tensor::Tensor derivative(derivative_alloc);
        if constexpr (sil::tensor::TensorNatIndex<OutIndex>) {
            sil::exterior::deriv<
                    OutIndex,
                    InIndex>(Kokkos::DefaultHostExecutionSpace(), derivative, tensor);
        } else {
            sil::exterior::deriv<
                    ddc::type_seq_element_t<
                            0,
                            ddc::type_seq_remove_t<
                                    sil::misc::to_type_seq_t<OutIndex>,
                                    sil::misc::to_type_seq_t<InIndex>>>,
                    InIndex>(Kokkos::DefaultHostExecutionSpace(), derivative, tensor);
        }
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
    EXPECT_EQ(
            derivative(ddc::DiscreteElement<DDimX> {2}, derivative.accessor().access_element<X>()),
            0.);

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
    EXPECT_EQ(
            derivative2(
                    ddc::DiscreteElement<DDimX> {2},
                    derivative2.accessor().access_element<X>()),
            0.);
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
                    ddc::DiscreteElement<DDimX, DDimY> {0, 2},
                    derivative.accessor().access_element<X>()),
            0.);
    EXPECT_EQ(
            derivative(
                    ddc::DiscreteElement<DDimX, DDimY> {0, 2},
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
    EXPECT_EQ(
            derivative(
                    ddc::DiscreteElement<DDimX, DDimY> {1, 2},
                    derivative.accessor().access_element<X>()),
            0.);
    EXPECT_EQ(
            derivative(
                    ddc::DiscreteElement<DDimX, DDimY> {1, 2},
                    derivative.accessor().access_element<Y>()),
            0.);
    EXPECT_EQ(
            derivative(
                    ddc::DiscreteElement<DDimX, DDimY> {2, 0},
                    derivative.accessor().access_element<X>()),
            0.);
    EXPECT_EQ(
            derivative(
                    ddc::DiscreteElement<DDimX, DDimY> {2, 0},
                    derivative.accessor().access_element<Y>()),
            0.);
    EXPECT_EQ(
            derivative(
                    ddc::DiscreteElement<DDimX, DDimY> {2, 1},
                    derivative.accessor().access_element<X>()),
            0.);
    EXPECT_EQ(
            derivative(
                    ddc::DiscreteElement<DDimX, DDimY> {2, 1},
                    derivative.accessor().access_element<Y>()),
            0.);
    EXPECT_EQ(
            derivative(
                    ddc::DiscreteElement<DDimX, DDimY> {2, 2},
                    derivative.accessor().access_element<X>()),
            0.);
    EXPECT_EQ(
            derivative(
                    ddc::DiscreteElement<DDimX, DDimY> {2, 2},
                    derivative.accessor().access_element<Y>()),
            0.);

    auto [alloc2, derivative2]
            = test_derivative<3, true, sil::tensor::TensorNaturalIndex<>, Mu2, DDimX, DDimY>();
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
                    ddc::DiscreteElement<DDimX, DDimY> {0, 2},
                    derivative2.accessor().access_element<X>()),
            0.);
    EXPECT_EQ(
            derivative2(
                    ddc::DiscreteElement<DDimX, DDimY> {0, 2},
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
    EXPECT_EQ(
            derivative2(
                    ddc::DiscreteElement<DDimX, DDimY> {1, 2},
                    derivative2.accessor().access_element<X>()),
            0.);
    EXPECT_EQ(
            derivative2(
                    ddc::DiscreteElement<DDimX, DDimY> {1, 2},
                    derivative2.accessor().access_element<Y>()),
            0.);
    EXPECT_EQ(
            derivative2(
                    ddc::DiscreteElement<DDimX, DDimY> {2, 0},
                    derivative2.accessor().access_element<X>()),
            0.);
    EXPECT_EQ(
            derivative2(
                    ddc::DiscreteElement<DDimX, DDimY> {2, 0},
                    derivative2.accessor().access_element<Y>()),
            0.);
    EXPECT_EQ(
            derivative2(
                    ddc::DiscreteElement<DDimX, DDimY> {2, 1},
                    derivative2.accessor().access_element<X>()),
            0.);
    EXPECT_EQ(
            derivative2(
                    ddc::DiscreteElement<DDimX, DDimY> {2, 1},
                    derivative2.accessor().access_element<Y>()),
            0.);
    EXPECT_EQ(
            derivative2(
                    ddc::DiscreteElement<DDimX, DDimY> {2, 2},
                    derivative2.accessor().access_element<X>()),
            0.);
    EXPECT_EQ(
            derivative2(
                    ddc::DiscreteElement<DDimX, DDimY> {2, 2},
                    derivative2.accessor().access_element<Y>()),
            0.);
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
            Mu2,
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

struct Nu3 : sil::tensor::TensorNaturalIndex<X, Y, Z>
{
};

TEST(ExteriorDerivative, 3DRotational)
{
    auto [alloc, derivative] = test_derivative<
            3,
            false,
            sil::tensor::TensorAntisymmetricIndex<Mu3>,
            sil::tensor::TensorAntisymmetricIndex<Nu3, Mu3>,
            DDimX,
            DDimY,
            DDimZ>();
    EXPECT_EQ(
            derivative(
                    ddc::DiscreteElement<DDimX, DDimY, DDimZ> {0, 1, 1},
                    derivative.accessor().access_element<X, Y>()),
            1.);
    EXPECT_EQ(
            derivative(
                    ddc::DiscreteElement<DDimX, DDimY, DDimZ> {0, 1, 1},
                    derivative.accessor().access_element<X, Z>()),
            1.);
    EXPECT_EQ(
            derivative(
                    ddc::DiscreteElement<DDimX, DDimY, DDimZ> {0, 1, 1},
                    derivative.accessor().access_element<Y, Z>()),
            0.);
    EXPECT_EQ(
            derivative(
                    ddc::DiscreteElement<DDimX, DDimY, DDimZ> {1, 0, 1},
                    derivative.accessor().access_element<X, Y>()),
            -1.);
    EXPECT_EQ(
            derivative(
                    ddc::DiscreteElement<DDimX, DDimY, DDimZ> {1, 0, 1},
                    derivative.accessor().access_element<X, Z>()),
            0.);
    EXPECT_EQ(
            derivative(
                    ddc::DiscreteElement<DDimX, DDimY, DDimZ> {1, 0, 1},
                    derivative.accessor().access_element<Y, Z>()),
            1.);
    EXPECT_EQ(
            derivative(
                    ddc::DiscreteElement<DDimX, DDimY, DDimZ> {1, 1, 0},
                    derivative.accessor().access_element<X, Y>()),
            0.);
    EXPECT_EQ(
            derivative(
                    ddc::DiscreteElement<DDimX, DDimY, DDimZ> {1, 1, 0},
                    derivative.accessor().access_element<X, Z>()),
            -1.);
    EXPECT_EQ(
            derivative(
                    ddc::DiscreteElement<DDimX, DDimY, DDimZ> {1, 1, 0},
                    derivative.accessor().access_element<Y, Z>()),
            -1.);

    auto [alloc2, derivative2] = test_derivative<
            3,
            true,
            sil::tensor::TensorAntisymmetricIndex<Mu3>,
            sil::tensor::TensorAntisymmetricIndex<Nu3, Mu3>,
            DDimX,
            DDimY,
            DDimZ>();
    EXPECT_EQ(
            derivative2(
                    ddc::DiscreteElement<DDimX, DDimY, DDimZ> {0, 1, 1},
                    derivative2.accessor().access_element<X, Y>()),
            1.);
    EXPECT_EQ(
            derivative2(
                    ddc::DiscreteElement<DDimX, DDimY, DDimZ> {0, 1, 1},
                    derivative2.accessor().access_element<X, Z>()),
            1.);
    EXPECT_EQ(
            derivative2(
                    ddc::DiscreteElement<DDimX, DDimY, DDimZ> {0, 1, 1},
                    derivative2.accessor().access_element<Y, Z>()),
            0.);
    EXPECT_EQ(
            derivative2(
                    ddc::DiscreteElement<DDimX, DDimY, DDimZ> {1, 0, 1},
                    derivative2.accessor().access_element<X, Y>()),
            -1.);
    EXPECT_EQ(
            derivative2(
                    ddc::DiscreteElement<DDimX, DDimY, DDimZ> {1, 0, 1},
                    derivative2.accessor().access_element<X, Z>()),
            0.);
    EXPECT_EQ(
            derivative2(
                    ddc::DiscreteElement<DDimX, DDimY, DDimZ> {1, 0, 1},
                    derivative2.accessor().access_element<Y, Z>()),
            1.);
    EXPECT_EQ(
            derivative2(
                    ddc::DiscreteElement<DDimX, DDimY, DDimZ> {1, 1, 0},
                    derivative2.accessor().access_element<X, Y>()),
            0.);
    EXPECT_EQ(
            derivative2(
                    ddc::DiscreteElement<DDimX, DDimY, DDimZ> {1, 1, 0},
                    derivative2.accessor().access_element<X, Z>()),
            -1.);
    EXPECT_EQ(
            derivative2(
                    ddc::DiscreteElement<DDimX, DDimY, DDimZ> {1, 1, 0},
                    derivative2.accessor().access_element<Y, Z>()),
            -1.);
}

struct Rho3 : sil::tensor::TensorNaturalIndex<X, Y, Z>
{
};

TEST(ExteriorDerivative, 3DDivergency)
{
    auto [alloc, derivative] = test_derivative<
            3,
            false,
            sil::tensor::TensorAntisymmetricIndex<Nu3, Mu3>,
            sil::tensor::TensorAntisymmetricIndex<Rho3, Nu3, Mu3>,
            DDimX,
            DDimY,
            DDimZ>();
    EXPECT_EQ(
            derivative(
                    ddc::DiscreteElement<DDimX, DDimY, DDimZ> {0, 1, 1},
                    derivative.accessor().access_element<X, Y, Z>()),
            1.);
    EXPECT_EQ(
            derivative(
                    ddc::DiscreteElement<DDimX, DDimY, DDimZ> {1, 0, 1},
                    derivative.accessor().access_element<X, Y, Z>()),
            -1.);
    EXPECT_EQ(
            derivative(
                    ddc::DiscreteElement<DDimX, DDimY, DDimZ> {1, 1, 0},
                    derivative.accessor().access_element<X, Y, Z>()),
            1.);
    EXPECT_EQ(
            derivative(
                    ddc::DiscreteElement<DDimX, DDimY, DDimZ> {1, 1, 1},
                    derivative.accessor().access_element<X, Y, Z>()),
            -1.);

    auto [alloc2, derivative2] = test_derivative<
            3,
            true,
            sil::tensor::TensorAntisymmetricIndex<Nu3, Mu3>,
            sil::tensor::TensorAntisymmetricIndex<Rho3, Nu3, Mu3>,
            DDimX,
            DDimY,
            DDimZ>();
    EXPECT_EQ(
            derivative2(
                    ddc::DiscreteElement<DDimX, DDimY, DDimZ> {0, 1, 1},
                    derivative2.accessor().access_element<X, Y, Z>()),
            1.);
    EXPECT_EQ(
            derivative2(
                    ddc::DiscreteElement<DDimX, DDimY, DDimZ> {1, 0, 1},
                    derivative2.accessor().access_element<X, Y, Z>()),
            -1.);
    EXPECT_EQ(
            derivative2(
                    ddc::DiscreteElement<DDimX, DDimY, DDimZ> {1, 1, 0},
                    derivative2.accessor().access_element<X, Y, Z>()),
            1.);
    EXPECT_EQ(
            derivative2(
                    ddc::DiscreteElement<DDimX, DDimY, DDimZ> {1, 1, 1},
                    derivative2.accessor().access_element<X, Y, Z>()),
            -1.);
}

struct Mu4 : sil::tensor::TensorNaturalIndex<T, X, Y, Z>
{
};

TEST(ExteriorDerivative, 4DGradient)
{
    auto [alloc, derivative] = test_derivative<
            3,
            false,
            sil::tensor::TensorAntisymmetricIndex<>,
            sil::tensor::TensorAntisymmetricIndex<Mu4>,
            DDimT,
            DDimX,
            DDimY,
            DDimZ>();
    EXPECT_EQ(
            derivative(
                    ddc::DiscreteElement<DDimT, DDimX, DDimY, DDimZ> {0, 0, 0, 0},
                    derivative.accessor().access_element<T>()),
            0.);
    EXPECT_EQ(
            derivative(
                    ddc::DiscreteElement<DDimT, DDimX, DDimY, DDimZ> {0, 1, 1, 1},
                    derivative.accessor().access_element<T>()),
            1.);
    EXPECT_EQ(
            derivative(
                    ddc::DiscreteElement<DDimT, DDimX, DDimY, DDimZ> {1, 0, 1, 1},
                    derivative.accessor().access_element<X>()),
            1.);
    EXPECT_EQ(
            derivative(
                    ddc::DiscreteElement<DDimT, DDimX, DDimY, DDimZ> {1, 1, 0, 1},
                    derivative.accessor().access_element<Y>()),
            1.);
    EXPECT_EQ(
            derivative(
                    ddc::DiscreteElement<DDimT, DDimX, DDimY, DDimZ> {1, 1, 1, 0},
                    derivative.accessor().access_element<Z>()),
            1.);

    auto [alloc2, derivative2] = test_derivative<
            3,
            true,
            sil::tensor::TensorAntisymmetricIndex<>,
            sil::tensor::TensorAntisymmetricIndex<Mu4>,
            DDimT,
            DDimX,
            DDimY,
            DDimZ>();
    EXPECT_EQ(
            derivative2(
                    ddc::DiscreteElement<DDimT, DDimX, DDimY, DDimZ> {0, 0, 0, 0},
                    derivative2.accessor().access_element<T>()),
            0.);
    EXPECT_EQ(
            derivative2(
                    ddc::DiscreteElement<DDimT, DDimX, DDimY, DDimZ> {0, 1, 1, 1},
                    derivative2.accessor().access_element<T>()),
            1.);
    EXPECT_EQ(
            derivative2(
                    ddc::DiscreteElement<DDimT, DDimX, DDimY, DDimZ> {1, 0, 1, 1},
                    derivative2.accessor().access_element<X>()),
            1.);
    EXPECT_EQ(
            derivative2(
                    ddc::DiscreteElement<DDimT, DDimX, DDimY, DDimZ> {1, 1, 0, 1},
                    derivative2.accessor().access_element<Y>()),
            1.);
    EXPECT_EQ(
            derivative2(
                    ddc::DiscreteElement<DDimT, DDimX, DDimY, DDimZ> {1, 1, 1, 0},
                    derivative2.accessor().access_element<Z>()),
            1.);
}

struct Nu4 : sil::tensor::TensorNaturalIndex<T, X, Y, Z>
{
};

TEST(ExteriorDerivative, 4DRotational)
{
    auto [alloc, derivative] = test_derivative<
            3,
            false,
            sil::tensor::TensorAntisymmetricIndex<Mu4>,
            sil::tensor::TensorAntisymmetricIndex<Nu4, Mu4>,
            DDimT,
            DDimX,
            DDimY,
            DDimZ>();

    auto [alloc2, derivative2] = test_derivative<
            3,
            true,
            sil::tensor::TensorAntisymmetricIndex<Mu4>,
            sil::tensor::TensorAntisymmetricIndex<Nu4, Mu4>,
            DDimT,
            DDimX,
            DDimY,
            DDimZ>();
}

struct Rho4 : sil::tensor::TensorNaturalIndex<T, X, Y, Z>
{
};

TEST(ExteriorDerivative, 4DDivergency)
{
    auto [alloc, derivative] = test_derivative<
            3,
            false,
            sil::tensor::TensorAntisymmetricIndex<Nu4, Mu4>,
            sil::tensor::TensorAntisymmetricIndex<Rho4, Nu4, Mu4>,
            DDimT,
            DDimX,
            DDimY,
            DDimZ>();

    auto [alloc2, derivative2] = test_derivative<
            3,
            true,
            sil::tensor::TensorAntisymmetricIndex<Nu4, Mu4>,
            sil::tensor::TensorAntisymmetricIndex<Rho4, Nu4, Mu4>,
            DDimT,
            DDimX,
            DDimY,
            DDimZ>();
}

struct Sigma4 : sil::tensor::TensorNaturalIndex<T, X, Y, Z>
{
};

TEST(ExteriorDerivative, 4DHyperDivergency)
{
    auto [alloc, derivative] = test_derivative<
            3,
            false,
            sil::tensor::TensorAntisymmetricIndex<Rho4, Nu4, Mu4>,
            sil::tensor::TensorAntisymmetricIndex<Sigma4, Rho4, Nu4, Mu4>,
            DDimT,
            DDimX,
            DDimY,
            DDimZ>();

    auto [alloc2, derivative2] = test_derivative<
            3,
            true,
            sil::tensor::TensorAntisymmetricIndex<Rho4, Nu4, Mu4>,
            sil::tensor::TensorAntisymmetricIndex<Sigma4, Rho4, Nu4, Mu4>,
            DDimT,
            DDimX,
            DDimY,
            DDimZ>();
}

struct DDimSpect : ddc::UniformPointSampling<T>
{
};

struct SpectNatIndex1 : sil::tensor::TensorNaturalIndex<T, X, Y, Z>
{
};

struct SpectNatIndex2 : sil::tensor::TensorNaturalIndex<T, X, Y, Z>
{
};

struct IndexSpect : sil::tensor::TensorSymmetricIndex<SpectNatIndex1, SpectNatIndex2>
{
};

TEST(ExteriorDerivative, 2DRotationalWithSpects)
{
    const std::size_t N = 3;
    [[maybe_unused]] sil::tensor::
            TensorAccessor<IndexSpect, sil::tensor::TensorAntisymmetricIndex<Mu2>> tensor_accessor;
    ddc::DiscreteDomain<
            sil::tensor::TensorAntisymmetricIndex<Mu2>,
            DDimSpect,
            DDimX,
            IndexSpect,
            DDimY>
            dom(tensor_accessor.domain(),
                ddc::DiscreteDomain(
                        sil::misc::filled_struct<ddc::DiscreteElement<DDimSpect, DDimX, DDimY>>(0),
                        sil::misc::filled_struct<ddc::DiscreteVector<DDimSpect, DDimX, DDimY>>(N)));
    ddc::Chunk tensor_alloc(dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor tensor(tensor_alloc);
    for (std::size_t i = 0; i < sil::tensor::TensorAntisymmetricIndex<Mu2>::mem_size(); ++i) {
        ddc::host_for_each(
                ddc::DiscreteDomain<DDimSpect, DDimX, IndexSpect, DDimY>(tensor.domain()),
                [&](auto elem) {
                    tensor
                            .mem(elem,
                                 ddc::DiscreteElement<sil::tensor::TensorAntisymmetricIndex<Mu2>>(
                                         i))
                            = i + 1.;
                });
        for (auto dim_spect_elem : ddc::DiscreteDomain<DDimSpect>(tensor.domain())) {
            for (auto index_spect_elem : ddc::DiscreteDomain<IndexSpect>(tensor.domain())) {
                tensor
                        .mem(dim_spect_elem,
                             index_spect_elem,
                             ddc::DiscreteElement<DDimX, DDimY> {1, 1},
                             ddc::DiscreteElement<sil::tensor::TensorAntisymmetricIndex<Mu2>>(i))
                        = i + 2.;
            }
        }
    }

    [[maybe_unused]] sil::tensor::TensorAccessor<
            IndexSpect,
            sil::tensor::TensorAntisymmetricIndex<Nu2, Mu2>> derivative_accessor;
    ddc::DiscreteDomain<
            sil::tensor::TensorAntisymmetricIndex<Nu2, Mu2>,
            DDimSpect,
            DDimX,
            IndexSpect,
            DDimY>
            derivative_dom(
                    derivative_accessor.domain(),
                    ddc::DiscreteDomain(
                            sil::misc::filled_struct<ddc::DiscreteElement<DDimSpect, DDimX, DDimY>>(
                                    0),
                            ddc::DiscreteVector<DDimSpect, DDimX, DDimY> {3, 2, 2}));
    ddc::Chunk derivative_alloc(derivative_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor derivative(derivative_alloc);
    sil::exterior::deriv<
            Nu2,
            sil::tensor::TensorAntisymmetricIndex<
                    Mu2>>(Kokkos::DefaultHostExecutionSpace(), derivative, tensor);

    for (auto dim_spect_elem : ddc::DiscreteDomain<DDimSpect>(tensor.domain())) {
        for (auto index_spect_elem : ddc::DiscreteDomain<IndexSpect>(tensor.domain())) {
            EXPECT_EQ(
                    derivative(
                            ddc::DiscreteElement<DDimX, DDimY> {0, 0},
                            dim_spect_elem,
                            index_spect_elem,
                            sil::tensor::TensorAccessor<
                                    sil::tensor::TensorAntisymmetricIndex<Nu2, Mu2>>::
                                    access_element<X, Y>()),
                    0.);
            EXPECT_EQ(
                    derivative(
                            ddc::DiscreteElement<DDimX, DDimY> {1, 0},
                            dim_spect_elem,
                            index_spect_elem,
                            sil::tensor::TensorAccessor<
                                    sil::tensor::TensorAntisymmetricIndex<Nu2, Mu2>>::
                                    access_element<X, Y>()),
                    -1.);
            EXPECT_EQ(
                    derivative(
                            ddc::DiscreteElement<DDimX, DDimY> {0, 1},
                            dim_spect_elem,
                            index_spect_elem,
                            sil::tensor::TensorAccessor<
                                    sil::tensor::TensorAntisymmetricIndex<Nu2, Mu2>>::
                                    access_element<X, Y>()),
                    1.);
            EXPECT_EQ(
                    derivative(
                            ddc::DiscreteElement<DDimX, DDimY> {1, 1},
                            dim_spect_elem,
                            index_spect_elem,
                            sil::tensor::TensorAccessor<
                                    sil::tensor::TensorAntisymmetricIndex<Nu2, Mu2>>::
                                    access_element<X, Y>()),
                    0.);
        }
    }
}
