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
