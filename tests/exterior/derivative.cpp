// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: GPL-3.0-or-later

#include <cmath>

#include <ddc/ddc.hpp>

#include <gtest/gtest.h>

#include "exterior.hpp"
#include "filled_struct.hpp"
#include "type_seq_conversion.hpp"

template <std::size_t N, class InIndex, class OutIndex, class... DDim>
static auto test_derivative()
{
    sil::tensor::TensorAccessor<InIndex> tensor_accessor;
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
                [&](auto elem) { tensor.mem(elem, ddc::DiscreteElement<InIndex>(i)) = i; });
        tensor
                .mem(sil::misc::filled_struct<ddc::DiscreteElement<DDim...>>(1),
                     ddc::DiscreteElement<InIndex>(i))
                = i + 1;
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
    auto [alloc, derivative] = test_derivative<3, Form1Index, Form2Index, DDimX, DDimY>();
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
}
