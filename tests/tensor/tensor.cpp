// SPDX-License-Identifier: GPL-3.0

#include <ddc/ddc.hpp>

#include <gtest/gtest.h>

#include "tensor.hpp"

struct X
{
};

struct Y
{
};

struct Z
{
};

struct Mu : sil::tensor::TensorNaturalIndex<Y, Z>
{
};

struct Nu : sil::tensor::TensorNaturalIndex<X, Y, Z>
{
};

struct Sigma : sil::tensor::FullTensorIndex<Mu, Nu>
{
};

TEST(Tensor, NaturalIndexing)
{
    sil::tensor::TensorHandler<Mu, Nu> tensor_handler;
    ddc::DiscreteDomain<Mu, Nu> tensor_dom = tensor_handler.get_domain();
    ddc::Chunk tensor_alloc(tensor_dom, ddc::HostAllocator<double>());
    ddc::ChunkSpan tensor = tensor_alloc.span_view();

    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 3; ++j) {
            tensor(ddc::DiscreteElement<Mu, Nu>(i, j)) = i * 3 + j;
        }
    }

    EXPECT_EQ(tensor(tensor_handler.get_element<Y, X>()), 0.);
    EXPECT_EQ(tensor(tensor_handler.get_element<Y, Y>()), 1.);
    EXPECT_EQ(tensor(tensor_handler.get_element<Y, Z>()), 2.);
    EXPECT_EQ(tensor(tensor_handler.get_element<Z, X>()), 3.);
    EXPECT_EQ(tensor(tensor_handler.get_element<Z, Y>()), 4.);
    EXPECT_EQ(tensor(tensor_handler.get_element<Z, Z>()), 5.);
}

TEST(Tensor, NonNaturalIndexing)
{
    sil::tensor::TensorHandler<Sigma> tensor_handler;
    ddc::DiscreteDomain<Sigma> tensor_dom = tensor_handler.get_domain();
    ddc::Chunk tensor_alloc(tensor_dom, ddc::HostAllocator<double>());
    ddc::ChunkSpan tensor = tensor_alloc.span_view();

    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 3; ++j) {
            tensor(ddc::DiscreteElement<Sigma>(i * 3 + j)) = i * 3 + j;
        }
    }

    EXPECT_EQ(tensor(tensor_handler.get_element<Y, X>()), 0.);
    EXPECT_EQ(tensor(tensor_handler.get_element<Y, Y>()), 1.);
    EXPECT_EQ(tensor(tensor_handler.get_element<Y, Z>()), 2.);
    EXPECT_EQ(tensor(tensor_handler.get_element<Z, X>()), 3.);
    EXPECT_EQ(tensor(tensor_handler.get_element<Z, Y>()), 4.);
    EXPECT_EQ(tensor(tensor_handler.get_element<Z, Z>()), 5.);
}
