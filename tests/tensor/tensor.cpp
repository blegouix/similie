// SPDX-License-Identifier: GPL-3.0

#include <cmath>

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

struct Alpha : sil::tensor::TensorNaturalIndex<Y, Z>
{
};

struct Beta : sil::tensor::TensorNaturalIndex<Y, Z>
{
};

struct Gamma : sil::tensor::TensorNaturalIndex<Y, Z>
{
};

struct Mu : sil::tensor::TensorNaturalIndex<X, Y, Z>
{
};

struct Nu : sil::tensor::TensorNaturalIndex<X, Y, Z>
{
};

TEST(Tensor, NaturalIndexing)
{
    sil::tensor::TensorAccessor<Alpha, Nu> tensor_accessor;
    ddc::DiscreteDomain<Alpha, Nu> tensor_dom = tensor_accessor.get_domain();
    ddc::Chunk tensor_alloc(tensor_dom, ddc::HostAllocator<double>());
    ddc::ChunkSpan tensor = tensor_alloc.span_view();

    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 3; ++j) {
            tensor(ddc::DiscreteElement<Alpha, Nu>(i, j)) = i * 3 + j;
        }
    }

    EXPECT_EQ(tensor(tensor_accessor.get_element<Y, X>()), 0.);
    EXPECT_EQ(tensor(tensor_accessor.get_element<Y, Y>()), 1.);
    EXPECT_EQ(tensor(tensor_accessor.get_element<Y, Z>()), 2.);
    EXPECT_EQ(tensor(tensor_accessor.get_element<Z, X>()), 3.);
    EXPECT_EQ(tensor(tensor_accessor.get_element<Z, Y>()), 4.);
    EXPECT_EQ(tensor(tensor_accessor.get_element<Z, Z>()), 5.);
}

struct Rho : sil::tensor::FullTensorIndex<Alpha, Nu>
{
};

TEST(Tensor, FullTensorIndexing)
{
    sil::tensor::TensorAccessor<Rho> tensor_accessor;
    ddc::DiscreteDomain<Rho> tensor_dom = tensor_accessor.get_domain();
    ddc::Chunk tensor_alloc(tensor_dom, ddc::HostAllocator<double>());
    ddc::ChunkSpan tensor = tensor_alloc.span_view();

    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 3; ++j) {
            tensor(ddc::DiscreteElement<Rho>(i * 3 + j)) = i * 3 + j;
        }
    }

    EXPECT_EQ(tensor(tensor_accessor.get_element<Y, X>()), 0.);
    EXPECT_EQ(tensor(tensor_accessor.get_element<Y, Y>()), 1.);
    EXPECT_EQ(tensor(tensor_accessor.get_element<Y, Z>()), 2.);
    EXPECT_EQ(tensor(tensor_accessor.get_element<Z, X>()), 3.);
    EXPECT_EQ(tensor(tensor_accessor.get_element<Z, Y>()), 4.);
    EXPECT_EQ(tensor(tensor_accessor.get_element<Z, Z>()), 5.);
}

struct Sigma : sil::tensor::SymmetricTensorIndex<Mu, Nu>
{
};

TEST(Tensor, SymmetricTensorIndexing)
{
    sil::tensor::TensorAccessor<Sigma> tensor_accessor;
    ddc::DiscreteDomain<Sigma> tensor_dom = tensor_accessor.get_domain();
    ddc::Chunk tensor_alloc(tensor_dom, ddc::HostAllocator<double>());
    ddc::ChunkSpan tensor = tensor_alloc.span_view();

    for (int i = 0; i < 6; ++i) {
        tensor(ddc::DiscreteElement<Sigma>(i)) = i;
    }

    EXPECT_EQ(tensor(tensor_accessor.get_element<X, X>()), 0.);
    EXPECT_EQ(tensor(tensor_accessor.get_element<X, Y>()), 1.);
    EXPECT_EQ(tensor(tensor_accessor.get_element<X, Z>()), 2.);
    EXPECT_EQ(tensor(tensor_accessor.get_element<Y, X>()), 1.);
    EXPECT_EQ(tensor(tensor_accessor.get_element<Y, Y>()), 3.);
    EXPECT_EQ(tensor(tensor_accessor.get_element<Y, Z>()), 4.);
    EXPECT_EQ(tensor(tensor_accessor.get_element<Z, X>()), 2.);
    EXPECT_EQ(tensor(tensor_accessor.get_element<Z, Y>()), 4.);
    EXPECT_EQ(tensor(tensor_accessor.get_element<Z, Z>()), 5.);
}
