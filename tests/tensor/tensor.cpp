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
    ddc::DiscreteDomain<Alpha, Nu> tensor_dom = tensor_accessor.domain();
    ddc::Chunk tensor_alloc(tensor_dom, ddc::HostAllocator<double>());
    ddc::ChunkSpan tensor = tensor_alloc.span_view();

    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 3; ++j) {
            tensor(ddc::DiscreteElement<Alpha, Nu>(i, j)) = i * 3 + j;
        }
    }

    EXPECT_EQ(tensor_accessor(tensor, tensor_accessor.element<Y, X>()), 0.);
    EXPECT_EQ(tensor_accessor(tensor, tensor_accessor.element<Y, Y>()), 1.);
    EXPECT_EQ(tensor_accessor(tensor, tensor_accessor.element<Y, Z>()), 2.);
    EXPECT_EQ(tensor_accessor(tensor, tensor_accessor.element<Z, X>()), 3.);
    EXPECT_EQ(tensor_accessor(tensor, tensor_accessor.element<Z, Y>()), 4.);
    EXPECT_EQ(tensor_accessor(tensor, tensor_accessor.element<Z, Z>()), 5.);
}

struct Rho : sil::tensor::FullTensorIndex<Alpha, Nu>
{
};

TEST(Tensor, FullTensorIndexing)
{
    sil::tensor::TensorAccessor<Rho> tensor_accessor;
    ddc::DiscreteDomain<Rho> tensor_dom = tensor_accessor.domain();
    ddc::Chunk tensor_alloc(tensor_dom, ddc::HostAllocator<double>());
    ddc::ChunkSpan tensor = tensor_alloc.span_view();

    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 3; ++j) {
            tensor(ddc::DiscreteElement<Rho>(i * 3 + j)) = i * 3 + j;
        }
    }

    EXPECT_EQ(tensor_accessor(tensor, tensor_accessor.element<Y, X>()), 0.);
    EXPECT_EQ(tensor_accessor(tensor, tensor_accessor.element<Y, Y>()), 1.);
    EXPECT_EQ(tensor_accessor(tensor, tensor_accessor.element<Y, Z>()), 2.);
    EXPECT_EQ(tensor_accessor(tensor, tensor_accessor.element<Z, X>()), 3.);
    EXPECT_EQ(tensor_accessor(tensor, tensor_accessor.element<Z, Y>()), 4.);
    EXPECT_EQ(tensor_accessor(tensor, tensor_accessor.element<Z, Z>()), 5.);
}

struct Sigma : sil::tensor::SymmetricTensorIndex<Mu, Nu>
{
};

TEST(Tensor, SymmetricTensorIndexing3x3)
{
    sil::tensor::TensorAccessor<Sigma> tensor_accessor;
    ddc::DiscreteDomain<Sigma> tensor_dom = tensor_accessor.domain();
    ddc::Chunk tensor_alloc(tensor_dom, ddc::HostAllocator<double>());
    ddc::ChunkSpan tensor = tensor_alloc.span_view();

    for (int i = 0; i < 6; ++i) {
        tensor(ddc::DiscreteElement<Sigma>(i)) = i;
    }

    EXPECT_EQ(tensor_accessor(tensor, tensor_accessor.element<X, X>()), 0.);
    EXPECT_EQ(tensor_accessor(tensor, tensor_accessor.element<X, Y>()), 1.);
    EXPECT_EQ(tensor_accessor(tensor, tensor_accessor.element<X, Z>()), 2.);
    EXPECT_EQ(tensor_accessor(tensor, tensor_accessor.element<Y, X>()), 1.);
    EXPECT_EQ(tensor_accessor(tensor, tensor_accessor.element<Y, Y>()), 3.);
    EXPECT_EQ(tensor_accessor(tensor, tensor_accessor.element<Y, Z>()), 4.);
    EXPECT_EQ(tensor_accessor(tensor, tensor_accessor.element<Z, X>()), 2.);
    EXPECT_EQ(tensor_accessor(tensor, tensor_accessor.element<Z, Y>()), 4.);
    EXPECT_EQ(tensor_accessor(tensor, tensor_accessor.element<Z, Z>()), 5.);
}

struct Tau : sil::tensor::SymmetricTensorIndex<Alpha, Beta, Gamma>
{
};

TEST(Tensor, SymmetricTensorIndexing2x2x2)
{
    sil::tensor::TensorAccessor<Tau> tensor_accessor;
    ddc::DiscreteDomain<Tau> tensor_dom = tensor_accessor.domain();
    ddc::Chunk tensor_alloc(tensor_dom, ddc::HostAllocator<double>());
    ddc::ChunkSpan tensor = tensor_alloc.span_view();

    for (int i = 0; i < 4; ++i) {
        tensor(ddc::DiscreteElement<Tau>(i)) = i;
    }

    EXPECT_EQ(tensor_accessor(tensor, tensor_accessor.element<Y, Y, Y>()), 0.);
    EXPECT_EQ(tensor_accessor(tensor, tensor_accessor.element<Y, Y, Z>()), 1.);
    EXPECT_EQ(tensor_accessor(tensor, tensor_accessor.element<Y, Z, Y>()), 1.);
    EXPECT_EQ(tensor_accessor(tensor, tensor_accessor.element<Y, Z, Z>()), 2.);
    EXPECT_EQ(tensor_accessor(tensor, tensor_accessor.element<Z, Y, Y>()), 1.);
    EXPECT_EQ(tensor_accessor(tensor, tensor_accessor.element<Z, Y, Z>()), 2.);
    EXPECT_EQ(tensor_accessor(tensor, tensor_accessor.element<Z, Z, Y>()), 2.);
    EXPECT_EQ(tensor_accessor(tensor, tensor_accessor.element<Z, Z, Z>()), 3.);
}

struct Upsilon : sil::tensor::FullTensorIndex<Mu, sil::tensor::SymmetricTensorIndex<Alpha, Beta>>
{
};

TEST(Tensor, PartiallySymmetricTensorIndexing3x2x2)
{
    sil::tensor::TensorAccessor<Upsilon> tensor_accessor;
    ddc::DiscreteDomain<Upsilon> tensor_dom = tensor_accessor.domain();
    ddc::Chunk tensor_alloc(tensor_dom, ddc::HostAllocator<double>());
    ddc::ChunkSpan tensor = tensor_alloc.span_view();

    for (int i = 0; i < 9; ++i) {
        tensor(ddc::DiscreteElement<Upsilon>(i)) = i;
    }

    EXPECT_EQ(tensor_accessor(tensor, tensor_accessor.element<X, Y, Y>()), 0.);
    EXPECT_EQ(tensor_accessor(tensor, tensor_accessor.element<X, Y, Z>()), 1.);
    EXPECT_EQ(tensor_accessor(tensor, tensor_accessor.element<X, Z, Y>()), 1.);
    EXPECT_EQ(tensor_accessor(tensor, tensor_accessor.element<X, Z, Z>()), 2.);
    EXPECT_EQ(tensor_accessor(tensor, tensor_accessor.element<Y, Y, Y>()), 3.);
    EXPECT_EQ(tensor_accessor(tensor, tensor_accessor.element<Y, Y, Z>()), 4.);
    EXPECT_EQ(tensor_accessor(tensor, tensor_accessor.element<Y, Z, Y>()), 4.);
    EXPECT_EQ(tensor_accessor(tensor, tensor_accessor.element<Y, Z, Z>()), 5.);
    EXPECT_EQ(tensor_accessor(tensor, tensor_accessor.element<Z, Y, Y>()), 6.);
    EXPECT_EQ(tensor_accessor(tensor, tensor_accessor.element<Z, Y, Z>()), 7.);
    EXPECT_EQ(tensor_accessor(tensor, tensor_accessor.element<Z, Z, Y>()), 7.);
    EXPECT_EQ(tensor_accessor(tensor, tensor_accessor.element<Z, Z, Z>()), 8.);
}
