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

struct T
{
};

struct Alpha : sil::tensor::TensorNaturalIndex<X, Y, Z>
{
};

struct Beta : sil::tensor::TensorNaturalIndex<X, Y, Z>
{
};

struct Gamma : sil::tensor::TensorNaturalIndex<X, Y, Z>
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
    ddc::DiscreteDomain<Alpha, Nu> tensor_dom = tensor_accessor.mem_domain();
    ddc::Chunk tensor_alloc(tensor_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor<
            double,
            ddc::DiscreteDomain<Alpha, Nu>,
            std::experimental::layout_right,
            Kokkos::DefaultHostExecutionSpace::memory_space>
            tensor(tensor_alloc);

    /*
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 3; ++j) {
            tensor(ddc::DiscreteElement<Alpha, Nu>(i, j)) = i * 3 + j;
        }
    }
    */

    tensor(tensor_accessor.element<Y, X>()) = 0.;
    tensor(tensor_accessor.element<Y, Y>()) = 1.;
    tensor(tensor_accessor.element<Y, Z>()) = 2.;
    tensor(tensor_accessor.element<Z, X>()) = 3.;
    tensor(tensor_accessor.element<Z, Y>()) = 4.;
    tensor(tensor_accessor.element<Z, Z>()) = 5.;

    EXPECT_EQ(tensor.get(tensor_accessor.element<Y, X>()), 0.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<Y, Y>()), 1.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<Y, Z>()), 2.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<Z, X>()), 3.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<Z, Y>()), 4.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<Z, Z>()), 5.);
}

struct Rho : sil::tensor::FullTensorIndex<Alpha, Nu>
{
};

TEST(Tensor, FullTensorIndexing)
{
    sil::tensor::TensorAccessor<Rho> tensor_accessor;
    ddc::DiscreteDomain<Rho> tensor_dom = tensor_accessor.mem_domain();
    ddc::Chunk tensor_alloc(tensor_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor<
            double,
            ddc::DiscreteDomain<Rho>,
            std::experimental::layout_right,
            Kokkos::DefaultHostExecutionSpace::memory_space>
            tensor(tensor_alloc);

    /*
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            tensor(ddc::DiscreteElement<Rho>(i * 3 + j)) = i * 3 + j;
        }
    }
    */

    tensor(tensor_accessor.element<X, X>()) = 0.;
    tensor(tensor_accessor.element<X, Y>()) = 1.;
    tensor(tensor_accessor.element<X, Z>()) = 2.;
    tensor(tensor_accessor.element<Y, X>()) = 3.;
    tensor(tensor_accessor.element<Y, Y>()) = 4.;
    tensor(tensor_accessor.element<Y, Z>()) = 5.;
    tensor(tensor_accessor.element<Z, X>()) = 6.;
    tensor(tensor_accessor.element<Z, Y>()) = 7.;
    tensor(tensor_accessor.element<Z, Z>()) = 8.;

    EXPECT_EQ(tensor.get(tensor_accessor.element<X, X>()), 0.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<X, Y>()), 1.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<X, Z>()), 2.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<Y, X>()), 3.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<Y, Y>()), 4.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<Y, Z>()), 5.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<Z, X>()), 6.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<Z, Y>()), 7.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<Z, Z>()), 8.);
}

struct IdIndex : sil::tensor::IdentityTensorIndex<Mu, Nu>
{
};

TEST(Tensor, IdentityTensorIndexing)
{
    sil::tensor::TensorAccessor<IdIndex> tensor_accessor;
    ddc::DiscreteDomain<IdIndex> tensor_dom = tensor_accessor.mem_domain();
    ddc::Chunk tensor_alloc(tensor_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor<
            double,
            ddc::DiscreteDomain<IdIndex>,
            std::experimental::layout_right,
            Kokkos::DefaultHostExecutionSpace::memory_space>
            tensor(tensor_alloc);

    EXPECT_EQ(tensor.get(tensor_accessor.element<X, X>()), 1.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<X, Y>()), 0.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<X, Z>()), 0.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<Y, X>()), 0.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<Y, Y>()), 1.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<Y, Z>()), 0.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<Z, X>()), 0.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<Z, Y>()), 0.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<Z, Z>()), 1.);
}

struct DiagIndex : sil::tensor::DiagonalTensorIndex<Mu, Nu>
{
};

TEST(Tensor, DiagonalTensorIndexing)
{
    sil::tensor::TensorAccessor<DiagIndex> tensor_accessor;
    ddc::DiscreteDomain<DiagIndex> tensor_dom = tensor_accessor.mem_domain();
    ddc::Chunk tensor_alloc(tensor_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor<
            double,
            ddc::DiscreteDomain<DiagIndex>,
            std::experimental::layout_right,
            Kokkos::DefaultHostExecutionSpace::memory_space>
            tensor(tensor_alloc);

    /*
    for (int i = 1; i < 4; ++i) {
            tensor(ddc::DiscreteElement<DiagIndex>(i)) = i;
    }
    */

    tensor(tensor_accessor.element<X, X>()) = 1.;
    tensor(tensor_accessor.element<Y, Y>()) = 2.;
    tensor(tensor_accessor.element<Z, Z>()) = 3.;

    EXPECT_EQ(tensor.get(tensor_accessor.element<X, X>()), 1.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<X, Y>()), 0.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<X, Z>()), 0.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<Y, X>()), 0.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<Y, Y>()), 2.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<Y, Z>()), 0.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<Z, X>()), 0.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<Z, Y>()), 0.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<Z, Z>()), 3.);
}

struct SymIndex : sil::tensor::SymmetricTensorIndex<Mu, Nu>
{
};

TEST(Tensor, SymmetricTensorIndexing3x3)
{
    sil::tensor::TensorAccessor<SymIndex> tensor_accessor;
    ddc::DiscreteDomain<SymIndex> tensor_dom = tensor_accessor.mem_domain();
    ddc::Chunk tensor_alloc(tensor_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor<
            double,
            ddc::DiscreteDomain<SymIndex>,
            std::experimental::layout_right,
            Kokkos::DefaultHostExecutionSpace::memory_space>
            tensor(tensor_alloc);


    /*
    for (int i = 0; i < 6; ++i) {
        tensor(ddc::DiscreteElement<SymIndex>(i)) = i;
    }
    */

    tensor(tensor_accessor.element<X, X>()) = 0.;
    tensor(tensor_accessor.element<X, Y>()) = 1.;
    tensor(tensor_accessor.element<X, Z>()) = 2.;
    tensor(tensor_accessor.element<Y, Y>()) = 3.;
    tensor(tensor_accessor.element<Y, Z>()) = 4.;
    tensor(tensor_accessor.element<Z, Z>()) = 5.;

    EXPECT_EQ(tensor.get(tensor_accessor.element<X, X>()), 0.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<X, Y>()), 1.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<X, Z>()), 2.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<Y, X>()), 1.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<Y, Y>()), 3.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<Y, Z>()), 4.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<Z, X>()), 2.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<Z, Y>()), 4.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<Z, Z>()), 5.);
}

struct SymIndex3x3x3 : sil::tensor::SymmetricTensorIndex<Alpha, Beta, Gamma>
{
};

TEST(Tensor, SymmetricTensorIndexing3x3x3)
{
    sil::tensor::TensorAccessor<SymIndex3x3x3> tensor_accessor;
    ddc::DiscreteDomain<SymIndex3x3x3> tensor_dom = tensor_accessor.mem_domain();
    ddc::Chunk tensor_alloc(tensor_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor<
            double,
            ddc::DiscreteDomain<SymIndex3x3x3>,
            std::experimental::layout_right,
            Kokkos::DefaultHostExecutionSpace::memory_space>
            tensor(tensor_alloc);

    /*
    for (int i = 0; i < 10; ++i) {
        tensor(ddc::DiscreteElement<SymIndex3x3x3>(i)) = i;
    }
    */

    tensor(tensor_accessor.element<X, X, X>()) = 0.;
    tensor(tensor_accessor.element<X, X, Y>()) = 1.;
    tensor(tensor_accessor.element<X, X, Z>()) = 2.;
    tensor(tensor_accessor.element<X, Y, Y>()) = 3.;
    tensor(tensor_accessor.element<X, Y, Z>()) = 4.;
    tensor(tensor_accessor.element<X, Z, Z>()) = 5.;
    tensor(tensor_accessor.element<Y, Y, Y>()) = 6.;
    tensor(tensor_accessor.element<Y, Y, Z>()) = 7.;
    tensor(tensor_accessor.element<Y, Z, Z>()) = 8.;
    tensor(tensor_accessor.element<Z, Z, Z>()) = 9.;

    EXPECT_EQ(tensor.get(tensor_accessor.element<X, X, X>()), 0.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<X, X, Y>()), 1.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<X, X, Z>()), 2.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<X, Y, X>()), 1.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<X, Y, Y>()), 3.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<X, Y, Z>()), 4.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<X, Z, X>()), 2.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<X, Z, Y>()), 4.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<X, Z, Z>()), 5.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<Y, X, X>()), 1.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<Y, X, Y>()), 3.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<Y, X, Z>()), 4.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<Y, Y, X>()), 3.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<Y, Y, Y>()), 6.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<Y, Y, Z>()), 7.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<Y, Z, X>()), 4.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<Y, Z, Y>()), 7.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<Y, Z, Z>()), 8.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<Z, X, X>()), 2.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<Z, X, Y>()), 4.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<Z, X, Z>()), 5.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<Z, Y, X>()), 4.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<Z, Y, Y>()), 7.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<Z, Y, Z>()), 8.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<Z, Z, X>()), 5.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<Z, Z, Y>()), 8.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<Z, Z, Z>()), 9.);
}

struct AntisymIndex : sil::tensor::AntisymmetricTensorIndex<Mu, Nu>
{
};

TEST(Tensor, AntisymmetricTensorIndexing3x3)
{
    sil::tensor::TensorAccessor<AntisymIndex> tensor_accessor;
    ddc::DiscreteDomain<AntisymIndex> tensor_dom = tensor_accessor.mem_domain();
    ddc::Chunk tensor_alloc(tensor_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor<
            double,
            ddc::DiscreteDomain<AntisymIndex>,
            std::experimental::layout_right,
            Kokkos::DefaultHostExecutionSpace::memory_space>
            tensor(tensor_alloc);

    /*
    for (int i = 1; i < 4; ++i) {
        tensor(ddc::DiscreteElement<AntisymIndex>(i)) = i;
    }
    */

    tensor(tensor_accessor.element<X, Y>()) = 1.;
    tensor(tensor_accessor.element<X, Z>()) = 2.;
    tensor(tensor_accessor.element<Y, Z>()) = 3.;

    EXPECT_EQ(tensor.get(tensor_accessor.element<X, X>()), 0.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<X, Y>()), 1.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<X, Z>()), 2.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<Y, X>()), -1.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<Y, Y>()), 0.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<Y, Z>()), 3.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<Z, X>()), -2.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<Z, Y>()), -3.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<Z, Z>()), 0.);
}

struct PartiallySymIndex
    : sil::tensor::FullTensorIndex<Mu, sil::tensor::SymmetricTensorIndex<Alpha, Beta>>
{
};

TEST(Tensor, PartiallySymmetricTensorIndexing3x3x3)
{
    sil::tensor::TensorAccessor<PartiallySymIndex> tensor_accessor;
    ddc::DiscreteDomain<PartiallySymIndex> tensor_dom = tensor_accessor.mem_domain();
    ddc::Chunk tensor_alloc(tensor_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor<
            double,
            ddc::DiscreteDomain<PartiallySymIndex>,
            std::experimental::layout_right,
            Kokkos::DefaultHostExecutionSpace::memory_space>
            tensor(tensor_alloc);

    /*
    for (int i = 0; i < 18; ++i) {
        tensor(ddc::DiscreteElement<PartiallySymIndex>(i)) = i;
    }
    */

    tensor(tensor_accessor.element<X, X, X>()) = 0.;
    tensor(tensor_accessor.element<X, X, Y>()) = 1.;
    tensor(tensor_accessor.element<X, X, Z>()) = 2.;
    tensor(tensor_accessor.element<X, Y, Y>()) = 3.;
    tensor(tensor_accessor.element<X, Y, Z>()) = 4.;
    tensor(tensor_accessor.element<X, Z, Z>()) = 5.;
    tensor(tensor_accessor.element<Y, X, X>()) = 6.;
    tensor(tensor_accessor.element<Y, X, Y>()) = 7.;
    tensor(tensor_accessor.element<Y, X, Z>()) = 8.;
    tensor(tensor_accessor.element<Y, Y, Y>()) = 9.;
    tensor(tensor_accessor.element<Y, Y, Z>()) = 10.;
    tensor(tensor_accessor.element<Y, Z, Z>()) = 11.;
    tensor(tensor_accessor.element<Z, X, X>()) = 12.;
    tensor(tensor_accessor.element<Z, X, Y>()) = 13.;
    tensor(tensor_accessor.element<Z, X, Z>()) = 14.;
    tensor(tensor_accessor.element<Z, Y, Y>()) = 15.;
    tensor(tensor_accessor.element<Z, Y, Z>()) = 16.;
    tensor(tensor_accessor.element<Z, Z, Z>()) = 17.;

    EXPECT_EQ(tensor.get(tensor_accessor.element<X, X, X>()), 0.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<X, X, Y>()), 1.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<X, X, Z>()), 2.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<X, Y, X>()), 1.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<X, Y, Y>()), 3.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<X, Y, Z>()), 4.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<X, Z, X>()), 2.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<X, Z, Y>()), 4.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<X, Z, Z>()), 5.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<Y, X, X>()), 6.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<Y, X, Y>()), 7.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<Y, X, Z>()), 8.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<Y, Y, X>()), 7.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<Y, Y, Y>()), 9.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<Y, Y, Z>()), 10.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<Y, Z, X>()), 8.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<Y, Z, Y>()), 10.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<Y, Z, Z>()), 11.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<Z, X, X>()), 12.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<Z, X, Y>()), 13.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<Z, X, Z>()), 14.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<Z, Y, X>()), 13.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<Z, Y, Y>()), 15.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<Z, Y, Z>()), 16.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<Z, Z, X>()), 14.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<Z, Z, Y>()), 16.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<Z, Z, Z>()), 17.);
}
