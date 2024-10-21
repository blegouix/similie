// SPDX-FileCopyrightText: 2024 Baptiste Legouix
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

struct Delta : sil::tensor::TensorNaturalIndex<X, Y, Z>
{
};

struct Mu : sil::tensor::TensorNaturalIndex<T, X, Y, Z>
{
};

struct Nu : sil::tensor::TensorNaturalIndex<T, X, Y, Z>
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
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 4; ++j) {
            tensor(ddc::DiscreteElement<Alpha, Nu>(i, j)) = i * 4 + j;
        }
    }
    */

    tensor(tensor_accessor.element<X, T>()) = 0.;
    tensor(tensor_accessor.element<X, X>()) = 1.;
    tensor(tensor_accessor.element<X, Y>()) = 2.;
    tensor(tensor_accessor.element<X, Z>()) = 3.;
    tensor(tensor_accessor.element<Y, T>()) = 4.;
    tensor(tensor_accessor.element<Y, X>()) = 5.;
    tensor(tensor_accessor.element<Y, Y>()) = 6.;
    tensor(tensor_accessor.element<Y, Z>()) = 7.;
    tensor(tensor_accessor.element<Z, T>()) = 8.;
    tensor(tensor_accessor.element<Z, X>()) = 9.;
    tensor(tensor_accessor.element<Z, Y>()) = 10.;
    tensor(tensor_accessor.element<Z, Z>()) = 11.;

    EXPECT_EQ(tensor.get(tensor_accessor.element<X, T>()), 0.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<X, X>()), 1.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<X, Y>()), 2.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<X, Z>()), 3.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<Y, T>()), 4.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<Y, X>()), 5.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<Y, Y>()), 6.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<Y, Z>()), 7.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<Z, T>()), 8.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<Z, X>()), 9.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<Z, Y>()), 10.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<Z, Z>()), 11.);
}

struct FullIndex : sil::tensor::FullTensorIndex<Alpha, Nu>
{
};

TEST(Tensor, FullTensorIndexing)
{
    sil::tensor::TensorAccessor<FullIndex> tensor_accessor;
    ddc::DiscreteDomain<FullIndex> tensor_dom = tensor_accessor.mem_domain();
    ddc::Chunk tensor_alloc(tensor_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor<
            double,
            ddc::DiscreteDomain<FullIndex>,
            std::experimental::layout_right,
            Kokkos::DefaultHostExecutionSpace::memory_space>
            tensor(tensor_alloc);

    /*
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 4; ++j) {
            tensor(ddc::DiscreteElement<FullIndex>(i * 4 + j)) = i * 4 + j;
        }
    }
    */

    tensor(tensor_accessor.element<X, T>()) = 0.;
    tensor(tensor_accessor.element<X, X>()) = 1.;
    tensor(tensor_accessor.element<X, Y>()) = 2.;
    tensor(tensor_accessor.element<X, Z>()) = 3.;
    tensor(tensor_accessor.element<Y, T>()) = 4.;
    tensor(tensor_accessor.element<Y, X>()) = 5.;
    tensor(tensor_accessor.element<Y, Y>()) = 6.;
    tensor(tensor_accessor.element<Y, Z>()) = 7.;
    tensor(tensor_accessor.element<Z, T>()) = 8.;
    tensor(tensor_accessor.element<Z, X>()) = 9.;
    tensor(tensor_accessor.element<Z, Y>()) = 10.;
    tensor(tensor_accessor.element<Z, Z>()) = 11.;

    EXPECT_EQ(tensor.get(tensor_accessor.element<X, T>()), 0.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<X, X>()), 1.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<X, Y>()), 2.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<X, Z>()), 3.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<Y, T>()), 4.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<Y, X>()), 5.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<Y, Y>()), 6.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<Y, Z>()), 7.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<Z, T>()), 8.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<Z, X>()), 9.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<Z, Y>()), 10.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<Z, Z>()), 11.);
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

    EXPECT_EQ(tensor.get(tensor_accessor.element<T, T>()), 1.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<T, X>()), 0.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<T, Y>()), 0.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<T, Z>()), 0.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<X, T>()), 0.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<X, X>()), 1.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<X, Y>()), 0.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<X, Z>()), 0.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<Y, T>()), 0.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<Y, X>()), 0.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<Y, Y>()), 1.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<Y, Z>()), 0.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<Z, T>()), 0.);
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
    for (int i = 1; i < 5; ++i) {
            tensor(ddc::DiscreteElement<DiagIndex>(i)) = i;
    }
    */

    tensor(tensor_accessor.element<T, T>()) = 1.;
    tensor(tensor_accessor.element<X, X>()) = 2.;
    tensor(tensor_accessor.element<Y, Y>()) = 3.;
    tensor(tensor_accessor.element<Z, Z>()) = 4.;

    EXPECT_EQ(tensor.get(tensor_accessor.element<T, T>()), 1.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<T, X>()), 0.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<T, Y>()), 0.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<T, Z>()), 0.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<X, T>()), 0.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<X, X>()), 2.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<X, Y>()), 0.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<X, Z>()), 0.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<Y, T>()), 0.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<Y, X>()), 0.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<Y, Y>()), 3.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<Y, Z>()), 0.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<Z, T>()), 0.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<Z, X>()), 0.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<Z, Y>()), 0.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<Z, Z>()), 4.);
}

struct SymIndex : sil::tensor::SymmetricTensorIndex<Mu, Nu>
{
};

TEST(Tensor, SymmetricTensorIndexing4x4)
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
    for (int i = 0; i < 10; ++i) {
        tensor(ddc::DiscreteElement<SymIndex>(i)) = i;
    }
    */

    tensor(tensor_accessor.element<T, T>()) = 0.;
    tensor(tensor_accessor.element<T, X>()) = 1.;
    tensor(tensor_accessor.element<T, Y>()) = 2.;
    tensor(tensor_accessor.element<T, Z>()) = 3.;
    tensor(tensor_accessor.element<X, X>()) = 4.;
    tensor(tensor_accessor.element<X, Y>()) = 5.;
    tensor(tensor_accessor.element<X, Z>()) = 6.;
    tensor(tensor_accessor.element<Y, Y>()) = 7.;
    tensor(tensor_accessor.element<Y, Z>()) = 8.;
    tensor(tensor_accessor.element<Z, Z>()) = 9.;

    EXPECT_EQ(tensor.get(tensor_accessor.element<T, T>()), 0.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<T, X>()), 1.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<T, Y>()), 2.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<T, Z>()), 3.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<X, T>()), 1.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<X, X>()), 4.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<X, Y>()), 5.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<X, Z>()), 6.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<Y, T>()), 2.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<Y, X>()), 5.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<Y, Y>()), 7.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<Y, Z>()), 8.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<Z, T>()), 3.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<Z, X>()), 6.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<Z, Y>()), 8.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<Z, Z>()), 9.);
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

TEST(Tensor, AntisymmetricTensorIndexing4x4)
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
    for (int i = 1; i < 7; ++i) {
        tensor(ddc::DiscreteElement<AntisymIndex>(i)) = i;
    }
    */

    tensor(tensor_accessor.element<T, X>()) = 1.;
    tensor(tensor_accessor.element<T, Y>()) = 2.;
    tensor(tensor_accessor.element<T, Z>()) = 3.;
    tensor(tensor_accessor.element<X, Y>()) = 4.;
    tensor(tensor_accessor.element<X, Z>()) = 5.;
    tensor(tensor_accessor.element<Y, Z>()) = 6.;

    EXPECT_EQ(tensor.get(tensor_accessor.element<T, T>()), 0.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<T, X>()), 1.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<T, Y>()), 2.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<T, Z>()), 3.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<X, T>()), -1.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<X, X>()), 0.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<X, Y>()), 4.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<X, Z>()), 5.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<Y, T>()), -2.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<Y, X>()), -4.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<Y, Y>()), 0.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<Y, Z>()), 6.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<Z, T>()), -3.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<Z, X>()), -5.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<Z, Y>()), -6.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<Z, Z>()), 0.);
}

struct PartiallySymIndex
    : sil::tensor::FullTensorIndex<Mu, sil::tensor::SymmetricTensorIndex<Alpha, Beta>>
{
};

TEST(Tensor, PartiallySymmetricTensorIndexing4x3x3)
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
    for (int i = 0; i < 24; ++i) {
        tensor(ddc::DiscreteElement<PartiallySymIndex>(i)) = i;
    }
    */

    tensor(tensor_accessor.element<T, X, X>()) = 0.;
    tensor(tensor_accessor.element<T, X, Y>()) = 1.;
    tensor(tensor_accessor.element<T, X, Z>()) = 2.;
    tensor(tensor_accessor.element<T, Y, Y>()) = 3.;
    tensor(tensor_accessor.element<T, Y, Z>()) = 4.;
    tensor(tensor_accessor.element<T, Z, Z>()) = 5.;
    tensor(tensor_accessor.element<X, X, X>()) = 6.;
    tensor(tensor_accessor.element<X, X, Y>()) = 7.;
    tensor(tensor_accessor.element<X, X, Z>()) = 8.;
    tensor(tensor_accessor.element<X, Y, Y>()) = 9.;
    tensor(tensor_accessor.element<X, Y, Z>()) = 10.;
    tensor(tensor_accessor.element<X, Z, Z>()) = 11.;
    tensor(tensor_accessor.element<Y, X, X>()) = 12.;
    tensor(tensor_accessor.element<Y, X, Y>()) = 13.;
    tensor(tensor_accessor.element<Y, X, Z>()) = 14.;
    tensor(tensor_accessor.element<Y, Y, Y>()) = 15.;
    tensor(tensor_accessor.element<Y, Y, Z>()) = 16.;
    tensor(tensor_accessor.element<Y, Z, Z>()) = 17.;
    tensor(tensor_accessor.element<Z, X, X>()) = 18.;
    tensor(tensor_accessor.element<Z, X, Y>()) = 19.;
    tensor(tensor_accessor.element<Z, X, Z>()) = 20.;
    tensor(tensor_accessor.element<Z, Y, Y>()) = 21.;
    tensor(tensor_accessor.element<Z, Y, Z>()) = 22.;
    tensor(tensor_accessor.element<Z, Z, Z>()) = 23.;

    EXPECT_EQ(tensor.get(tensor_accessor.element<T, X, X>()), 0.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<T, X, Y>()), 1.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<T, X, Z>()), 2.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<T, Y, X>()), 1.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<T, Y, Y>()), 3.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<T, Y, Z>()), 4.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<T, Z, X>()), 2.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<T, Z, Y>()), 4.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<T, Z, Z>()), 5.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<X, X, X>()), 6.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<X, X, Y>()), 7.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<X, X, Z>()), 8.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<X, Y, X>()), 7.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<X, Y, Y>()), 9.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<X, Y, Z>()), 10.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<X, Z, X>()), 8.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<X, Z, Y>()), 10.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<X, Z, Z>()), 11.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<Y, X, X>()), 12.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<Y, X, Y>()), 13.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<Y, X, Z>()), 14.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<Y, Y, X>()), 13.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<Y, Y, Y>()), 15.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<Y, Y, Z>()), 16.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<Y, Z, X>()), 14.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<Y, Z, Y>()), 16.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<Y, Z, Z>()), 17.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<Z, X, X>()), 18.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<Z, X, Y>()), 19.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<Z, X, Z>()), 20.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<Z, Y, X>()), 19.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<Z, Y, Y>()), 21.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<Z, Y, Z>()), 22.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<Z, Z, X>()), 20.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<Z, Z, Y>()), 22.);
    EXPECT_EQ(tensor.get(tensor_accessor.element<Z, Z, Z>()), 23.);
}

struct YoungTableauIndex
    : sil::tensor::YoungTableauTensorIndex<
              sil::young_tableau::YoungTableau<
                      3,
                      sil::young_tableau::YoungTableauSeq<std::index_sequence<1, 2, 3>>>,
              Alpha,
              Beta,
              Gamma>
{
};

TEST(Tensor, YoungTableauIndexing)
{
    sil::tensor::TensorAccessor<YoungTableauIndex> tensor_accessor;
    ddc::DiscreteDomain<YoungTableauIndex> tensor_dom = tensor_accessor.mem_domain();
    ddc::Chunk tensor_alloc(tensor_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor<
            double,
            ddc::DiscreteDomain<YoungTableauIndex>,
            std::experimental::layout_right,
            Kokkos::DefaultHostExecutionSpace::memory_space>
            tensor(tensor_alloc);
    sil::young_tableau::
            YoungTableau<3, sil::young_tableau::YoungTableauSeq<std::index_sequence<1, 2, 3>>>
                    young_tableau {};
    young_tableau.print_u();
    // SymIndex3x3x3::young_tableau().print_representation_absent();

    /*
    for (int i = 0; i < 10; ++i) {
        tensor(ddc::DiscreteElement<SymIndex3x3x3>(i)) = i;
    }
    */

    /*
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
*/
}

TEST(TensorProd, SimpleContractionRank3xRank2)
{
    sil::tensor::TensorAccessor<Alpha, Beta, Gamma> tensor_accessor1;
    ddc::DiscreteDomain<Alpha, Beta, Gamma> tensor1_dom = tensor_accessor1.mem_domain();
    ddc::Chunk tensor1_alloc(tensor1_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor<
            double,
            ddc::DiscreteDomain<Alpha, Beta, Gamma>,
            std::experimental::layout_right,
            Kokkos::DefaultHostExecutionSpace::memory_space>
            tensor1(tensor1_alloc);

    tensor1(tensor_accessor1.element<X, X, X>()) = 0.;
    tensor1(tensor_accessor1.element<X, X, Y>()) = 1.;
    tensor1(tensor_accessor1.element<X, X, Z>()) = 2.;
    tensor1(tensor_accessor1.element<X, Y, X>()) = 3.;
    tensor1(tensor_accessor1.element<X, Y, Y>()) = 4.;
    tensor1(tensor_accessor1.element<X, Y, Z>()) = 5.;
    tensor1(tensor_accessor1.element<X, Z, X>()) = 6.;
    tensor1(tensor_accessor1.element<X, Z, Y>()) = 7.;
    tensor1(tensor_accessor1.element<X, Z, Z>()) = 8.;
    tensor1(tensor_accessor1.element<Y, X, X>()) = 9.;
    tensor1(tensor_accessor1.element<Y, X, Y>()) = 10.;
    tensor1(tensor_accessor1.element<Y, X, Z>()) = 11.;
    tensor1(tensor_accessor1.element<Y, Y, X>()) = 12.;
    tensor1(tensor_accessor1.element<Y, Y, Y>()) = 13.;
    tensor1(tensor_accessor1.element<Y, Y, Z>()) = 14.;
    tensor1(tensor_accessor1.element<Y, Z, X>()) = 15.;
    tensor1(tensor_accessor1.element<Y, Z, Y>()) = 16.;
    tensor1(tensor_accessor1.element<Y, Z, Z>()) = 17.;
    tensor1(tensor_accessor1.element<Z, X, X>()) = 18.;
    tensor1(tensor_accessor1.element<Z, X, Y>()) = 19.;
    tensor1(tensor_accessor1.element<Z, X, Z>()) = 20.;
    tensor1(tensor_accessor1.element<Z, Y, X>()) = 21.;
    tensor1(tensor_accessor1.element<Z, Y, Y>()) = 22.;
    tensor1(tensor_accessor1.element<Z, Y, Z>()) = 23.;
    tensor1(tensor_accessor1.element<Z, Z, X>()) = 24.;
    tensor1(tensor_accessor1.element<Z, Z, Y>()) = 25.;
    tensor1(tensor_accessor1.element<Z, Z, Z>()) = 26.;


    sil::tensor::TensorAccessor<Gamma, Delta> tensor_accessor2;
    ddc::DiscreteDomain<Gamma, Delta> tensor2_dom = tensor_accessor2.mem_domain();
    ddc::Chunk tensor2_alloc(tensor2_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor<
            double,
            ddc::DiscreteDomain<Gamma, Delta>,
            std::experimental::layout_right,
            Kokkos::DefaultHostExecutionSpace::memory_space>
            tensor2(tensor2_alloc);

    tensor2(tensor_accessor2.element<X, X>()) = 0.;
    tensor2(tensor_accessor2.element<X, Y>()) = 1.;
    tensor2(tensor_accessor2.element<X, Z>()) = 2.;
    tensor2(tensor_accessor2.element<Y, X>()) = 3.;
    tensor2(tensor_accessor2.element<Y, Y>()) = 4.;
    tensor2(tensor_accessor2.element<Y, Z>()) = 5.;
    tensor2(tensor_accessor2.element<Z, X>()) = 6.;
    tensor2(tensor_accessor2.element<Z, Y>()) = 7.;
    tensor2(tensor_accessor2.element<Z, Z>()) = 8.;

    sil::tensor::TensorAccessor<Alpha, Beta, Delta> prod_tensor_accessor;
    ddc::DiscreteDomain<Alpha, Beta, Delta> prod_tensor_dom = prod_tensor_accessor.mem_domain();

    ddc::Chunk prod_tensor_alloc(prod_tensor_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor<
            double,
            ddc::DiscreteDomain<Alpha, Beta, Delta>,
            std::experimental::layout_right,
            Kokkos::DefaultHostExecutionSpace::memory_space>
            prod_tensor(prod_tensor_alloc);

    sil::tensor::tensor_prod(prod_tensor, tensor1, tensor2);

    EXPECT_EQ(prod_tensor.get(prod_tensor_accessor.element<X, X, X>()), 15.);
    EXPECT_EQ(prod_tensor.get(prod_tensor_accessor.element<X, X, Y>()), 18.);
    EXPECT_EQ(prod_tensor.get(prod_tensor_accessor.element<X, X, Z>()), 21.);
    EXPECT_EQ(prod_tensor.get(prod_tensor_accessor.element<X, Y, X>()), 42.);
    EXPECT_EQ(prod_tensor.get(prod_tensor_accessor.element<X, Y, Y>()), 54.);
    EXPECT_EQ(prod_tensor.get(prod_tensor_accessor.element<X, Y, Z>()), 66.);
    EXPECT_EQ(prod_tensor.get(prod_tensor_accessor.element<X, Z, X>()), 69.);
    EXPECT_EQ(prod_tensor.get(prod_tensor_accessor.element<X, Z, Y>()), 90.);
    EXPECT_EQ(prod_tensor.get(prod_tensor_accessor.element<X, Z, Z>()), 111.);
    EXPECT_EQ(prod_tensor.get(prod_tensor_accessor.element<Y, X, X>()), 96.);
    EXPECT_EQ(prod_tensor.get(prod_tensor_accessor.element<Y, X, Y>()), 126.);
    EXPECT_EQ(prod_tensor.get(prod_tensor_accessor.element<Y, X, Z>()), 156.);
    EXPECT_EQ(prod_tensor.get(prod_tensor_accessor.element<Y, Y, X>()), 123.);
    EXPECT_EQ(prod_tensor.get(prod_tensor_accessor.element<Y, Y, Y>()), 162.);
    EXPECT_EQ(prod_tensor.get(prod_tensor_accessor.element<Y, Y, Z>()), 201.);
    EXPECT_EQ(prod_tensor.get(prod_tensor_accessor.element<Y, Z, X>()), 150.);
    EXPECT_EQ(prod_tensor.get(prod_tensor_accessor.element<Y, Z, Y>()), 198.);
    EXPECT_EQ(prod_tensor.get(prod_tensor_accessor.element<Y, Z, Z>()), 246.);
    EXPECT_EQ(prod_tensor.get(prod_tensor_accessor.element<Z, X, X>()), 177.);
    EXPECT_EQ(prod_tensor.get(prod_tensor_accessor.element<Z, X, Y>()), 234.);
    EXPECT_EQ(prod_tensor.get(prod_tensor_accessor.element<Z, X, Z>()), 291.);
    EXPECT_EQ(prod_tensor.get(prod_tensor_accessor.element<Z, Y, X>()), 204.);
    EXPECT_EQ(prod_tensor.get(prod_tensor_accessor.element<Z, Y, Y>()), 270.);
    EXPECT_EQ(prod_tensor.get(prod_tensor_accessor.element<Z, Y, Z>()), 336.);
    EXPECT_EQ(prod_tensor.get(prod_tensor_accessor.element<Z, Z, X>()), 231.);
    EXPECT_EQ(prod_tensor.get(prod_tensor_accessor.element<Z, Z, Y>()), 306.);
    EXPECT_EQ(prod_tensor.get(prod_tensor_accessor.element<Z, Z, Z>()), 381.);
}

TEST(TensorProd, DoubleContractionRank3xRank3)
{
    sil::tensor::TensorAccessor<Alpha, Beta, Gamma> tensor_accessor1;
    ddc::DiscreteDomain<Alpha, Beta, Gamma> tensor1_dom = tensor_accessor1.mem_domain();
    ddc::Chunk tensor1_alloc(tensor1_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor<
            double,
            ddc::DiscreteDomain<Alpha, Beta, Gamma>,
            std::experimental::layout_right,
            Kokkos::DefaultHostExecutionSpace::memory_space>
            tensor1(tensor1_alloc);

    tensor1(tensor_accessor1.element<X, X, X>()) = 0.;
    tensor1(tensor_accessor1.element<X, X, Y>()) = 1.;
    tensor1(tensor_accessor1.element<X, X, Z>()) = 2.;
    tensor1(tensor_accessor1.element<X, Y, X>()) = 3.;
    tensor1(tensor_accessor1.element<X, Y, Y>()) = 4.;
    tensor1(tensor_accessor1.element<X, Y, Z>()) = 5.;
    tensor1(tensor_accessor1.element<X, Z, X>()) = 6.;
    tensor1(tensor_accessor1.element<X, Z, Y>()) = 7.;
    tensor1(tensor_accessor1.element<X, Z, Z>()) = 8.;
    tensor1(tensor_accessor1.element<Y, X, X>()) = 9.;
    tensor1(tensor_accessor1.element<Y, X, Y>()) = 10.;
    tensor1(tensor_accessor1.element<Y, X, Z>()) = 11.;
    tensor1(tensor_accessor1.element<Y, Y, X>()) = 12.;
    tensor1(tensor_accessor1.element<Y, Y, Y>()) = 13.;
    tensor1(tensor_accessor1.element<Y, Y, Z>()) = 14.;
    tensor1(tensor_accessor1.element<Y, Z, X>()) = 15.;
    tensor1(tensor_accessor1.element<Y, Z, Y>()) = 16.;
    tensor1(tensor_accessor1.element<Y, Z, Z>()) = 17.;
    tensor1(tensor_accessor1.element<Z, X, X>()) = 18.;
    tensor1(tensor_accessor1.element<Z, X, Y>()) = 19.;
    tensor1(tensor_accessor1.element<Z, X, Z>()) = 20.;
    tensor1(tensor_accessor1.element<Z, Y, X>()) = 21.;
    tensor1(tensor_accessor1.element<Z, Y, Y>()) = 22.;
    tensor1(tensor_accessor1.element<Z, Y, Z>()) = 23.;
    tensor1(tensor_accessor1.element<Z, Z, X>()) = 24.;
    tensor1(tensor_accessor1.element<Z, Z, Y>()) = 25.;
    tensor1(tensor_accessor1.element<Z, Z, Z>()) = 26.;


    sil::tensor::TensorAccessor<Beta, Gamma, Delta> tensor_accessor2;
    ddc::DiscreteDomain<Beta, Gamma, Delta> tensor2_dom = tensor_accessor2.mem_domain();
    ddc::Chunk tensor2_alloc(tensor2_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor<
            double,
            ddc::DiscreteDomain<Beta, Gamma, Delta>,
            std::experimental::layout_right,
            Kokkos::DefaultHostExecutionSpace::memory_space>
            tensor2(tensor2_alloc);

    tensor2(tensor_accessor2.element<X, X, X>()) = 0.;
    tensor2(tensor_accessor2.element<X, X, Y>()) = 1.;
    tensor2(tensor_accessor2.element<X, X, Z>()) = 2.;
    tensor2(tensor_accessor2.element<X, Y, X>()) = 3.;
    tensor2(tensor_accessor2.element<X, Y, Y>()) = 4.;
    tensor2(tensor_accessor2.element<X, Y, Z>()) = 5.;
    tensor2(tensor_accessor2.element<X, Z, X>()) = 6.;
    tensor2(tensor_accessor2.element<X, Z, Y>()) = 7.;
    tensor2(tensor_accessor2.element<X, Z, Z>()) = 8.;
    tensor2(tensor_accessor2.element<Y, X, X>()) = 9.;
    tensor2(tensor_accessor2.element<Y, X, Y>()) = 10.;
    tensor2(tensor_accessor2.element<Y, X, Z>()) = 11.;
    tensor2(tensor_accessor2.element<Y, Y, X>()) = 12.;
    tensor2(tensor_accessor2.element<Y, Y, Y>()) = 13.;
    tensor2(tensor_accessor2.element<Y, Y, Z>()) = 14.;
    tensor2(tensor_accessor2.element<Y, Z, X>()) = 15.;
    tensor2(tensor_accessor2.element<Y, Z, Y>()) = 16.;
    tensor2(tensor_accessor2.element<Y, Z, Z>()) = 17.;
    tensor2(tensor_accessor2.element<Z, X, X>()) = 18.;
    tensor2(tensor_accessor2.element<Z, X, Y>()) = 19.;
    tensor2(tensor_accessor2.element<Z, X, Z>()) = 20.;
    tensor2(tensor_accessor2.element<Z, Y, X>()) = 21.;
    tensor2(tensor_accessor2.element<Z, Y, Y>()) = 22.;
    tensor2(tensor_accessor2.element<Z, Y, Z>()) = 23.;
    tensor2(tensor_accessor2.element<Z, Z, X>()) = 24.;
    tensor2(tensor_accessor2.element<Z, Z, Y>()) = 25.;
    tensor2(tensor_accessor2.element<Z, Z, Z>()) = 26.;

    sil::tensor::TensorAccessor<Alpha, Delta> prod_tensor_accessor;
    ddc::DiscreteDomain<Alpha, Delta> prod_tensor_dom = prod_tensor_accessor.mem_domain();

    ddc::Chunk prod_tensor_alloc(prod_tensor_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor<
            double,
            ddc::DiscreteDomain<Alpha, Delta>,
            std::experimental::layout_right,
            Kokkos::DefaultHostExecutionSpace::memory_space>
            prod_tensor(prod_tensor_alloc);

    sil::tensor::tensor_prod(prod_tensor, tensor1, tensor2);

    EXPECT_EQ(prod_tensor.get(prod_tensor_accessor.element<X, X>()), 612.);
    EXPECT_EQ(prod_tensor.get(prod_tensor_accessor.element<X, Y>()), 648.);
    EXPECT_EQ(prod_tensor.get(prod_tensor_accessor.element<X, Z>()), 684.);
    EXPECT_EQ(prod_tensor.get(prod_tensor_accessor.element<Y, X>()), 1584.);
    EXPECT_EQ(prod_tensor.get(prod_tensor_accessor.element<Y, Y>()), 1701.);
    EXPECT_EQ(prod_tensor.get(prod_tensor_accessor.element<Y, Z>()), 1818.);
    EXPECT_EQ(prod_tensor.get(prod_tensor_accessor.element<Z, X>()), 2556.);
    EXPECT_EQ(prod_tensor.get(prod_tensor_accessor.element<Z, Y>()), 2754.);
    EXPECT_EQ(prod_tensor.get(prod_tensor_accessor.element<Z, Z>()), 2952.);
}

TEST(TensorPrint, Rank3xRank3)
{
    sil::tensor::TensorAccessor<Alpha, Beta, Gamma> tensor_accessor;
    ddc::DiscreteDomain<Alpha, Beta, Gamma> tensor_dom = tensor_accessor.mem_domain();
    ddc::Chunk tensor_alloc(tensor_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor<
            double,
            ddc::DiscreteDomain<Alpha, Beta, Gamma>,
            std::experimental::layout_right,
            Kokkos::DefaultHostExecutionSpace::memory_space>
            tensor(tensor_alloc);

    tensor(tensor_accessor.element<X, X, X>()) = 0.;
    tensor(tensor_accessor.element<X, X, Y>()) = 1.;
    tensor(tensor_accessor.element<X, X, Z>()) = 2.;
    tensor(tensor_accessor.element<X, Y, X>()) = 3.;
    tensor(tensor_accessor.element<X, Y, Y>()) = 4.;
    tensor(tensor_accessor.element<X, Y, Z>()) = 5.;
    tensor(tensor_accessor.element<X, Z, X>()) = 6.;
    tensor(tensor_accessor.element<X, Z, Y>()) = 7.;
    tensor(tensor_accessor.element<X, Z, Z>()) = 8.;
    tensor(tensor_accessor.element<Y, X, X>()) = 9.;
    tensor(tensor_accessor.element<Y, X, Y>()) = 10.;
    tensor(tensor_accessor.element<Y, X, Z>()) = 11.;
    tensor(tensor_accessor.element<Y, Y, X>()) = 12.;
    tensor(tensor_accessor.element<Y, Y, Y>()) = 13.;
    tensor(tensor_accessor.element<Y, Y, Z>()) = 14.;
    tensor(tensor_accessor.element<Y, Z, X>()) = 15.;
    tensor(tensor_accessor.element<Y, Z, Y>()) = 16.;
    tensor(tensor_accessor.element<Y, Z, Z>()) = 17.;
    tensor(tensor_accessor.element<Z, X, X>()) = 18.;
    tensor(tensor_accessor.element<Z, X, Y>()) = 19.;
    tensor(tensor_accessor.element<Z, X, Z>()) = 20.;
    tensor(tensor_accessor.element<Z, Y, X>()) = 21.;
    tensor(tensor_accessor.element<Z, Y, Y>()) = 22.;
    tensor(tensor_accessor.element<Z, Y, Z>()) = 23.;
    tensor(tensor_accessor.element<Z, Z, X>()) = 24.;
    tensor(tensor_accessor.element<Z, Z, Y>()) = 25.;
    tensor(tensor_accessor.element<Z, Z, Z>()) = 26.;

    std::cout << tensor;
}
