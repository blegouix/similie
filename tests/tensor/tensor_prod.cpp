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

struct Alpha : sil::tensor::TensorNaturalIndex<X, Y, Z>
{
};

struct Beta : sil::tensor::TensorNaturalIndex<X, Y, Z>
{
};

struct Gamma : sil::tensor::TensorNaturalIndex<X, Y, Z>
{
};

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
