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

struct Delta : sil::tensor::TensorNaturalIndex<X, Y, Z>
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
            Kokkos::layout_right,
            Kokkos::DefaultHostExecutionSpace::memory_space>
            tensor1(tensor1_alloc);

    tensor1(tensor1.access_element<X, X, X>()) = 0.;
    tensor1(tensor1.access_element<X, X, Y>()) = 1.;
    tensor1(tensor1.access_element<X, X, Z>()) = 2.;
    tensor1(tensor1.access_element<X, Y, X>()) = 3.;
    tensor1(tensor1.access_element<X, Y, Y>()) = 4.;
    tensor1(tensor1.access_element<X, Y, Z>()) = 5.;
    tensor1(tensor1.access_element<X, Z, X>()) = 6.;
    tensor1(tensor1.access_element<X, Z, Y>()) = 7.;
    tensor1(tensor1.access_element<X, Z, Z>()) = 8.;
    tensor1(tensor1.access_element<Y, X, X>()) = 9.;
    tensor1(tensor1.access_element<Y, X, Y>()) = 10.;
    tensor1(tensor1.access_element<Y, X, Z>()) = 11.;
    tensor1(tensor1.access_element<Y, Y, X>()) = 12.;
    tensor1(tensor1.access_element<Y, Y, Y>()) = 13.;
    tensor1(tensor1.access_element<Y, Y, Z>()) = 14.;
    tensor1(tensor1.access_element<Y, Z, X>()) = 15.;
    tensor1(tensor1.access_element<Y, Z, Y>()) = 16.;
    tensor1(tensor1.access_element<Y, Z, Z>()) = 17.;
    tensor1(tensor1.access_element<Z, X, X>()) = 18.;
    tensor1(tensor1.access_element<Z, X, Y>()) = 19.;
    tensor1(tensor1.access_element<Z, X, Z>()) = 20.;
    tensor1(tensor1.access_element<Z, Y, X>()) = 21.;
    tensor1(tensor1.access_element<Z, Y, Y>()) = 22.;
    tensor1(tensor1.access_element<Z, Y, Z>()) = 23.;
    tensor1(tensor1.access_element<Z, Z, X>()) = 24.;
    tensor1(tensor1.access_element<Z, Z, Y>()) = 25.;
    tensor1(tensor1.access_element<Z, Z, Z>()) = 26.;


    sil::tensor::TensorAccessor<Gamma, Delta> tensor_accessor2;
    ddc::DiscreteDomain<Gamma, Delta> tensor2_dom = tensor_accessor2.mem_domain();
    ddc::Chunk tensor2_alloc(tensor2_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor<
            double,
            ddc::DiscreteDomain<Gamma, Delta>,
            Kokkos::layout_right,
            Kokkos::DefaultHostExecutionSpace::memory_space>
            tensor2(tensor2_alloc);

    tensor2(tensor2.access_element<X, X>()) = 0.;
    tensor2(tensor2.access_element<X, Y>()) = 1.;
    tensor2(tensor2.access_element<X, Z>()) = 2.;
    tensor2(tensor2.access_element<Y, X>()) = 3.;
    tensor2(tensor2.access_element<Y, Y>()) = 4.;
    tensor2(tensor2.access_element<Y, Z>()) = 5.;
    tensor2(tensor2.access_element<Z, X>()) = 6.;
    tensor2(tensor2.access_element<Z, Y>()) = 7.;
    tensor2(tensor2.access_element<Z, Z>()) = 8.;

    sil::tensor::TensorAccessor<Alpha, Beta, Delta> prod_tensor_accessor;
    ddc::DiscreteDomain<Alpha, Beta, Delta> prod_tensor_dom = prod_tensor_accessor.mem_domain();

    ddc::Chunk prod_tensor_alloc(prod_tensor_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor<
            double,
            ddc::DiscreteDomain<Alpha, Beta, Delta>,
            Kokkos::layout_right,
            Kokkos::DefaultHostExecutionSpace::memory_space>
            prod_tensor(prod_tensor_alloc);

    sil::tensor::natural_tensor_prod(prod_tensor, tensor1, tensor2);

    EXPECT_EQ(prod_tensor.get(prod_tensor.access_element<X, X, X>()), 15.);
    EXPECT_EQ(prod_tensor.get(prod_tensor.access_element<X, X, Y>()), 18.);
    EXPECT_EQ(prod_tensor.get(prod_tensor.access_element<X, X, Z>()), 21.);
    EXPECT_EQ(prod_tensor.get(prod_tensor.access_element<X, Y, X>()), 42.);
    EXPECT_EQ(prod_tensor.get(prod_tensor.access_element<X, Y, Y>()), 54.);
    EXPECT_EQ(prod_tensor.get(prod_tensor.access_element<X, Y, Z>()), 66.);
    EXPECT_EQ(prod_tensor.get(prod_tensor.access_element<X, Z, X>()), 69.);
    EXPECT_EQ(prod_tensor.get(prod_tensor.access_element<X, Z, Y>()), 90.);
    EXPECT_EQ(prod_tensor.get(prod_tensor.access_element<X, Z, Z>()), 111.);
    EXPECT_EQ(prod_tensor.get(prod_tensor.access_element<Y, X, X>()), 96.);
    EXPECT_EQ(prod_tensor.get(prod_tensor.access_element<Y, X, Y>()), 126.);
    EXPECT_EQ(prod_tensor.get(prod_tensor.access_element<Y, X, Z>()), 156.);
    EXPECT_EQ(prod_tensor.get(prod_tensor.access_element<Y, Y, X>()), 123.);
    EXPECT_EQ(prod_tensor.get(prod_tensor.access_element<Y, Y, Y>()), 162.);
    EXPECT_EQ(prod_tensor.get(prod_tensor.access_element<Y, Y, Z>()), 201.);
    EXPECT_EQ(prod_tensor.get(prod_tensor.access_element<Y, Z, X>()), 150.);
    EXPECT_EQ(prod_tensor.get(prod_tensor.access_element<Y, Z, Y>()), 198.);
    EXPECT_EQ(prod_tensor.get(prod_tensor.access_element<Y, Z, Z>()), 246.);
    EXPECT_EQ(prod_tensor.get(prod_tensor.access_element<Z, X, X>()), 177.);
    EXPECT_EQ(prod_tensor.get(prod_tensor.access_element<Z, X, Y>()), 234.);
    EXPECT_EQ(prod_tensor.get(prod_tensor.access_element<Z, X, Z>()), 291.);
    EXPECT_EQ(prod_tensor.get(prod_tensor.access_element<Z, Y, X>()), 204.);
    EXPECT_EQ(prod_tensor.get(prod_tensor.access_element<Z, Y, Y>()), 270.);
    EXPECT_EQ(prod_tensor.get(prod_tensor.access_element<Z, Y, Z>()), 336.);
    EXPECT_EQ(prod_tensor.get(prod_tensor.access_element<Z, Z, X>()), 231.);
    EXPECT_EQ(prod_tensor.get(prod_tensor.access_element<Z, Z, Y>()), 306.);
    EXPECT_EQ(prod_tensor.get(prod_tensor.access_element<Z, Z, Z>()), 381.);
}

TEST(TensorProd, DoubleContractionRank3xRank3)
{
    sil::tensor::TensorAccessor<Alpha, Beta, Gamma> tensor_accessor1;
    ddc::DiscreteDomain<Alpha, Beta, Gamma> tensor1_dom = tensor_accessor1.mem_domain();
    ddc::Chunk tensor1_alloc(tensor1_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor<
            double,
            ddc::DiscreteDomain<Alpha, Beta, Gamma>,
            Kokkos::layout_right,
            Kokkos::DefaultHostExecutionSpace::memory_space>
            tensor1(tensor1_alloc);

    tensor1(tensor1.access_element<X, X, X>()) = 0.;
    tensor1(tensor1.access_element<X, X, Y>()) = 1.;
    tensor1(tensor1.access_element<X, X, Z>()) = 2.;
    tensor1(tensor1.access_element<X, Y, X>()) = 3.;
    tensor1(tensor1.access_element<X, Y, Y>()) = 4.;
    tensor1(tensor1.access_element<X, Y, Z>()) = 5.;
    tensor1(tensor1.access_element<X, Z, X>()) = 6.;
    tensor1(tensor1.access_element<X, Z, Y>()) = 7.;
    tensor1(tensor1.access_element<X, Z, Z>()) = 8.;
    tensor1(tensor1.access_element<Y, X, X>()) = 9.;
    tensor1(tensor1.access_element<Y, X, Y>()) = 10.;
    tensor1(tensor1.access_element<Y, X, Z>()) = 11.;
    tensor1(tensor1.access_element<Y, Y, X>()) = 12.;
    tensor1(tensor1.access_element<Y, Y, Y>()) = 13.;
    tensor1(tensor1.access_element<Y, Y, Z>()) = 14.;
    tensor1(tensor1.access_element<Y, Z, X>()) = 15.;
    tensor1(tensor1.access_element<Y, Z, Y>()) = 16.;
    tensor1(tensor1.access_element<Y, Z, Z>()) = 17.;
    tensor1(tensor1.access_element<Z, X, X>()) = 18.;
    tensor1(tensor1.access_element<Z, X, Y>()) = 19.;
    tensor1(tensor1.access_element<Z, X, Z>()) = 20.;
    tensor1(tensor1.access_element<Z, Y, X>()) = 21.;
    tensor1(tensor1.access_element<Z, Y, Y>()) = 22.;
    tensor1(tensor1.access_element<Z, Y, Z>()) = 23.;
    tensor1(tensor1.access_element<Z, Z, X>()) = 24.;
    tensor1(tensor1.access_element<Z, Z, Y>()) = 25.;
    tensor1(tensor1.access_element<Z, Z, Z>()) = 26.;


    sil::tensor::TensorAccessor<Beta, Gamma, Delta> tensor_accessor2;
    ddc::DiscreteDomain<Beta, Gamma, Delta> tensor2_dom = tensor_accessor2.mem_domain();
    ddc::Chunk tensor2_alloc(tensor2_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor<
            double,
            ddc::DiscreteDomain<Beta, Gamma, Delta>,
            Kokkos::layout_right,
            Kokkos::DefaultHostExecutionSpace::memory_space>
            tensor2(tensor2_alloc);

    tensor2(tensor2.access_element<X, X, X>()) = 0.;
    tensor2(tensor2.access_element<X, X, Y>()) = 1.;
    tensor2(tensor2.access_element<X, X, Z>()) = 2.;
    tensor2(tensor2.access_element<X, Y, X>()) = 3.;
    tensor2(tensor2.access_element<X, Y, Y>()) = 4.;
    tensor2(tensor2.access_element<X, Y, Z>()) = 5.;
    tensor2(tensor2.access_element<X, Z, X>()) = 6.;
    tensor2(tensor2.access_element<X, Z, Y>()) = 7.;
    tensor2(tensor2.access_element<X, Z, Z>()) = 8.;
    tensor2(tensor2.access_element<Y, X, X>()) = 9.;
    tensor2(tensor2.access_element<Y, X, Y>()) = 10.;
    tensor2(tensor2.access_element<Y, X, Z>()) = 11.;
    tensor2(tensor2.access_element<Y, Y, X>()) = 12.;
    tensor2(tensor2.access_element<Y, Y, Y>()) = 13.;
    tensor2(tensor2.access_element<Y, Y, Z>()) = 14.;
    tensor2(tensor2.access_element<Y, Z, X>()) = 15.;
    tensor2(tensor2.access_element<Y, Z, Y>()) = 16.;
    tensor2(tensor2.access_element<Y, Z, Z>()) = 17.;
    tensor2(tensor2.access_element<Z, X, X>()) = 18.;
    tensor2(tensor2.access_element<Z, X, Y>()) = 19.;
    tensor2(tensor2.access_element<Z, X, Z>()) = 20.;
    tensor2(tensor2.access_element<Z, Y, X>()) = 21.;
    tensor2(tensor2.access_element<Z, Y, Y>()) = 22.;
    tensor2(tensor2.access_element<Z, Y, Z>()) = 23.;
    tensor2(tensor2.access_element<Z, Z, X>()) = 24.;
    tensor2(tensor2.access_element<Z, Z, Y>()) = 25.;
    tensor2(tensor2.access_element<Z, Z, Z>()) = 26.;

    sil::tensor::TensorAccessor<Alpha, Delta> prod_tensor_accessor;
    ddc::DiscreteDomain<Alpha, Delta> prod_tensor_dom = prod_tensor_accessor.mem_domain();

    ddc::Chunk prod_tensor_alloc(prod_tensor_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor<
            double,
            ddc::DiscreteDomain<Alpha, Delta>,
            Kokkos::layout_right,
            Kokkos::DefaultHostExecutionSpace::memory_space>
            prod_tensor(prod_tensor_alloc);

    sil::tensor::natural_tensor_prod(prod_tensor, tensor1, tensor2);

    EXPECT_EQ(prod_tensor.get(prod_tensor.access_element<X, X>()), 612.);
    EXPECT_EQ(prod_tensor.get(prod_tensor.access_element<X, Y>()), 648.);
    EXPECT_EQ(prod_tensor.get(prod_tensor.access_element<X, Z>()), 684.);
    EXPECT_EQ(prod_tensor.get(prod_tensor.access_element<Y, X>()), 1584.);
    EXPECT_EQ(prod_tensor.get(prod_tensor.access_element<Y, Y>()), 1701.);
    EXPECT_EQ(prod_tensor.get(prod_tensor.access_element<Y, Z>()), 1818.);
    EXPECT_EQ(prod_tensor.get(prod_tensor.access_element<Z, X>()), 2556.);
    EXPECT_EQ(prod_tensor.get(prod_tensor.access_element<Z, Y>()), 2754.);
    EXPECT_EQ(prod_tensor.get(prod_tensor.access_element<Z, Z>()), 2952.);
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

TEST(TensorProd, DoubleContractionYoungIndexedxNaturalIndex)
{
    sil::tensor::TensorAccessor<Alpha, Beta, Gamma> natural_accessor;
    ddc::DiscreteDomain<Alpha, Beta, Gamma> natural_dom = natural_accessor.mem_domain();
    ddc::Chunk natural_alloc(natural_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor<
            double,
            ddc::DiscreteDomain<Alpha, Beta, Gamma>,
            Kokkos::layout_right,
            Kokkos::DefaultHostExecutionSpace::memory_space>
            natural(natural_alloc);

    natural(natural_accessor.access_element<X, X, X>()) = 0.;
    natural(natural_accessor.access_element<X, X, Y>()) = 1.;
    natural(natural_accessor.access_element<X, X, Z>()) = 2.;
    natural(natural_accessor.access_element<X, Y, X>()) = 1.;
    natural(natural_accessor.access_element<X, Y, Y>()) = 3.;
    natural(natural_accessor.access_element<X, Y, Z>()) = 4.;
    natural(natural_accessor.access_element<X, Z, X>()) = 2.;
    natural(natural_accessor.access_element<X, Z, Y>()) = 4.;
    natural(natural_accessor.access_element<X, Z, Z>()) = 5.;
    natural(natural_accessor.access_element<Y, X, X>()) = 1.;
    natural(natural_accessor.access_element<Y, X, Y>()) = 3.;
    natural(natural_accessor.access_element<Y, X, Z>()) = 4.;
    natural(natural_accessor.access_element<Y, Y, X>()) = 3.;
    natural(natural_accessor.access_element<Y, Y, Y>()) = 6.;
    natural(natural_accessor.access_element<Y, Y, Z>()) = 7.;
    natural(natural_accessor.access_element<Y, Z, X>()) = 4.;
    natural(natural_accessor.access_element<Y, Z, Y>()) = 7.;
    natural(natural_accessor.access_element<Y, Z, Z>()) = 8.;
    natural(natural_accessor.access_element<Z, X, X>()) = 2.;
    natural(natural_accessor.access_element<Z, X, Y>()) = 4.;
    natural(natural_accessor.access_element<Z, X, Z>()) = 5.;
    natural(natural_accessor.access_element<Z, Y, X>()) = 4.;
    natural(natural_accessor.access_element<Z, Y, Y>()) = 7.;
    natural(natural_accessor.access_element<Z, Y, Z>()) = 8.;
    natural(natural_accessor.access_element<Z, Z, X>()) = 5.;
    natural(natural_accessor.access_element<Z, Z, Y>()) = 8.;
    natural(natural_accessor.access_element<Z, Z, Z>()) = 9.;

    sil::tensor::TensorAccessor<YoungTableauIndex> tensor_accessor1;
    ddc::DiscreteDomain<YoungTableauIndex> tensor1_dom = tensor_accessor1.mem_domain();
    ddc::Chunk tensor1_alloc(tensor1_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor<
            double,
            ddc::DiscreteDomain<YoungTableauIndex>,
            Kokkos::layout_right,
            Kokkos::DefaultHostExecutionSpace::memory_space>
            tensor1(tensor1_alloc);

    sil::tensor::compress(tensor1, natural);

    sil::tensor::TensorAccessor<Beta, Gamma, Delta> tensor_accessor2;
    ddc::DiscreteDomain<Beta, Gamma, Delta> tensor2_dom = tensor_accessor2.mem_domain();
    ddc::Chunk tensor2_alloc(tensor2_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor<
            double,
            ddc::DiscreteDomain<Beta, Gamma, Delta>,
            Kokkos::layout_right,
            Kokkos::DefaultHostExecutionSpace::memory_space>
            tensor2(tensor2_alloc);

    tensor2(tensor2.access_element<X, X, X>()) = 0.;
    tensor2(tensor2.access_element<X, X, Y>()) = 1.;
    tensor2(tensor2.access_element<X, X, Z>()) = 2.;
    tensor2(tensor2.access_element<X, Y, X>()) = 3.;
    tensor2(tensor2.access_element<X, Y, Y>()) = 4.;
    tensor2(tensor2.access_element<X, Y, Z>()) = 5.;
    tensor2(tensor2.access_element<X, Z, X>()) = 6.;
    tensor2(tensor2.access_element<X, Z, Y>()) = 7.;
    tensor2(tensor2.access_element<X, Z, Z>()) = 8.;
    tensor2(tensor2.access_element<Y, X, X>()) = 9.;
    tensor2(tensor2.access_element<Y, X, Y>()) = 10.;
    tensor2(tensor2.access_element<Y, X, Z>()) = 11.;
    tensor2(tensor2.access_element<Y, Y, X>()) = 12.;
    tensor2(tensor2.access_element<Y, Y, Y>()) = 13.;
    tensor2(tensor2.access_element<Y, Y, Z>()) = 14.;
    tensor2(tensor2.access_element<Y, Z, X>()) = 15.;
    tensor2(tensor2.access_element<Y, Z, Y>()) = 16.;
    tensor2(tensor2.access_element<Y, Z, Z>()) = 17.;
    tensor2(tensor2.access_element<Z, X, X>()) = 18.;
    tensor2(tensor2.access_element<Z, X, Y>()) = 19.;
    tensor2(tensor2.access_element<Z, X, Z>()) = 20.;
    tensor2(tensor2.access_element<Z, Y, X>()) = 21.;
    tensor2(tensor2.access_element<Z, Y, Y>()) = 22.;
    tensor2(tensor2.access_element<Z, Y, Z>()) = 23.;
    tensor2(tensor2.access_element<Z, Z, X>()) = 24.;
    tensor2(tensor2.access_element<Z, Z, Y>()) = 25.;
    tensor2(tensor2.access_element<Z, Z, Z>()) = 26.;

    sil::tensor::TensorAccessor<Alpha, Delta> prod_tensor_accessor;
    ddc::DiscreteDomain<Alpha, Delta> prod_tensor_dom = prod_tensor_accessor.mem_domain();

    ddc::Chunk prod_tensor_alloc(prod_tensor_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor<
            double,
            ddc::DiscreteDomain<Alpha, Delta>,
            Kokkos::layout_right,
            Kokkos::DefaultHostExecutionSpace::memory_space>
            prod_tensor(prod_tensor_alloc);

    sil::tensor::tensor_prod(prod_tensor, tensor1, tensor2);

    EXPECT_DOUBLE_EQ(prod_tensor.get(prod_tensor.access_element<X, X>()), 360.);
    EXPECT_DOUBLE_EQ(prod_tensor.get(prod_tensor.access_element<X, Y>()), 382.);
    EXPECT_DOUBLE_EQ(prod_tensor.get(prod_tensor.access_element<X, Z>()), 404.);
    EXPECT_DOUBLE_EQ(prod_tensor.get(prod_tensor.access_element<Y, X>()), 648.);
    EXPECT_DOUBLE_EQ(prod_tensor.get(prod_tensor.access_element<Y, Y>()), 691.);
    EXPECT_DOUBLE_EQ(prod_tensor.get(prod_tensor.access_element<Y, Z>()), 734.);
    EXPECT_DOUBLE_EQ(prod_tensor.get(prod_tensor.access_element<Z, X>()), 756.);
    EXPECT_DOUBLE_EQ(prod_tensor.get(prod_tensor.access_element<Z, Y>()), 808.);
    EXPECT_DOUBLE_EQ(prod_tensor.get(prod_tensor.access_element<Z, Z>()), 860.);
}
