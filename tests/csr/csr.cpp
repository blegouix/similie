// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: GPL-3.0

#include <cmath>

#include <ddc/ddc.hpp>

#include <gtest/gtest.h>

#include "csr.hpp"
#include "csr_dynamic.hpp"

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

TEST(CsrDynamic, Csr2Dense)
{
    sil::tensor::TensorAccessor<Alpha, Beta, Gamma> tensor_accessor;
    ddc::DiscreteDomain<Alpha, Beta, Gamma> tensor_dom = tensor_accessor.mem_domain();
    ddc::Chunk tensor_alloc(tensor_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor<
            double,
            ddc::DiscreteDomain<Alpha, Beta, Gamma>,
            Kokkos::layout_right,
            Kokkos::DefaultHostExecutionSpace::memory_space>
            tensor(tensor_alloc);

    ddc::parallel_fill(tensor, 0.);
    tensor(tensor_accessor.element<X, X, Y>()) = 1.;
    tensor(tensor_accessor.element<X, Z, Y>()) = 2.;
    tensor(tensor_accessor.element<Y, X, X>()) = 3.;
    tensor(tensor_accessor.element<Y, X, Z>()) = 4.;
    tensor(tensor_accessor.element<Y, Z, Z>()) = 5.;
    tensor(tensor_accessor.element<Z, X, Z>()) = 6.;
    tensor(tensor_accessor.element<Z, Y, Y>()) = 7.;
    tensor(tensor_accessor.element<Z, X, Y>()) = 8.;
    tensor(tensor_accessor.element<Z, Z, Z>()) = 9.;

    sil::csr::CsrDynamic<Alpha, Beta, Gamma> csr(tensor_dom);

    csr.push_back(tensor[ddc::DiscreteElement<Alpha>(0)]);
    csr.push_back(tensor[ddc::DiscreteElement<Alpha>(1)]);
    csr.push_back(tensor[ddc::DiscreteElement<Alpha>(2)]);

    ddc::Chunk dense_tensor_alloc(tensor_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor<
            double,
            ddc::DiscreteDomain<Alpha, Beta, Gamma>,
            Kokkos::layout_right,
            Kokkos::DefaultHostExecutionSpace::memory_space>
            dense_tensor(dense_tensor_alloc);
    sil::csr::csr2dense(dense_tensor, csr);

    EXPECT_EQ(dense_tensor.get(tensor_accessor.element<X, X, X>()), 0.);
    EXPECT_EQ(dense_tensor.get(tensor_accessor.element<X, X, Y>()), 1.);
    EXPECT_EQ(dense_tensor.get(tensor_accessor.element<X, X, Z>()), 0.);
    EXPECT_EQ(dense_tensor.get(tensor_accessor.element<X, Y, X>()), 0.);
    EXPECT_EQ(dense_tensor.get(tensor_accessor.element<X, Y, Y>()), 0.);
    EXPECT_EQ(dense_tensor.get(tensor_accessor.element<X, Y, Z>()), 0.);
    EXPECT_EQ(dense_tensor.get(tensor_accessor.element<X, Z, X>()), 0.);
    EXPECT_EQ(dense_tensor.get(tensor_accessor.element<X, Z, Y>()), 2.);
    EXPECT_EQ(dense_tensor.get(tensor_accessor.element<X, Z, Z>()), 0.);
    EXPECT_EQ(dense_tensor.get(tensor_accessor.element<Y, X, X>()), 3.);
    EXPECT_EQ(dense_tensor.get(tensor_accessor.element<Y, X, Y>()), 0.);
    EXPECT_EQ(dense_tensor.get(tensor_accessor.element<Y, X, Z>()), 4.);
    EXPECT_EQ(dense_tensor.get(tensor_accessor.element<Y, Y, X>()), 0.);
    EXPECT_EQ(dense_tensor.get(tensor_accessor.element<Y, Y, Y>()), 0.);
    EXPECT_EQ(dense_tensor.get(tensor_accessor.element<Y, Y, Z>()), 0.);
    EXPECT_EQ(dense_tensor.get(tensor_accessor.element<Y, Z, X>()), 0.);
    EXPECT_EQ(dense_tensor.get(tensor_accessor.element<Y, Z, Y>()), 0.);
    EXPECT_EQ(dense_tensor.get(tensor_accessor.element<Y, Z, Z>()), 5.);
    EXPECT_EQ(dense_tensor.get(tensor_accessor.element<Z, X, X>()), 0.);
    EXPECT_EQ(dense_tensor.get(tensor_accessor.element<Z, X, Y>()), 8.);
    EXPECT_EQ(dense_tensor.get(tensor_accessor.element<Z, X, Z>()), 6.);
    EXPECT_EQ(dense_tensor.get(tensor_accessor.element<Z, Y, X>()), 0.);
    EXPECT_EQ(dense_tensor.get(tensor_accessor.element<Z, Y, Y>()), 7.);
    EXPECT_EQ(dense_tensor.get(tensor_accessor.element<Z, Y, Z>()), 0.);
    EXPECT_EQ(dense_tensor.get(tensor_accessor.element<Z, Z, X>()), 0.);
    EXPECT_EQ(dense_tensor.get(tensor_accessor.element<Z, Z, Y>()), 0.);
    EXPECT_EQ(dense_tensor.get(tensor_accessor.element<Z, Z, Z>()), 9.);
}

TEST(Csr, CsrDenseProducts)
{
    sil::tensor::TensorAccessor<Alpha, Beta, Gamma> tensor_accessor;
    ddc::DiscreteDomain<Alpha, Beta, Gamma> tensor_dom = tensor_accessor.mem_domain();
    ddc::Chunk tensor_alloc(tensor_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor<
            double,
            ddc::DiscreteDomain<Alpha, Beta, Gamma>,
            Kokkos::layout_right,
            Kokkos::DefaultHostExecutionSpace::memory_space>
            tensor(tensor_alloc);

    ddc::parallel_fill(tensor, 0.);
    tensor(tensor_accessor.element<X, X, Y>()) = 1.;
    tensor(tensor_accessor.element<X, Z, Y>()) = 2.;
    tensor(tensor_accessor.element<Y, X, X>()) = 3.;
    tensor(tensor_accessor.element<Y, X, Z>()) = 4.;
    tensor(tensor_accessor.element<Y, Z, Z>()) = 5.;
    tensor(tensor_accessor.element<Z, X, Z>()) = 6.;
    tensor(tensor_accessor.element<Z, Y, Y>()) = 7.;
    tensor(tensor_accessor.element<Z, X, Y>()) = 8.;
    tensor(tensor_accessor.element<Z, Z, Z>()) = 9.;

    sil::csr::CsrDynamic<Alpha, Beta, Gamma> csr_dyn(tensor_dom);

    csr_dyn.push_back(tensor[ddc::DiscreteElement<Alpha>(0)]);
    csr_dyn.push_back(tensor[ddc::DiscreteElement<Alpha>(1)]);
    csr_dyn.push_back(tensor[ddc::DiscreteElement<Alpha>(2)]);

    sil::csr::Csr<9, Alpha, Beta, Gamma> csr(csr_dyn);

    sil::tensor::TensorAccessor<Beta, Gamma> right_tensor_accessor;
    ddc::DiscreteDomain<Beta, Gamma> right_tensor_dom = right_tensor_accessor.mem_domain();
    ddc::Chunk right_tensor_alloc(right_tensor_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor<
            double,
            ddc::DiscreteDomain<Beta, Gamma>,
            Kokkos::layout_right,
            Kokkos::DefaultHostExecutionSpace::memory_space>
            right_tensor(right_tensor_alloc);
    ddc::parallel_fill(right_tensor, 1.);

    sil::tensor::TensorAccessor<Alpha> right_prod_accessor;
    ddc::DiscreteDomain<Alpha> right_prod_dom = right_prod_accessor.mem_domain();
    ddc::Chunk right_prod_alloc(right_prod_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor<
            double,
            ddc::DiscreteDomain<Alpha>,
            Kokkos::layout_right,
            Kokkos::DefaultHostExecutionSpace::memory_space>
            right_prod(right_prod_alloc);

    sil::csr::tensor_prod(right_prod, csr, right_tensor);

    EXPECT_EQ(right_prod.get(right_prod_accessor.element<X>()), 3.);
    EXPECT_EQ(right_prod.get(right_prod_accessor.element<Y>()), 12.);
    EXPECT_EQ(right_prod.get(right_prod_accessor.element<Z>()), 30.);

    sil::tensor::TensorAccessor<Alpha> left_vector_accessor;
    ddc::DiscreteDomain<Alpha> left_vector_dom = left_vector_accessor.mem_domain();
    ddc::Chunk left_vector_alloc(left_vector_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor<
            double,
            ddc::DiscreteDomain<Alpha>,
            Kokkos::layout_right,
            Kokkos::DefaultHostExecutionSpace::memory_space>
            left_vector(left_vector_alloc);
    ddc::parallel_fill(left_vector, 1.);

    sil::tensor::TensorAccessor<Beta, Gamma> left_prod_accessor;
    ddc::DiscreteDomain<Beta, Gamma> left_prod_dom = left_prod_accessor.mem_domain();
    ddc::Chunk left_prod_alloc(left_prod_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor<
            double,
            ddc::DiscreteDomain<Beta, Gamma>,
            Kokkos::layout_right,
            Kokkos::DefaultHostExecutionSpace::memory_space>
            left_prod(left_prod_alloc);

    sil::csr::tensor_prod(left_prod, left_vector, csr);

    EXPECT_EQ(left_prod.get(left_prod_accessor.element<X, X>()), 3.);
    EXPECT_EQ(left_prod.get(left_prod_accessor.element<X, Y>()), 9.);
    EXPECT_EQ(left_prod.get(left_prod_accessor.element<X, Z>()), 10.);
    EXPECT_EQ(left_prod.get(left_prod_accessor.element<Y, X>()), 0.);
    EXPECT_EQ(left_prod.get(left_prod_accessor.element<Y, Y>()), 7.);
    EXPECT_EQ(left_prod.get(left_prod_accessor.element<Y, Z>()), 0.);
    EXPECT_EQ(left_prod.get(left_prod_accessor.element<Z, X>()), 0.);
    EXPECT_EQ(left_prod.get(left_prod_accessor.element<Z, Y>()), 2.);
    EXPECT_EQ(left_prod.get(left_prod_accessor.element<Z, Z>()), 14.);
}
