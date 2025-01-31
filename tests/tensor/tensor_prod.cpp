// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: MIT

#include <cmath>

#include <ddc/ddc.hpp>

#include <gtest/gtest.h>

#include "tensor.hpp"

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
    [[maybe_unused]] sil::tensor::TensorAccessor<Alpha, Beta, Gamma> tensor_accessor1;
    ddc::DiscreteDomain<Alpha, Beta, Gamma> tensor1_dom = tensor_accessor1.domain();
    ddc::Chunk tensor1_alloc(tensor1_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor tensor1(tensor1_alloc);

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


    [[maybe_unused]] sil::tensor::TensorAccessor<Gamma, Delta> tensor_accessor2;
    ddc::DiscreteDomain<Gamma, Delta> tensor2_dom = tensor_accessor2.domain();
    ddc::Chunk tensor2_alloc(tensor2_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor tensor2(tensor2_alloc);

    tensor2(tensor2.access_element<X, X>()) = 0.;
    tensor2(tensor2.access_element<X, Y>()) = 1.;
    tensor2(tensor2.access_element<X, Z>()) = 2.;
    tensor2(tensor2.access_element<Y, X>()) = 3.;
    tensor2(tensor2.access_element<Y, Y>()) = 4.;
    tensor2(tensor2.access_element<Y, Z>()) = 5.;
    tensor2(tensor2.access_element<Z, X>()) = 6.;
    tensor2(tensor2.access_element<Z, Y>()) = 7.;
    tensor2(tensor2.access_element<Z, Z>()) = 8.;

    [[maybe_unused]] sil::tensor::TensorAccessor<Alpha, Beta, Delta> prod_tensor_accessor;
    ddc::DiscreteDomain<Alpha, Beta, Delta> prod_tensor_dom = prod_tensor_accessor.domain();

    ddc::Chunk prod_tensor_alloc(prod_tensor_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor prod_tensor(prod_tensor_alloc);

    sil::tensor::tensor_prod(prod_tensor, tensor1, tensor2);

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
    [[maybe_unused]] sil::tensor::TensorAccessor<Alpha, Beta, Gamma> tensor_accessor1;
    ddc::DiscreteDomain<Alpha, Beta, Gamma> tensor1_dom = tensor_accessor1.domain();
    ddc::Chunk tensor1_alloc(tensor1_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor tensor1(tensor1_alloc);

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


    [[maybe_unused]] sil::tensor::TensorAccessor<Beta, Gamma, Delta> tensor_accessor2;
    ddc::DiscreteDomain<Beta, Gamma, Delta> tensor2_dom = tensor_accessor2.domain();
    ddc::Chunk tensor2_alloc(tensor2_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor tensor2(tensor2_alloc);

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

    [[maybe_unused]] sil::tensor::TensorAccessor<Alpha, Delta> prod_tensor_accessor;
    ddc::DiscreteDomain<Alpha, Delta> prod_tensor_dom = prod_tensor_accessor.domain();

    ddc::Chunk prod_tensor_alloc(prod_tensor_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor prod_tensor(prod_tensor_alloc);

    sil::tensor::tensor_prod(prod_tensor, tensor1, tensor2);

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

// Warning : the following tests compute the results of the tensor product of
// two (anti)symmetric tensors as an (anti)symmetric tensor, which is not correct.
// The test is thus performed only on the components which are expected to be correct.
struct Mu : sil::tensor::TensorNaturalIndex<T, X, Y, Z>
{
};

struct Nu : sil::tensor::TensorNaturalIndex<T, X, Y, Z>
{
};

struct Rho : sil::tensor::TensorNaturalIndex<T, X, Y, Z>
{
};

using Rank2SymIndex1 = sil::tensor::
        TensorSymmetricIndex<sil::tensor::Contravariant<Mu>, sil::tensor::Covariant<Nu>>;
using Rank2SymIndex2 = sil::tensor::
        TensorSymmetricIndex<sil::tensor::Contravariant<Nu>, sil::tensor::Contravariant<Rho>>;
using Rank2SymIndex3 = sil::tensor::
        TensorSymmetricIndex<sil::tensor::Contravariant<Mu>, sil::tensor::Contravariant<Rho>>;

TEST(TensorProd, SimpleContractionRank2SymIndexedxRank2SymIndexed)
{
    [[maybe_unused]] sil::tensor::TensorAccessor<Rank2SymIndex1> tensor_accessor1;
    ddc::DiscreteDomain<Rank2SymIndex1> tensor1_dom = tensor_accessor1.domain();
    ddc::Chunk tensor1_alloc(tensor1_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor tensor1(tensor1_alloc);

    tensor1(tensor1.access_element<T, T>()) = 1.;
    tensor1(tensor1.access_element<T, X>()) = 2.;
    tensor1(tensor1.access_element<T, Y>()) = 3.;
    tensor1(tensor1.access_element<T, Z>()) = 4.;
    tensor1(tensor1.access_element<X, X>()) = 5.;
    tensor1(tensor1.access_element<X, Y>()) = 6.;
    tensor1(tensor1.access_element<X, Z>()) = 7.;
    tensor1(tensor1.access_element<Y, Y>()) = 8.;
    tensor1(tensor1.access_element<Y, Z>()) = 9.;
    tensor1(tensor1.access_element<Z, Z>()) = 10.;


    [[maybe_unused]] sil::tensor::TensorAccessor<Rank2SymIndex2> tensor_accessor2;
    ddc::DiscreteDomain<Rank2SymIndex2> tensor2_dom = tensor_accessor2.domain();
    ddc::Chunk tensor2_alloc(tensor2_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor tensor2(tensor2_alloc);

    tensor2(tensor2.access_element<T, T>()) = 11.;
    tensor2(tensor2.access_element<T, X>()) = 12.;
    tensor2(tensor2.access_element<T, Y>()) = 13.;
    tensor2(tensor2.access_element<T, Z>()) = 14.;
    tensor2(tensor2.access_element<X, X>()) = 15.;
    tensor2(tensor2.access_element<X, Y>()) = 16.;
    tensor2(tensor2.access_element<X, Z>()) = 17.;
    tensor2(tensor2.access_element<Y, Y>()) = 18.;
    tensor2(tensor2.access_element<Y, Z>()) = 19.;
    tensor2(tensor2.access_element<Z, Z>()) = 20.;


    [[maybe_unused]] sil::tensor::TensorAccessor<Rank2SymIndex3> prod_tensor_accessor;
    ddc::DiscreteDomain<Rank2SymIndex3> prod_tensor_dom = prod_tensor_accessor.domain();

    ddc::Chunk prod_tensor_alloc(prod_tensor_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor prod_tensor(prod_tensor_alloc);

    sil::tensor::tensor_prod(prod_tensor, tensor1, tensor2);

    EXPECT_EQ(prod_tensor.get(prod_tensor.access_element<T, T>()), 130.);
    EXPECT_EQ(prod_tensor.get(prod_tensor.access_element<T, X>()), 158.);
    EXPECT_EQ(prod_tensor.get(prod_tensor.access_element<T, Y>()), 175.);
    EXPECT_EQ(prod_tensor.get(prod_tensor.access_element<T, Z>()), 185.);
    EXPECT_EQ(prod_tensor.get(prod_tensor.access_element<X, X>()), 314.);
    EXPECT_EQ(prod_tensor.get(prod_tensor.access_element<X, Y>()), 347.);
    EXPECT_EQ(prod_tensor.get(prod_tensor.access_element<X, Z>()), 367.);
    EXPECT_EQ(prod_tensor.get(prod_tensor.access_element<Y, Y>()), 450.);
    EXPECT_EQ(prod_tensor.get(prod_tensor.access_element<Y, Z>()), 476.);
    EXPECT_EQ(prod_tensor.get(prod_tensor.access_element<Z, Z>()), 546.);
}

using Rank2AntisymIndex1 = sil::tensor::
        TensorAntisymmetricIndex<sil::tensor::Contravariant<Mu>, sil::tensor::Covariant<Nu>>;
using Rank2AntisymIndex2 = sil::tensor::
        TensorAntisymmetricIndex<sil::tensor::Contravariant<Nu>, sil::tensor::Contravariant<Rho>>;
using Rank2AntisymIndex3 = sil::tensor::
        TensorAntisymmetricIndex<sil::tensor::Contravariant<Mu>, sil::tensor::Contravariant<Rho>>;

TEST(TensorProd, SimpleContractionRank2AntisymIndexedxRank2AntisymIndexed)
{
    [[maybe_unused]] sil::tensor::TensorAccessor<Rank2AntisymIndex1> tensor_accessor1;
    ddc::DiscreteDomain<Rank2AntisymIndex1> tensor1_dom = tensor_accessor1.domain();
    ddc::Chunk tensor1_alloc(tensor1_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor tensor1(tensor1_alloc);

    tensor1(tensor1.access_element<T, X>()) = 1.;
    tensor1(tensor1.access_element<T, Y>()) = 2.;
    tensor1(tensor1.access_element<T, Z>()) = 3.;
    tensor1(tensor1.access_element<X, Y>()) = 4.;
    tensor1(tensor1.access_element<X, Z>()) = 5.;
    tensor1(tensor1.access_element<Y, Z>()) = 6.;


    [[maybe_unused]] sil::tensor::TensorAccessor<Rank2AntisymIndex2> tensor_accessor2;
    ddc::DiscreteDomain<Rank2AntisymIndex2> tensor2_dom = tensor_accessor2.domain();
    ddc::Chunk tensor2_alloc(tensor2_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor tensor2(tensor2_alloc);

    tensor2(tensor2.access_element<T, X>()) = 7.;
    tensor2(tensor2.access_element<T, Y>()) = 8.;
    tensor2(tensor2.access_element<T, Z>()) = 9.;
    tensor2(tensor2.access_element<X, Y>()) = 10.;
    tensor2(tensor2.access_element<X, Z>()) = 11.;
    tensor2(tensor2.access_element<Y, Z>()) = 12.;


    [[maybe_unused]] sil::tensor::TensorAccessor<Rank2AntisymIndex3> prod_tensor_accessor;
    ddc::DiscreteDomain<Rank2AntisymIndex3> prod_tensor_dom = prod_tensor_accessor.domain();

    ddc::Chunk prod_tensor_alloc(prod_tensor_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor prod_tensor(prod_tensor_alloc);

    sil::tensor::tensor_prod(prod_tensor, tensor1, tensor2);

    EXPECT_EQ(prod_tensor.get(prod_tensor.access_element<T, X>()), -53.);
    EXPECT_EQ(prod_tensor.get(prod_tensor.access_element<T, Y>()), -26.);
    EXPECT_EQ(prod_tensor.get(prod_tensor.access_element<T, Z>()), 35.);
    EXPECT_EQ(prod_tensor.get(prod_tensor.access_element<X, Y>()), -68.);
    EXPECT_EQ(prod_tensor.get(prod_tensor.access_element<X, Z>()), 39.);
    EXPECT_EQ(prod_tensor.get(prod_tensor.access_element<Y, Z>()), -62.);
}

struct Sigma : sil::tensor::TensorNaturalIndex<T, X, Y, Z>
{
};

using Rank3SymIndex1 = sil::tensor::TensorSymmetricIndex<
        sil::tensor::Contravariant<Mu>,
        sil::tensor::Contravariant<Nu>,
        sil::tensor::Covariant<Rho>>;
using Rank3SymIndex2 = sil::tensor::
        TensorSymmetricIndex<sil::tensor::Contravariant<Rho>, sil::tensor::Contravariant<Sigma>>;
using Rank3SymIndex3 = sil::tensor::TensorSymmetricIndex<
        sil::tensor::Contravariant<Mu>,
        sil::tensor::Contravariant<Nu>,
        sil::tensor::Contravariant<Sigma>>;

TEST(TensorProd, SimpleContractionRank3SymIndexedxRank2SymIndexed)
{
    [[maybe_unused]] sil::tensor::TensorAccessor<Rank3SymIndex1> tensor_accessor1;
    ddc::DiscreteDomain<Rank3SymIndex1> tensor1_dom = tensor_accessor1.domain();
    ddc::Chunk tensor1_alloc(tensor1_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor tensor1(tensor1_alloc);

    tensor1(tensor1.access_element<T, T, T>()) = 1.;
    tensor1(tensor1.access_element<T, T, X>()) = 2.;
    tensor1(tensor1.access_element<T, T, Y>()) = 3.;
    tensor1(tensor1.access_element<T, T, Z>()) = 4.;
    tensor1(tensor1.access_element<T, X, X>()) = 5.;
    tensor1(tensor1.access_element<T, X, Y>()) = 6.;
    tensor1(tensor1.access_element<T, X, Z>()) = 7.;
    tensor1(tensor1.access_element<T, Y, Y>()) = 8.;
    tensor1(tensor1.access_element<T, Y, Z>()) = 9.;
    tensor1(tensor1.access_element<T, Z, Z>()) = 10.;
    tensor1(tensor1.access_element<X, X, X>()) = 11.;
    tensor1(tensor1.access_element<X, X, Y>()) = 12.;
    tensor1(tensor1.access_element<X, X, Z>()) = 13.;
    tensor1(tensor1.access_element<X, Y, Y>()) = 14.;
    tensor1(tensor1.access_element<X, Y, Z>()) = 15.;
    tensor1(tensor1.access_element<X, Z, Z>()) = 16.;
    tensor1(tensor1.access_element<Y, Y, Y>()) = 17.;
    tensor1(tensor1.access_element<Y, Y, Z>()) = 18.;
    tensor1(tensor1.access_element<Y, Z, Z>()) = 19.;
    tensor1(tensor1.access_element<Z, Z, Z>()) = 20.;


    [[maybe_unused]] sil::tensor::TensorAccessor<Rank3SymIndex2> tensor_accessor2;
    ddc::DiscreteDomain<Rank3SymIndex2> tensor2_dom = tensor_accessor2.domain();
    ddc::Chunk tensor2_alloc(tensor2_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor tensor2(tensor2_alloc);

    tensor2(tensor2.access_element<T, T>()) = 21.;
    tensor2(tensor2.access_element<T, X>()) = 22.;
    tensor2(tensor2.access_element<T, Y>()) = 23.;
    tensor2(tensor2.access_element<T, Z>()) = 24.;
    tensor2(tensor2.access_element<X, X>()) = 25.;
    tensor2(tensor2.access_element<X, Y>()) = 26.;
    tensor2(tensor2.access_element<X, Z>()) = 27.;
    tensor2(tensor2.access_element<Y, Y>()) = 28.;
    tensor2(tensor2.access_element<Y, Z>()) = 29.;
    tensor2(tensor2.access_element<Z, Z>()) = 30.;


    [[maybe_unused]] sil::tensor::TensorAccessor<Rank3SymIndex3> prod_tensor_accessor;
    ddc::DiscreteDomain<Rank3SymIndex3> prod_tensor_dom = prod_tensor_accessor.domain();

    ddc::Chunk prod_tensor_alloc(prod_tensor_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor prod_tensor(prod_tensor_alloc);

    sil::tensor::tensor_prod(prod_tensor, tensor1, tensor2);

    EXPECT_EQ(prod_tensor.get(prod_tensor.access_element<T, T, T>()), 230.);
    EXPECT_EQ(prod_tensor.get(prod_tensor.access_element<T, T, X>()), 258.);
    EXPECT_EQ(prod_tensor.get(prod_tensor.access_element<T, T, Y>()), 275.);
    EXPECT_EQ(prod_tensor.get(prod_tensor.access_element<T, T, Z>()), 285.);
    EXPECT_EQ(prod_tensor.get(prod_tensor.access_element<T, X, X>()), 514.);
    EXPECT_EQ(prod_tensor.get(prod_tensor.access_element<T, X, Y>()), 547.);
    EXPECT_EQ(prod_tensor.get(prod_tensor.access_element<T, X, Z>()), 567.);
    EXPECT_EQ(prod_tensor.get(prod_tensor.access_element<T, Y, Y>()), 710.);
    EXPECT_EQ(prod_tensor.get(prod_tensor.access_element<T, Y, Z>()), 736.);
    EXPECT_EQ(prod_tensor.get(prod_tensor.access_element<T, Z, Z>()), 846.);
    EXPECT_EQ(prod_tensor.get(prod_tensor.access_element<X, X, X>()), 1048.);
    EXPECT_EQ(prod_tensor.get(prod_tensor.access_element<X, X, Y>()), 1114.);
    EXPECT_EQ(prod_tensor.get(prod_tensor.access_element<X, X, Z>()), 1155.);
    EXPECT_EQ(prod_tensor.get(prod_tensor.access_element<X, Y, Y>()), 1277.);
    EXPECT_EQ(prod_tensor.get(prod_tensor.access_element<X, Y, Z>()), 1324.);
    EXPECT_EQ(prod_tensor.get(prod_tensor.access_element<X, Z, Z>()), 1434.);
    EXPECT_EQ(prod_tensor.get(prod_tensor.access_element<Y, Y, Y>()), 1546.);
    EXPECT_EQ(prod_tensor.get(prod_tensor.access_element<Y, Y, Z>()), 1603.);
    EXPECT_EQ(prod_tensor.get(prod_tensor.access_element<Y, Z, Z>()), 1713.);
    EXPECT_EQ(prod_tensor.get(prod_tensor.access_element<Z, Z, Z>()), 1823.);
}

using Rank3AntisymIndex1 = sil::tensor::TensorAntisymmetricIndex<
        sil::tensor::Contravariant<Mu>,
        sil::tensor::Contravariant<Nu>,
        sil::tensor::Covariant<Rho>>;
using Rank3AntisymIndex2 = sil::tensor::TensorAntisymmetricIndex<
        sil::tensor::Contravariant<Rho>,
        sil::tensor::Contravariant<Sigma>>;
using Rank3AntisymIndex3 = sil::tensor::TensorAntisymmetricIndex<
        sil::tensor::Contravariant<Mu>,
        sil::tensor::Contravariant<Nu>,
        sil::tensor::Contravariant<Sigma>>;

TEST(TensorProd, SimpleContractionRank3AntisymIndexedxRank2AntisymIndexed)
{
    [[maybe_unused]] sil::tensor::TensorAccessor<Rank3AntisymIndex1> tensor_accessor1;
    ddc::DiscreteDomain<Rank3AntisymIndex1> tensor1_dom = tensor_accessor1.domain();
    ddc::Chunk tensor1_alloc(tensor1_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor tensor1(tensor1_alloc);

    tensor1(tensor1.access_element<T, X, Y>()) = 1.;
    tensor1(tensor1.access_element<T, X, Z>()) = 2.;
    tensor1(tensor1.access_element<T, Y, Z>()) = 3.;
    tensor1(tensor1.access_element<X, Y, Z>()) = 4.;


    [[maybe_unused]] sil::tensor::TensorAccessor<Rank3AntisymIndex2> tensor_accessor2;
    ddc::DiscreteDomain<Rank3AntisymIndex2> tensor2_dom = tensor_accessor2.domain();
    ddc::Chunk tensor2_alloc(tensor2_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor tensor2(tensor2_alloc);

    tensor2(tensor2.access_element<T, X>()) = 5.;
    tensor2(tensor2.access_element<T, Y>()) = 6.;
    tensor2(tensor2.access_element<T, Z>()) = 7.;
    tensor2(tensor2.access_element<X, Y>()) = 8.;
    tensor2(tensor2.access_element<X, Z>()) = 9.;
    tensor2(tensor2.access_element<Y, Z>()) = 10.;


    [[maybe_unused]] sil::tensor::TensorAccessor<Rank3AntisymIndex3> prod_tensor_accessor;
    ddc::DiscreteDomain<Rank3AntisymIndex3> prod_tensor_dom = prod_tensor_accessor.domain();

    ddc::Chunk prod_tensor_alloc(prod_tensor_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor prod_tensor(prod_tensor_alloc);

    sil::tensor::tensor_prod(prod_tensor, tensor1, tensor2);

    EXPECT_EQ(prod_tensor.get(prod_tensor.access_element<T, X, Y>()), -8.);
    EXPECT_EQ(prod_tensor.get(prod_tensor.access_element<T, X, Z>()), 24.);
    EXPECT_EQ(prod_tensor.get(prod_tensor.access_element<T, Y, Z>()), 32.);
    EXPECT_EQ(prod_tensor.get(prod_tensor.access_element<X, Y, Z>()), 19.);
}

#if defined BUILD_YOUNG_TABLEAU
using YoungTableauIndex = sil::tensor::TensorYoungTableauIndex<
        sil::young_tableau::
                YoungTableau<3, sil::young_tableau::YoungTableauSeq<std::index_sequence<1, 2, 3>>>,
        Alpha,
        Beta,
        Gamma>;

TEST(TensorProd, DoubleContractionYoungIndexedxNaturalIndex)
{
    [[maybe_unused]] sil::tensor::TensorAccessor<Alpha, Beta, Gamma> natural_accessor;
    ddc::DiscreteDomain<Alpha, Beta, Gamma> natural_dom = natural_accessor.domain();
    ddc::Chunk natural_alloc(natural_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor natural(natural_alloc);

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

    [[maybe_unused]] sil::tensor::TensorAccessor<YoungTableauIndex> tensor_accessor1;
    ddc::DiscreteDomain<YoungTableauIndex> tensor1_dom = tensor_accessor1.domain();
    ddc::Chunk tensor1_alloc(tensor1_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor tensor1(tensor1_alloc);

    sil::tensor::compress(tensor1, natural);

    [[maybe_unused]] sil::tensor::TensorAccessor<Beta, Gamma, Delta> tensor_accessor2;
    ddc::DiscreteDomain<Beta, Gamma, Delta> tensor2_dom = tensor_accessor2.domain();
    ddc::Chunk tensor2_alloc(tensor2_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor tensor2(tensor2_alloc);

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

    [[maybe_unused]] sil::tensor::TensorAccessor<Alpha, Delta> prod_tensor_accessor;
    ddc::DiscreteDomain<Alpha, Delta> prod_tensor_dom = prod_tensor_accessor.domain();

    ddc::Chunk prod_tensor_alloc(prod_tensor_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor prod_tensor(prod_tensor_alloc);

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
#endif
