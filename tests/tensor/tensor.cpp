// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: GPL-3.0-or-later

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
            Kokkos::layout_right,
            Kokkos::DefaultHostExecutionSpace::memory_space>
            tensor(tensor_alloc);

    /*
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 4; ++j) {
            tensor(ddc::DiscreteElement<Alpha, Nu>(i, j)) = i * 4 + j;
        }
    }
    */

    tensor(tensor.access_element<X, T>()) = 0.;
    tensor(tensor.access_element<X, X>()) = 1.;
    tensor(tensor.access_element<X, Y>()) = 2.;
    tensor(tensor.access_element<X, Z>()) = 3.;
    tensor(tensor.access_element<Y, T>()) = 4.;
    tensor(tensor.access_element<Y, X>()) = 5.;
    tensor(tensor.access_element<Y, Y>()) = 6.;
    tensor(tensor.access_element<Y, Z>()) = 7.;
    tensor(tensor.access_element<Z, T>()) = 8.;
    tensor(tensor.access_element<Z, X>()) = 9.;
    tensor(tensor.access_element<Z, Y>()) = 10.;
    tensor(tensor.access_element<Z, Z>()) = 11.;

    EXPECT_EQ(tensor.get(tensor.access_element<X, T>()), 0.);
    EXPECT_EQ(tensor.get(tensor.access_element<X, X>()), 1.);
    EXPECT_EQ(tensor.get(tensor.access_element<X, Y>()), 2.);
    EXPECT_EQ(tensor.get(tensor.access_element<X, Z>()), 3.);
    EXPECT_EQ(tensor.get(tensor.access_element<Y, T>()), 4.);
    EXPECT_EQ(tensor.get(tensor.access_element<Y, X>()), 5.);
    EXPECT_EQ(tensor.get(tensor.access_element<Y, Y>()), 6.);
    EXPECT_EQ(tensor.get(tensor.access_element<Y, Z>()), 7.);
    EXPECT_EQ(tensor.get(tensor.access_element<Z, T>()), 8.);
    EXPECT_EQ(tensor.get(tensor.access_element<Z, X>()), 9.);
    EXPECT_EQ(tensor.get(tensor.access_element<Z, Y>()), 10.);
    EXPECT_EQ(tensor.get(tensor.access_element<Z, Z>()), 11.);
}

using FullIndex = sil::tensor::TensorFullIndex<Alpha, Nu>;

TEST(Tensor, TensorFullIndexing)
{
    sil::tensor::TensorAccessor<FullIndex> tensor_accessor;
    ddc::DiscreteDomain<FullIndex> tensor_dom = tensor_accessor.mem_domain();
    ddc::Chunk tensor_alloc(tensor_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor<
            double,
            ddc::DiscreteDomain<FullIndex>,
            Kokkos::layout_right,
            Kokkos::DefaultHostExecutionSpace::memory_space>
            tensor(tensor_alloc);

    /*
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 4; ++j) {
            tensor(ddc::DiscreteElement<FullIndex>(i * 4 + j)) = i * 4 + j;
        }
    }
    */

    tensor(tensor.access_element<X, T>()) = 0.;
    tensor(tensor.access_element<X, X>()) = 1.;
    tensor(tensor.access_element<X, Y>()) = 2.;
    tensor(tensor.access_element<X, Z>()) = 3.;
    tensor(tensor.access_element<Y, T>()) = 4.;
    tensor(tensor.access_element<Y, X>()) = 5.;
    tensor(tensor.access_element<Y, Y>()) = 6.;
    tensor(tensor.access_element<Y, Z>()) = 7.;
    tensor(tensor.access_element<Z, T>()) = 8.;
    tensor(tensor.access_element<Z, X>()) = 9.;
    tensor(tensor.access_element<Z, Y>()) = 10.;
    tensor(tensor.access_element<Z, Z>()) = 11.;

    EXPECT_EQ(tensor.get(tensor.access_element<X, T>()), 0.);
    EXPECT_EQ(tensor.get(tensor.access_element<X, X>()), 1.);
    EXPECT_EQ(tensor.get(tensor.access_element<X, Y>()), 2.);
    EXPECT_EQ(tensor.get(tensor.access_element<X, Z>()), 3.);
    EXPECT_EQ(tensor.get(tensor.access_element<Y, T>()), 4.);
    EXPECT_EQ(tensor.get(tensor.access_element<Y, X>()), 5.);
    EXPECT_EQ(tensor.get(tensor.access_element<Y, Y>()), 6.);
    EXPECT_EQ(tensor.get(tensor.access_element<Y, Z>()), 7.);
    EXPECT_EQ(tensor.get(tensor.access_element<Z, T>()), 8.);
    EXPECT_EQ(tensor.get(tensor.access_element<Z, X>()), 9.);
    EXPECT_EQ(tensor.get(tensor.access_element<Z, Y>()), 10.);
    EXPECT_EQ(tensor.get(tensor.access_element<Z, Z>()), 11.);
}

using IdIndex = sil::tensor::TensorIdentityIndex<Mu, Nu>;

TEST(Tensor, TensorIdentityIndexing)
{
    sil::tensor::TensorAccessor<IdIndex> tensor_accessor;
    ddc::DiscreteDomain<IdIndex> tensor_dom = tensor_accessor.mem_domain();
    ddc::Chunk tensor_alloc(tensor_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor<
            double,
            ddc::DiscreteDomain<IdIndex>,
            Kokkos::layout_right,
            Kokkos::DefaultHostExecutionSpace::memory_space>
            tensor(tensor_alloc);

    EXPECT_EQ(tensor.get(tensor.access_element<T, T>()), 1.);
    EXPECT_EQ(tensor.get(tensor.access_element<T, X>()), 0.);
    EXPECT_EQ(tensor.get(tensor.access_element<T, Y>()), 0.);
    EXPECT_EQ(tensor.get(tensor.access_element<T, Z>()), 0.);
    EXPECT_EQ(tensor.get(tensor.access_element<X, T>()), 0.);
    EXPECT_EQ(tensor.get(tensor.access_element<X, X>()), 1.);
    EXPECT_EQ(tensor.get(tensor.access_element<X, Y>()), 0.);
    EXPECT_EQ(tensor.get(tensor.access_element<X, Z>()), 0.);
    EXPECT_EQ(tensor.get(tensor.access_element<Y, T>()), 0.);
    EXPECT_EQ(tensor.get(tensor.access_element<Y, X>()), 0.);
    EXPECT_EQ(tensor.get(tensor.access_element<Y, Y>()), 1.);
    EXPECT_EQ(tensor.get(tensor.access_element<Y, Z>()), 0.);
    EXPECT_EQ(tensor.get(tensor.access_element<Z, T>()), 0.);
    EXPECT_EQ(tensor.get(tensor.access_element<Z, X>()), 0.);
    EXPECT_EQ(tensor.get(tensor.access_element<Z, Y>()), 0.);
    EXPECT_EQ(tensor.get(tensor.access_element<Z, Z>()), 1.);
}

using LorentzianSignIndex
        = sil::tensor::TensorLorentzianSignIndex<std::integral_constant<std::size_t, 2>, Mu, Nu>;

TEST(Tensor, TensorLorentzianSignIndexing)
{
    sil::tensor::TensorAccessor<LorentzianSignIndex> tensor_accessor;
    ddc::DiscreteDomain<LorentzianSignIndex> tensor_dom = tensor_accessor.mem_domain();
    ddc::Chunk tensor_alloc(tensor_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor<
            double,
            ddc::DiscreteDomain<LorentzianSignIndex>,
            Kokkos::layout_right,
            Kokkos::DefaultHostExecutionSpace::memory_space>
            tensor(tensor_alloc);

    EXPECT_EQ(tensor.get(tensor.access_element<T, T>()), -1.);
    EXPECT_EQ(tensor.get(tensor.access_element<T, X>()), 0.);
    EXPECT_EQ(tensor.get(tensor.access_element<T, Y>()), 0.);
    EXPECT_EQ(tensor.get(tensor.access_element<T, Z>()), 0.);
    EXPECT_EQ(tensor.get(tensor.access_element<X, T>()), 0.);
    EXPECT_EQ(tensor.get(tensor.access_element<X, X>()), -1.);
    EXPECT_EQ(tensor.get(tensor.access_element<X, Y>()), 0.);
    EXPECT_EQ(tensor.get(tensor.access_element<X, Z>()), 0.);
    EXPECT_EQ(tensor.get(tensor.access_element<Y, T>()), 0.);
    EXPECT_EQ(tensor.get(tensor.access_element<Y, X>()), 0.);
    EXPECT_EQ(tensor.get(tensor.access_element<Y, Y>()), 1.);
    EXPECT_EQ(tensor.get(tensor.access_element<Y, Z>()), 0.);
    EXPECT_EQ(tensor.get(tensor.access_element<Z, T>()), 0.);
    EXPECT_EQ(tensor.get(tensor.access_element<Z, X>()), 0.);
    EXPECT_EQ(tensor.get(tensor.access_element<Z, Y>()), 0.);
    EXPECT_EQ(tensor.get(tensor.access_element<Z, Z>()), 1.);
}

using DiagIndex = sil::tensor::TensorDiagonalIndex<Mu, Nu>;

TEST(Tensor, TensorDiagonalIndexing)
{
    sil::tensor::TensorAccessor<DiagIndex> tensor_accessor;
    ddc::DiscreteDomain<DiagIndex> tensor_dom = tensor_accessor.mem_domain();
    ddc::Chunk tensor_alloc(tensor_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor<
            double,
            ddc::DiscreteDomain<DiagIndex>,
            Kokkos::layout_right,
            Kokkos::DefaultHostExecutionSpace::memory_space>
            tensor(tensor_alloc);

    /*
    for (int i = 1; i < 5; ++i) {
            tensor(ddc::DiscreteElement<DiagIndex>(i)) = i;
    }
    */

    tensor(tensor.access_element<T, T>()) = 1.;
    tensor(tensor.access_element<X, X>()) = 2.;
    tensor(tensor.access_element<Y, Y>()) = 3.;
    tensor(tensor.access_element<Z, Z>()) = 4.;

    EXPECT_EQ(tensor.get(tensor.access_element<T, T>()), 1.);
    EXPECT_EQ(tensor.get(tensor.access_element<T, X>()), 0.);
    EXPECT_EQ(tensor.get(tensor.access_element<T, Y>()), 0.);
    EXPECT_EQ(tensor.get(tensor.access_element<T, Z>()), 0.);
    EXPECT_EQ(tensor.get(tensor.access_element<X, T>()), 0.);
    EXPECT_EQ(tensor.get(tensor.access_element<X, X>()), 2.);
    EXPECT_EQ(tensor.get(tensor.access_element<X, Y>()), 0.);
    EXPECT_EQ(tensor.get(tensor.access_element<X, Z>()), 0.);
    EXPECT_EQ(tensor.get(tensor.access_element<Y, T>()), 0.);
    EXPECT_EQ(tensor.get(tensor.access_element<Y, X>()), 0.);
    EXPECT_EQ(tensor.get(tensor.access_element<Y, Y>()), 3.);
    EXPECT_EQ(tensor.get(tensor.access_element<Y, Z>()), 0.);
    EXPECT_EQ(tensor.get(tensor.access_element<Z, T>()), 0.);
    EXPECT_EQ(tensor.get(tensor.access_element<Z, X>()), 0.);
    EXPECT_EQ(tensor.get(tensor.access_element<Z, Y>()), 0.);
    EXPECT_EQ(tensor.get(tensor.access_element<Z, Z>()), 4.);
}

using SymIndex = sil::tensor::TensorSymmetricIndex<Mu, Nu>;

TEST(Tensor, TensorSymmetricIndexing4x4)
{
    sil::tensor::TensorAccessor<SymIndex> tensor_accessor;
    ddc::DiscreteDomain<SymIndex> tensor_dom = tensor_accessor.mem_domain();
    ddc::Chunk tensor_alloc(tensor_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor<
            double,
            ddc::DiscreteDomain<SymIndex>,
            Kokkos::layout_right,
            Kokkos::DefaultHostExecutionSpace::memory_space>
            tensor(tensor_alloc);


    /*
    for (int i = 0; i < 10; ++i) {
        tensor(ddc::DiscreteElement<SymIndex>(i)) = i;
    }
    */

    tensor(tensor.access_element<T, T>()) = 0.;
    tensor(tensor.access_element<T, X>()) = 1.;
    tensor(tensor.access_element<T, Y>()) = 2.;
    tensor(tensor.access_element<T, Z>()) = 3.;
    tensor(tensor.access_element<X, X>()) = 4.;
    tensor(tensor.access_element<X, Y>()) = 5.;
    tensor(tensor.access_element<X, Z>()) = 6.;
    tensor(tensor.access_element<Y, Y>()) = 7.;
    tensor(tensor.access_element<Y, Z>()) = 8.;
    tensor(tensor.access_element<Z, Z>()) = 9.;

    EXPECT_EQ(tensor.get(tensor.access_element<T, T>()), 0.);
    EXPECT_EQ(tensor.get(tensor.access_element<T, X>()), 1.);
    EXPECT_EQ(tensor.get(tensor.access_element<T, Y>()), 2.);
    EXPECT_EQ(tensor.get(tensor.access_element<T, Z>()), 3.);
    EXPECT_EQ(tensor.get(tensor.access_element<X, T>()), 1.);
    EXPECT_EQ(tensor.get(tensor.access_element<X, X>()), 4.);
    EXPECT_EQ(tensor.get(tensor.access_element<X, Y>()), 5.);
    EXPECT_EQ(tensor.get(tensor.access_element<X, Z>()), 6.);
    EXPECT_EQ(tensor.get(tensor.access_element<Y, T>()), 2.);
    EXPECT_EQ(tensor.get(tensor.access_element<Y, X>()), 5.);
    EXPECT_EQ(tensor.get(tensor.access_element<Y, Y>()), 7.);
    EXPECT_EQ(tensor.get(tensor.access_element<Y, Z>()), 8.);
    EXPECT_EQ(tensor.get(tensor.access_element<Z, T>()), 3.);
    EXPECT_EQ(tensor.get(tensor.access_element<Z, X>()), 6.);
    EXPECT_EQ(tensor.get(tensor.access_element<Z, Y>()), 8.);
    EXPECT_EQ(tensor.get(tensor.access_element<Z, Z>()), 9.);
}

using SymIndex3x3x3 = sil::tensor::TensorSymmetricIndex<Alpha, Beta, Gamma>;

TEST(Tensor, TensorSymmetricIndexing3x3x3)
{
    sil::tensor::TensorAccessor<SymIndex3x3x3> tensor_accessor;
    ddc::DiscreteDomain<SymIndex3x3x3> tensor_dom = tensor_accessor.mem_domain();
    ddc::Chunk tensor_alloc(tensor_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor<
            double,
            ddc::DiscreteDomain<SymIndex3x3x3>,
            Kokkos::layout_right,
            Kokkos::DefaultHostExecutionSpace::memory_space>
            tensor(tensor_alloc);

    /*
    for (int i = 0; i < 10; ++i) {
        tensor(ddc::DiscreteElement<SymIndex3x3x3>(i)) = i;
    }
    */

    tensor(tensor.access_element<X, X, X>()) = 0.;
    tensor(tensor.access_element<X, X, Y>()) = 1.;
    tensor(tensor.access_element<X, X, Z>()) = 2.;
    tensor(tensor.access_element<X, Y, Y>()) = 3.;
    tensor(tensor.access_element<X, Y, Z>()) = 4.;
    tensor(tensor.access_element<X, Z, Z>()) = 5.;
    tensor(tensor.access_element<Y, Y, Y>()) = 6.;
    tensor(tensor.access_element<Y, Y, Z>()) = 7.;
    tensor(tensor.access_element<Y, Z, Z>()) = 8.;
    tensor(tensor.access_element<Z, Z, Z>()) = 9.;

    EXPECT_EQ(tensor.get(tensor.access_element<X, X, X>()), 0.);
    EXPECT_EQ(tensor.get(tensor.access_element<X, X, Y>()), 1.);
    EXPECT_EQ(tensor.get(tensor.access_element<X, X, Z>()), 2.);
    EXPECT_EQ(tensor.get(tensor.access_element<X, Y, X>()), 1.);
    EXPECT_EQ(tensor.get(tensor.access_element<X, Y, Y>()), 3.);
    EXPECT_EQ(tensor.get(tensor.access_element<X, Y, Z>()), 4.);
    EXPECT_EQ(tensor.get(tensor.access_element<X, Z, X>()), 2.);
    EXPECT_EQ(tensor.get(tensor.access_element<X, Z, Y>()), 4.);
    EXPECT_EQ(tensor.get(tensor.access_element<X, Z, Z>()), 5.);
    EXPECT_EQ(tensor.get(tensor.access_element<Y, X, X>()), 1.);
    EXPECT_EQ(tensor.get(tensor.access_element<Y, X, Y>()), 3.);
    EXPECT_EQ(tensor.get(tensor.access_element<Y, X, Z>()), 4.);
    EXPECT_EQ(tensor.get(tensor.access_element<Y, Y, X>()), 3.);
    EXPECT_EQ(tensor.get(tensor.access_element<Y, Y, Y>()), 6.);
    EXPECT_EQ(tensor.get(tensor.access_element<Y, Y, Z>()), 7.);
    EXPECT_EQ(tensor.get(tensor.access_element<Y, Z, X>()), 4.);
    EXPECT_EQ(tensor.get(tensor.access_element<Y, Z, Y>()), 7.);
    EXPECT_EQ(tensor.get(tensor.access_element<Y, Z, Z>()), 8.);
    EXPECT_EQ(tensor.get(tensor.access_element<Z, X, X>()), 2.);
    EXPECT_EQ(tensor.get(tensor.access_element<Z, X, Y>()), 4.);
    EXPECT_EQ(tensor.get(tensor.access_element<Z, X, Z>()), 5.);
    EXPECT_EQ(tensor.get(tensor.access_element<Z, Y, X>()), 4.);
    EXPECT_EQ(tensor.get(tensor.access_element<Z, Y, Y>()), 7.);
    EXPECT_EQ(tensor.get(tensor.access_element<Z, Y, Z>()), 8.);
    EXPECT_EQ(tensor.get(tensor.access_element<Z, Z, X>()), 5.);
    EXPECT_EQ(tensor.get(tensor.access_element<Z, Z, Y>()), 8.);
    EXPECT_EQ(tensor.get(tensor.access_element<Z, Z, Z>()), 9.);
}

using AntisymIndex = sil::tensor::TensorAntisymmetricIndex<Mu, Nu>;

TEST(Tensor, TensorAntisymmetricIndexing4x4)
{
    sil::tensor::TensorAccessor<AntisymIndex> tensor_accessor;
    ddc::DiscreteDomain<AntisymIndex> tensor_dom = tensor_accessor.mem_domain();
    ddc::Chunk tensor_alloc(tensor_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor<
            double,
            ddc::DiscreteDomain<AntisymIndex>,
            Kokkos::layout_right,
            Kokkos::DefaultHostExecutionSpace::memory_space>
            tensor(tensor_alloc);

    /*
    for (int i = 1; i < 7; ++i) {
        tensor(ddc::DiscreteElement<AntisymIndex>(i)) = i;
    }
    */

    tensor(tensor.access_element<T, X>()) = 1.;
    tensor(tensor.access_element<T, Y>()) = 2.;
    tensor(tensor.access_element<T, Z>()) = 3.;
    tensor(tensor.access_element<X, Y>()) = 4.;
    tensor(tensor.access_element<X, Z>()) = 5.;
    tensor(tensor.access_element<Y, Z>()) = 6.;

    EXPECT_EQ(tensor.get(tensor.access_element<T, T>()), 0.);
    EXPECT_EQ(tensor.get(tensor.access_element<T, X>()), 1.);
    EXPECT_EQ(tensor.get(tensor.access_element<T, Y>()), 2.);
    EXPECT_EQ(tensor.get(tensor.access_element<T, Z>()), 3.);
    EXPECT_EQ(tensor.get(tensor.access_element<X, T>()), -1.);
    EXPECT_EQ(tensor.get(tensor.access_element<X, X>()), 0.);
    EXPECT_EQ(tensor.get(tensor.access_element<X, Y>()), 4.);
    EXPECT_EQ(tensor.get(tensor.access_element<X, Z>()), 5.);
    EXPECT_EQ(tensor.get(tensor.access_element<Y, T>()), -2.);
    EXPECT_EQ(tensor.get(tensor.access_element<Y, X>()), -4.);
    EXPECT_EQ(tensor.get(tensor.access_element<Y, Y>()), 0.);
    EXPECT_EQ(tensor.get(tensor.access_element<Y, Z>()), 6.);
    EXPECT_EQ(tensor.get(tensor.access_element<Z, T>()), -3.);
    EXPECT_EQ(tensor.get(tensor.access_element<Z, X>()), -5.);
    EXPECT_EQ(tensor.get(tensor.access_element<Z, Y>()), -6.);
    EXPECT_EQ(tensor.get(tensor.access_element<Z, Z>()), 0.);
}

using LeviCivitaIndex = sil::tensor::TensorLeviCivitaIndex<Alpha, Beta, Gamma>;

TEST(Tensor, TensorLeviCivitaIndexing)
{
    sil::tensor::TensorAccessor<LeviCivitaIndex> tensor_accessor;
    ddc::DiscreteDomain<LeviCivitaIndex> tensor_dom = tensor_accessor.mem_domain();
    ddc::Chunk tensor_alloc(tensor_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor<
            double,
            ddc::DiscreteDomain<LeviCivitaIndex>,
            Kokkos::layout_right,
            Kokkos::DefaultHostExecutionSpace::memory_space>
            tensor(tensor_alloc);

    EXPECT_EQ(tensor.get(tensor.access_element<X, X, X>()), 0.);
    EXPECT_EQ(tensor.get(tensor.access_element<X, X, Y>()), 0.);
    EXPECT_EQ(tensor.get(tensor.access_element<X, X, Z>()), 0.);
    EXPECT_EQ(tensor.get(tensor.access_element<X, Y, X>()), 0.);
    EXPECT_EQ(tensor.get(tensor.access_element<X, Y, Y>()), 0.);
    EXPECT_EQ(tensor.get(tensor.access_element<X, Y, Z>()), 1.);
    EXPECT_EQ(tensor.get(tensor.access_element<X, Z, X>()), 0.);
    EXPECT_EQ(tensor.get(tensor.access_element<X, Z, Y>()), -1.);
    EXPECT_EQ(tensor.get(tensor.access_element<X, Z, Z>()), 0.);
    EXPECT_EQ(tensor.get(tensor.access_element<Y, X, X>()), 0.);
    EXPECT_EQ(tensor.get(tensor.access_element<Y, X, Y>()), 0.);
    EXPECT_EQ(tensor.get(tensor.access_element<Y, X, Z>()), -1.);
    EXPECT_EQ(tensor.get(tensor.access_element<Y, Y, X>()), 0.);
    EXPECT_EQ(tensor.get(tensor.access_element<Y, Y, Y>()), 0.);
    EXPECT_EQ(tensor.get(tensor.access_element<Y, Y, Z>()), 0.);
    EXPECT_EQ(tensor.get(tensor.access_element<Y, Z, X>()), 1.);
    EXPECT_EQ(tensor.get(tensor.access_element<Y, Z, Y>()), 0.);
    EXPECT_EQ(tensor.get(tensor.access_element<Y, Z, Z>()), 0.);
    EXPECT_EQ(tensor.get(tensor.access_element<Z, X, X>()), 0.);
    EXPECT_EQ(tensor.get(tensor.access_element<Z, X, Y>()), 1.);
    EXPECT_EQ(tensor.get(tensor.access_element<Z, X, Z>()), 0.);
    EXPECT_EQ(tensor.get(tensor.access_element<Z, Y, X>()), -1.);
    EXPECT_EQ(tensor.get(tensor.access_element<Z, Y, Y>()), 0.);
    EXPECT_EQ(tensor.get(tensor.access_element<Z, Y, Z>()), 0.);
    EXPECT_EQ(tensor.get(tensor.access_element<Z, Z, X>()), 0.);
    EXPECT_EQ(tensor.get(tensor.access_element<Z, Z, Y>()), 0.);
    EXPECT_EQ(tensor.get(tensor.access_element<Z, Z, Z>()), 0.);
}

using SymIndex3x3 = sil::tensor::TensorSymmetricIndex<Alpha, Beta>;

TEST(Tensor, PartiallyTensorSymmetricIndexing4x3x3)
{
    sil::tensor::TensorAccessor<Mu, SymIndex3x3> tensor_accessor;
    ddc::DiscreteDomain<Mu, SymIndex3x3> tensor_dom = tensor_accessor.mem_domain();
    ddc::Chunk tensor_alloc(tensor_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor<
            double,
            ddc::DiscreteDomain<Mu, SymIndex3x3>,
            Kokkos::layout_right,
            Kokkos::DefaultHostExecutionSpace::memory_space>
            tensor(tensor_alloc);

    /*
    for (int i = 0; i < 24; ++i) {
        tensor(ddc::DiscreteElement<PartiallySymIndex>(i)) = i;
    }
    */

    tensor(tensor.access_element<T, X, X>()) = 0.;
    tensor(tensor.access_element<T, X, Y>()) = 1.;
    tensor(tensor.access_element<T, X, Z>()) = 2.;
    tensor(tensor.access_element<T, Y, Y>()) = 3.;
    tensor(tensor.access_element<T, Y, Z>()) = 4.;
    tensor(tensor.access_element<T, Z, Z>()) = 5.;
    tensor(tensor.access_element<X, X, X>()) = 6.;
    tensor(tensor.access_element<X, X, Y>()) = 7.;
    tensor(tensor.access_element<X, X, Z>()) = 8.;
    tensor(tensor.access_element<X, Y, Y>()) = 9.;
    tensor(tensor.access_element<X, Y, Z>()) = 10.;
    tensor(tensor.access_element<X, Z, Z>()) = 11.;
    tensor(tensor.access_element<Y, X, X>()) = 12.;
    tensor(tensor.access_element<Y, X, Y>()) = 13.;
    tensor(tensor.access_element<Y, X, Z>()) = 14.;
    tensor(tensor.access_element<Y, Y, Y>()) = 15.;
    tensor(tensor.access_element<Y, Y, Z>()) = 16.;
    tensor(tensor.access_element<Y, Z, Z>()) = 17.;
    tensor(tensor.access_element<Z, X, X>()) = 18.;
    tensor(tensor.access_element<Z, X, Y>()) = 19.;
    tensor(tensor.access_element<Z, X, Z>()) = 20.;
    tensor(tensor.access_element<Z, Y, Y>()) = 21.;
    tensor(tensor.access_element<Z, Y, Z>()) = 22.;
    tensor(tensor.access_element<Z, Z, Z>()) = 23.;

    EXPECT_EQ(tensor.get(tensor.access_element<T, X, X>()), 0.);
    EXPECT_EQ(tensor.get(tensor.access_element<T, X, Y>()), 1.);
    EXPECT_EQ(tensor.get(tensor.access_element<T, X, Z>()), 2.);
    EXPECT_EQ(tensor.get(tensor.access_element<T, Y, X>()), 1.);
    EXPECT_EQ(tensor.get(tensor.access_element<T, Y, Y>()), 3.);
    EXPECT_EQ(tensor.get(tensor.access_element<T, Y, Z>()), 4.);
    EXPECT_EQ(tensor.get(tensor.access_element<T, Z, X>()), 2.);
    EXPECT_EQ(tensor.get(tensor.access_element<T, Z, Y>()), 4.);
    EXPECT_EQ(tensor.get(tensor.access_element<T, Z, Z>()), 5.);
    EXPECT_EQ(tensor.get(tensor.access_element<X, X, X>()), 6.);
    EXPECT_EQ(tensor.get(tensor.access_element<X, X, Y>()), 7.);
    EXPECT_EQ(tensor.get(tensor.access_element<X, X, Z>()), 8.);
    EXPECT_EQ(tensor.get(tensor.access_element<X, Y, X>()), 7.);
    EXPECT_EQ(tensor.get(tensor.access_element<X, Y, Y>()), 9.);
    EXPECT_EQ(tensor.get(tensor.access_element<X, Y, Z>()), 10.);
    EXPECT_EQ(tensor.get(tensor.access_element<X, Z, X>()), 8.);
    EXPECT_EQ(tensor.get(tensor.access_element<X, Z, Y>()), 10.);
    EXPECT_EQ(tensor.get(tensor.access_element<X, Z, Z>()), 11.);
    EXPECT_EQ(tensor.get(tensor.access_element<Y, X, X>()), 12.);
    EXPECT_EQ(tensor.get(tensor.access_element<Y, X, Y>()), 13.);
    EXPECT_EQ(tensor.get(tensor.access_element<Y, X, Z>()), 14.);
    EXPECT_EQ(tensor.get(tensor.access_element<Y, Y, X>()), 13.);
    EXPECT_EQ(tensor.get(tensor.access_element<Y, Y, Y>()), 15.);
    EXPECT_EQ(tensor.get(tensor.access_element<Y, Y, Z>()), 16.);
    EXPECT_EQ(tensor.get(tensor.access_element<Y, Z, X>()), 14.);
    EXPECT_EQ(tensor.get(tensor.access_element<Y, Z, Y>()), 16.);
    EXPECT_EQ(tensor.get(tensor.access_element<Y, Z, Z>()), 17.);
    EXPECT_EQ(tensor.get(tensor.access_element<Z, X, X>()), 18.);
    EXPECT_EQ(tensor.get(tensor.access_element<Z, X, Y>()), 19.);
    EXPECT_EQ(tensor.get(tensor.access_element<Z, X, Z>()), 20.);
    EXPECT_EQ(tensor.get(tensor.access_element<Z, Y, X>()), 19.);
    EXPECT_EQ(tensor.get(tensor.access_element<Z, Y, Y>()), 21.);
    EXPECT_EQ(tensor.get(tensor.access_element<Z, Y, Z>()), 22.);
    EXPECT_EQ(tensor.get(tensor.access_element<Z, Z, X>()), 20.);
    EXPECT_EQ(tensor.get(tensor.access_element<Z, Z, Y>()), 22.);
    EXPECT_EQ(tensor.get(tensor.access_element<Z, Z, Z>()), 23.);
}

using YoungTableauIndex = sil::tensor::TensorYoungTableauIndex<
        sil::young_tableau::
                YoungTableau<3, sil::young_tableau::YoungTableauSeq<std::index_sequence<1, 2, 3>>>,
        Alpha,
        Beta,
        Gamma>;

TEST(Tensor, YoungTableauIndexing)
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

    sil::tensor::TensorAccessor<YoungTableauIndex> tensor_accessor;
    ddc::DiscreteDomain<YoungTableauIndex> tensor_dom = tensor_accessor.mem_domain();
    ddc::Chunk tensor_alloc(tensor_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor<
            double,
            ddc::DiscreteDomain<YoungTableauIndex>,
            Kokkos::layout_right,
            Kokkos::DefaultHostExecutionSpace::memory_space>
            tensor(tensor_alloc);

    sil::tensor::compress(tensor, natural);

    EXPECT_DOUBLE_EQ(tensor.get(tensor.access_element<X, X, X>()), 0.);
    EXPECT_DOUBLE_EQ(tensor.get(tensor.access_element<X, X, Y>()), 1.);
    EXPECT_DOUBLE_EQ(tensor.get(tensor.access_element<X, X, Z>()), 2.);
    EXPECT_DOUBLE_EQ(tensor.get(tensor.access_element<X, Y, X>()), 1.);
    EXPECT_DOUBLE_EQ(tensor.get(tensor.access_element<X, Y, Y>()), 3.);
    EXPECT_DOUBLE_EQ(tensor.get(tensor.access_element<X, Y, Z>()), 4.);
    EXPECT_DOUBLE_EQ(tensor.get(tensor.access_element<X, Z, X>()), 2.);
    EXPECT_DOUBLE_EQ(tensor.get(tensor.access_element<X, Z, Y>()), 4.);
    EXPECT_DOUBLE_EQ(tensor.get(tensor.access_element<X, Z, Z>()), 5.);
    EXPECT_DOUBLE_EQ(tensor.get(tensor.access_element<Y, X, X>()), 1.);
    EXPECT_DOUBLE_EQ(tensor.get(tensor.access_element<Y, X, Y>()), 3.);
    EXPECT_DOUBLE_EQ(tensor.get(tensor.access_element<Y, X, Z>()), 4.);
    EXPECT_DOUBLE_EQ(tensor.get(tensor.access_element<Y, Y, X>()), 3.);
    EXPECT_DOUBLE_EQ(tensor.get(tensor.access_element<Y, Y, Y>()), 6.);
    EXPECT_DOUBLE_EQ(tensor.get(tensor.access_element<Y, Y, Z>()), 7.);
    EXPECT_DOUBLE_EQ(tensor.get(tensor.access_element<Y, Z, X>()), 4.);
    EXPECT_DOUBLE_EQ(tensor.get(tensor.access_element<Y, Z, Y>()), 7.);
    EXPECT_DOUBLE_EQ(tensor.get(tensor.access_element<Y, Z, Z>()), 8.);
    EXPECT_DOUBLE_EQ(tensor.get(tensor.access_element<Z, X, X>()), 2.);
    EXPECT_DOUBLE_EQ(tensor.get(tensor.access_element<Z, X, Y>()), 4.);
    EXPECT_DOUBLE_EQ(tensor.get(tensor.access_element<Z, X, Z>()), 5.);
    EXPECT_DOUBLE_EQ(tensor.get(tensor.access_element<Z, Y, X>()), 4.);
    EXPECT_DOUBLE_EQ(tensor.get(tensor.access_element<Z, Y, Y>()), 7.);
    EXPECT_DOUBLE_EQ(tensor.get(tensor.access_element<Z, Y, Z>()), 8.);
    EXPECT_DOUBLE_EQ(tensor.get(tensor.access_element<Z, Z, X>()), 5.);
    EXPECT_DOUBLE_EQ(tensor.get(tensor.access_element<Z, Z, Y>()), 8.);
    EXPECT_DOUBLE_EQ(tensor.get(tensor.access_element<Z, Z, Z>()), 9.);

    ddc::Chunk uncompressed_alloc(natural_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor<
            double,
            ddc::DiscreteDomain<Alpha, Beta, Gamma>,
            Kokkos::layout_right,
            Kokkos::DefaultHostExecutionSpace::memory_space>
            uncompressed(uncompressed_alloc);

    sil::tensor::uncompress(uncompressed, tensor);

    EXPECT_DOUBLE_EQ(uncompressed.get(natural_accessor.access_element<X, X, X>()), 0.);
    EXPECT_DOUBLE_EQ(uncompressed.get(natural_accessor.access_element<X, X, Y>()), 1.);
    EXPECT_DOUBLE_EQ(uncompressed.get(natural_accessor.access_element<X, X, Z>()), 2.);
    EXPECT_DOUBLE_EQ(uncompressed.get(natural_accessor.access_element<X, Y, X>()), 1.);
    EXPECT_DOUBLE_EQ(uncompressed.get(natural_accessor.access_element<X, Y, Y>()), 3.);
    EXPECT_DOUBLE_EQ(uncompressed.get(natural_accessor.access_element<X, Y, Z>()), 4.);
    EXPECT_DOUBLE_EQ(uncompressed.get(natural_accessor.access_element<X, Z, X>()), 2.);
    EXPECT_DOUBLE_EQ(uncompressed.get(natural_accessor.access_element<X, Z, Y>()), 4.);
    EXPECT_DOUBLE_EQ(uncompressed.get(natural_accessor.access_element<X, Z, Z>()), 5.);
    EXPECT_DOUBLE_EQ(uncompressed.get(natural_accessor.access_element<Y, X, X>()), 1.);
    EXPECT_DOUBLE_EQ(uncompressed.get(natural_accessor.access_element<Y, X, Y>()), 3.);
    EXPECT_DOUBLE_EQ(uncompressed.get(natural_accessor.access_element<Y, X, Z>()), 4.);
    EXPECT_DOUBLE_EQ(uncompressed.get(natural_accessor.access_element<Y, Y, X>()), 3.);
    EXPECT_DOUBLE_EQ(uncompressed.get(natural_accessor.access_element<Y, Y, Y>()), 6.);
    EXPECT_DOUBLE_EQ(uncompressed.get(natural_accessor.access_element<Y, Y, Z>()), 7.);
    EXPECT_DOUBLE_EQ(uncompressed.get(natural_accessor.access_element<Y, Z, X>()), 4.);
    EXPECT_DOUBLE_EQ(uncompressed.get(natural_accessor.access_element<Y, Z, Y>()), 7.);
    EXPECT_DOUBLE_EQ(uncompressed.get(natural_accessor.access_element<Y, Z, Z>()), 8.);
    EXPECT_DOUBLE_EQ(uncompressed.get(natural_accessor.access_element<Z, X, X>()), 2.);
    EXPECT_DOUBLE_EQ(uncompressed.get(natural_accessor.access_element<Z, X, Y>()), 4.);
    EXPECT_DOUBLE_EQ(uncompressed.get(natural_accessor.access_element<Z, X, Z>()), 5.);
    EXPECT_DOUBLE_EQ(uncompressed.get(natural_accessor.access_element<Z, Y, X>()), 4.);
    EXPECT_DOUBLE_EQ(uncompressed.get(natural_accessor.access_element<Z, Y, Y>()), 7.);
    EXPECT_DOUBLE_EQ(uncompressed.get(natural_accessor.access_element<Z, Y, Z>()), 8.);
    EXPECT_DOUBLE_EQ(uncompressed.get(natural_accessor.access_element<Z, Z, X>()), 5.);
    EXPECT_DOUBLE_EQ(uncompressed.get(natural_accessor.access_element<Z, Z, Y>()), 8.);
    EXPECT_DOUBLE_EQ(uncompressed.get(natural_accessor.access_element<Z, Z, Z>()), 9.);
}

TEST(Tensor, Determinant)
{
    sil::tensor::TensorAccessor<SymIndex> tensor_accessor;
    ddc::DiscreteDomain<SymIndex> tensor_dom = tensor_accessor.mem_domain();
    ddc::Chunk tensor_alloc(tensor_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor<
            double,
            ddc::DiscreteDomain<SymIndex>,
            Kokkos::layout_right,
            Kokkos::DefaultHostExecutionSpace::memory_space>
            tensor(tensor_alloc);

    tensor(tensor.access_element<T, T>()) = 1.;
    tensor(tensor.access_element<T, X>()) = 2.;
    tensor(tensor.access_element<T, Y>()) = 3.;
    tensor(tensor.access_element<T, Z>()) = 4.;
    tensor(tensor.access_element<X, X>()) = 5.;
    tensor(tensor.access_element<X, Y>()) = 6.;
    tensor(tensor.access_element<X, Z>()) = 7.;
    tensor(tensor.access_element<Y, Y>()) = 8.;
    tensor(tensor.access_element<Y, Z>()) = 9.;
    tensor(tensor.access_element<Z, Z>()) = 10.;

    EXPECT_EQ(determinant(tensor), -2.);
}
