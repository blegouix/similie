// SPDX-License-Identifier: GPL-3.0

#include <cmath>

#include <gtest/gtest.h>

#include "young_tableau.hpp"
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

struct Mu : sil::tensor::TensorNaturalIndex<T, X, Y, Z>
{
};

struct Nu : sil::tensor::TensorNaturalIndex<T, X, Y, Z>
{
};

TEST(YoungTableau, IrrepDim1_2)
{
    sil::young_tableau::
            YoungTableau<4, sil::young_tableau::YoungTableauSeq<std::index_sequence<1, 2>>>
                    young_tableau;

    EXPECT_EQ(young_tableau.irrep_dim(), 10);

    sil::tensor::TensorAccessor<Mu, Nu> tensor_accessor;
    ddc::DiscreteDomain<Mu, Nu> tensor_dom = tensor_accessor.mem_domain();
    ddc::Chunk tensor_alloc(tensor_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor<
            double,
            ddc::DiscreteDomain<Mu, Nu>,
            std::experimental::layout_right,
            Kokkos::DefaultHostExecutionSpace::memory_space>
            tensor(tensor_alloc);

    tensor(tensor_accessor.element<T, T>()) = 0.; 
    tensor(tensor_accessor.element<T, X>()) = 1.; 
    tensor(tensor_accessor.element<T, Y>()) = 2.; 
    tensor(tensor_accessor.element<T, Z>()) = 3.;
    tensor(tensor_accessor.element<X, T>()) = 4.; 
    tensor(tensor_accessor.element<X, X>()) = 5.; 
    tensor(tensor_accessor.element<X, Y>()) = 6.; 
    tensor(tensor_accessor.element<X, Z>()) = 7.; 
    tensor(tensor_accessor.element<Y, T>()) = 8.; 
    tensor(tensor_accessor.element<Y, X>()) = 9.; 
    tensor(tensor_accessor.element<Y, Y>()) = 10; 
    tensor(tensor_accessor.element<Y, Z>()) = 11; 
    tensor(tensor_accessor.element<Z, T>()) = 12.; 
    tensor(tensor_accessor.element<Z, X>()) = 13.; 
    tensor(tensor_accessor.element<Z, Y>()) = 14.;
    tensor(tensor_accessor.element<Z, Z>()) = 15.;

    auto [proj_alloc, proj] = young_tableau.projector(); 
    
    std::cout << proj;
}

TEST(YoungTableau, IrrepDim1l2)
{
    sil::young_tableau::YoungTableau<
            4,
            sil::young_tableau::YoungTableauSeq<std::index_sequence<1>, std::index_sequence<2>>>
            young_tableau;

    EXPECT_EQ(young_tableau.irrep_dim(), 6);

    auto [proj_alloc, proj] = young_tableau.projector(); 
    
    std::cout << proj;
}

TEST(YoungTableau, IrrepDim1_2_3)
{
    sil::young_tableau::
            YoungTableau<4, sil::young_tableau::YoungTableauSeq<std::index_sequence<1, 2, 3>>>
                    young_tableau;

    EXPECT_EQ(young_tableau.irrep_dim(), 20);

    auto [proj_alloc, proj] = young_tableau.projector(); 
    
    std::cout << proj;
}

TEST(YoungTableau, IrrepDim1l2l3)
{
    sil::young_tableau::YoungTableau<
            4,
            sil::young_tableau::YoungTableauSeq<
                    std::index_sequence<1>,
                    std::index_sequence<2>,
                    std::index_sequence<3>>>
            young_tableau;

    EXPECT_EQ(young_tableau.irrep_dim(), 4);
}

TEST(YoungTableau, IrrepDim1_2l3)
{
    sil::young_tableau::YoungTableau<
            4,
            sil::young_tableau::YoungTableauSeq<std::index_sequence<1, 2>, std::index_sequence<3>>>
            young_tableau;

    EXPECT_EQ(young_tableau.irrep_dim(), 20);
}

TEST(YoungTableau, IrrepDim1_3l2)
{
    sil::young_tableau::YoungTableau<
            4,
            sil::young_tableau::YoungTableauSeq<std::index_sequence<1, 3>, std::index_sequence<2>>>
            young_tableau;

    EXPECT_EQ(young_tableau.irrep_dim(), 20);
}

TEST(YoungTableau, IrrepDim1l3_2l4)
{
    sil::young_tableau::YoungTableau<
            4,
            sil::young_tableau::
                    YoungTableauSeq<std::index_sequence<1, 3>, std::index_sequence<2, 4>>>
            young_tableau;

    EXPECT_EQ(young_tableau.irrep_dim(), 20);
}
