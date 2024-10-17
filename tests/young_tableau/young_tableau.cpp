// SPDX-License-Identifier: GPL-3.0

#include <cmath>

#include <gtest/gtest.h>

#include "tensor.hpp"
#include "young_tableau.hpp"

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

TEST(YoungTableau, 1_2)
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

    auto [proj_alloc, proj] = young_tableau.projector<Mu, Nu>();

    std::cout << proj;

    ddc::Chunk prod_alloc(tensor_prod_domain(proj, tensor), ddc::HostAllocator<double>());
    sil::tensor::Tensor<
            double,
            sil::tensor::tensor_prod_domain_t<
                    typename decltype(young_tableau)::projector_domain<Mu, Nu>,
                    ddc::DiscreteDomain<Mu, Nu>>,
            std::experimental::layout_right,
            Kokkos::DefaultHostExecutionSpace::memory_space>
            prod(prod_alloc);

    sil::tensor::tensor_prod(prod, proj, tensor);

    sil::tensor::tensor_accessor_for_domain_t<sil::tensor::tensor_prod_domain_t<
                    typename decltype(young_tableau)::projector_domain<Mu, Nu>,
                    ddc::DiscreteDomain<Mu, Nu>>> tensor_accessor_prod;
    EXPECT_EQ(prod.get(tensor_accessor_prod.element<T, X>()), prod.get(tensor_accessor_prod.element<X, T>()));
    EXPECT_EQ(prod.get(tensor_accessor_prod.element<T, Y>()), prod.get(tensor_accessor_prod.element<Y, T>()));
    EXPECT_EQ(prod.get(tensor_accessor_prod.element<T, Z>()), prod.get(tensor_accessor_prod.element<Z, T>()));
    EXPECT_EQ(prod.get(tensor_accessor_prod.element<X, Y>()), prod.get(tensor_accessor_prod.element<Y, X>()));
    EXPECT_EQ(prod.get(tensor_accessor_prod.element<X, Z>()), prod.get(tensor_accessor_prod.element<Z, X>()));
    EXPECT_EQ(prod.get(tensor_accessor_prod.element<Y, Z>()), prod.get(tensor_accessor_prod.element<Z, Y>()));
}

TEST(YoungTableau, 1l2)
{
    sil::young_tableau::YoungTableau<
            4,
            sil::young_tableau::YoungTableauSeq<std::index_sequence<1>, std::index_sequence<2>>>
            young_tableau;

    EXPECT_EQ(young_tableau.irrep_dim(), 6);

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

    auto [proj_alloc, proj] = young_tableau.projector<Mu, Nu>();

    std::cout << proj;

    ddc::Chunk prod_alloc(tensor_prod_domain(proj, tensor), ddc::HostAllocator<double>());
    sil::tensor::Tensor<
            double,
            sil::tensor::tensor_prod_domain_t<
                    typename decltype(young_tableau)::projector_domain<Mu, Nu>,
                    ddc::DiscreteDomain<Mu, Nu>>,
            std::experimental::layout_right,
            Kokkos::DefaultHostExecutionSpace::memory_space>
            prod(prod_alloc);

    sil::tensor::tensor_prod(prod, proj, tensor);

    sil::tensor::tensor_accessor_for_domain_t<sil::tensor::tensor_prod_domain_t<
                    typename decltype(young_tableau)::projector_domain<Mu, Nu>,
                    ddc::DiscreteDomain<Mu, Nu>>> tensor_accessor_prod;
    EXPECT_EQ(prod.get(tensor_accessor_prod.element<T, T>()), 0.);
    EXPECT_EQ(prod.get(tensor_accessor_prod.element<T, X>()), -prod.get(tensor_accessor_prod.element<X, T>()));
    EXPECT_EQ(prod.get(tensor_accessor_prod.element<T, Y>()), -prod.get(tensor_accessor_prod.element<Y, T>()));
    EXPECT_EQ(prod.get(tensor_accessor_prod.element<T, Z>()), -prod.get(tensor_accessor_prod.element<Z, T>()));
    EXPECT_EQ(prod.get(tensor_accessor_prod.element<X, X>()), 0.);
    EXPECT_EQ(prod.get(tensor_accessor_prod.element<X, Y>()), -prod.get(tensor_accessor_prod.element<Y, X>()));
    EXPECT_EQ(prod.get(tensor_accessor_prod.element<X, Z>()), -prod.get(tensor_accessor_prod.element<Z, X>()));
    EXPECT_EQ(prod.get(tensor_accessor_prod.element<Y, Y>()), 0.);
    EXPECT_EQ(prod.get(tensor_accessor_prod.element<Y, Z>()), -prod.get(tensor_accessor_prod.element<Z, Y>()));
    EXPECT_EQ(prod.get(tensor_accessor_prod.element<Z, Z>()), 0.);
}

TEST(YoungTableau, 1_2_3)
{
    sil::young_tableau::
            YoungTableau<3, sil::young_tableau::YoungTableauSeq<std::index_sequence<1, 2, 3>>>
                    young_tableau;

    EXPECT_EQ(young_tableau.irrep_dim(), 10);

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
    tensor(tensor_accessor.element<X, Y, Y>()) = 3.;
    tensor(tensor_accessor.element<X, Y, Z>()) = 4.;
    tensor(tensor_accessor.element<X, Z, Z>()) = 5.;
    tensor(tensor_accessor.element<Y, Y, Y>()) = 6.;
    tensor(tensor_accessor.element<Y, Y, Z>()) = 7.;
    tensor(tensor_accessor.element<Y, Z, Z>()) = 8.;
    tensor(tensor_accessor.element<Z, Z, Z>()) = 9.;

    auto [proj_alloc, proj] = young_tableau.projector<Alpha, Beta, Gamma>();

    ddc::Chunk prod_alloc(tensor_prod_domain(proj, tensor), ddc::HostAllocator<double>());
    sil::tensor::Tensor<
            double,
            sil::tensor::tensor_prod_domain_t<
                    typename decltype(young_tableau)::projector_domain<Alpha, Beta, Gamma>,
                    ddc::DiscreteDomain<Alpha, Beta, Gamma>>,
            std::experimental::layout_right,
            Kokkos::DefaultHostExecutionSpace::memory_space>
            prod(prod_alloc);

    sil::tensor::tensor_prod(prod, proj, tensor);

    sil::tensor::tensor_accessor_for_domain_t<sil::tensor::tensor_prod_domain_t<
                    typename decltype(young_tableau)::projector_domain<Alpha, Beta, Gamma>,
                    ddc::DiscreteDomain<Alpha, Beta, Gamma>>> tensor_accessor_prod;

    EXPECT_EQ(prod.get(tensor_accessor_prod.element<X, X, Y>()), prod.get(tensor_accessor_prod.element<X, Y, X>()));
    EXPECT_EQ(prod.get(tensor_accessor_prod.element<X, Y, X>()), prod.get(tensor_accessor_prod.element<Y, X, X>()));
    EXPECT_EQ(prod.get(tensor_accessor_prod.element<X, X, Z>()), prod.get(tensor_accessor_prod.element<X, Z, X>()));
    EXPECT_EQ(prod.get(tensor_accessor_prod.element<X, Z, X>()), prod.get(tensor_accessor_prod.element<Z, X, X>()));
    EXPECT_EQ(prod.get(tensor_accessor_prod.element<Y, Y, X>()), prod.get(tensor_accessor_prod.element<Y, X, Y>()));
    EXPECT_EQ(prod.get(tensor_accessor_prod.element<Y, X, Y>()), prod.get(tensor_accessor_prod.element<X, Y, Y>()));
    EXPECT_EQ(prod.get(tensor_accessor_prod.element<Y, Y, Z>()), prod.get(tensor_accessor_prod.element<Y, Z, Y>()));
    EXPECT_EQ(prod.get(tensor_accessor_prod.element<Y, Z, Y>()), prod.get(tensor_accessor_prod.element<Z, Y, Y>()));
    EXPECT_EQ(prod.get(tensor_accessor_prod.element<Z, Z, X>()), prod.get(tensor_accessor_prod.element<Z, X, Z>()));
    EXPECT_EQ(prod.get(tensor_accessor_prod.element<Z, X, Z>()), prod.get(tensor_accessor_prod.element<X, Z, Z>()));
    EXPECT_EQ(prod.get(tensor_accessor_prod.element<Z, Z, Y>()), prod.get(tensor_accessor_prod.element<Z, Y, Z>()));
    EXPECT_EQ(prod.get(tensor_accessor_prod.element<X, Y, Z>()), prod.get(tensor_accessor_prod.element<X, Z, Y>()));
    EXPECT_EQ(prod.get(tensor_accessor_prod.element<X, Z, Y>()), prod.get(tensor_accessor_prod.element<Y, X, Z>()));
    EXPECT_EQ(prod.get(tensor_accessor_prod.element<Y, X, Z>()), prod.get(tensor_accessor_prod.element<Y, Z, X>()));
    EXPECT_EQ(prod.get(tensor_accessor_prod.element<Y, Z, X>()), prod.get(tensor_accessor_prod.element<Z, X, Y>()));
    EXPECT_EQ(prod.get(tensor_accessor_prod.element<Z, X, Y>()), prod.get(tensor_accessor_prod.element<Z, Y, X>()));
}

TEST(YoungTableau, 1l2l3)
{
    sil::young_tableau::YoungTableau<
            4,
            sil::young_tableau::YoungTableauSeq<
                    std::index_sequence<1>,
                    std::index_sequence<2>,
                    std::index_sequence<3>>>
            young_tableau;

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
    tensor(tensor_accessor.element<X, Y, Y>()) = 3.;
    tensor(tensor_accessor.element<X, Y, Z>()) = 4.;
    tensor(tensor_accessor.element<X, Z, Z>()) = 5.;
    tensor(tensor_accessor.element<Y, Y, Y>()) = 6.;
    tensor(tensor_accessor.element<Y, Y, Z>()) = 7.;
    tensor(tensor_accessor.element<Y, Z, Z>()) = 8.;
    tensor(tensor_accessor.element<Z, Z, Z>()) = 9.;

    auto [proj_alloc, proj] = young_tableau.projector<Alpha, Beta, Gamma>();

    ddc::Chunk prod_alloc(tensor_prod_domain(proj, tensor), ddc::HostAllocator<double>());
    sil::tensor::Tensor<
            double,
            sil::tensor::tensor_prod_domain_t<
                    typename decltype(young_tableau)::projector_domain<Alpha, Beta, Gamma>,
                    ddc::DiscreteDomain<Alpha, Beta, Gamma>>,
            std::experimental::layout_right,
            Kokkos::DefaultHostExecutionSpace::memory_space>
            prod(prod_alloc);

    sil::tensor::tensor_prod(prod, proj, tensor);

    sil::tensor::tensor_accessor_for_domain_t<sil::tensor::tensor_prod_domain_t<
                    typename decltype(young_tableau)::projector_domain<Alpha, Beta, Gamma>,
                    ddc::DiscreteDomain<Alpha, Beta, Gamma>>> tensor_accessor_prod;

    EXPECT_EQ(prod.get(tensor_accessor_prod.element<X, X, Y>()), -prod.get(tensor_accessor_prod.element<X, Y, X>()));
    EXPECT_EQ(prod.get(tensor_accessor_prod.element<X, Y, X>()), -prod.get(tensor_accessor_prod.element<Y, X, X>()));
    EXPECT_EQ(prod.get(tensor_accessor_prod.element<X, X, Z>()), -prod.get(tensor_accessor_prod.element<X, Z, X>()));
    EXPECT_EQ(prod.get(tensor_accessor_prod.element<X, Z, X>()), -prod.get(tensor_accessor_prod.element<Z, X, X>()));
    EXPECT_EQ(prod.get(tensor_accessor_prod.element<Y, Y, X>()), -prod.get(tensor_accessor_prod.element<Y, X, Y>()));
    EXPECT_EQ(prod.get(tensor_accessor_prod.element<Y, X, Y>()), -prod.get(tensor_accessor_prod.element<X, Y, Y>()));
    EXPECT_EQ(prod.get(tensor_accessor_prod.element<Y, Y, Z>()), -prod.get(tensor_accessor_prod.element<Y, Z, Y>()));
    EXPECT_EQ(prod.get(tensor_accessor_prod.element<Y, Z, Y>()), -prod.get(tensor_accessor_prod.element<Z, Y, Y>()));
    EXPECT_EQ(prod.get(tensor_accessor_prod.element<Z, Z, X>()), -prod.get(tensor_accessor_prod.element<Z, X, Z>()));
    EXPECT_EQ(prod.get(tensor_accessor_prod.element<Z, X, Z>()), -prod.get(tensor_accessor_prod.element<X, Z, Z>()));
    EXPECT_EQ(prod.get(tensor_accessor_prod.element<Z, Z, Y>()), -prod.get(tensor_accessor_prod.element<Z, Y, Z>()));
    EXPECT_EQ(prod.get(tensor_accessor_prod.element<X, Y, Z>()), -prod.get(tensor_accessor_prod.element<X, Z, Y>()));
    EXPECT_EQ(prod.get(tensor_accessor_prod.element<X, Z, Y>()), prod.get(tensor_accessor_prod.element<Y, X, Z>()));
    EXPECT_EQ(prod.get(tensor_accessor_prod.element<Y, X, Z>()), -prod.get(tensor_accessor_prod.element<Y, Z, X>()));
    EXPECT_EQ(prod.get(tensor_accessor_prod.element<Y, Z, X>()), prod.get(tensor_accessor_prod.element<Z, X, Y>()));
    EXPECT_EQ(prod.get(tensor_accessor_prod.element<Z, X, Y>()), -prod.get(tensor_accessor_prod.element<Z, Y, X>()));
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
