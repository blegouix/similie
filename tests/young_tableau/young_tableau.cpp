// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: GPL-3.0-or-later

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

    [[maybe_unused]] sil::tensor::TensorAccessor<Mu, Nu> tensor_accessor;
    ddc::DiscreteDomain<Mu, Nu> tensor_dom = tensor_accessor.domain();
    ddc::Chunk tensor_alloc(tensor_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor tensor(tensor_alloc);

    tensor(tensor.access_element<T, T>()) = 0.;
    tensor(tensor.access_element<T, X>()) = 1.;
    tensor(tensor.access_element<T, Y>()) = 2.;
    tensor(tensor.access_element<T, Z>()) = 3.;
    tensor(tensor.access_element<X, T>()) = 4.;
    tensor(tensor.access_element<X, X>()) = 5.;
    tensor(tensor.access_element<X, Y>()) = 6.;
    tensor(tensor.access_element<X, Z>()) = 7.;
    tensor(tensor.access_element<Y, T>()) = 8.;
    tensor(tensor.access_element<Y, X>()) = 9.;
    tensor(tensor.access_element<Y, Y>()) = 10;
    tensor(tensor.access_element<Y, Z>()) = 11;
    tensor(tensor.access_element<Z, T>()) = 12.;
    tensor(tensor.access_element<Z, X>()) = 13.;
    tensor(tensor.access_element<Z, Y>()) = 14.;
    tensor(tensor.access_element<Z, Z>()) = 15.;

    auto [proj_alloc, proj] = young_tableau.projector<Mu, Nu>();

    ddc::Chunk prod_alloc(
            natural_tensor_prod_domain(proj.domain(), tensor.domain()),
            ddc::HostAllocator<double>());
    sil::tensor::Tensor prod(prod_alloc);

    sil::tensor::tensor_prod(prod, proj, tensor);

    sil::tensor::tensor_accessor_for_domain_t<sil::tensor::natural_tensor_prod_domain_t<
            typename decltype(young_tableau)::projector_domain<Mu, Nu>,
            ddc::DiscreteDomain<Mu, Nu>>>
            tensor_accessor_prod;
    EXPECT_EQ(
            prod.get(tensor_accessor_prod.access_element<T, X>()),
            prod.get(tensor_accessor_prod.access_element<X, T>()));
    EXPECT_EQ(
            prod.get(tensor_accessor_prod.access_element<T, Y>()),
            prod.get(tensor_accessor_prod.access_element<Y, T>()));
    EXPECT_EQ(
            prod.get(tensor_accessor_prod.access_element<T, Z>()),
            prod.get(tensor_accessor_prod.access_element<Z, T>()));
    EXPECT_EQ(
            prod.get(tensor_accessor_prod.access_element<X, Y>()),
            prod.get(tensor_accessor_prod.access_element<Y, X>()));
    EXPECT_EQ(
            prod.get(tensor_accessor_prod.access_element<X, Z>()),
            prod.get(tensor_accessor_prod.access_element<Z, X>()));
    EXPECT_EQ(
            prod.get(tensor_accessor_prod.access_element<Y, Z>()),
            prod.get(tensor_accessor_prod.access_element<Z, Y>()));
}

TEST(YoungTableau, 1l2)
{
    sil::young_tableau::YoungTableau<
            4,
            sil::young_tableau::YoungTableauSeq<std::index_sequence<1>, std::index_sequence<2>>>
            young_tableau;

    EXPECT_EQ(young_tableau.irrep_dim(), 6);

    [[maybe_unused]] sil::tensor::TensorAccessor<Mu, Nu> tensor_accessor;
    ddc::DiscreteDomain<Mu, Nu> tensor_dom = tensor_accessor.domain();
    ddc::Chunk tensor_alloc(tensor_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor tensor(tensor_alloc);

    tensor(tensor.access_element<T, T>()) = 0.;
    tensor(tensor.access_element<T, X>()) = 1.;
    tensor(tensor.access_element<T, Y>()) = 2.;
    tensor(tensor.access_element<T, Z>()) = 3.;
    tensor(tensor.access_element<X, T>()) = 4.;
    tensor(tensor.access_element<X, X>()) = 5.;
    tensor(tensor.access_element<X, Y>()) = 6.;
    tensor(tensor.access_element<X, Z>()) = 7.;
    tensor(tensor.access_element<Y, T>()) = 8.;
    tensor(tensor.access_element<Y, X>()) = 9.;
    tensor(tensor.access_element<Y, Y>()) = 10;
    tensor(tensor.access_element<Y, Z>()) = 11;
    tensor(tensor.access_element<Z, T>()) = 12.;
    tensor(tensor.access_element<Z, X>()) = 13.;
    tensor(tensor.access_element<Z, Y>()) = 14.;
    tensor(tensor.access_element<Z, Z>()) = 15.;

    auto [proj_alloc, proj] = young_tableau.projector<Mu, Nu>();

    ddc::Chunk prod_alloc(
            natural_tensor_prod_domain(proj.domain(), tensor.domain()),
            ddc::HostAllocator<double>());
    sil::tensor::Tensor prod(prod_alloc);

    sil::tensor::tensor_prod(prod, proj, tensor);

    sil::tensor::tensor_accessor_for_domain_t<sil::tensor::natural_tensor_prod_domain_t<
            typename decltype(young_tableau)::projector_domain<Mu, Nu>,
            ddc::DiscreteDomain<Mu, Nu>>>
            tensor_accessor_prod;
    EXPECT_EQ(prod.get(tensor_accessor_prod.access_element<T, T>()), 0.);
    EXPECT_EQ(
            prod.get(tensor_accessor_prod.access_element<T, X>()),
            -prod.get(tensor_accessor_prod.access_element<X, T>()));
    EXPECT_EQ(
            prod.get(tensor_accessor_prod.access_element<T, Y>()),
            -prod.get(tensor_accessor_prod.access_element<Y, T>()));
    EXPECT_EQ(
            prod.get(tensor_accessor_prod.access_element<T, Z>()),
            -prod.get(tensor_accessor_prod.access_element<Z, T>()));
    EXPECT_EQ(prod.get(tensor_accessor_prod.access_element<X, X>()), 0.);
    EXPECT_EQ(
            prod.get(tensor_accessor_prod.access_element<X, Y>()),
            -prod.get(tensor_accessor_prod.access_element<Y, X>()));
    EXPECT_EQ(
            prod.get(tensor_accessor_prod.access_element<X, Z>()),
            -prod.get(tensor_accessor_prod.access_element<Z, X>()));
    EXPECT_EQ(prod.get(tensor_accessor_prod.access_element<Y, Y>()), 0.);
    EXPECT_EQ(
            prod.get(tensor_accessor_prod.access_element<Y, Z>()),
            -prod.get(tensor_accessor_prod.access_element<Z, Y>()));
    EXPECT_EQ(prod.get(tensor_accessor_prod.access_element<Z, Z>()), 0.);
}

TEST(YoungTableau, 1_2_3)
{
    sil::young_tableau::
            YoungTableau<3, sil::young_tableau::YoungTableauSeq<std::index_sequence<1, 2, 3>>>
                    young_tableau;

    EXPECT_EQ(young_tableau.irrep_dim(), 10);

    [[maybe_unused]] sil::tensor::TensorAccessor<Alpha, Beta, Gamma> tensor_accessor;
    ddc::DiscreteDomain<Alpha, Beta, Gamma> tensor_dom = tensor_accessor.domain();
    ddc::Chunk tensor_alloc(tensor_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor tensor(tensor_alloc);

    tensor(tensor.access_element<X, X, X>()) = 0.;
    tensor(tensor.access_element<X, X, Y>()) = 1.;
    tensor(tensor.access_element<X, X, Z>()) = 2.;
    tensor(tensor.access_element<X, Y, X>()) = 3.;
    tensor(tensor.access_element<X, Y, Y>()) = 4.;
    tensor(tensor.access_element<X, Y, Z>()) = 5.;
    tensor(tensor.access_element<X, Z, X>()) = 6.;
    tensor(tensor.access_element<X, Z, Y>()) = 7.;
    tensor(tensor.access_element<X, Z, Z>()) = 8.;
    tensor(tensor.access_element<Y, X, X>()) = 9.;
    tensor(tensor.access_element<Y, X, Y>()) = 10.;
    tensor(tensor.access_element<Y, X, Z>()) = 11.;
    tensor(tensor.access_element<Y, Y, X>()) = 12.;
    tensor(tensor.access_element<Y, Y, Y>()) = 13.;
    tensor(tensor.access_element<Y, Y, Z>()) = 14.;
    tensor(tensor.access_element<Y, Z, X>()) = 15.;
    tensor(tensor.access_element<Y, Z, Y>()) = 16.;
    tensor(tensor.access_element<Y, Z, Z>()) = 17.;
    tensor(tensor.access_element<Z, X, X>()) = 18.;
    tensor(tensor.access_element<Z, X, Y>()) = 19.;
    tensor(tensor.access_element<Z, X, Z>()) = 20.;
    tensor(tensor.access_element<Z, Y, X>()) = 21.;
    tensor(tensor.access_element<Z, Y, Y>()) = 22.;
    tensor(tensor.access_element<Z, Y, Z>()) = 23.;
    tensor(tensor.access_element<Z, Z, X>()) = 24.;
    tensor(tensor.access_element<Z, Z, Y>()) = 25.;
    tensor(tensor.access_element<Z, Z, Z>()) = 26.;

    auto [proj_alloc, proj] = young_tableau.projector<Alpha, Beta, Gamma>();

    ddc::Chunk prod_alloc(
            natural_tensor_prod_domain(proj.domain(), tensor.domain()),
            ddc::HostAllocator<double>());
    sil::tensor::Tensor prod(prod_alloc);

    sil::tensor::tensor_prod(prod, proj, tensor);

    sil::tensor::tensor_accessor_for_domain_t<sil::tensor::natural_tensor_prod_domain_t<
            typename decltype(young_tableau)::projector_domain<Alpha, Beta, Gamma>,
            ddc::DiscreteDomain<Alpha, Beta, Gamma>>>
            tensor_accessor_prod;

    EXPECT_EQ(
            prod.get(tensor_accessor_prod.access_element<X, X, Y>()),
            prod.get(tensor_accessor_prod.access_element<X, Y, X>()));
    EXPECT_EQ(
            prod.get(tensor_accessor_prod.access_element<X, Y, X>()),
            prod.get(tensor_accessor_prod.access_element<Y, X, X>()));
    EXPECT_EQ(
            prod.get(tensor_accessor_prod.access_element<X, X, Z>()),
            prod.get(tensor_accessor_prod.access_element<X, Z, X>()));
    EXPECT_EQ(
            prod.get(tensor_accessor_prod.access_element<X, Z, X>()),
            prod.get(tensor_accessor_prod.access_element<Z, X, X>()));
    EXPECT_EQ(
            prod.get(tensor_accessor_prod.access_element<Y, Y, X>()),
            prod.get(tensor_accessor_prod.access_element<Y, X, Y>()));
    EXPECT_EQ(
            prod.get(tensor_accessor_prod.access_element<Y, X, Y>()),
            prod.get(tensor_accessor_prod.access_element<X, Y, Y>()));
    EXPECT_EQ(
            prod.get(tensor_accessor_prod.access_element<Y, Y, Z>()),
            prod.get(tensor_accessor_prod.access_element<Y, Z, Y>()));
    EXPECT_EQ(
            prod.get(tensor_accessor_prod.access_element<Y, Z, Y>()),
            prod.get(tensor_accessor_prod.access_element<Z, Y, Y>()));
    EXPECT_EQ(
            prod.get(tensor_accessor_prod.access_element<Z, Z, X>()),
            prod.get(tensor_accessor_prod.access_element<Z, X, Z>()));
    EXPECT_EQ(
            prod.get(tensor_accessor_prod.access_element<Z, X, Z>()),
            prod.get(tensor_accessor_prod.access_element<X, Z, Z>()));
    EXPECT_EQ(
            prod.get(tensor_accessor_prod.access_element<Z, Z, Y>()),
            prod.get(tensor_accessor_prod.access_element<Z, Y, Z>()));
    EXPECT_EQ(
            prod.get(tensor_accessor_prod.access_element<X, Y, Z>()),
            prod.get(tensor_accessor_prod.access_element<X, Z, Y>()));
    EXPECT_EQ(
            prod.get(tensor_accessor_prod.access_element<X, Z, Y>()),
            prod.get(tensor_accessor_prod.access_element<Y, X, Z>()));
    EXPECT_EQ(
            prod.get(tensor_accessor_prod.access_element<Y, X, Z>()),
            prod.get(tensor_accessor_prod.access_element<Y, Z, X>()));
    EXPECT_EQ(
            prod.get(tensor_accessor_prod.access_element<Y, Z, X>()),
            prod.get(tensor_accessor_prod.access_element<Z, X, Y>()));
    EXPECT_EQ(
            prod.get(tensor_accessor_prod.access_element<Z, X, Y>()),
            prod.get(tensor_accessor_prod.access_element<Z, Y, X>()));
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

    [[maybe_unused]] sil::tensor::TensorAccessor<Alpha, Beta, Gamma> tensor_accessor;
    ddc::DiscreteDomain<Alpha, Beta, Gamma> tensor_dom = tensor_accessor.domain();
    ddc::Chunk tensor_alloc(tensor_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor tensor(tensor_alloc);

    tensor(tensor.access_element<X, X, X>()) = 0.;
    tensor(tensor.access_element<X, X, Y>()) = 1.;
    tensor(tensor.access_element<X, X, Z>()) = 2.;
    tensor(tensor.access_element<X, Y, X>()) = 3.;
    tensor(tensor.access_element<X, Y, Y>()) = 4.;
    tensor(tensor.access_element<X, Y, Z>()) = 5.;
    tensor(tensor.access_element<X, Z, X>()) = 6.;
    tensor(tensor.access_element<X, Z, Y>()) = 7.;
    tensor(tensor.access_element<X, Z, Z>()) = 8.;
    tensor(tensor.access_element<Y, X, X>()) = 9.;
    tensor(tensor.access_element<Y, X, Y>()) = 10.;
    tensor(tensor.access_element<Y, X, Z>()) = 11.;
    tensor(tensor.access_element<Y, Y, X>()) = 12.;
    tensor(tensor.access_element<Y, Y, Y>()) = 13.;
    tensor(tensor.access_element<Y, Y, Z>()) = 14.;
    tensor(tensor.access_element<Y, Z, X>()) = 15.;
    tensor(tensor.access_element<Y, Z, Y>()) = 16.;
    tensor(tensor.access_element<Y, Z, Z>()) = 17.;
    tensor(tensor.access_element<Z, X, X>()) = 18.;
    tensor(tensor.access_element<Z, X, Y>()) = 19.;
    tensor(tensor.access_element<Z, X, Z>()) = 20.;
    tensor(tensor.access_element<Z, Y, X>()) = 21.;
    tensor(tensor.access_element<Z, Y, Y>()) = 22.;
    tensor(tensor.access_element<Z, Y, Z>()) = 23.;
    tensor(tensor.access_element<Z, Z, X>()) = 24.;
    tensor(tensor.access_element<Z, Z, Y>()) = 25.;
    tensor(tensor.access_element<Z, Z, Z>()) = 26.;

    auto [proj_alloc, proj] = young_tableau.projector<Alpha, Beta, Gamma>();

    ddc::Chunk prod_alloc(
            natural_tensor_prod_domain(proj.domain(), tensor.domain()),
            ddc::HostAllocator<double>());
    sil::tensor::Tensor prod(prod_alloc);

    sil::tensor::tensor_prod(prod, proj, tensor);

    sil::tensor::tensor_accessor_for_domain_t<sil::tensor::natural_tensor_prod_domain_t<
            typename decltype(young_tableau)::projector_domain<Alpha, Beta, Gamma>,
            ddc::DiscreteDomain<Alpha, Beta, Gamma>>>
            tensor_accessor_prod;

    EXPECT_EQ(prod.get(tensor_accessor_prod.access_element<X, X, X>()), 0.);
    EXPECT_EQ(prod.get(tensor_accessor_prod.access_element<Y, Y, Y>()), 0.);
    EXPECT_EQ(prod.get(tensor_accessor_prod.access_element<Z, Z, Z>()), 0.);
    EXPECT_EQ(
            prod.get(tensor_accessor_prod.access_element<X, X, Y>()),
            -prod.get(tensor_accessor_prod.access_element<X, Y, X>()));
    EXPECT_EQ(
            prod.get(tensor_accessor_prod.access_element<X, Y, X>()),
            -prod.get(tensor_accessor_prod.access_element<Y, X, X>()));
    EXPECT_EQ(
            prod.get(tensor_accessor_prod.access_element<X, X, Z>()),
            -prod.get(tensor_accessor_prod.access_element<X, Z, X>()));
    EXPECT_EQ(
            prod.get(tensor_accessor_prod.access_element<X, Z, X>()),
            -prod.get(tensor_accessor_prod.access_element<Z, X, X>()));
    EXPECT_EQ(
            prod.get(tensor_accessor_prod.access_element<Y, Y, X>()),
            -prod.get(tensor_accessor_prod.access_element<Y, X, Y>()));
    EXPECT_EQ(
            prod.get(tensor_accessor_prod.access_element<Y, X, Y>()),
            -prod.get(tensor_accessor_prod.access_element<X, Y, Y>()));
    EXPECT_EQ(
            prod.get(tensor_accessor_prod.access_element<Y, Y, Z>()),
            -prod.get(tensor_accessor_prod.access_element<Y, Z, Y>()));
    EXPECT_EQ(
            prod.get(tensor_accessor_prod.access_element<Y, Z, Y>()),
            -prod.get(tensor_accessor_prod.access_element<Z, Y, Y>()));
    EXPECT_EQ(
            prod.get(tensor_accessor_prod.access_element<Z, Z, X>()),
            -prod.get(tensor_accessor_prod.access_element<Z, X, Z>()));
    EXPECT_EQ(
            prod.get(tensor_accessor_prod.access_element<Z, X, Z>()),
            -prod.get(tensor_accessor_prod.access_element<X, Z, Z>()));
    EXPECT_EQ(
            prod.get(tensor_accessor_prod.access_element<Z, Z, Y>()),
            -prod.get(tensor_accessor_prod.access_element<Z, Y, Z>()));
    EXPECT_EQ(
            prod.get(tensor_accessor_prod.access_element<X, Y, Z>()),
            -prod.get(tensor_accessor_prod.access_element<X, Z, Y>()));
    EXPECT_EQ(
            prod.get(tensor_accessor_prod.access_element<X, Z, Y>()),
            prod.get(tensor_accessor_prod.access_element<Y, X, Z>()));
    EXPECT_EQ(
            prod.get(tensor_accessor_prod.access_element<Y, X, Z>()),
            -prod.get(tensor_accessor_prod.access_element<Y, Z, X>()));
    EXPECT_EQ(
            prod.get(tensor_accessor_prod.access_element<Y, Z, X>()),
            prod.get(tensor_accessor_prod.access_element<Z, X, Y>()));
    EXPECT_EQ(
            prod.get(tensor_accessor_prod.access_element<Z, X, Y>()),
            -prod.get(tensor_accessor_prod.access_element<Z, Y, X>()));
}

TEST(YoungTableau, 1_2l3)
{
    sil::young_tableau::YoungTableau<
            3,
            sil::young_tableau::YoungTableauSeq<std::index_sequence<1, 2>, std::index_sequence<3>>>
            young_tableau;

    EXPECT_EQ(young_tableau.irrep_dim(), 8);

    [[maybe_unused]] sil::tensor::TensorAccessor<Alpha, Beta, Gamma> tensor_accessor;
    ddc::DiscreteDomain<Alpha, Beta, Gamma> tensor_dom = tensor_accessor.domain();
    ddc::Chunk tensor_alloc(tensor_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor tensor(tensor_alloc);

    tensor(tensor.access_element<X, X, X>()) = 0.;
    tensor(tensor.access_element<X, X, Y>()) = 1.;
    tensor(tensor.access_element<X, X, Z>()) = 2.;
    tensor(tensor.access_element<X, Y, X>()) = 3.;
    tensor(tensor.access_element<X, Y, Y>()) = 4.;
    tensor(tensor.access_element<X, Y, Z>()) = 5.;
    tensor(tensor.access_element<X, Z, X>()) = 6.;
    tensor(tensor.access_element<X, Z, Y>()) = 7.;
    tensor(tensor.access_element<X, Z, Z>()) = 8.;
    tensor(tensor.access_element<Y, X, X>()) = 9.;
    tensor(tensor.access_element<Y, X, Y>()) = 10.;
    tensor(tensor.access_element<Y, X, Z>()) = 11.;
    tensor(tensor.access_element<Y, Y, X>()) = 12.;
    tensor(tensor.access_element<Y, Y, Y>()) = 13.;
    tensor(tensor.access_element<Y, Y, Z>()) = 14.;
    tensor(tensor.access_element<Y, Z, X>()) = 15.;
    tensor(tensor.access_element<Y, Z, Y>()) = 16.;
    tensor(tensor.access_element<Y, Z, Z>()) = 17.;
    tensor(tensor.access_element<Z, X, X>()) = 18.;
    tensor(tensor.access_element<Z, X, Y>()) = 19.;
    tensor(tensor.access_element<Z, X, Z>()) = 20.;
    tensor(tensor.access_element<Z, Y, X>()) = 21.;
    tensor(tensor.access_element<Z, Y, Y>()) = 22.;
    tensor(tensor.access_element<Z, Y, Z>()) = 23.;
    tensor(tensor.access_element<Z, Z, X>()) = 24.;
    tensor(tensor.access_element<Z, Z, Y>()) = 25.;
    tensor(tensor.access_element<Z, Z, Z>()) = 26.;

    auto [proj_alloc, proj] = young_tableau.projector<Alpha, Beta, Gamma>();

    ddc::Chunk prod_alloc(
            natural_tensor_prod_domain(proj.domain(), tensor.domain()),
            ddc::HostAllocator<double>());
    sil::tensor::Tensor prod(prod_alloc);

    sil::tensor::tensor_prod(prod, proj, tensor);

    sil::tensor::tensor_accessor_for_domain_t<sil::tensor::natural_tensor_prod_domain_t<
            typename decltype(young_tableau)::projector_domain<Alpha, Beta, Gamma>,
            ddc::DiscreteDomain<Alpha, Beta, Gamma>>>
            tensor_accessor_prod;

    EXPECT_EQ(
            prod.get(tensor_accessor_prod.access_element<X, Y, Z>()),
            -prod.get(tensor_accessor_prod.access_element<Z, Y, X>()));
    // TODO understand better the symmetry of rank-3 mixed tensor and test it.
}

TEST(YoungTableau, 1_3l2)
{
    sil::young_tableau::YoungTableau<
            3,
            sil::young_tableau::YoungTableauSeq<std::index_sequence<1, 3>, std::index_sequence<2>>>
            young_tableau;

    EXPECT_EQ(young_tableau.irrep_dim(), 8);

    [[maybe_unused]] sil::tensor::TensorAccessor<Alpha, Beta, Gamma> tensor_accessor;
    ddc::DiscreteDomain<Alpha, Beta, Gamma> tensor_dom = tensor_accessor.domain();
    ddc::Chunk tensor_alloc(tensor_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor tensor(tensor_alloc);

    tensor(tensor.access_element<X, X, X>()) = 0.;
    tensor(tensor.access_element<X, X, Y>()) = 1.;
    tensor(tensor.access_element<X, X, Z>()) = 2.;
    tensor(tensor.access_element<X, Y, X>()) = 3.;
    tensor(tensor.access_element<X, Y, Y>()) = 4.;
    tensor(tensor.access_element<X, Y, Z>()) = 5.;
    tensor(tensor.access_element<X, Z, X>()) = 6.;
    tensor(tensor.access_element<X, Z, Y>()) = 7.;
    tensor(tensor.access_element<X, Z, Z>()) = 8.;
    tensor(tensor.access_element<Y, X, X>()) = 9.;
    tensor(tensor.access_element<Y, X, Y>()) = 10.;
    tensor(tensor.access_element<Y, X, Z>()) = 11.;
    tensor(tensor.access_element<Y, Y, X>()) = 12.;
    tensor(tensor.access_element<Y, Y, Y>()) = 13.;
    tensor(tensor.access_element<Y, Y, Z>()) = 14.;
    tensor(tensor.access_element<Y, Z, X>()) = 15.;
    tensor(tensor.access_element<Y, Z, Y>()) = 16.;
    tensor(tensor.access_element<Y, Z, Z>()) = 17.;
    tensor(tensor.access_element<Z, X, X>()) = 18.;
    tensor(tensor.access_element<Z, X, Y>()) = 19.;
    tensor(tensor.access_element<Z, X, Z>()) = 20.;
    tensor(tensor.access_element<Z, Y, X>()) = 21.;
    tensor(tensor.access_element<Z, Y, Y>()) = 22.;
    tensor(tensor.access_element<Z, Y, Z>()) = 23.;
    tensor(tensor.access_element<Z, Z, X>()) = 24.;
    tensor(tensor.access_element<Z, Z, Y>()) = 25.;
    tensor(tensor.access_element<Z, Z, Z>()) = 26.;

    auto [proj_alloc, proj] = young_tableau.projector<Alpha, Beta, Gamma>();

    ddc::Chunk prod_alloc(
            natural_tensor_prod_domain(proj.domain(), tensor.domain()),
            ddc::HostAllocator<double>());
    sil::tensor::Tensor prod(prod_alloc);

    sil::tensor::tensor_prod(prod, proj, tensor);

    sil::tensor::tensor_accessor_for_domain_t<sil::tensor::natural_tensor_prod_domain_t<
            typename decltype(young_tableau)::projector_domain<Alpha, Beta, Gamma>,
            ddc::DiscreteDomain<Alpha, Beta, Gamma>>>
            tensor_accessor_prod;

    EXPECT_EQ(
            prod.get(tensor_accessor_prod.access_element<X, Y, Z>()),
            -prod.get(tensor_accessor_prod.access_element<Y, X, Z>()));
    // TODO understand better the symmetry of rank-3 mixed tensor and test it.
}

TEST(YoungTableau, 1l3_2l4)
{
    sil::young_tableau::YoungTableau<
            3,
            sil::young_tableau::
                    YoungTableauSeq<std::index_sequence<1, 3>, std::index_sequence<2, 4>>>
            young_tableau;

    EXPECT_EQ(young_tableau.irrep_dim(), 6);

    [[maybe_unused]] sil::tensor::TensorAccessor<Alpha, Beta, Gamma, Delta> tensor_accessor;
    ddc::DiscreteDomain<Alpha, Beta, Gamma, Delta> tensor_dom = tensor_accessor.domain();
    ddc::Chunk tensor_alloc(tensor_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor tensor(tensor_alloc);

    tensor(tensor.access_element<X, X, X, X>()) = 0.;
    tensor(tensor.access_element<X, X, X, Y>()) = 1.;
    tensor(tensor.access_element<X, X, X, Z>()) = 2.;
    tensor(tensor.access_element<X, X, Y, X>()) = 3.;
    tensor(tensor.access_element<X, X, Y, Y>()) = 4.;
    tensor(tensor.access_element<X, X, Y, Z>()) = 5.;
    tensor(tensor.access_element<X, X, Z, X>()) = 6.;
    tensor(tensor.access_element<X, X, Z, Y>()) = 7.;
    tensor(tensor.access_element<X, X, Z, Z>()) = 8.;
    tensor(tensor.access_element<X, Y, X, X>()) = 9.;
    tensor(tensor.access_element<X, Y, X, Y>()) = 10.;
    tensor(tensor.access_element<X, Y, X, Z>()) = 11.;
    tensor(tensor.access_element<X, Y, Y, X>()) = 12.;
    tensor(tensor.access_element<X, Y, Y, Y>()) = 13.;
    tensor(tensor.access_element<X, Y, Y, Z>()) = 14.;
    tensor(tensor.access_element<X, Y, Z, X>()) = 15.;
    tensor(tensor.access_element<X, Y, Z, Y>()) = 16.;
    tensor(tensor.access_element<X, Y, Z, Z>()) = 17.;
    tensor(tensor.access_element<X, Z, X, X>()) = 18.;
    tensor(tensor.access_element<X, Z, X, Y>()) = 19.;
    tensor(tensor.access_element<X, Z, X, Z>()) = 20.;
    tensor(tensor.access_element<X, Z, Y, X>()) = 21.;
    tensor(tensor.access_element<X, Z, Y, Y>()) = 22.;
    tensor(tensor.access_element<X, Z, Y, Z>()) = 23.;
    tensor(tensor.access_element<X, Z, Z, X>()) = 24.;
    tensor(tensor.access_element<X, Z, Z, Y>()) = 25.;
    tensor(tensor.access_element<X, Z, Z, Z>()) = 26.;
    tensor(tensor.access_element<Y, X, X, X>()) = 27.;
    tensor(tensor.access_element<Y, X, X, Y>()) = 28.;
    tensor(tensor.access_element<Y, X, X, Z>()) = 29.;
    tensor(tensor.access_element<Y, X, Y, X>()) = 30.;
    tensor(tensor.access_element<Y, X, Y, Y>()) = 31.;
    tensor(tensor.access_element<Y, X, Y, Z>()) = 32.;
    tensor(tensor.access_element<Y, X, Z, X>()) = 33.;
    tensor(tensor.access_element<Y, X, Z, Y>()) = 34.;
    tensor(tensor.access_element<Y, X, Z, Z>()) = 35.;
    tensor(tensor.access_element<Y, Y, X, X>()) = 36.;
    tensor(tensor.access_element<Y, Y, X, Y>()) = 37.;
    tensor(tensor.access_element<Y, Y, X, Z>()) = 38.;
    tensor(tensor.access_element<Y, Y, Y, X>()) = 39.;
    tensor(tensor.access_element<Y, Y, Y, Y>()) = 40.;
    tensor(tensor.access_element<Y, Y, Y, Z>()) = 41.;
    tensor(tensor.access_element<Y, Y, Z, X>()) = 42.;
    tensor(tensor.access_element<Y, Y, Z, Y>()) = 43.;
    tensor(tensor.access_element<Y, Y, Z, Z>()) = 44.;
    tensor(tensor.access_element<Y, Z, X, X>()) = 45.;
    tensor(tensor.access_element<Y, Z, X, Y>()) = 46.;
    tensor(tensor.access_element<Y, Z, X, Z>()) = 47.;
    tensor(tensor.access_element<Y, Z, Y, X>()) = 48.;
    tensor(tensor.access_element<Y, Z, Y, Y>()) = 49.;
    tensor(tensor.access_element<Y, Z, Y, Z>()) = 50.;
    tensor(tensor.access_element<Y, Z, Z, X>()) = 51.;
    tensor(tensor.access_element<Y, Z, Z, Y>()) = 52.;
    tensor(tensor.access_element<Y, Z, Z, Z>()) = 53.;
    tensor(tensor.access_element<Z, X, X, X>()) = 54.;
    tensor(tensor.access_element<Z, X, X, Y>()) = 55.;
    tensor(tensor.access_element<Z, X, X, Z>()) = 56.;
    tensor(tensor.access_element<Z, X, Y, X>()) = 57.;
    tensor(tensor.access_element<Z, X, Y, Y>()) = 58.;
    tensor(tensor.access_element<Z, X, Y, Z>()) = 59.;
    tensor(tensor.access_element<Z, X, Z, X>()) = 60.;
    tensor(tensor.access_element<Z, X, Z, Y>()) = 61.;
    tensor(tensor.access_element<Z, X, Z, Z>()) = 62.;
    tensor(tensor.access_element<Z, Y, X, X>()) = 63.;
    tensor(tensor.access_element<Z, Y, X, Y>()) = 64.;
    tensor(tensor.access_element<Z, Y, X, Z>()) = 65.;
    tensor(tensor.access_element<Z, Y, Y, X>()) = 66.;
    tensor(tensor.access_element<Z, Y, Y, Y>()) = 67.;
    tensor(tensor.access_element<Z, Y, Y, Z>()) = 68.;
    tensor(tensor.access_element<Z, Y, Z, X>()) = 69.;
    tensor(tensor.access_element<Z, Y, Z, Y>()) = 70.;
    tensor(tensor.access_element<Z, Y, Z, Z>()) = 71.;
    tensor(tensor.access_element<Z, Z, X, X>()) = 72.;
    tensor(tensor.access_element<Z, Z, X, Y>()) = 73.;
    tensor(tensor.access_element<Z, Z, X, Z>()) = 74.;
    tensor(tensor.access_element<Z, Z, Y, X>()) = 75.;
    tensor(tensor.access_element<Z, Z, Y, Y>()) = 76.;
    tensor(tensor.access_element<Z, Z, Y, Z>()) = 77.;
    tensor(tensor.access_element<Z, Z, Z, X>()) = 78.;
    tensor(tensor.access_element<Z, Z, Z, Y>()) = 79.;
    tensor(tensor.access_element<Z, Z, Z, Z>()) = 80.;

    auto [proj_alloc, proj] = young_tableau.projector<Alpha, Beta, Gamma, Delta>();

    ddc::Chunk prod_alloc(
            natural_tensor_prod_domain(proj.domain(), tensor.domain()),
            ddc::HostAllocator<double>());
    sil::tensor::Tensor prod(prod_alloc);

    sil::tensor::tensor_prod(prod, proj, tensor);

    sil::tensor::tensor_accessor_for_domain_t<sil::tensor::natural_tensor_prod_domain_t<
            typename decltype(young_tableau)::projector_domain<Alpha, Beta, Gamma, Delta>,
            ddc::DiscreteDomain<Alpha, Beta, Gamma, Delta>>>
            tensor_accessor_prod;

    EXPECT_EQ(
            prod.get(tensor_accessor_prod.access_element<X, Y, Z, X>()),
            -prod.get(tensor_accessor_prod.access_element<Y, X, Z, X>()));
    EXPECT_EQ(
            prod.get(tensor_accessor_prod.access_element<X, Y, Z, X>()),
            -prod.get(tensor_accessor_prod.access_element<X, Y, X, Z>()));
    EXPECT_EQ(
            prod.get(tensor_accessor_prod.access_element<X, Y, Z, X>()),
            prod.get(tensor_accessor_prod.access_element<Z, X, X, Y>()));
    EXPECT_EQ(
            prod.get(tensor_accessor_prod.access_element<X, Y, Z, X>())
                    + prod.get(tensor_accessor_prod.access_element<X, X, Y, Z>())
                    + prod.get(tensor_accessor_prod.access_element<X, Z, X, Y>()),
            0.);
}
