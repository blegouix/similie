// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: MIT

#include <cmath>

#include <ddc/ddc.hpp>

#include <gtest/gtest.h>
#include <similie/exterior/hodge_star.hpp>
#include <similie/tensor/levi_civita_tensor.hpp>
#include <similie/tensor/metric.hpp>
#include <similie/tensor/symmetric_tensor.hpp>

struct X
{
};

struct Y
{
};

struct Z
{
};

struct DDimX : ddc::UniformPointSampling<X>
{
};

struct DDimY : ddc::UniformPointSampling<Y>
{
};

struct DDimZ : ddc::UniformPointSampling<Z>
{
};

struct Mu : sil::tensor::TensorNaturalIndex<X, Y, Z>
{
};

struct Mu1 : sil::tensor::TensorNaturalIndex<X>
{
};

struct Mu2 : sil::tensor::TensorNaturalIndex<X, Y>
{
};

struct Nu : sil::tensor::TensorNaturalIndex<X, Y, Z>
{
};

struct Nu2 : sil::tensor::TensorNaturalIndex<X, Y>
{
};

struct Rho : sil::tensor::TensorNaturalIndex<X, Y, Z>
{
};

using MuUp = sil::tensor::Contravariant<Mu>;
using MuLow = sil::tensor::Covariant<Mu>;
using NuUp = sil::tensor::Contravariant<Nu>;
using NuLow = sil::tensor::Covariant<Nu>;
using RhoUp = sil::tensor::Contravariant<Rho>;
using RhoLow = sil::tensor::Covariant<Rho>;
using Mu1Low = sil::tensor::Covariant<Mu1>;
using Mu2Up = sil::tensor::Contravariant<Mu2>;
using Mu2Low = sil::tensor::Covariant<Mu2>;
using Nu2Up = sil::tensor::Contravariant<Nu2>;
using Nu2Low = sil::tensor::Covariant<Nu2>;

using MetricIndex = sil::tensor::TensorSymmetricIndex<
        sil::tensor::Contravariant<sil::tensor::MetricIndex1<X, Y, Z>>,
        sil::tensor::Contravariant<sil::tensor::MetricIndex2<X, Y, Z>>>;
using MetricIndex1D = sil::tensor::TensorSymmetricIndex<
        sil::tensor::Contravariant<sil::tensor::MetricIndex1<X>>,
        sil::tensor::Contravariant<sil::tensor::MetricIndex2<X>>>;
using MetricIndex2D = sil::tensor::TensorSymmetricIndex<
        sil::tensor::Contravariant<sil::tensor::MetricIndex1<X, Y>>,
        sil::tensor::Contravariant<sil::tensor::MetricIndex2<X, Y>>>;

using HodgeStarDomain = sil::exterior::
        hodge_star_domain_t<ddc::detail::TypeSeq<MuUp, NuUp>, ddc::detail::TypeSeq<RhoLow>>;
using HodgeStarDomain2 = sil::exterior::
        hodge_star_domain_t<ddc::detail::TypeSeq<RhoUp>, ddc::detail::TypeSeq<MuLow, NuLow>>;
using HodgeStarDomain1D
        = sil::exterior::hodge_star_domain_t<ddc::detail::TypeSeq<>, ddc::detail::TypeSeq<Mu1Low>>;

TEST(HodgeStar, CenteredDualMeshMeasure)
{
    ddc::Coordinate<X> lower_bound(0.);
    ddc::Coordinate<X> upper_bound(4.);
    ddc::DiscreteVector<DDimX> nb_points(5);
    ddc::DiscreteDomain<DDimX> mesh_x = ddc::init_discrete_space<DDimX>(
            DDimX::init<DDimX>(lower_bound, upper_bound, nb_points));

    [[maybe_unused]] sil::tensor::TensorAccessor<MetricIndex1D> metric_accessor;
    ddc::DiscreteDomain<DDimX, MetricIndex1D> metric_dom(mesh_x, metric_accessor.domain());
    ddc::Chunk metric_alloc(metric_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor metric(metric_alloc);
    ddc::host_for_each(mesh_x, [&](ddc::DiscreteElement<DDimX> elem) {
        metric(elem, metric.accessor().access_element<X, X>()) = 1.;
    });

    [[maybe_unused]] sil::tensor::tensor_accessor_for_domain_t<HodgeStarDomain1D>
            hodge_star_accessor;
    ddc::cartesian_prod_t<ddc::DiscreteDomain<DDimX>, HodgeStarDomain1D>
            hodge_star_dom(metric.non_indices_domain(), hodge_star_accessor.domain());
    ddc::Chunk hodge_star_alloc(hodge_star_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor hodge_star(hodge_star_alloc);

    sil::exterior::fill_hodge_star<
            MetricIndex1D,
            ddc::detail::TypeSeq<>,
            ddc::detail::TypeSeq<Mu1Low>>(Kokkos::DefaultHostExecutionSpace(), hodge_star, metric);

    ddc::host_for_each(mesh_x, [&](ddc::DiscreteElement<DDimX> elem) {
        double const expected = (elem == mesh_x.front() || elem == mesh_x.back()) ? 0.5 : 1.;
        EXPECT_DOUBLE_EQ(hodge_star(elem, hodge_star.accessor().access_element<X>()), expected);
    });

    ddc::detail::g_discrete_space_dual<DDimX>.reset();
}

TEST(HodgeStar, Test)
{
    ddc::Coordinate<X> lower_bound_x(0.);
    ddc::Coordinate<X> upper_bound_x(2.);
    ddc::Coordinate<Y> lower_bound_y(0.);
    ddc::Coordinate<Y> upper_bound_y(2.);
    ddc::DiscreteVector<DDimX> nb_points_x(3);
    ddc::DiscreteVector<DDimY> nb_points_y(3);
    ddc::DiscreteDomain<DDimX> mesh_x = ddc::init_discrete_space<DDimX>(
            DDimX::init<DDimX>(lower_bound_x, upper_bound_x, nb_points_x));
    ddc::DiscreteDomain<DDimY> mesh_y = ddc::init_discrete_space<DDimY>(
            DDimY::init<DDimY>(lower_bound_y, upper_bound_y, nb_points_y));
    ddc::DiscreteDomain<DDimX, DDimY> mesh_xy(mesh_x, mesh_y);

    using HodgeStarDomain2D = sil::exterior::
            hodge_star_domain_t<ddc::detail::TypeSeq<Mu2Up>, ddc::detail::TypeSeq<Nu2Low>>;
    using HodgeStarDomain2DInv = sil::exterior::
            hodge_star_domain_t<ddc::detail::TypeSeq<Nu2Up>, ddc::detail::TypeSeq<Mu2Low>>;

    [[maybe_unused]] sil::tensor::TensorAccessor<MetricIndex2D> metric_accessor;
    ddc::DiscreteDomain<DDimX, DDimY, MetricIndex2D> metric_dom(mesh_xy, metric_accessor.domain());
    ddc::Chunk metric_alloc(metric_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor metric(metric_alloc);
    ddc::host_for_each(mesh_xy, [&](ddc::DiscreteElement<DDimX, DDimY> elem) {
        metric(elem, metric.accessor().access_element<X, X>()) = 1.;
        metric(elem, metric.accessor().access_element<X, Y>()) = 0.;
        metric(elem, metric.accessor().access_element<Y, Y>()) = 1.;
    });

    [[maybe_unused]] sil::tensor::tensor_accessor_for_domain_t<HodgeStarDomain2D>
            hodge_star_accessor;
    ddc::cartesian_prod_t<ddc::DiscreteDomain<DDimX, DDimY>, HodgeStarDomain2D>
            hodge_star_dom(metric.non_indices_domain(), hodge_star_accessor.domain());
    ddc::Chunk hodge_star_alloc(hodge_star_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor hodge_star(hodge_star_alloc);

    sil::exterior::fill_hodge_star<
            MetricIndex2D,
            ddc::detail::TypeSeq<Mu2Up>,
            ddc::detail::TypeSeq<Nu2Low>>(Kokkos::DefaultHostExecutionSpace(), hodge_star, metric);

    [[maybe_unused]] sil::tensor::TensorAccessor<Mu2Low> form_accessor;
    ddc::DiscreteDomain<DDimX, DDimY, Mu2Low>
            form_dom(metric.non_indices_domain(), form_accessor.domain());
    ddc::Chunk form_alloc(form_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor form(form_alloc);
    ddc::host_for_each(mesh_xy, [&](ddc::DiscreteElement<DDimX, DDimY> elem) {
        form(elem, form.accessor().access_element<X>()) = 1.;
        form(elem, form.accessor().access_element<Y>()) = 2.;
    });

    [[maybe_unused]] sil::tensor::TensorAccessor<Nu2Low> dual_form_accessor;
    ddc::DiscreteDomain<DDimX, DDimY, Nu2Low>
            dual_form_dom(metric.non_indices_domain(), dual_form_accessor.domain());
    ddc::Chunk dual_form_alloc(dual_form_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor dual_form(dual_form_alloc);
    ddc::DiscreteElement<DDimX, DDimY> const interior_elem(1, 1);

    ddc::host_for_each(mesh_xy, [&](ddc::DiscreteElement<DDimX, DDimY> elem) {
        sil::tensor::tensor_prod(dual_form[elem], hodge_star[elem], form[elem]);
        if (elem == interior_elem) {
            EXPECT_DOUBLE_EQ(dual_form(elem, dual_form.accessor().access_element<X>()), -2.);
            EXPECT_DOUBLE_EQ(dual_form(elem, dual_form.accessor().access_element<Y>()), 1.);
        }
    });

    [[maybe_unused]] sil::tensor::tensor_accessor_for_domain_t<HodgeStarDomain2DInv>
            hodge_star_accessor2;
    ddc::cartesian_prod_t<ddc::DiscreteDomain<DDimX, DDimY>, HodgeStarDomain2DInv>
            hodge_star_dom2(metric.non_indices_domain(), hodge_star_accessor2.domain());
    ddc::Chunk hodge_star_alloc2(hodge_star_dom2, ddc::HostAllocator<double>());
    sil::tensor::Tensor hodge_star2(hodge_star_alloc2);

    sil::exterior::fill_hodge_star<
            MetricIndex2D,
            ddc::detail::TypeSeq<Nu2Up>,
            ddc::detail::TypeSeq<Mu2Low>>(Kokkos::DefaultHostExecutionSpace(), hodge_star2, metric);

    ddc::parallel_fill(form, 0.);
    ddc::host_for_each(mesh_xy, [&](ddc::DiscreteElement<DDimX, DDimY> elem) {
        sil::tensor::tensor_prod(form[elem], hodge_star2[elem], dual_form[elem]);
        if (elem == interior_elem) {
            EXPECT_DOUBLE_EQ(form(elem, form.accessor().access_element<X>()), -1.);
            EXPECT_DOUBLE_EQ(form(elem, form.accessor().access_element<Y>()), -2.);
        }
    });

    ddc::detail::g_discrete_space_dual<DDimX>.reset();
    ddc::detail::g_discrete_space_dual<DDimY>.reset();
}
