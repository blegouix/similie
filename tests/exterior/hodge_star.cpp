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

struct Nu : sil::tensor::TensorNaturalIndex<X, Y, Z>
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

using MetricIndex = sil::tensor::TensorSymmetricIndex<
        sil::tensor::Contravariant<sil::tensor::MetricIndex1<X, Y, Z>>,
        sil::tensor::Contravariant<sil::tensor::MetricIndex2<X, Y, Z>>>;

using HodgeStarDomain = sil::exterior::
        hodge_star_domain_t<ddc::detail::TypeSeq<MuUp, NuUp>, ddc::detail::TypeSeq<RhoLow>>;
using HodgeStarDomain2 = sil::exterior::
        hodge_star_domain_t<ddc::detail::TypeSeq<RhoUp>, ddc::detail::TypeSeq<MuLow, NuLow>>;

TEST(HodgeStar, Test)
{
    ddc::DiscreteDomain<DDimX, DDimY>
            mesh_xy(ddc::DiscreteElement<DDimX, DDimY>(0, 0),
                    ddc::DiscreteVector<DDimX, DDimY>(3, 3));

    [[maybe_unused]] sil::tensor::TensorAccessor<MetricIndex> metric_accessor;
    ddc::DiscreteDomain<DDimX, DDimY, MetricIndex> metric_dom(mesh_xy, metric_accessor.domain());
    ddc::Chunk metric_alloc(metric_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor metric(metric_alloc);
    ddc::host_for_each(mesh_xy, [&](ddc::DiscreteElement<DDimX, DDimY> elem) {
        metric(elem, metric.accessor().access_element<X, X>()) = 4.;
        metric(elem, metric.accessor().access_element<X, Y>()) = 1.;
        metric(elem, metric.accessor().access_element<X, Z>()) = 2.;
        metric(elem, metric.accessor().access_element<Y, Y>()) = 5.;
        metric(elem, metric.accessor().access_element<Y, Z>()) = 3.;
        metric(elem, metric.accessor().access_element<Z, Z>()) = 6.;
    });

    [[maybe_unused]] sil::tensor::tensor_accessor_for_domain_t<HodgeStarDomain> hodge_star_accessor;
    ddc::cartesian_prod_t<ddc::DiscreteDomain<DDimX, DDimY>, HodgeStarDomain>
            hodge_star_dom(metric.non_indices_domain(), hodge_star_accessor.domain());
    ddc::Chunk hodge_star_alloc(hodge_star_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor hodge_star(hodge_star_alloc);

    sil::exterior::fill_hodge_star<
            MetricIndex,
            ddc::detail::TypeSeq<MuUp, NuUp>,
            ddc::detail::TypeSeq<RhoLow>>(Kokkos::DefaultHostExecutionSpace(), hodge_star, metric);

    [[maybe_unused]] sil::tensor::TensorAccessor<
            sil::tensor::TensorAntisymmetricIndex<MuLow, NuLow>> form_accessor;
    ddc::DiscreteDomain<DDimX, DDimY, sil::tensor::TensorAntisymmetricIndex<MuLow, NuLow>>
            form_dom(metric.non_indices_domain(), form_accessor.domain());
    ddc::Chunk form_alloc(form_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor form(form_alloc);
    ddc::host_for_each(mesh_xy, [&](ddc::DiscreteElement<DDimX, DDimY> elem) {
        form(elem, form.accessor().access_element<X, Y>()) = 1.;
        form(elem, form.accessor().access_element<X, Z>()) = 2.;
        form(elem, form.accessor().access_element<Y, Z>()) = 3.;
    });

    [[maybe_unused]] sil::tensor::TensorAccessor<RhoLow> dual_form_accessor;
    ddc::DiscreteDomain<DDimX, DDimY, RhoLow>
            dual_form_dom(metric.non_indices_domain(), dual_form_accessor.domain());
    ddc::Chunk dual_form_alloc(dual_form_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor dual_form(dual_form_alloc);

    ddc::host_for_each(mesh_xy, [&](ddc::DiscreteElement<DDimX, DDimY> elem) {
        sil::tensor::tensor_prod(dual_form[elem], hodge_star[elem], form[elem]);
    });

    [[maybe_unused]] sil::tensor::tensor_accessor_for_domain_t<HodgeStarDomain2>
            hodge_star_accessor2;
    ddc::cartesian_prod_t<ddc::DiscreteDomain<DDimX, DDimY>, HodgeStarDomain2>
            hodge_star_dom2(metric.non_indices_domain(), hodge_star_accessor2.domain());
    ddc::Chunk hodge_star_alloc2(hodge_star_dom2, ddc::HostAllocator<double>());
    sil::tensor::Tensor hodge_star2(hodge_star_alloc2);

    sil::exterior::fill_hodge_star<
            MetricIndex,
            ddc::detail::TypeSeq<RhoUp>,
            ddc::detail::TypeSeq<
                    MuLow,
                    NuLow>>(Kokkos::DefaultHostExecutionSpace(), hodge_star2, metric);

    ddc::parallel_fill(form, 0.);
    ddc::host_for_each(mesh_xy, [&](ddc::DiscreteElement<DDimX, DDimY> elem) {
        sil::tensor::tensor_prod(form[elem], hodge_star2[elem], dual_form[elem]);
        EXPECT_DOUBLE_EQ(form(elem, form.accessor().access_element<X, Y>()), 1.);
        EXPECT_DOUBLE_EQ(form(elem, form.accessor().access_element<X, Z>()), 2.);
        EXPECT_DOUBLE_EQ(form(elem, form.accessor().access_element<Y, Z>()), 3.);
    });
}
