// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: MIT

#include <ddc/ddc.hpp>

#include <gtest/gtest.h>
#include <similie/exterior/hodge_star.hpp>
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
        sil::tensor::Covariant<sil::tensor::MetricIndex1<X, Y, Z>>,
        sil::tensor::Covariant<sil::tensor::MetricIndex2<X, Y, Z>>>;

using PositionIndex = sil::tensor::Contravariant<sil::tensor::TensorNaturalIndex<X, Y, Z>>;

using HodgeStarDomain = sil::exterior::
        hodge_star_domain_t<ddc::detail::TypeSeq<MuUp, NuUp>, ddc::detail::TypeSeq<RhoLow>>;
using HodgeStarDomain2 = sil::exterior::
        hodge_star_domain_t<ddc::detail::TypeSeq<RhoUp>, ddc::detail::TypeSeq<MuLow, NuLow>>;

TEST(HodgeStar, Euclidean3D)
{
    ddc::Coordinate<X, Y, Z> lower_bounds(0., 0., 0.);
    ddc::Coordinate<X, Y, Z> upper_bounds(2., 2., 2.);
    ddc::DiscreteVector<DDimX, DDimY, DDimZ> nb_cells(2, 2, 2);
    ddc::DiscreteDomain<DDimX> mesh_x = ddc::init_discrete_space<DDimX>(DDimX::init<DDimX>(
            ddc::Coordinate<X>(lower_bounds),
            ddc::Coordinate<X>(upper_bounds),
            ddc::DiscreteVector<DDimX>(nb_cells)));
    ddc::DiscreteDomain<DDimY> mesh_y = ddc::init_discrete_space<DDimY>(DDimY::init<DDimY>(
            ddc::Coordinate<Y>(lower_bounds),
            ddc::Coordinate<Y>(upper_bounds),
            ddc::DiscreteVector<DDimY>(nb_cells)));
    ddc::DiscreteDomain<DDimZ> mesh_z = ddc::init_discrete_space<DDimZ>(DDimZ::init<DDimZ>(
            ddc::Coordinate<Z>(lower_bounds),
            ddc::Coordinate<Z>(upper_bounds),
            ddc::DiscreteVector<DDimZ>(nb_cells)));
    ddc::DiscreteDomain<DDimX, DDimY, DDimZ> mesh_xyz(mesh_x, mesh_y, mesh_z);

    [[maybe_unused]] sil::tensor::TensorAccessor<PositionIndex> position_accessor;
    ddc::DiscreteDomain<DDimX, DDimY, DDimZ, PositionIndex>
            position_dom(mesh_xyz, position_accessor.domain());
    ddc::Chunk position_alloc(position_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor position(position_alloc);

    ddc::host_for_each(mesh_xyz, [&](ddc::DiscreteElement<DDimX, DDimY, DDimZ> elem) {
        position(elem, position.accessor().access_element<X>())
                = static_cast<double>(ddc::coordinate(ddc::DiscreteElement<DDimX>(elem)));
        position(elem, position.accessor().access_element<Y>())
                = static_cast<double>(ddc::coordinate(ddc::DiscreteElement<DDimY>(elem)));
        position(elem, position.accessor().access_element<Z>())
                = static_cast<double>(ddc::coordinate(ddc::DiscreteElement<DDimZ>(elem)));
    });

    [[maybe_unused]] sil::tensor::TensorAccessor<MetricIndex> metric_accessor;
    ddc::DiscreteDomain<DDimX, DDimY, DDimZ, MetricIndex>
            metric_dom(mesh_xyz, metric_accessor.domain());
    ddc::Chunk metric_alloc(metric_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor metric(metric_alloc);

    ddc::host_for_each(mesh_xyz, [&](ddc::DiscreteElement<DDimX, DDimY, DDimZ> elem) {
        metric(elem, metric.accessor().access_element<X, X>()) = 1.;
        metric(elem, metric.accessor().access_element<X, Y>()) = 0.;
        metric(elem, metric.accessor().access_element<X, Z>()) = 0.;
        metric(elem, metric.accessor().access_element<Y, Y>()) = 1.;
        metric(elem, metric.accessor().access_element<Y, Z>()) = 0.;
        metric(elem, metric.accessor().access_element<Z, Z>()) = 1.;
    });

    [[maybe_unused]] sil::tensor::tensor_accessor_for_domain_t<HodgeStarDomain> hodge_star_accessor;
    ddc::cartesian_prod_t<ddc::DiscreteDomain<DDimX, DDimY, DDimZ>, HodgeStarDomain>
            hodge_star_dom(mesh_xyz, hodge_star_accessor.domain());
    ddc::Chunk hodge_star_alloc(hodge_star_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor hodge_star(hodge_star_alloc);

    sil::exterior::fill_hodge_star<
            ddc::detail::TypeSeq<MuUp, NuUp>,
            ddc::detail::TypeSeq<
                    RhoLow>>(Kokkos::DefaultHostExecutionSpace(), hodge_star, metric, position);

    [[maybe_unused]] sil::tensor::TensorAccessor<
            sil::tensor::TensorAntisymmetricIndex<MuLow, NuLow>> form_accessor;
    ddc::DiscreteDomain<DDimX, DDimY, DDimZ, sil::tensor::TensorAntisymmetricIndex<MuLow, NuLow>>
            form_dom(mesh_xyz, form_accessor.domain());
    ddc::Chunk form_alloc(form_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor form(form_alloc);

    ddc::parallel_fill(form, 0.);
    ddc::host_for_each(mesh_xyz, [&](ddc::DiscreteElement<DDimX, DDimY, DDimZ> elem) {
        form(elem, form.accessor().access_element<X, Y>()) = 3.;
    });

    [[maybe_unused]] sil::tensor::TensorAccessor<RhoLow> dual_form_accessor;
    ddc::DiscreteDomain<DDimX, DDimY, DDimZ, RhoLow>
            dual_form_dom(mesh_xyz, dual_form_accessor.domain());
    ddc::Chunk dual_form_alloc(dual_form_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor dual_form(dual_form_alloc);

    ddc::host_for_each(mesh_xyz, [&](ddc::DiscreteElement<DDimX, DDimY, DDimZ> elem) {
        sil::tensor::tensor_prod(dual_form[elem], hodge_star[elem], form[elem]);
        EXPECT_DOUBLE_EQ(dual_form(elem, dual_form.accessor().access_element<Z>()), 1.5);
    });

    [[maybe_unused]] sil::tensor::tensor_accessor_for_domain_t<HodgeStarDomain2>
            hodge_star_accessor2;
    ddc::cartesian_prod_t<ddc::DiscreteDomain<DDimX, DDimY, DDimZ>, HodgeStarDomain2>
            hodge_star_dom2(mesh_xyz, hodge_star_accessor2.domain());
    ddc::Chunk hodge_star_alloc2(hodge_star_dom2, ddc::HostAllocator<double>());
    sil::tensor::Tensor hodge_star2(hodge_star_alloc2);

    sil::exterior::fill_hodge_star<
            ddc::detail::TypeSeq<RhoUp>,
            ddc::detail::TypeSeq<
                    MuLow,
                    NuLow>>(Kokkos::DefaultHostExecutionSpace(), hodge_star2, metric, position);

    ddc::parallel_fill(form, 0.);
    ddc::host_for_each(mesh_xyz, [&](ddc::DiscreteElement<DDimX, DDimY, DDimZ> elem) {
        sil::tensor::tensor_prod(form[elem], hodge_star2[elem], dual_form[elem]);
        EXPECT_DOUBLE_EQ(form(elem, form.accessor().access_element<X, Y>()), 3.);
    });

    ddc::detail::g_discrete_space_dual<DDimX>.reset();
    ddc::detail::g_discrete_space_dual<DDimY>.reset();
    ddc::detail::g_discrete_space_dual<DDimZ>.reset();
}
