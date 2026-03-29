// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: MIT

#include <cmath>

#include <ddc/ddc.hpp>

#include <gtest/gtest.h>
#include <similie/exterior/hodge_star.hpp>
#include <similie/mesher/dualizer.hpp>
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

using DummyIndex = sil::tensor::Covariant<sil::tensor::ScalarIndex>;

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

TEST(HodgeStar, TensorForm1In2D)
{
    using XDualizer = sil::mesher::HalfShiftDualizer<X>;
    using YDualizer = sil::mesher::HalfShiftDualizer<Y>;
    using DDimXDual = sil::mesher::dual_discrete_dimension_t<XDualizer, DDimX>;
    using DDimYDual = sil::mesher::dual_discrete_dimension_t<YDualizer, DDimY>;

    auto const x_dom = ddc::init_discrete_space<DDimX>(DDimX::init<DDimX>(
            ddc::Coordinate<X>(0.),
            ddc::Coordinate<X>(1.),
            ddc::DiscreteVector<DDimX>(4)));
    auto const y_dom = ddc::init_discrete_space<DDimY>(DDimY::init<DDimY>(
            ddc::Coordinate<Y>(0.),
            ddc::Coordinate<Y>(1.),
            ddc::DiscreteVector<DDimY>(4)));
    ddc::DiscreteDomain<DDimX, DDimY> const mesh(x_dom, y_dom);
    XDualizer const x_dualizer;
    YDualizer const y_dualizer;
    ddc::DiscreteDomain<DDimXDual, DDimY> const x_face_dom = x_dualizer(mesh);
    ddc::DiscreteDomain<DDimX, DDimYDual> const y_face_dom = y_dualizer(mesh);

    [[maybe_unused]] sil::tensor::TensorAccessor<DummyIndex> scalar_accessor;

    ddc::Chunk primal_x_alloc(
            ddc::DiscreteDomain<DDimXDual, DDimY, DummyIndex>(x_face_dom, scalar_accessor.domain()),
            ddc::HostAllocator<double>());
    sil::tensor::Tensor primal_x(primal_x_alloc);
    ddc::Chunk primal_y_alloc(
            ddc::DiscreteDomain<DDimX, DDimYDual, DummyIndex>(y_face_dom, scalar_accessor.domain()),
            ddc::HostAllocator<double>());
    sil::tensor::Tensor primal_y(primal_y_alloc);

    ddc::parallel_for_each(
            Kokkos::DefaultHostExecutionSpace(),
            primal_x.domain(),
            KOKKOS_LAMBDA(ddc::DiscreteElement<DDimXDual, DDimY, DummyIndex> elem) {
                primal_x(elem) = 2.;
            });
    ddc::parallel_for_each(
            Kokkos::DefaultHostExecutionSpace(),
            primal_y.domain(),
            KOKKOS_LAMBDA(ddc::DiscreteElement<DDimX, DDimYDual, DummyIndex> elem) {
                primal_y(elem) = 3.;
            });

    auto primal_form = sil::exterior::make_tensor_form(
            sil::exterior::component<X>(primal_x),
            sil::exterior::component<Y>(primal_y));

    ddc::Chunk dual_x_alloc(
            ddc::DiscreteDomain<DDimX, DDimYDual, DummyIndex>(y_face_dom, scalar_accessor.domain()),
            ddc::HostAllocator<double>());
    sil::tensor::Tensor dual_x(dual_x_alloc);
    ddc::Chunk dual_y_alloc(
            ddc::DiscreteDomain<DDimXDual, DDimY, DummyIndex>(x_face_dom, scalar_accessor.domain()),
            ddc::HostAllocator<double>());
    sil::tensor::Tensor dual_y(dual_y_alloc);
    auto dual_form = sil::exterior::make_tensor_form<sil::exterior::DualSupport>(
            sil::exterior::component<X>(dual_x),
            sil::exterior::component<Y>(dual_y));

    sil::exterior::hodge_star(Kokkos::DefaultHostExecutionSpace(), dual_form, primal_form, 0);

    EXPECT_DOUBLE_EQ(
            dual_x(y_face_dom.front(), ddc::DiscreteElement<DummyIndex>(0)),
            -3.);
    EXPECT_DOUBLE_EQ(
            dual_y(x_face_dom.front(), ddc::DiscreteElement<DummyIndex>(0)),
            2.);

    ddc::Chunk primal_back_x_alloc(
            ddc::DiscreteDomain<DDimXDual, DDimY, DummyIndex>(x_face_dom, scalar_accessor.domain()),
            ddc::HostAllocator<double>());
    sil::tensor::Tensor primal_back_x(primal_back_x_alloc);
    ddc::Chunk primal_back_y_alloc(
            ddc::DiscreteDomain<DDimX, DDimYDual, DummyIndex>(y_face_dom, scalar_accessor.domain()),
            ddc::HostAllocator<double>());
    sil::tensor::Tensor primal_back_y(primal_back_y_alloc);
    auto primal_back = sil::exterior::make_tensor_form(
            sil::exterior::component<X>(primal_back_x),
            sil::exterior::component<Y>(primal_back_y));

    sil::exterior::hodge_star(Kokkos::DefaultHostExecutionSpace(), primal_back, dual_form, 0);

    EXPECT_DOUBLE_EQ(
            primal_back_x(x_face_dom.front(), ddc::DiscreteElement<DummyIndex>(0)),
            -2.);
    EXPECT_DOUBLE_EQ(
            primal_back_y(y_face_dom.front(), ddc::DiscreteElement<DummyIndex>(0)),
            -3.);
}

TEST(HodgeStar, ScalarTensor0And2In2D)
{
    struct X2
    {
    };
    struct Y2
    {
    };
    struct DDimX2 : ddc::UniformPointSampling<X2>
    {
    };
    struct DDimY2 : ddc::UniformPointSampling<Y2>
    {
    };
    using ScalarIndex = sil::tensor::Covariant<sil::tensor::ScalarIndex>;
    using Metric2DIndex = sil::tensor::TensorDiagonalIndex<
            sil::tensor::Contravariant<sil::tensor::MetricIndex1<X2, Y2>>,
            sil::tensor::Contravariant<sil::tensor::MetricIndex2<X2, Y2>>>;
    using XDualizer = sil::mesher::HalfShiftDualizer<X2>;
    using YDualizer = sil::mesher::HalfShiftDualizer<Y2>;
    using DDimXDual = sil::mesher::dual_discrete_dimension_t<XDualizer, DDimX2>;
    using DDimYDual = sil::mesher::dual_discrete_dimension_t<YDualizer, DDimY2>;

    auto const x_dom = ddc::init_discrete_space<DDimX2>(DDimX2::init<DDimX2>(
            ddc::Coordinate<X2>(0.),
            ddc::Coordinate<X2>(1.),
            ddc::DiscreteVector<DDimX2>(4)));
    auto const y_dom = ddc::init_discrete_space<DDimY2>(DDimY2::init<DDimY2>(
            ddc::Coordinate<Y2>(0.),
            ddc::Coordinate<Y2>(1.),
            ddc::DiscreteVector<DDimY2>(4)));
    ddc::DiscreteDomain<DDimX2, DDimY2> const mesh(x_dom, y_dom);
    XDualizer const x_dualizer;
    YDualizer const y_dualizer;
    auto const dual_mesh = y_dualizer(x_dualizer(mesh));

    [[maybe_unused]] sil::tensor::TensorAccessor<ScalarIndex> scalar_accessor;
    ddc::Chunk scalar_alloc(
            ddc::DiscreteDomain<DDimX2, DDimY2, ScalarIndex>(mesh, scalar_accessor.domain()),
            ddc::HostAllocator<double>());
    sil::tensor::Tensor scalar(scalar_alloc);
    ddc::parallel_for_each(
            Kokkos::DefaultHostExecutionSpace(),
            scalar.domain(),
            KOKKOS_LAMBDA(ddc::DiscreteElement<DDimX2, DDimY2, ScalarIndex> elem) { scalar(elem) = 2.; });

    [[maybe_unused]] sil::tensor::TensorAccessor<Metric2DIndex> metric_accessor;
    ddc::Chunk metric_alloc(
            ddc::DiscreteDomain<DDimX2, DDimY2, Metric2DIndex>(mesh, metric_accessor.domain()),
            ddc::HostAllocator<double>());
    sil::tensor::Tensor metric(metric_alloc);
    ddc::parallel_for_each(
            Kokkos::DefaultHostExecutionSpace(),
            mesh,
            KOKKOS_LAMBDA(ddc::DiscreteElement<DDimX2, DDimY2> elem) {
                metric(elem, metric.accessor().access_element<X2, X2>()) = 4.;
                metric(elem, metric.accessor().access_element<Y2, Y2>()) = 9.;
            });

    ddc::Chunk dual_scalar_alloc(
            ddc::DiscreteDomain<DDimXDual, DDimYDual, ScalarIndex>(dual_mesh, scalar_accessor.domain()),
            ddc::HostAllocator<double>());
    sil::tensor::Tensor dual_scalar(dual_scalar_alloc);

    sil::exterior::hodge_star<Metric2DIndex>(
            Kokkos::DefaultHostExecutionSpace(),
            dual_scalar,
            scalar,
            metric);

    EXPECT_DOUBLE_EQ(dual_scalar(dual_scalar.domain().front()), 1. / 3.);

    ddc::DiscreteDomain<DDimX2, DDimY2> const primal_submesh(
            mesh.front(),
            ddc::DiscreteVector<DDimX2, DDimY2>(
                    dual_mesh.template extent<DDimXDual>().value(),
                    dual_mesh.template extent<DDimYDual>().value()));
    ddc::Chunk scalar_back_alloc(
            ddc::DiscreteDomain<DDimX2, DDimY2, ScalarIndex>(primal_submesh, scalar_accessor.domain()),
            ddc::HostAllocator<double>());
    sil::tensor::Tensor scalar_back(scalar_back_alloc);
    ddc::parallel_fill(scalar_back, 0.);
    sil::exterior::hodge_star<Metric2DIndex>(
            Kokkos::DefaultHostExecutionSpace(),
            scalar_back,
            dual_scalar,
            metric);

    EXPECT_DOUBLE_EQ(scalar_back(scalar_back.domain().front()), 2.);
}
