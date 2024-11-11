// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: GPL-3.0-or-later

#include <ddc/ddc.hpp>

#include <gtest/gtest.h>

#include "tensor.hpp"

struct X
{
};

struct Y
{
};

struct DDimX : ddc::UniformPointSampling<X>
{
};

struct DDimY : ddc::UniformPointSampling<Y>
{
};

struct I : sil::tensor::TensorNaturalIndex<X, Y>
{
};

struct J : sil::tensor::TensorNaturalIndex<X, Y>
{
};

using MetricLikeIndex = sil::tensor::TensorSymmetricIndex<I, J>;

TEST(TensorField, MetricLike)
{
    ddc::DiscreteDomain<DDimX, DDimY>
            mesh_xy(ddc::DiscreteElement<DDimX, DDimY>(0, 0),
                    ddc::DiscreteVector<DDimX, DDimY>(10, 10));

    sil::tensor::TensorAccessor<MetricLikeIndex> metric_accessor;
    ddc::DiscreteDomain<DDimX, DDimY, MetricLikeIndex>
            metric_dom(mesh_xy, metric_accessor.mem_domain());

    ddc::Chunk metric_alloc(metric_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor<
            double,
            ddc::DiscreteDomain<DDimX, DDimY, MetricLikeIndex>,
            Kokkos::layout_right,
            Kokkos::DefaultHostExecutionSpace::memory_space>
            metric(metric_alloc);
    ddc::for_each(mesh_xy, [&](ddc::DiscreteElement<DDimX, DDimY> elem) {
        metric(elem, metric.accessor().access_element<X, X>()) = 1.;
        metric(elem, metric.accessor().access_element<X, Y>()) = 2.;
        metric(elem, metric.accessor().access_element<Y, Y>()) = 3.;
    });
    ddc::for_each(mesh_xy, [&](ddc::DiscreteElement<DDimX, DDimY> elem) {
        EXPECT_EQ(metric.get(elem, metric.accessor().access_element<X, X>()), 1.);
        EXPECT_EQ(metric.get(elem, metric.accessor().access_element<X, Y>()), 2.);
        EXPECT_EQ(metric.get(elem, metric.accessor().access_element<Y, Y>()), 3.);
    });
}

using ILow = sil::tensor::TensorCovariantNaturalIndex<I>;
using JLow = sil::tensor::TensorCovariantNaturalIndex<J>;

using MetricIndex = sil::tensor::
        TensorSymmetricIndex<sil::tensor::MetricIndex1<X, Y>, sil::tensor::MetricIndex2<X, Y>>;

TEST(TensorField, Metric)
{
    ddc::DiscreteDomain<DDimX, DDimY>
            mesh_xy(ddc::DiscreteElement<DDimX, DDimY>(0, 0),
                    ddc::DiscreteVector<DDimX, DDimY>(10, 10));

    sil::tensor::TensorAccessor<MetricIndex> metric_accessor;
    ddc::DiscreteDomain<DDimX, DDimY, MetricIndex>
            metric_dom(mesh_xy, metric_accessor.mem_domain());
    ddc::Chunk metric_alloc(metric_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor<
            double,
            ddc::DiscreteDomain<DDimX, DDimY, MetricIndex>,
            Kokkos::layout_right,
            Kokkos::DefaultHostExecutionSpace::memory_space>
            metric(metric_alloc);
    auto g_i_j = sil::tensor::relabelize_metric<ILow, JLow>(metric);
    ddc::for_each(mesh_xy, [&](ddc::DiscreteElement<DDimX, DDimY> elem) {
        g_i_j(elem, g_i_j.accessor().access_element<X, X>()) = 1.;
        g_i_j(elem, g_i_j.accessor().access_element<X, Y>()) = 2.;
        g_i_j(elem, g_i_j.accessor().access_element<Y, Y>()) = 3.;
    });
    ddc::for_each(mesh_xy, [&](ddc::DiscreteElement<DDimX, DDimY> elem) {
        EXPECT_EQ(g_i_j.get(elem, g_i_j.accessor().access_element<X, X>()), 1.);
        EXPECT_EQ(g_i_j.get(elem, g_i_j.accessor().access_element<X, Y>()), 2.);
        EXPECT_EQ(g_i_j.get(elem, g_i_j.accessor().access_element<Y, X>()), 2.);
        EXPECT_EQ(g_i_j.get(elem, g_i_j.accessor().access_element<Y, Y>()), 3.);
    });
}

struct K : sil::tensor::TensorNaturalIndex<X, Y>
{
};

using KLow = sil::tensor::TensorCovariantNaturalIndex<K>;
using KUp = sil::tensor::upper<KLow>;

TEST(TensorField, ChristoffelLike)
{
    ddc::DiscreteDomain<DDimX, DDimY>
            mesh_xy(ddc::DiscreteElement<DDimX, DDimY>(0, 0),
                    ddc::DiscreteVector<DDimX, DDimY>(10, 10));

    sil::tensor::TensorAccessor<MetricIndex> metric_accessor;
    ddc::DiscreteDomain<DDimX, DDimY, MetricIndex>
            metric_dom(mesh_xy, metric_accessor.mem_domain());
    ddc::Chunk metric_alloc(metric_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor<
            double,
            ddc::DiscreteDomain<DDimX, DDimY, MetricIndex>,
            Kokkos::layout_right,
            Kokkos::DefaultHostExecutionSpace::memory_space>
            metric(metric_alloc);
    ddc::for_each(mesh_xy, [&](ddc::DiscreteElement<DDimX, DDimY> elem) {
        metric(elem, metric.accessor().access_element<X, X>()) = 1.;
        metric(elem, metric.accessor().access_element<X, Y>()) = 0.;
        metric(elem, metric.accessor().access_element<Y, Y>()) = 2.;
    });

    sil::tensor::TensorAccessor<KUp, sil::tensor::TensorSymmetricIndex<ILow, JLow>>
            christoffel_1st_accessor;
    ddc::DiscreteDomain<DDimX, DDimY, KUp, sil::tensor::TensorSymmetricIndex<ILow, JLow>>
            christoffel_1st_dom(mesh_xy, christoffel_1st_accessor.mem_domain());
    ddc::Chunk christoffel_1st_alloc(christoffel_1st_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor<
            double,
            ddc::DiscreteDomain<DDimX, DDimY, KUp, sil::tensor::TensorSymmetricIndex<ILow, JLow>>,
            Kokkos::layout_right,
            Kokkos::DefaultHostExecutionSpace::memory_space>
            christoffel_1st(christoffel_1st_alloc);
    ddc::for_each(mesh_xy, [&](ddc::DiscreteElement<DDimX, DDimY> elem) {
        christoffel_1st(elem, christoffel_1st.accessor().access_element<X, X, X>()) = 1.;
        christoffel_1st(elem, christoffel_1st.accessor().access_element<X, X, Y>()) = 2.;
        christoffel_1st(elem, christoffel_1st.accessor().access_element<X, Y, Y>()) = 3.;
        christoffel_1st(elem, christoffel_1st.accessor().access_element<Y, X, X>()) = 4.;
        christoffel_1st(elem, christoffel_1st.accessor().access_element<Y, X, Y>()) = 5.;
        christoffel_1st(elem, christoffel_1st.accessor().access_element<Y, Y, Y>()) = 6.;
    });
    ddc::for_each(mesh_xy, [&](ddc::DiscreteElement<DDimX, DDimY> elem) {
        EXPECT_EQ(
                christoffel_1st.get(elem, christoffel_1st.accessor().access_element<X, X, X>()),
                1.);
        EXPECT_EQ(
                christoffel_1st.get(elem, christoffel_1st.accessor().access_element<X, X, Y>()),
                2.);
        EXPECT_EQ(
                christoffel_1st.get(elem, christoffel_1st.accessor().access_element<X, Y, X>()),
                2.);
        EXPECT_EQ(
                christoffel_1st.get(elem, christoffel_1st.accessor().access_element<X, Y, Y>()),
                3.);
        EXPECT_EQ(
                christoffel_1st.get(elem, christoffel_1st.accessor().access_element<Y, X, X>()),
                4.);
        EXPECT_EQ(
                christoffel_1st.get(elem, christoffel_1st.accessor().access_element<Y, X, Y>()),
                5.);
        EXPECT_EQ(
                christoffel_1st.get(elem, christoffel_1st.accessor().access_element<Y, Y, X>()),
                5.);
        EXPECT_EQ(
                christoffel_1st.get(elem, christoffel_1st.accessor().access_element<Y, Y, Y>()),
                6.);
    });
    auto christoffel_2nd
            = sil::tensor::inplace_apply_metric<MetricIndex, KLow, KUp>(christoffel_1st, metric);
    std::cout << christoffel_2nd;
}
