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

TEST(Metric, MetricLike)
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

using MetricIndex = sil::tensor::TensorSymmetricIndex<
        sil::tensor::TensorCovariantNaturalIndex<sil::tensor::MetricIndex1<X, Y>>,
        sil::tensor::TensorCovariantNaturalIndex<sil::tensor::MetricIndex2<X, Y>>>;

TEST(Metric, Covariant)
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

TEST(Metric, ChristoffelLike)
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
            = sil::tensor::inplace_apply_metric<MetricIndex, KLow>(christoffel_1st, metric);
    ddc::for_each(mesh_xy, [&](ddc::DiscreteElement<DDimX, DDimY> elem) {
        EXPECT_EQ(
                christoffel_2nd.get(elem, christoffel_2nd.accessor().access_element<X, X, X>()),
                1.);
        EXPECT_EQ(
                christoffel_2nd.get(elem, christoffel_2nd.accessor().access_element<X, X, Y>()),
                2.);
        EXPECT_EQ(
                christoffel_2nd.get(elem, christoffel_2nd.accessor().access_element<X, Y, X>()),
                2.);
        EXPECT_EQ(
                christoffel_2nd.get(elem, christoffel_2nd.accessor().access_element<X, Y, Y>()),
                3.);
        EXPECT_EQ(
                christoffel_2nd.get(elem, christoffel_2nd.accessor().access_element<Y, X, X>()),
                8.);
        EXPECT_EQ(
                christoffel_2nd.get(elem, christoffel_2nd.accessor().access_element<Y, X, Y>()),
                10.);
        EXPECT_EQ(
                christoffel_2nd.get(elem, christoffel_2nd.accessor().access_element<Y, Y, X>()),
                10.);
        EXPECT_EQ(
                christoffel_2nd.get(elem, christoffel_2nd.accessor().access_element<Y, Y, Y>()),
                12.);
    });
}

// TODO test for metric_prod

using InvMetricIndex = sil::tensor::TensorSymmetricIndex<
        sil::tensor::TensorContravariantNaturalIndex<sil::tensor::MetricIndex1<X, Y>>,
        sil::tensor::TensorContravariantNaturalIndex<sil::tensor::MetricIndex2<X, Y>>>;

using IdIndex = sil::tensor::TensorSymmetricIndex<
        sil::tensor::TensorCovariantNaturalIndex<sil::tensor::MetricIndex1<X, Y>>,
        sil::tensor::TensorContravariantNaturalIndex<sil::tensor::MetricIndex2<X, Y>>>;

TEST(Metric, Inverse)
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
        metric(elem, metric.accessor().access_element<X, Y>()) = 2.;
        metric(elem, metric.accessor().access_element<Y, Y>()) = 3.;
    });

    sil::tensor::TensorAccessor<InvMetricIndex> inv_metric_accessor;
    ddc::DiscreteDomain<DDimX, DDimY, InvMetricIndex>
            inv_metric_dom(mesh_xy, inv_metric_accessor.mem_domain());
    ddc::Chunk inv_metric_alloc(inv_metric_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor<
            double,
            ddc::DiscreteDomain<DDimX, DDimY, InvMetricIndex>,
            Kokkos::layout_right,
            Kokkos::DefaultHostExecutionSpace::memory_space>
            inv_metric(inv_metric_alloc);

    sil::tensor::fill_inverse_metric<MetricIndex>(inv_metric, metric);

    sil::tensor::TensorAccessor<IdIndex> identity_accessor;
    ddc::DiscreteDomain<DDimX, DDimY, IdIndex>
            identity_dom(mesh_xy, identity_accessor.mem_domain());
    ddc::Chunk identity_alloc(identity_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor<
            double,
            ddc::DiscreteDomain<DDimX, DDimY, IdIndex>,
            Kokkos::layout_right,
            Kokkos::DefaultHostExecutionSpace::memory_space>
            identity(identity_alloc);

    ddc::parallel_for_each(Kokkos::DefaultHostExecutionSpace(), mesh_xy, [&](ddc::DiscreteElement<DDimX, DDimY> elem) {
        sil::tensor::tensor_prod(
                identity[elem],
                sil::tensor::relabelize_index_of<sil::tensor::MetricIndex2<X, Y>, I>(inv_metric[elem]), sil::tensor::relabelize_index_of<sil::tensor::MetricIndex1<X, Y>, I>(metric[elem]));
    });
    std::cout << metric;
    std::cout << inv_metric;
    std::cout << identity;
    /*
    ddc::for_each(mesh_xy, [&](ddc::DiscreteElement<DDimX, DDimY> elem) {
        EXPECT_EQ(
                identity.get(elem, identity.accessor().access_element<X, X>()),
                1.);
    });
    */
}
