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

struct Mu : sil::tensor::TensorNaturalIndex<X, Y>
{
};

struct Nu : sil::tensor::TensorNaturalIndex<X, Y>
{
};

using MetricLikeIndex = sil::tensor::TensorSymmetricIndex<Mu, Nu>;

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
    std::cout << metric;
    ddc::for_each(mesh_xy, [&](ddc::DiscreteElement<DDimX, DDimY> elem) {
        EXPECT_EQ(metric.get(elem, metric.accessor().access_element<X, X>()), 0.);
        /*
        EXPECT_EQ(metric.get(elem, metric.accessor().access_element<X, Y>()), 2.);
        EXPECT_EQ(metric.get(elem, metric.accessor().access_element<Y, Y>()), 3.);
        */
    });
}
