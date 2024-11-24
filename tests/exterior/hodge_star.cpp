// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: GPL-3.0-or-later

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

using MuLow = sil::tensor::TensorContravariantNaturalIndex<Mu>;
using NuLow = sil::tensor::TensorContravariantNaturalIndex<Nu>;
using RhoUp = sil::tensor::TensorCovariantNaturalIndex<Rho>;

using MetricIndex = sil::tensor::TensorSymmetricIndex<
        sil::tensor::TensorCovariantNaturalIndex<sil::tensor::MetricIndex1<X, Y, Z>>,
        sil::tensor::TensorCovariantNaturalIndex<sil::tensor::MetricIndex2<X, Y, Z>>>;

using LeviCivitaIndex = sil::tensor::TensorLeviCivitaIndex<Mu, Nu, Rho>;

using HodgeStarIndex = sil::exterior::
        hodge_star_index_t<ddc::detail::TypeSeq<MuLow, NuLow>, ddc::detail::TypeSeq<RhoUp>>;

TEST(HodgeStar, Test)
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
        metric(elem, metric.accessor().access_element<X, X>()) = 4.;
        metric(elem, metric.accessor().access_element<X, Y>()) = 1.;
        metric(elem, metric.accessor().access_element<X, Z>()) = 2.;
        metric(elem, metric.accessor().access_element<Y, Y>()) = 5.;
        metric(elem, metric.accessor().access_element<Y, Z>()) = 3.;
        metric(elem, metric.accessor().access_element<Z, Z>()) = 6.;
    });

    sil::tensor::TensorAccessor<HodgeStarIndex> hodge_star_accessor;
    ddc::DiscreteDomain<DDimX, DDimY, HodgeStarIndex>
            hodge_star_dom(metric.non_indices_domain(), hodge_star_accessor.mem_domain());
    ddc::Chunk hodge_star_alloc(hodge_star_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor<
            double,
            ddc::DiscreteDomain<DDimX, DDimY, HodgeStarIndex>,
            Kokkos::layout_right,
            Kokkos::DefaultHostExecutionSpace::memory_space>
            hodge_star(hodge_star_alloc);

    sil::exterior::fill_hodge_star<
            MetricIndex,
            ddc::detail::TypeSeq<Mu, Nu>,
            ddc::detail::TypeSeq<Rho>>(hodge_star, metric);
    std::cout << hodge_star;
}
