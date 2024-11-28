// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: GPL-3.0-or-later

#include <ddc/ddc.hpp>
#include <ddc/kernels/splines.hpp>

#include <similie/similie.hpp>

static constexpr std::size_t s_degree = 3;

// Labelize the dimensions of space
struct X
{
    static constexpr bool PERIODIC = false;
};

struct Y
{
    static constexpr bool PERIODIC = false;
};

// Declare a metric
using MetricIndex = sil::tensor::TensorSymmetricIndex<
        sil::tensor::TensorCovariantNaturalIndex<sil::tensor::MetricIndex1<X, Y>>,
        sil::tensor::TensorCovariantNaturalIndex<sil::tensor::MetricIndex2<X, Y>>>;

using MesherXY = sil::mesher::Mesher<s_degree, X, Y>;

struct BSplinesX : MesherXY::template bsplines_type<X>
{
};

struct DDimX : MesherXY::template discrete_dimension_type<X>
{
};

struct BSplinesY : MesherXY::template bsplines_type<Y>
{
};

struct DDimY : MesherXY::template discrete_dimension_type<Y>
{
};

// Declare natural indices taking values in {X, Y}
struct Mu : sil::tensor::TensorNaturalIndex<X, Y>
{
};

struct Nu : sil::tensor::TensorNaturalIndex<X, Y>
{
};

struct Rho : sil::tensor::TensorNaturalIndex<X, Y>
{
};

// Declare upper (contravariant) indices
using MuLow = sil::tensor::TensorCovariantNaturalIndex<Mu>;
using MuUp = sil::tensor::TensorContravariantNaturalIndex<Mu>;
using NuLow = sil::tensor::TensorCovariantNaturalIndex<Nu>;
using NuUp = sil::tensor::TensorContravariantNaturalIndex<Nu>;
using RhoLow = sil::tensor::TensorCovariantNaturalIndex<Rho>;
using RhoUp = sil::tensor::TensorContravariantNaturalIndex<Rho>;

using HodgeStarDomain = sil::exterior::
        hodge_star_domain_t<ddc::detail::TypeSeq<MuUp, NuUp>, ddc::detail::TypeSeq<RhoLow>>;

int main(int argc, char** argv)
{
    Kokkos::ScopeGuard const kokkos_scope(argc, argv);
    ddc::ScopeGuard const ddc_scope(argc, argv);

    printf("start example\n");

    MesherXY mesher;
    ddc::Coordinate<X, Y> lower_bounds(0., 0.);
    ddc::Coordinate<X, Y> upper_bounds(1., 1.);
    ddc::DiscreteVector<DDimX, DDimY> nb_cells(10, 10);
    ddc::DiscreteDomain<DDimX, DDimY> mesh_xy = mesher.template mesh<
            ddc::detail::TypeSeq<DDimX, DDimY>,
            ddc::detail::TypeSeq<BSplinesX, BSplinesY>>(lower_bounds, upper_bounds, nb_cells);

    // Allocate and instantiate a metric tensor field.
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
        metric(elem, metric.accessor().access_element<Y, Y>()) = 1.;
    });

    // Invert metric
    sil::tensor::TensorAccessor<sil::tensor::upper<MetricIndex>> inv_metric_accessor;
    ddc::DiscreteDomain<DDimX, DDimY, sil::tensor::upper<MetricIndex>>
            inv_metric_dom(mesh_xy, inv_metric_accessor.mem_domain());
    ddc::Chunk inv_metric_alloc(inv_metric_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor<
            double,
            ddc::DiscreteDomain<DDimX, DDimY, sil::tensor::upper<MetricIndex>>,
            Kokkos::layout_right,
            Kokkos::DefaultHostExecutionSpace::memory_space>
            inv_metric(inv_metric_alloc);
    sil::tensor::fill_inverse_metric<MetricIndex>(inv_metric, metric);
    // auto gmunu = sil::tensor::relabelize_metric<MuUp, NuUp>(inv_metric);

    // Hodge star
    sil::tensor::tensor_accessor_for_domain_t<HodgeStarDomain> hodge_star_accessor;
    ddc::cartesian_prod_t<ddc::DiscreteDomain<DDimX, DDimY>, HodgeStarDomain>
            hodge_star_dom(metric.non_indices_domain(), hodge_star_accessor.mem_domain());
    ddc::Chunk hodge_star_alloc(hodge_star_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor<
            double,
            ddc::cartesian_prod_t<ddc::DiscreteDomain<DDimX, DDimY>, HodgeStarDomain>,
            Kokkos::layout_right,
            Kokkos::DefaultHostExecutionSpace::memory_space>
            hodge_star(hodge_star_alloc);

    sil::exterior::fill_hodge_star<
            MetricIndex,
            ddc::detail::TypeSeq<MuLow, NuLow>,
            ddc::detail::TypeSeq<RhoUp>>(hodge_star, inv_metric);
    std::cout << hodge_star;
}
