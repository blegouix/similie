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
// using MetricIndex = sil::tensor::TensorSymmetricIndex<
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

// Declare indices
using MuLow = sil::tensor::TensorCovariantNaturalIndex<Mu>;
using MuUp = sil::tensor::TensorContravariantNaturalIndex<Mu>;
using NuLow = sil::tensor::TensorCovariantNaturalIndex<Nu>;
using NuUp = sil::tensor::TensorContravariantNaturalIndex<Nu>;
using RhoLow = sil::tensor::TensorCovariantNaturalIndex<Rho>;
using RhoUp = sil::tensor::TensorContravariantNaturalIndex<Rho>;

using HodgeStarDomain = sil::exterior::
        hodge_star_domain_t<ddc::detail::TypeSeq<MuUp>, ddc::detail::TypeSeq<NuLow>>;
using HodgeStarDomain2 = sil::exterior::
        hodge_star_domain_t<ddc::detail::TypeSeq<RhoUp, NuUp>, ddc::detail::TypeSeq<>>;

using DummyIndex = sil::tensor::TensorCovariantNaturalIndex<sil::tensor::TensorNaturalIndex<>>;

int main(int argc, char** argv)
{
    Kokkos::ScopeGuard const kokkos_scope(argc, argv);
    ddc::ScopeGuard const ddc_scope(argc, argv);

    printf("start example\n");

    MesherXY mesher;
    ddc::Coordinate<X, Y> lower_bounds(-5., -5.);
    ddc::Coordinate<X, Y> upper_bounds(5., 5.);
    ddc::DiscreteVector<DDimX, DDimY> nb_cells(40, 40);
    ddc::DiscreteDomain<DDimX, DDimY> mesh_xy = mesher.template mesh<
            ddc::detail::TypeSeq<DDimX, DDimY>,
            ddc::detail::TypeSeq<BSplinesX, BSplinesY>>(lower_bounds, upper_bounds, nb_cells);

    // Allocate and instantiate a metric tensor field.
    [[maybe_unused]] sil::tensor::TensorAccessor<MetricIndex> metric_accessor;
    ddc::DiscreteDomain<DDimX, DDimY, MetricIndex>
            metric_dom(mesh_xy, metric_accessor.mem_domain());
    ddc::Chunk metric_alloc(metric_dom, ddc::DeviceAllocator<double>());
    sil::tensor::Tensor<
            double,
            ddc::DiscreteDomain<DDimX, DDimY, MetricIndex>,
            Kokkos::layout_right,
            Kokkos::DefaultExecutionSpace::memory_space>
            metric(metric_alloc);
    ddc::parallel_for_each(
            Kokkos::DefaultExecutionSpace(),
            mesh_xy,
            KOKKOS_LAMBDA(ddc::DiscreteElement<DDimX, DDimY> elem) {
                metric(elem, metric.accessor().access_element<X, X>()) = 1.;
                metric(elem, metric.accessor().access_element<X, Y>()) = 0.;
                metric(elem, metric.accessor().access_element<Y, Y>()) = 1.;
            });

    // Invert metric
    [[maybe_unused]] sil::tensor::TensorAccessor<sil::tensor::upper<MetricIndex>>
            inv_metric_accessor;
    ddc::DiscreteDomain<DDimX, DDimY, sil::tensor::upper<MetricIndex>>
            inv_metric_dom(mesh_xy, inv_metric_accessor.mem_domain());
    ddc::Chunk inv_metric_alloc(inv_metric_dom, ddc::DeviceAllocator<double>());
    sil::tensor::Tensor<
            double,
            ddc::DiscreteDomain<DDimX, DDimY, sil::tensor::upper<MetricIndex>>,
            Kokkos::layout_right,
            Kokkos::DefaultExecutionSpace::memory_space>
            inv_metric(inv_metric_alloc);
    sil::tensor::fill_inverse_metric<
            MetricIndex>(Kokkos::DefaultExecutionSpace(), inv_metric, metric);

    ddc::parallel_for_each(
            Kokkos::DefaultExecutionSpace(),
            mesh_xy,
            KOKKOS_LAMBDA(ddc::DiscreteElement<DDimX, DDimY> elem) {
                Kokkos::
                        printf("%f ",
                               inv_metric(elem, inv_metric.accessor().access_element<X, X>()));
                Kokkos::
                        printf("%f ",
                               inv_metric(elem, inv_metric.accessor().access_element<X, Y>()));
                Kokkos::
                        printf("%f \n",
                               inv_metric(elem, inv_metric.accessor().access_element<Y, Y>()));
            });
    /*
    // Potential
    [[maybe_unused]] sil::tensor::TensorAccessor<DummyIndex> potential_accessor;
    ddc::DiscreteDomain<DDimX, DDimY, DummyIndex>
            potential_dom(metric.non_indices_domain(), potential_accessor.mem_domain());
    ddc::Chunk potential_alloc(potential_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor<
            double,
            ddc::DiscreteDomain<DDimX, DDimY, DummyIndex>,
            Kokkos::layout_right,
            Kokkos::DefaultHostExecutionSpace::memory_space>
            potential(potential_alloc);

    ddc::for_each(potential.domain(), [&](auto elem) {
        double const r = Kokkos::sqrt(
                static_cast<double>(
                        ddc::coordinate(ddc::DiscreteElement<DDimX>(elem))
                        * ddc::coordinate(ddc::DiscreteElement<DDimX>(elem)))
                + static_cast<double>(
                        ddc::coordinate(ddc::DiscreteElement<DDimY>(elem))
                        * ddc::coordinate(ddc::DiscreteElement<DDimY>(elem))));
        if (r <= 1) {
            potential.mem(elem) = -Kokkos::numbers::pi_v<double> / 2 * r * r;
        } else {
            potential.mem(elem) = -Kokkos::numbers::pi_v<double> * Kokkos::log(r);
        }
    });

    std::cout << "Potential:" << std::endl;
    std::cout << potential[potential_accessor.mem_domain().front()] << std::endl;

    // Gradient
    [[maybe_unused]] sil::tensor::TensorAccessor<MuLow> gradient_accessor;
    ddc::DiscreteDomain<DDimX, DDimY, MuLow> gradient_dom(mesh_xy, gradient_accessor.mem_domain());
    ddc::Chunk gradient_alloc(gradient_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor<
            double,
            ddc::DiscreteDomain<DDimX, DDimY, MuLow>,
            Kokkos::layout_right,
            Kokkos::DefaultHostExecutionSpace::memory_space>
            gradient(gradient_alloc);
    sil::exterior::deriv<MuLow, DummyIndex>(gradient, potential);

    // Hodge star
    [[maybe_unused]] sil::tensor::tensor_accessor_for_domain_t<HodgeStarDomain> hodge_star_accessor;
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
            sil::tensor::upper<MetricIndex>,
            ddc::detail::TypeSeq<MuUp>,
            ddc::detail::TypeSeq<NuLow>>(hodge_star, inv_metric);

    // Dual gradient
    [[maybe_unused]] sil::tensor::TensorAccessor<NuLow> dual_gradient_accessor;
    ddc::DiscreteDomain<DDimX, DDimY, NuLow>
            dual_gradient_dom(mesh_xy, dual_gradient_accessor.mem_domain());
    ddc::Chunk dual_gradient_alloc(dual_gradient_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor<
            double,
            ddc::DiscreteDomain<DDimX, DDimY, NuLow>,
            Kokkos::layout_right,
            Kokkos::DefaultHostExecutionSpace::memory_space>
            dual_gradient(dual_gradient_alloc);

    ddc::for_each(dual_gradient.non_indices_domain(), [&](ddc::DiscreteElement<DDimX, DDimY> elem) {
        sil::tensor::tensor_prod(dual_gradient[elem], gradient[elem], hodge_star[elem]);
    });

    // Dual Laplacian
    [[maybe_unused]] sil::tensor::TensorAccessor<
            sil::tensor::TensorAntisymmetricIndex<RhoLow, NuLow>> dual_laplacian_accessor;
    ddc::DiscreteDomain<DDimX, DDimY, sil::tensor::TensorAntisymmetricIndex<RhoLow, NuLow>>
            dual_laplacian_dom(
                    mesh_xy.remove_last(ddc::DiscreteVector<DDimX, DDimY> {1, 1}),
                    dual_laplacian_accessor.mem_domain());
    ddc::Chunk dual_laplacian_alloc(dual_laplacian_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor<
            double,
            ddc::DiscreteDomain<DDimX, DDimY, sil::tensor::TensorAntisymmetricIndex<RhoLow, NuLow>>,
            Kokkos::layout_right,
            Kokkos::DefaultHostExecutionSpace::memory_space>
            dual_laplacian(dual_laplacian_alloc);
    sil::exterior::deriv<RhoLow, NuLow>(dual_laplacian, dual_gradient);

    // Hodge star 2
    [[maybe_unused]] sil::tensor::tensor_accessor_for_domain_t<HodgeStarDomain2>
            hodge_star_accessor2;
    ddc::cartesian_prod_t<ddc::DiscreteDomain<DDimX, DDimY>, HodgeStarDomain2>
            hodge_star_dom2(metric.non_indices_domain(), hodge_star_accessor2.mem_domain());
    ddc::Chunk hodge_star_alloc2(hodge_star_dom2, ddc::HostAllocator<double>());
    sil::tensor::Tensor<
            double,
            ddc::cartesian_prod_t<ddc::DiscreteDomain<DDimX, DDimY>, HodgeStarDomain2>,
            Kokkos::layout_right,
            Kokkos::DefaultHostExecutionSpace::memory_space>
            hodge_star2(hodge_star_alloc2);

    sil::exterior::fill_hodge_star<
            sil::tensor::upper<MetricIndex>,
            ddc::detail::TypeSeq<RhoUp, NuUp>,
            ddc::detail::TypeSeq<>>(hodge_star2, inv_metric);

    // Laplacian
    [[maybe_unused]] sil::tensor::TensorAccessor<DummyIndex> laplacian_accessor;
    ddc::DiscreteDomain<DDimX, DDimY, DummyIndex> laplacian_dom(
            mesh_xy.remove_last(ddc::DiscreteVector<DDimX, DDimY> {1, 1}),
            laplacian_accessor.mem_domain());
    ddc::Chunk laplacian_alloc(laplacian_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor<
            double,
            ddc::DiscreteDomain<DDimX, DDimY, DummyIndex>,
            Kokkos::layout_right,
            Kokkos::DefaultHostExecutionSpace::memory_space>
            laplacian(laplacian_alloc);

    ddc::parallel_fill(laplacian, 0.);
    ddc::for_each(laplacian.non_indices_domain(), [&](ddc::DiscreteElement<DDimX, DDimY> elem) {
        sil::tensor::tensor_prod(laplacian[elem], dual_laplacian[elem], hodge_star2[elem]);
    });

    std::cout << "Laplacian:" << std::endl;
    std::cout << laplacian[laplacian_accessor.mem_domain().front()] << std::endl;
    */
}
