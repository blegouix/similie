// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: GPL-3.0-or-later

#include <cmath>

#include <ddc/ddc.hpp>

#include <gtest/gtest.h>
#include <similie/tensor/identity_tensor.hpp>

#include "exterior.hpp"

template <class... CDim>
using MetricIndex = sil::tensor::TensorIdentityIndex<
        sil::tensor::TensorContravariantNaturalIndex<sil::tensor::MetricIndex1<CDim...>>,
        sil::tensor::TensorContravariantNaturalIndex<sil::tensor::MetricIndex2<CDim...>>>;

// std::size_t N ?
template <class InterestIndex, class Index, class... DDim>
static auto test_derivative(auto potential)
{
    // Allocate and instantiate an inverse metric tensor field.
    [[maybe_unused]] sil::tensor::TensorAccessor<
            MetricIndex<typename DDim::continuous_dimension_type...>> inv_metric_accessor;
    ddc::DiscreteDomain<DDim..., MetricIndex<typename DDim::continuous_dimension_type...>>
            inv_metric_dom(potential.non_indices_domain(), inv_metric_accessor.mem_domain());
    ddc::Chunk inv_metric_alloc(inv_metric_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor inv_metric(inv_metric_alloc);

    // Allocate and compute Laplacian
    [[maybe_unused]] sil::tensor::TensorAccessor<Index> laplacian_accessor;
    ddc::DiscreteDomain<DDim..., Index> laplacian_dom(
            potential.non_indices_domain().remove_last(
                    ddc::DiscreteVector<DDim...>(ddc::DiscreteVector<DDim>(1)...)),
            laplacian_accessor.mem_domain());
    ddc::Chunk laplacian_alloc(laplacian_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor laplacian(laplacian_alloc);

    sil::exterior::laplacian<
            MetricIndex<typename DDim::continuous_dimension_type...>,
            InterestIndex,
            Index>(Kokkos::DefaultHostExecutionSpace(), laplacian, potential, inv_metric);
    Kokkos::fence();

    return std::make_pair(std::move(laplacian_alloc), laplacian);
}

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

struct Mu2 : sil::tensor::TensorNaturalIndex<X, Y>
{
};

TEST(Laplacian, 2D0Form)
{
    ddc::Coordinate<X, Y> lower_bounds(-5., -5.);
    ddc::Coordinate<X, Y> upper_bounds(5., 5.);
    ddc::DiscreteVector<DDimX, DDimY> nb_cells(50, 50);
    ddc::DiscreteDomain<DDimX> mesh_x = ddc::init_discrete_space<DDimX>(DDimX::init<DDimX>(
            ddc::Coordinate<X>(lower_bounds),
            ddc::Coordinate<X>(upper_bounds),
            ddc::DiscreteVector<DDimX>(nb_cells)));
    ddc::DiscreteDomain<DDimY> mesh_y = ddc::init_discrete_space<DDimY>(DDimY::init<DDimY>(
            ddc::Coordinate<Y>(lower_bounds),
            ddc::Coordinate<Y>(upper_bounds),
            ddc::DiscreteVector<DDimY>(nb_cells)));
    ddc::DiscreteDomain<DDimX, DDimY> mesh_xy(mesh_x, mesh_y);

    // Potential
    [[maybe_unused]] sil::tensor::TensorAccessor<
            sil::tensor::TensorCovariantNaturalIndex<sil::tensor::ScalarIndex>> potential_accessor;
    ddc::DiscreteDomain<
            DDimX,
            DDimY,
            sil::tensor::TensorCovariantNaturalIndex<sil::tensor::ScalarIndex>>
            potential_dom(mesh_xy, potential_accessor.mem_domain());
    ddc::Chunk potential_alloc(potential_dom, ddc::DeviceAllocator<double>());
    sil::tensor::Tensor potential(potential_alloc);

    double const R = 2.;
    double const L = ddc::coordinate(ddc::DiscreteElement<DDimX>(potential.domain().back()))
                     - ddc::coordinate(ddc::DiscreteElement<DDimX>(potential.domain().front()));
    double const alpha = (static_cast<double>(nb_cells.template get<DDimX>())
                          * static_cast<double>(nb_cells.template get<DDimY>()))
                         / L / 2 / L / 2;
    ddc::parallel_for_each(
            Kokkos::DefaultHostExecutionSpace(),
            potential.domain(),
            [&](ddc::DiscreteElement<
                    DDimX,
                    DDimY,
                    sil::tensor::TensorCovariantNaturalIndex<sil::tensor::ScalarIndex>> elem) {
                double const r = Kokkos::sqrt(
                        static_cast<double>(
                                ddc::coordinate(ddc::DiscreteElement<DDimX>(elem))
                                * ddc::coordinate(ddc::DiscreteElement<DDimX>(elem)))
                        + static_cast<double>(
                                ddc::coordinate(ddc::DiscreteElement<DDimY>(elem))
                                * ddc::coordinate(ddc::DiscreteElement<DDimY>(elem))));
                if (r <= R) {
                    potential.mem(elem) = -alpha * r * r;
                } else {
                    potential.mem(elem) = alpha * R * R * (2 * Kokkos::log(R / r) - 1);
                }
            });


    auto [alloc, laplacian] = test_derivative<
            sil::tensor::TensorCovariantNaturalIndex<Mu2>,
            sil::tensor::TensorCovariantNaturalIndex<sil::tensor::ScalarIndex>,
            DDimX,
            DDimY>(potential);

    ddc::parallel_for_each(
            Kokkos::DefaultHostExecutionSpace(),
            laplacian.template domain<DDimX>().remove_last(ddc::DiscreteVector<DDimX>(1)),
            [&](ddc::DiscreteElement<DDimX> elem) {
                double const value = laplacian(
                        elem,
                        ddc::DiscreteElement<DDimY> {
                                static_cast<std::size_t>(nb_cells.template get<DDimY>()) / 2},
                        ddc::DiscreteElement<sil::tensor::TensorCovariantNaturalIndex<
                                sil::tensor::ScalarIndex>> {0});
                if (ddc::coordinate(elem) < -1.2 * R || ddc::coordinate(elem) > 1.2 * R) {
                    EXPECT_NEAR(value, 0., .5);
                } else if (ddc::coordinate(elem) > -.8 * R && ddc::coordinate(elem) < .8 * R) {
                    EXPECT_NEAR(value, 1., .5);
                }
            });
    ddc::detail::g_discrete_space_dual<DDimX>.reset();
    ddc::detail::g_discrete_space_dual<DDimY>.reset();
}

TEST(Laplacian, 2D1Form)
{
    ddc::Coordinate<X, Y> lower_bounds(-5., -5.);
    ddc::Coordinate<X, Y> upper_bounds(5., 5.);
    ddc::DiscreteVector<DDimX, DDimY> nb_cells(50, 50);
    ddc::DiscreteDomain<DDimX> mesh_x = ddc::init_discrete_space<DDimX>(DDimX::init<DDimX>(
            ddc::Coordinate<X>(lower_bounds),
            ddc::Coordinate<X>(upper_bounds),
            ddc::DiscreteVector<DDimX>(nb_cells)));
    ddc::DiscreteDomain<DDimY> mesh_y = ddc::init_discrete_space<DDimY>(DDimY::init<DDimY>(
            ddc::Coordinate<Y>(lower_bounds),
            ddc::Coordinate<Y>(upper_bounds),
            ddc::DiscreteVector<DDimY>(nb_cells)));
    ddc::DiscreteDomain<DDimX, DDimY> mesh_xy(mesh_x, mesh_y);

    // Potential
    [[maybe_unused]] sil::tensor::TensorAccessor<sil::tensor::TensorCovariantNaturalIndex<Mu2>>
            potential_accessor;
    ddc::DiscreteDomain<DDimX, DDimY, sil::tensor::TensorCovariantNaturalIndex<Mu2>>
            potential_dom(mesh_xy, potential_accessor.mem_domain());
    ddc::Chunk potential_alloc(potential_dom, ddc::DeviceAllocator<double>());
    sil::tensor::Tensor potential(potential_alloc);

    double const R = 2.;
    double const L = ddc::coordinate(ddc::DiscreteElement<DDimX>(potential.domain().back()))
                     - ddc::coordinate(ddc::DiscreteElement<DDimX>(potential.domain().front()));
    double const alpha = (static_cast<double>(nb_cells.template get<DDimX>())
                          * static_cast<double>(nb_cells.template get<DDimY>()))
                         / L / 2 / L / 2;
    ddc::parallel_for_each(
            Kokkos::DefaultHostExecutionSpace(),
            potential.non_indices_domain(),
            [&](ddc::DiscreteElement<DDimX, DDimY> elem) {
                double const r = Kokkos::sqrt(
                        static_cast<double>(
                                ddc::coordinate(ddc::DiscreteElement<DDimX>(elem))
                                * ddc::coordinate(ddc::DiscreteElement<DDimX>(elem)))
                        + static_cast<double>(
                                ddc::coordinate(ddc::DiscreteElement<DDimY>(elem))
                                * ddc::coordinate(ddc::DiscreteElement<DDimY>(elem))));
                double const theta = Kokkos::
                        atan2(ddc::coordinate(ddc::DiscreteElement<DDimY>(elem)),
                              ddc::coordinate(ddc::DiscreteElement<DDimX>(elem)));
                if (r <= R) {
                    potential.mem(elem, potential_accessor.access_element<X>())
                            = alpha * r * r * Kokkos::sin(theta);
                    potential.mem(elem, potential_accessor.access_element<Y>())
                            = -alpha * r * r * Kokkos::cos(theta);
                } else {
                    potential.mem(elem, potential_accessor.access_element<X>())
                            = -alpha * R * R * (2 * Kokkos::log(R / r) - 1) * Kokkos::sin(theta);
                    potential.mem(elem, potential_accessor.access_element<Y>())
                            = alpha * R * R * (2 * Kokkos::log(R / r) - 1) * Kokkos::cos(theta);
                }
            });


    auto [alloc, laplacian] = test_derivative<
            sil::tensor::TensorCovariantNaturalIndex<Mu2>,
            sil::tensor::TensorCovariantNaturalIndex<Mu2>,
            DDimX,
            DDimY>(potential);

    ddc::parallel_for_each(
            Kokkos::DefaultHostExecutionSpace(),
            laplacian.template domain<DDimX>().remove_last(ddc::DiscreteVector<DDimX>(1)),
            [&](ddc::DiscreteElement<DDimX> elem) {
                double const value = laplacian(
                        elem,
                        ddc::DiscreteElement<DDimY> {
                                static_cast<std::size_t>(nb_cells.template get<DDimY>()) / 2},
                        laplacian.accessor().access_element<Y>());
                if (ddc::coordinate(elem) < -1.2 * R || ddc::coordinate(elem) > 1.2 * R) {
                    EXPECT_NEAR(value, 0., .5);
                } else if (ddc::coordinate(elem) > -.8 * R && ddc::coordinate(elem) < -.2 * R) {
                    EXPECT_NEAR(value, -1., .5);
                } else if (ddc::coordinate(elem) > .2 * R && ddc::coordinate(elem) < .8 * R) {
                    EXPECT_NEAR(value, 1., .5);
                }
            });
    ddc::detail::g_discrete_space_dual<DDimX>.reset();
    ddc::detail::g_discrete_space_dual<DDimY>.reset();
}
