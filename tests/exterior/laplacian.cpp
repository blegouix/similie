// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: MIT

#include <cmath>

#include <ddc/ddc.hpp>

#include <gtest/gtest.h>
#include <similie/tensor/identity_tensor.hpp>

#include "exterior.hpp"

template <class... CDim>
using MetricIndex = sil::tensor::TensorIdentityIndex<
        sil::tensor::Covariant<sil::tensor::MetricIndex1<CDim...>>,
        sil::tensor::Covariant<sil::tensor::MetricIndex2<CDim...>>>;

template <class InterestIndex, class Index, class... DDim>
static auto test_derivative(auto potential)
{
    using PositionIndex = sil::tensor::Contravariant<
            sil::tensor::TensorNaturalIndex<typename DDim::continuous_dimension_type...>>;

    // Allocate and instantiate an identity metric tensor field.
    [[maybe_unused]] sil::tensor::TensorAccessor<
            MetricIndex<typename DDim::continuous_dimension_type...>> metric_accessor;
    ddc::DiscreteDomain<DDim..., MetricIndex<typename DDim::continuous_dimension_type...>>
            metric_dom(potential.non_indices_domain(), metric_accessor.domain());
    ddc::Chunk metric_alloc(metric_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor metric(metric_alloc);

    // Allocate and instantiate the position field from the mesh coordinates.
    [[maybe_unused]] sil::tensor::TensorAccessor<PositionIndex> position_accessor;
    ddc::DiscreteDomain<DDim..., PositionIndex>
            position_dom(potential.non_indices_domain(), position_accessor.domain());
    ddc::Chunk position_alloc(position_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor position(position_alloc);

    ddc::host_for_each(
            potential.non_indices_domain(),
            [&](typename decltype(potential)::non_indices_domain_t::discrete_element_type elem) {
                ((position
                          .mem(elem,
                               position_accessor.template access_element<
                                       typename DDim::continuous_dimension_type>())
                  = static_cast<double>(ddc::coordinate(ddc::DiscreteElement<DDim>(elem)))),
                 ...);
            });

    // Allocate and compute Laplacian
    [[maybe_unused]] sil::tensor::TensorAccessor<Index> laplacian_accessor;
    ddc::DiscreteDomain<DDim..., Index> laplacian_dom(
            potential.non_indices_domain().remove_last(
                    ddc::DiscreteVector<DDim...>(ddc::DiscreteVector<DDim>(1)...)),
            laplacian_accessor.domain());
    ddc::Chunk laplacian_alloc(laplacian_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor laplacian(laplacian_alloc);

    sil::exterior::laplacian<
            MetricIndex<typename DDim::continuous_dimension_type...>,
            InterestIndex,
            Index>(Kokkos::DefaultHostExecutionSpace(), laplacian, potential, metric, position);
    Kokkos::fence();

    return std::make_pair(std::move(laplacian_alloc), laplacian);
}

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

struct Mu1 : sil::tensor::TensorNaturalIndex<X>
{
};

TEST(Laplacian, 1D0Form)
{
    ddc::Coordinate<X> lower_bounds(-5.);
    ddc::Coordinate<X> upper_bounds(5.);
    ddc::DiscreteVector<DDimX> nb_cells(1000);
    ddc::DiscreteDomain<DDimX> mesh_x = ddc::init_discrete_space<DDimX>(
            DDimX::init<DDimX>(lower_bounds, upper_bounds, nb_cells));

    // Potential
    [[maybe_unused]] sil::tensor::TensorAccessor<sil::tensor::Covariant<sil::tensor::ScalarIndex>>
            potential_accessor;
    ddc::DiscreteDomain<DDimX, sil::tensor::Covariant<sil::tensor::ScalarIndex>>
            potential_dom(mesh_x, potential_accessor.domain());
    ddc::Chunk potential_alloc(potential_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor potential(potential_alloc);

    double const R = 2.;
    double const alpha = 0.5;
    ddc::host_for_each(
            potential.domain(),
            [&](ddc::DiscreteElement<DDimX, sil::tensor::Covariant<sil::tensor::ScalarIndex>>
                        elem) {
                double const r = Kokkos::abs(
                        static_cast<double>(ddc::coordinate(ddc::DiscreteElement<DDimX>(elem))));
                if (r <= R) {
                    potential.mem(elem) = -alpha * r * r;
                } else {
                    potential.mem(elem) = -alpha * R * (2 * r - R);
                }
            });


    auto [alloc, laplacian] = test_derivative<
            sil::tensor::Covariant<Mu1>,
            sil::tensor::Covariant<sil::tensor::ScalarIndex>,
            DDimX>(potential);

    ddc::host_for_each(
            laplacian.template domain<DDimX>().remove_last(ddc::DiscreteVector<DDimX>(1)),
            [&](ddc::DiscreteElement<DDimX> elem) {
                double const value = laplacian(
                        elem,
                        ddc::DiscreteElement<sil::tensor::Covariant<sil::tensor::ScalarIndex>> {0});
                if (ddc::coordinate(elem) < -1.2 * R || ddc::coordinate(elem) > 1.2 * R) {
                    EXPECT_NEAR(value, 0., 1e-2);
                } else if (ddc::coordinate(elem) > -.8 * R && ddc::coordinate(elem) < .8 * R) {
                    EXPECT_NEAR(value, 1., 1e-2);
                }
            });
    ddc::detail::g_discrete_space_dual<DDimX>.reset();
}

TEST(Laplacian, 1D1Form)
{
    ddc::Coordinate<X> lower_bounds(-5.);
    ddc::Coordinate<X> upper_bounds(5.);
    ddc::DiscreteVector<DDimX> nb_cells(1000);
    ddc::DiscreteDomain<DDimX> mesh_x = ddc::init_discrete_space<DDimX>(
            DDimX::init<DDimX>(lower_bounds, upper_bounds, nb_cells));

    // Potential
    [[maybe_unused]] sil::tensor::TensorAccessor<sil::tensor::Covariant<Mu1>> potential_accessor;
    ddc::DiscreteDomain<DDimX, sil::tensor::Covariant<Mu1>>
            potential_dom(mesh_x, potential_accessor.domain());
    ddc::Chunk potential_alloc(potential_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor potential(potential_alloc);

    double const R = 2.;
    double const alpha = 0.5;
    ddc::host_for_each(
            potential.domain(),
            [&](ddc::DiscreteElement<DDimX, sil::tensor::Covariant<Mu1>> elem) {
                double const r = Kokkos::abs(
                        static_cast<double>(ddc::coordinate(ddc::DiscreteElement<DDimX>(elem))));
                if (r <= R) {
                    potential.mem(elem) = -alpha * r * r;
                } else {
                    potential.mem(elem) = -alpha * R * (2 * r - R);
                }
            });


    auto [alloc, laplacian]
            = test_derivative<sil::tensor::Covariant<Mu1>, sil::tensor::Covariant<Mu1>, DDimX>(
                    potential);

    ddc::host_for_each(
            laplacian.template domain<DDimX>().remove_last(ddc::DiscreteVector<DDimX>(1)),
            [&](ddc::DiscreteElement<DDimX> elem) {
                double const value = laplacian(elem, laplacian.accessor().access_element<X>());
                if (ddc::coordinate(elem) < -1.2 * R || ddc::coordinate(elem) > 1.2 * R) {
                    EXPECT_NEAR(value, 0., 1e-2);
                } else if (ddc::coordinate(elem) > -.8 * R && ddc::coordinate(elem) < .8 * R) {
                    EXPECT_NEAR(value, 1., 1e-2);
                }
            });
    ddc::detail::g_discrete_space_dual<DDimX>.reset();
}

struct Mu2 : sil::tensor::TensorNaturalIndex<X, Y>
{
};

TEST(Laplacian, 2D0Form)
{
    ddc::Coordinate<X, Y> lower_bounds(-5., -5.);
    ddc::Coordinate<X, Y> upper_bounds(5., 5.);
    ddc::DiscreteVector<DDimX, DDimY> nb_cells(31, 31);
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
    [[maybe_unused]] sil::tensor::TensorAccessor<sil::tensor::Covariant<sil::tensor::ScalarIndex>>
            potential_accessor;
    ddc::DiscreteDomain<DDimX, DDimY, sil::tensor::Covariant<sil::tensor::ScalarIndex>>
            potential_dom(mesh_xy, potential_accessor.domain());
    ddc::Chunk potential_alloc(potential_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor potential(potential_alloc);

    double const R = 2.;
    double const alpha = 0.25;
    ddc::host_for_each(
            potential.domain(),
            [&](ddc::DiscreteElement<DDimX, DDimY, sil::tensor::Covariant<sil::tensor::ScalarIndex>>
                        elem) {
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
            sil::tensor::Covariant<Mu2>,
            sil::tensor::Covariant<sil::tensor::ScalarIndex>,
            DDimX,
            DDimY>(potential);

    ddc::host_for_each(
            laplacian.template domain<DDimX>().remove_last(ddc::DiscreteVector<DDimX>(1)),
            [&](ddc::DiscreteElement<DDimX> elem) {
                double const value = laplacian(
                        elem,
                        ddc::DiscreteElement<DDimY> {
                                static_cast<std::size_t>(nb_cells.template get<DDimY>()) / 2},
                        ddc::DiscreteElement<sil::tensor::Covariant<sil::tensor::ScalarIndex>> {0});
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
    ddc::DiscreteVector<DDimX, DDimY> nb_cells(31, 31);
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
    [[maybe_unused]] sil::tensor::TensorAccessor<sil::tensor::Covariant<Mu2>> potential_accessor;
    ddc::DiscreteDomain<DDimX, DDimY, sil::tensor::Covariant<Mu2>>
            potential_dom(mesh_xy, potential_accessor.domain());
    ddc::Chunk potential_alloc(potential_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor potential(potential_alloc);

    double const R = 2.;
    ddc::host_for_each(
            potential.non_indices_domain(),
            [&](ddc::DiscreteElement<DDimX, DDimY> elem) {
                double const x_coord = ddc::coordinate(ddc::DiscreteElement<DDimX>(elem));
                double const y_coord = ddc::coordinate(ddc::DiscreteElement<DDimY>(elem));
                double const r = Kokkos::sqrt(
                        static_cast<double>(x_coord * x_coord)
                        + static_cast<double>(y_coord * y_coord));
                if (r <= R) {
                    double const factor = r / 3. - R / 2.;
                    potential.mem(elem, potential_accessor.access_element<X>()) = y_coord * factor;
                    potential.mem(elem, potential_accessor.access_element<Y>()) = -x_coord * factor;
                } else {
                    double const factor = -R * R * R / (6. * r * r);
                    potential.mem(elem, potential_accessor.access_element<X>()) = y_coord * factor;
                    potential.mem(elem, potential_accessor.access_element<Y>()) = -x_coord * factor;
                }
            });


    auto [alloc, laplacian] = test_derivative<
            sil::tensor::Covariant<Mu2>,
            sil::tensor::Covariant<Mu2>,
            DDimX,
            DDimY>(potential);

    ddc::host_for_each(
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

struct Mu3 : sil::tensor::TensorNaturalIndex<X, Y, Z>
{
};

TEST(Laplacian, 3D1Form)
{
    ddc::Coordinate<X, Y, Z> lower_bounds(-5., -5., -5.);
    ddc::Coordinate<X, Y, Z> upper_bounds(5., 5., 5.);
    ddc::DiscreteVector<DDimX, DDimY, DDimZ> nb_cells(21, 21, 21);
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

    // Potential
    [[maybe_unused]] sil::tensor::TensorAccessor<sil::tensor::Covariant<Mu3>> potential_accessor;
    ddc::DiscreteDomain<DDimX, DDimY, DDimZ, sil::tensor::Covariant<Mu3>>
            potential_dom(mesh_xyz, potential_accessor.domain());
    ddc::Chunk potential_alloc(potential_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor potential(potential_alloc);

    double const R = 2.;
    double const alpha = 0.2;
    ddc::host_for_each(
            potential.non_indices_domain(),
            [&](ddc::DiscreteElement<DDimX, DDimY, DDimZ> elem) {
                double const r = Kokkos::sqrt(
                        static_cast<double>(
                                ddc::coordinate(ddc::DiscreteElement<DDimX>(elem))
                                * ddc::coordinate(ddc::DiscreteElement<DDimX>(elem)))
                        + static_cast<double>(
                                ddc::coordinate(ddc::DiscreteElement<DDimY>(elem))
                                * ddc::coordinate(ddc::DiscreteElement<DDimY>(elem)))
                        + static_cast<double>(
                                ddc::coordinate(ddc::DiscreteElement<DDimZ>(elem))
                                * ddc::coordinate(ddc::DiscreteElement<DDimZ>(elem))));
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
                            = alpha * R * R * R * R / (r * r) * Kokkos::sin(theta);
                    potential.mem(elem, potential_accessor.access_element<Y>())
                            = -alpha * R * R * R * R / (r * r) * Kokkos::cos(theta);
                }
                potential.mem(elem, potential_accessor.access_element<Z>()) = 0.;
            });


    auto [alloc, laplacian] = test_derivative<
            sil::tensor::Covariant<Mu3>,
            sil::tensor::Covariant<Mu3>,
            DDimX,
            DDimY,
            DDimZ>(potential);

    ddc::host_for_each(
            laplacian.template domain<DDimX>().remove_last(ddc::DiscreteVector<DDimX>(1)),
            [&](ddc::DiscreteElement<DDimX> elem) {
                double const value = laplacian(
                        elem,
                        ddc::DiscreteElement<DDimY> {
                                static_cast<std::size_t>(nb_cells.template get<DDimY>()) / 2},
                        ddc::DiscreteElement<DDimZ> {
                                static_cast<std::size_t>(nb_cells.template get<DDimZ>()) / 2},

                        laplacian.accessor().access_element<Y>());
                if (ddc::coordinate(elem) < -1.3 * R || ddc::coordinate(elem) > 1.3 * R) {
                    EXPECT_NEAR(value, 0., .5);
                } else if (ddc::coordinate(elem) > -.7 * R && ddc::coordinate(elem) < -.3 * R) {
                    EXPECT_NEAR(value, -1., .5);
                } else if (ddc::coordinate(elem) > .3 * R && ddc::coordinate(elem) < .7 * R) {
                    EXPECT_NEAR(value, 1., .5);
                }
            });
    ddc::detail::g_discrete_space_dual<DDimX>.reset();
    ddc::detail::g_discrete_space_dual<DDimY>.reset();
}
