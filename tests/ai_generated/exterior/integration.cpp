// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#include <gtest/gtest.h>
#include <similie/exterior/exterior.hpp>

namespace {

struct X
{
};
struct Y
{
};
struct Z
{
};
struct GridX
{
    using continuous_dimension_type = X;
};
struct GridY
{
    using continuous_dimension_type = Y;
};
struct GridZ
{
};
using First = sil::tensor::Covariant<sil::tensor::TensorNaturalIndex<X, Y>>;
using Second = sil::tensor::Covariant<sil::tensor::prime<sil::tensor::TensorNaturalIndex<X, Y>>>;
using Area = sil::tensor::TensorAntisymmetricIndex<First, Second>;
using Position = sil::tensor::Contravariant<sil::tensor::TensorNaturalIndex<X, Y>>;
using Scalar = sil::tensor::Covariant<sil::tensor::ScalarIndex>;
using Memory = Kokkos::DefaultExecutionSpace::memory_space;

template <class Index>
auto allocate_field()
{
    [[maybe_unused]] sil::tensor::TensorAccessor<Index> accessor;
    return ddc::
            Chunk(ddc::DiscreteDomain<GridX, GridY, Index>(
                          ddc::DiscreteDomain<GridX, GridY>(
                                  ddc::DiscreteElement<GridX, GridY>(2, 3),
                                  ddc::DiscreteVector<GridX, GridY>(3, 3)),
                          accessor.domain()),
                  ddc::DeviceAllocator<double>());
}

template <std::size_t Degree>
auto device_chain(std::initializer_list<sil::exterior::Simplex<Degree, GridX, GridY>> cells)
{
    using Cell = sil::exterior::Simplex<Degree, GridX, GridY>;
    Kokkos::View<Cell*, Memory> allocation("integration_chain", cells.size());
    auto host = Kokkos::create_mirror_view(allocation);
    std::size_t i = 0;
    for (Cell const& cell : cells)
        host(i++) = cell;
    Kokkos::deep_copy(allocation, host);
    // Avoid inspecting device memory in Chain's checking constructor.
    sil::exterior::Chain<Cell, Kokkos::LayoutRight, Memory> chain(allocation, std::size_t(0));
    chain += cells.size();
    return chain;
}

TEST(Integration, ReducedCochainOrientationAndNonzeroOrigin)
{
    ddc::Chunk allocation = allocate_field<First>();
    sil::tensor::Tensor field(allocation);
    auto host_allocation = ddc::create_mirror_view_and_copy(Kokkos::HostSpace(), field);
    sil::tensor::Tensor host(host_allocation);
    ddc::host_for_each(host.non_indices_domain(), [&](auto elem) {
        host(elem, host.accessor().template access_element<X>()) = 3.0;
        host(elem, host.accessor().template access_element<Y>()) = 7.0;
    });
    ddc::parallel_deepcopy(field, host);
    using Edge = sil::exterior::Simplex<1, GridX, GridY>;
    auto chain = device_chain<1>(
            {Edge(ddc::DiscreteElement<GridX, GridY>(2, 3),
                  ddc::DiscreteVector<GridX, GridY>(1, 0)),
             Edge(ddc::DiscreteElement<GridX, GridY>(3, 3),
                  ddc::DiscreteVector<GridX, GridY>(0, 1)),
             Edge(ddc::DiscreteElement<GridX, GridY>(2, 4),
                  ddc::DiscreteVector<GridX, GridY>(1, 0),
                  true)});
    EXPECT_DOUBLE_EQ(
            sil::exterior::integrate<First>(Kokkos::DefaultExecutionSpace(), field, chain),
            7.0);
}

TEST(Integration, EmptyChainAndZeroForms)
{
    ddc::Chunk allocation = allocate_field<Scalar>();
    sil::tensor::Tensor field(allocation);
    ddc::parallel_fill(field, 4.0);
    auto empty = device_chain<0>({});
    EXPECT_DOUBLE_EQ(
            sil::exterior::integrate<Scalar>(Kokkos::DefaultExecutionSpace(), field, empty),
            0.0);
    EXPECT_DOUBLE_EQ(
            sil::exterior::integrate<
                    ddc::TypeSeq<>>(Kokkos::DefaultExecutionSpace(), field, field, empty),
            0.0);
    using Point = sil::exterior::Simplex<0, GridX, GridY>;
    auto points = device_chain<0>(
            {Point(ddc::DiscreteElement<GridX, GridY>(2, 3), ddc::DiscreteVector<>()),
             Point(ddc::DiscreteElement<GridX, GridY>(3, 3), ddc::DiscreteVector<>())});
    EXPECT_DOUBLE_EQ(
            sil::exterior::integrate<Scalar>(Kokkos::DefaultExecutionSpace(), field, points),
            8.0);
    EXPECT_DOUBLE_EQ(
            sil::exterior::integrate<
                    ddc::TypeSeq<>>(Kokkos::DefaultExecutionSpace(), field, field, points),
            8.0);
}

TEST(Integration, PhysicalFormsUseMappedReduction)
{
    ddc::Chunk positions_alloc = allocate_field<Position>();
    ddc::Chunk form_alloc = allocate_field<First>();
    ddc::Chunk area_alloc = allocate_field<Area>();
    sil::tensor::Tensor position(positions_alloc);
    sil::tensor::Tensor form(form_alloc);
    sil::tensor::Tensor area(area_alloc);
    auto positions_host_allocation
            = ddc::create_mirror_view_and_copy(Kokkos::HostSpace(), position);
    sil::tensor::Tensor positions_host(positions_host_allocation);
    auto form_host_allocation = ddc::create_mirror_view_and_copy(Kokkos::HostSpace(), form);
    sil::tensor::Tensor form_host(form_host_allocation);
    auto area_host_allocation = ddc::create_mirror_view_and_copy(Kokkos::HostSpace(), area);
    sil::tensor::Tensor area_host(area_host_allocation);
    ddc::host_for_each(positions_host.non_indices_domain(), [&](auto elem) {
        double const i = elem.template uid<GridX>();
        double const j = elem.template uid<GridY>();
        positions_host(elem, positions_host.accessor().template access_element<X>()) = i + j;
        positions_host(elem, positions_host.accessor().template access_element<Y>()) = 2.0 * j;
        form_host(elem, form_host.accessor().template access_element<X>()) = 3.0;
        form_host(elem, form_host.accessor().template access_element<Y>()) = -1.0;
        area_host(elem, area_host.accessor().template access_element<X, Y>()) = 4.0;
    });
    ddc::parallel_deepcopy(position, positions_host);
    ddc::parallel_deepcopy(form, form_host);
    ddc::parallel_deepcopy(area, area_host);
    using Edge = sil::exterior::Simplex<1, GridX, GridY>;
    // The vertical edge lies on the upper-x boundary: no x-forward cell exists.
    auto edges = device_chain<1>(
            {Edge(ddc::DiscreteElement<GridX, GridY>(4, 3),
                  ddc::DiscreteVector<GridX, GridY>(0, 1))});
    EXPECT_DOUBLE_EQ(
            sil::exterior::integrate<
                    ddc::TypeSeq<First>>(Kokkos::DefaultExecutionSpace(), form, position, edges),
            1.0);
    using Cell = sil::exterior::Simplex<2, GridX, GridY>;
    auto cells = device_chain<2>(
            {Cell(ddc::DiscreteElement<GridX, GridY>(2, 3),
                  ddc::DiscreteVector<GridX, GridY>(1, 1),
                  true)});
    EXPECT_DOUBLE_EQ(
            (sil::exterior::integrate<ddc::TypeSeq<
                     First,
                     Second>>(Kokkos::DefaultExecutionSpace(), area, position, cells)),
            -8.0);
}

TEST(Integration, DiscreteStokes)
{
    ddc::Chunk one_alloc = allocate_field<First>();
    using Derivative = sil::exterior::coboundary_index_t<Second, First>;
    ddc::Chunk two_alloc = allocate_field<Derivative>();
    sil::tensor::Tensor one(one_alloc);
    sil::tensor::Tensor two(two_alloc);
    auto host_allocation = ddc::create_mirror_view_and_copy(Kokkos::HostSpace(), one);
    sil::tensor::Tensor host(host_allocation);
    ddc::host_for_each(host.non_indices_domain(), [&](auto elem) {
        host(elem, host.accessor().template access_element<X>()) = 0.0;
        host(elem, host.accessor().template access_element<Y>()) = elem.template uid<GridX>();
    });
    ddc::parallel_deepcopy(one, host);
    sil::exterior::deriv<Second, First>(
            Kokkos::DefaultExecutionSpace(),
            two,
            one,
            sil::tensor::ClampExtrapolationRule());
    using Cell = sil::exterior::Simplex<2, GridX, GridY>;
    Cell const
            cell(ddc::DiscreteElement<GridX, GridY>(2, 3), ddc::DiscreteVector<GridX, GridY>(1, 1));
    auto cells = device_chain<2>({cell});
    auto boundary = sil::exterior::boundary<Kokkos::HostSpace>(cell);
    auto edges = device_chain<1>({boundary[0], boundary[1], boundary[2], boundary[3]});
    double const boundary_integral
            = sil::exterior::integrate<First>(Kokkos::DefaultExecutionSpace(), one, edges);
    EXPECT_DOUBLE_EQ(boundary_integral, 1.0);
    EXPECT_DOUBLE_EQ(
            sil::exterior::integrate<Derivative>(Kokkos::DefaultExecutionSpace(), two, cells),
            boundary_integral);
}

} // namespace
