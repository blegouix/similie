// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#include <gtest/gtest.h>
#include <similie/exterior/reduction_and_reconstruction/vertex_form_transfer.hpp>

#include "local_staging_fixture.hpp"

namespace {
using namespace local_staging_test;
using FormTwo = sil::tensor::TensorAntisymmetricIndex<sil::tensor::Covariant<Other>, One>;

struct TransferChecks
{
    static void check(bool warped)
    {
        Domain const geometry_domain(
                ddc::DiscreteElement<GridX, GridY, GridZ>(0, 0, 0),
                ddc::DiscreteVector<GridX, GridY, GridZ>(5, 5, 5));
        Domain const
                cells(ddc::DiscreteElement<GridX, GridY, GridZ>(1, 1, 1),
                      ddc::DiscreteVector<GridX, GridY, GridZ>(3, 3, 3));
        Domain const interior(
                ddc::DiscreteElement<GridX, GridY, GridZ>(1, 1, 1),
                ddc::DiscreteVector<GridX, GridY, GridZ>(2, 2, 2));
        Domain const vertices(
                ddc::DiscreteElement<GridX, GridY, GridZ>(2, 2, 2),
                ddc::DiscreteVector<GridX, GridY, GridZ>(2, 2, 2));
        Allocation<PositionIndex> position_alloc = allocate<PositionIndex>(geometry_domain);
        Allocation<One> nodal_alloc = allocate<One>(geometry_domain),
                        potential_alloc = allocate<One>(geometry_domain),
                        cochain_alloc = allocate<One>(cells),
                        potential_cochain_alloc = allocate<One>(cells),
                        observed_alloc = allocate<One>(vertices);
        Allocation<FormTwo> two_alloc = allocate<FormTwo>(geometry_domain),
                            two_cochain_alloc = allocate<FormTwo>(cells),
                            derivative_alloc = allocate<FormTwo>(interior),
                            observed_two_alloc = allocate<FormTwo>(vertices);
        Field<PositionIndex> const position(position_alloc.span_view());
        Field<One> const nodal(nodal_alloc.span_view()), potential(potential_alloc.span_view()),
                cochain(cochain_alloc.span_view()),
                potential_cochain(potential_cochain_alloc.span_view()),
                observed(observed_alloc.span_view());
        Field<FormTwo> const two(two_alloc.span_view()), two_cochain(two_cochain_alloc.span_view()),
                derivative(derivative_alloc.span_view()),
                observed_two(observed_two_alloc.span_view());
        ddc::parallel_for_each(
                Exec(),
                geometry_domain,
                KOKKOS_LAMBDA(Domain::discrete_element_type element) {
                    double const i = element.uid<GridX>(), j = element.uid<GridY>(),
                                 k = element.uid<GridZ>();
                    double const u = i + 0.02 * i * i, v = j + 0.03 * j * j, w = k + 0.01 * k * k;
                    double const x = 2 * u + 0.3 * v + (warped ? 0.04 * u * v : 0);
                    position.mem(element, ddc::DiscreteElement<PositionIndex>(0)) = x;
                    position.mem(element, ddc::DiscreteElement<PositionIndex>(1))
                            = 3 * v + 0.2 * w + (warped ? 0.02 * v * w : 0);
                    position.mem(element, ddc::DiscreteElement<PositionIndex>(2)) = 4 * w + 0.1 * u;
                    for (int c = 0; c < 3; ++c) {
                        nodal.mem(element, ddc::DiscreteElement<One>(c)) = c + 1;
                        potential.mem(element, ddc::DiscreteElement<One>(c)) = c == 1 ? x : 0;
                    }
                });
        ddc::parallel_for_each(
                Exec(),
                two.domain(),
                KOKKOS_LAMBDA(FullDomain<FormTwo>::discrete_element_type element) {
                    std::array const axes = sil::exterior::detail::flat_natural_elem_ids(
                            two.accessor().canonical_natural_element(
                                    ddc::DiscreteElement<FormTwo>(element)));
                    two.mem(element) = axes[0] == 0 && axes[1] == 1 ? 1 : 0;
                });
        ddc::parallel_for_each(
                Exec(),
                cells,
                KOKKOS_LAMBDA(Domain::discrete_element_type element) {
                    sil::exterior::VertexFormProjection<ddc::TypeSeq<One>>::
                    operator()(cochain[element], nodal, position, element);
                    sil::exterior::VertexFormProjection<ddc::TypeSeq<One>>::
                    operator()(potential_cochain[element], potential, position, element);
                    sil::exterior::VertexFormProjection<
                            ddc::TypeSeq<sil::tensor::Covariant<Other>, One>>::
                    operator()(two_cochain[element], two, position, element);
                });
        sil::exterior::
                deriv<sil::tensor::Covariant<Other>, One>(Exec(), derivative, potential_cochain);
        ddc::parallel_for_each(
                Exec(),
                vertices,
                KOKKOS_LAMBDA(Domain::discrete_element_type element) {
                    sil::exterior::VertexFormReconstruction<ddc::TypeSeq<One>>::
                    operator()(observed[element], cochain, position, element);
                    sil::exterior::VertexFormReconstruction<
                            ddc::TypeSeq<sil::tensor::Covariant<Other>, One>>::
                    operator()(observed_two[element], two_cochain, position, element);
                });
        double const one_error = ddc::parallel_transform_reduce(
                Exec(),
                observed.domain(),
                0.0,
                ddc::reducer::sum<double>(),
                KOKKOS_LAMBDA(FullDomain<One>::discrete_element_type element) {
                    double const difference = observed.mem(element) - (element.uid<One>() + 1);
                    return difference * difference;
                });
        double const stokes_error = ddc::parallel_transform_reduce(
                Exec(),
                derivative.domain(),
                0.0,
                ddc::reducer::sum<double>(),
                KOKKOS_LAMBDA(FullDomain<FormTwo>::discrete_element_type element) {
                    double const difference = derivative.mem(element) - two_cochain.mem(element);
                    return difference * difference;
                });
        EXPECT_LT(one_error, 1.e-24);
        EXPECT_LT(stokes_error, 1.e-24);
        if (!warped) {
            double const two_error = ddc::parallel_transform_reduce(
                    Exec(),
                    observed_two.domain(),
                    0.0,
                    ddc::reducer::sum<double>(),
                    KOKKOS_LAMBDA(FullDomain<FormTwo>::discrete_element_type element) {
                        double const difference = observed_two.mem(element) - two.mem(element);
                        return difference * difference;
                    });
            EXPECT_LT(two_error, 1.e-24);
        }
    }
};

TEST(VertexFormTransfer, ObliqueNonuniformCells)
{
    TransferChecks::check(false);
}

TEST(VertexFormTransfer, WarpedCellsPreserveStokesAndConstantOneForms)
{
    TransferChecks::check(true);
}

} // namespace
