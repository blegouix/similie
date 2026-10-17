// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#include <ddc/ddc.hpp>

#include <gtest/gtest.h>

#include "exterior.hpp"

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
    using continuous_dimension_type = Z;
};
struct Direction : sil::tensor::TensorNaturalIndex<X, Y, Z>
{
};
struct OtherDirection : sil::tensor::TensorNaturalIndex<X, Y, Z>
{
};
struct ThirdDirection : sil::tensor::TensorNaturalIndex<X, Y, Z>
{
};
using Scalar = sil::tensor::Covariant<sil::tensor::ScalarIndex>;
using OneForm = sil::tensor::Covariant<Direction>;
using TwoForm
        = sil::tensor::TensorAntisymmetricIndex<OneForm, sil::tensor::Covariant<OtherDirection>>;
using ThreeForm = sil::tensor::TensorAntisymmetricIndex<
        OneForm,
        sil::tensor::Covariant<OtherDirection>,
        sil::tensor::Covariant<ThirdDirection>>;
using PositionIndex = sil::tensor::Contravariant<Direction>;
using SpatialDomain = ddc::DiscreteDomain<GridX, GridY, GridZ>;

using MetricIndex = sil::tensor::TensorIdentityIndex<
        sil::tensor::Covariant<sil::tensor::MetricIndex1<X, Y, Z>>,
        sil::tensor::Covariant<sil::tensor::MetricIndex2<X, Y, Z>>>;
template <class Index>
using Field = sil::tensor::Tensor<
        double,
        ddc::DiscreteDomain<GridX, GridY, GridZ, Index>,
        Kokkos::layout_right,
        Kokkos::DefaultExecutionSpace::memory_space>;

static_assert(!sil::exterior::StagedOperatorConcept<void (*)(int, double), int, double>);
static_assert(!sil::exterior::StagedOperatorConcept<int (*)(int, char*), int, double>);

template <class Form, sil::exterior::CellComplex Complex>
void check_reconstruction(bool singular)
{
    using MemorySpace = Kokkos::DefaultExecutionSpace::memory_space;
    static_assert(sil::exterior::StagedOperatorConcept<
                  sil::exterior::StagedLaplacian<
                          MetricIndex,
                          OneForm,
                          Form,
                          Field<Form>,
                          Field<MetricIndex>,
                          Field<PositionIndex>,
                          Kokkos::DefaultExecutionSpace>,
                  Field<Form>>);
    if constexpr (Form::rank() > 0)
        static_assert(sil::exterior::StagedOperatorConcept<
                      sil::exterior::StagedCodifferential<
                              MetricIndex,
                              OneForm,
                              Form,
                              Field<Form>,
                              Field<MetricIndex>,
                              Field<PositionIndex>,
                              Kokkos::DefaultExecutionSpace>,
                      sil::exterior::codifferential_tensor_t<OneForm, Form, Field<Form>>,
                      Field<Form>>);
    SpatialDomain const geometry_domain(
            ddc::DiscreteElement<GridX, GridY, GridZ>(0, 0, 0),
            ddc::DiscreteVector<GridX, GridY, GridZ>(4, 4, 4));
    SpatialDomain const
            domain(ddc::DiscreteElement<GridX, GridY, GridZ>(1, 1, 1),
                   ddc::DiscreteVector<GridX, GridY, GridZ>(2, 2, 2));
    sil::tensor::TensorAccessor<PositionIndex> const position_accessor;
    ddc::DiscreteDomain<GridX, GridY, GridZ, PositionIndex> const
            position_domain(geometry_domain, position_accessor.domain());
    ddc::Chunk position_alloc(position_domain, ddc::KokkosAllocator<double, MemorySpace>());
    sil::tensor::Tensor position(position_alloc);
    ddc::parallel_for_each(
            Kokkos::DefaultExecutionSpace(),
            geometry_domain,
            KOKKOS_LAMBDA(SpatialDomain::discrete_element_type elem) {
                double const x = elem.template uid<GridX>(), y = elem.template uid<GridY>(),
                             z = elem.template uid<GridZ>();
                position(elem, position.accessor().template access_element<X>())
                        = 1.7 * x + 0.2 * y + 0.1 * z + 0.03 * x * y;
                position(elem, position.accessor().template access_element<Y>())
                        = 0.4 * x + 1.3 * y + 0.2 * z + 0.02 * y * z;
                position(elem, position.accessor().template access_element<Z>())
                        = singular ? 0.0 : 0.1 * x + 0.3 * y + 1.1 * z + 0.01 * x * z;
            });
    sil::tensor::TensorAccessor<Form> const accessor;
    ddc::DiscreteDomain<GridX, GridY, GridZ, Form> const form_domain(domain, accessor.domain());
    ddc::Chunk form_alloc(form_domain, ddc::KokkosAllocator<double, MemorySpace>());
    ddc::Chunk reduced_alloc(form_domain, ddc::KokkosAllocator<double, MemorySpace>());
    ddc::Chunk reconstructed_alloc(form_domain, ddc::KokkosAllocator<double, MemorySpace>());
    ddc::Chunk staged_alloc(form_domain, ddc::KokkosAllocator<double, MemorySpace>());
    sil::tensor::Tensor const form(form_alloc), reduced(reduced_alloc),
            reconstructed(reconstructed_alloc);
    using Indices = ddc::to_type_seq_t<typename decltype(accessor)::natural_domain_t>;
    ddc::parallel_for_each(
            Kokkos::DefaultExecutionSpace(),
            domain,
            KOKKOS_LAMBDA(SpatialDomain::discrete_element_type elem) {
                ddc::device_for_each(accessor.domain(), [&](auto component) {
                    form.mem(elem, component) = 0.4 + 0.2 * component.template uid<Form>()
                                                + 0.01 * elem.template uid<GridX>()
                                                - 0.03 * elem.template uid<GridY>()
                                                + 0.02 * elem.template uid<GridZ>();
                });
                sil::exterior::Reduction<Indices, decltype(position), decltype(elem), Complex>::
                operator()(reduced[elem], form[elem], position, elem);
                sil::exterior::
                        Reconstruction<Indices, decltype(position), decltype(elem), Complex>::
                        operator()(reconstructed[elem], reduced[elem], position, elem);
            });
    sil::tensor::Tensor staged_result(staged_alloc);
    auto staged = sil::exterior::make_staged_reconstruction<
            Indices,
            Complex>(Kokkos::DefaultExecutionSpace(), staged_result, reduced, position);
    static_assert(sil::exterior::StagedOperatorConcept<
                  decltype(staged),
                  decltype(staged_result),
                  decltype(reduced)>);
    staged(staged_result, reduced);
    sil::tensor::tensor_accessor_for_domain_t<sil::exterior::reconstruction_domain_t<Indices>> const
            reconstruction_accessor;
    sil::misc::
            cartesian_prod_t<SpatialDomain, sil::exterior::reconstruction_domain_t<Indices>> const
                    operator_domain(domain, reconstruction_accessor.domain());
    ddc::Chunk operator_alloc(operator_domain, ddc::KokkosAllocator<double, MemorySpace>());
    sil::tensor::Tensor reconstruction_operator(operator_alloc);
    sil::exterior::fill_reconstruction_operator<
            Indices,
            decltype(reconstruction_operator),
            decltype(position),
            Kokkos::DefaultExecutionSpace,
            Complex>(Kokkos::DefaultExecutionSpace(), reconstruction_operator, position);
    Kokkos::View<double*> coefficient_errors("reconstruction_coefficients", domain.size());
    ddc::parallel_for_each(
            Kokkos::DefaultExecutionSpace(),
            domain,
            KOKKOS_LAMBDA(SpatialDomain::discrete_element_type elem) {
                double error = 0.0;
                ddc::device_for_each(reconstruction_accessor.domain(), [&](auto component) {
                    double const expected = sil::exterior::
                            Reconstruction<Indices, decltype(position), decltype(elem), Complex>::
                                    value(position,
                                          elem,
                                          reconstruction_accessor.canonical_natural_element(
                                                  component));
                    error += Kokkos::abs(reconstruction_operator.mem(elem, component) - expected);
                });
                coefficient_errors(
                        (elem.template uid<GridX>() - 1) * 4 + (elem.template uid<GridY>() - 1) * 2
                        + elem.template uid<GridZ>() - 1)
                        = error;
            });
    auto const expected = Kokkos::
            create_mirror_view_and_copy(Kokkos::HostSpace(), form_alloc.allocation_kokkos_view());
    auto const actual = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(),
            reconstructed_alloc.allocation_kokkos_view());
    auto const staged_host = Kokkos::
            create_mirror_view_and_copy(Kokkos::HostSpace(), staged_alloc.allocation_kokkos_view());
    auto const errors
            = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), coefficient_errors);
    for (std::size_t i = 0; i < actual.size(); ++i)
        EXPECT_NEAR(
                actual.data()[i],
                singular && Form::rank() != 0 ? 0.0 : expected.data()[i],
                1.e-12);
    for (std::size_t i = 0; i < errors.size(); ++i)
        EXPECT_NEAR(errors(i), 0.0, 1.e-13);
    for (std::size_t i = 0; i < actual.size(); ++i)
        EXPECT_NEAR(staged_host.data()[i], actual.data()[i], 1.e-13);

    // Cochain values change independently of the cached geometry coefficients.
    ddc::parallel_for_each(
            Kokkos::DefaultExecutionSpace(),
            form_domain,
            KOKKOS_LAMBDA(typename decltype(form_domain)::discrete_element_type elem) {
                reduced.mem(elem) *= 1.75;
            });
    staged(staged_result, reduced);
    auto const updated = Kokkos::
            create_mirror_view_and_copy(Kokkos::HostSpace(), staged_alloc.allocation_kokkos_view());
    for (std::size_t i = 0; i < actual.size(); ++i)
        EXPECT_NEAR(updated.data()[i], 1.75 * actual.data()[i], 1.e-12);

    // Geometry changes require rebuilding the staged operator. The old operator
    // keeps its original coefficients, while the new one follows the scaled chart.
    ddc::parallel_for_each(
            Kokkos::DefaultExecutionSpace(),
            position_domain,
            KOKKOS_LAMBDA(typename decltype(position_domain)::discrete_element_type elem) {
                position.mem(elem) *= 2.;
            });
    staged(staged_result, reduced);
    auto const unchanged = Kokkos::
            create_mirror_view_and_copy(Kokkos::HostSpace(), staged_alloc.allocation_kokkos_view());
    for (std::size_t i = 0; i < actual.size(); ++i)
        EXPECT_NEAR(unchanged.data()[i], updated.data()[i], 1.e-13);
    auto refreshed = sil::exterior::make_staged_reconstruction<
            Indices,
            Complex>(Kokkos::DefaultExecutionSpace(), staged_result, reduced, position);
    refreshed(staged_result, reduced);
    auto const refreshed_host = Kokkos::
            create_mirror_view_and_copy(Kokkos::HostSpace(), staged_alloc.allocation_kokkos_view());
    for (std::size_t i = 0; i < actual.size(); ++i)
        EXPECT_NEAR(refreshed_host.data()[i], updated.data()[i] / (1UL << Form::rank()), 1.e-12);
}

TEST(Reconstruction, CurvilinearThreeDimensionalFormsOnDevice)
{
    check_reconstruction<Scalar, sil::exterior::CellComplex::Primal>(false);
    check_reconstruction<OneForm, sil::exterior::CellComplex::Primal>(false);
    check_reconstruction<TwoForm, sil::exterior::CellComplex::Primal>(false);
    check_reconstruction<ThreeForm, sil::exterior::CellComplex::Primal>(false);
    check_reconstruction<Scalar, sil::exterior::CellComplex::CircumcentricDual>(false);
    check_reconstruction<OneForm, sil::exterior::CellComplex::CircumcentricDual>(false);
    check_reconstruction<TwoForm, sil::exterior::CellComplex::CircumcentricDual>(false);
    check_reconstruction<ThreeForm, sil::exterior::CellComplex::CircumcentricDual>(false);
}

TEST(Reconstruction, SingularGeometryOnDevice)
{
    check_reconstruction<Scalar, sil::exterior::CellComplex::Primal>(true);
    check_reconstruction<OneForm, sil::exterior::CellComplex::Primal>(true);
    check_reconstruction<TwoForm, sil::exterior::CellComplex::Primal>(true);
    check_reconstruction<ThreeForm, sil::exterior::CellComplex::Primal>(true);
}
} // namespace
