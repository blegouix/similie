// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#include <gtest/gtest.h>

#include "local_staging_fixture.hpp"

namespace {
using namespace local_staging_test;

template <class Form>
void expect_equal(Field<Form> const& actual, Field<Form> const& expected)
{
    Kokkos::View<double*, Memory, Kokkos::MemoryUnmanaged> const
            actual_view(actual.data_handle(), actual.size()),
            expected_view(expected.data_handle(), expected.size());
    Kokkos::View<double*, Kokkos::HostSpace> const actual_host
            = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), actual_view);
    Kokkos::View<double*, Kokkos::HostSpace> const expected_host
            = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), expected_view);
    for (std::size_t i = 0; i < actual.size(); ++i)
        EXPECT_NEAR(actual_host(i), expected_host(i), 3.e-12 * (1. + std::abs(expected_host(i))));
}

template <class Form, sil::exterior::CellComplex Complex>
void check_scaling(double spacing = 0.75, double origin = 0.)
{
    using Indices = ddc::to_type_seq_t<sil::tensor::natural_domain_t<Form>>;
    using Reduction = sil::exterior::Reduction<
            Indices,
            Field<PositionIndex>,
            Domain::discrete_element_type,
            Complex,
            sil::tensor::TensorScalingIndex>;
    using Reconstruction = sil::exterior::Reconstruction<
            Indices,
            Field<PositionIndex>,
            Domain::discrete_element_type,
            Complex,
            sil::tensor::TensorScalingIndex>;
    using ReferenceReduction = sil::exterior::
            Reduction<Indices, Field<PositionIndex>, Domain::discrete_element_type, Complex>;
    using ReferenceReconstruction = sil::exterior::
            Reconstruction<Indices, Field<PositionIndex>, Domain::discrete_element_type, Complex>;
    static_assert(
            sil::exterior::detail::LocalCoefficientIndex<
                    typename Reconstruction::source_index_type,
                    typename Reconstruction::target_index_type,
                    sil::tensor::TensorScalingIndex>::mem_size()
            == 1);
    Allocation<PositionIndex> positions = allocate<PositionIndex>(halo());
    Allocation<MetricIndex> metrics = allocate<MetricIndex>(domain());
    Allocation<Form> inputs = allocate<Form>(domain());
    Allocation<Form> expected_alloc = allocate<Form>(domain());
    Allocation<Form> direct_alloc = allocate<Form>(domain());
    Allocation<Form> cached_alloc = allocate<Form>(domain());
    Field<PositionIndex> const position(positions);
    Field<MetricIndex> const metric(metrics);
    Field<Form> const input(inputs), expected(expected_alloc), direct(direct_alloc),
            cached(cached_alloc);
    // Isotropic Cartesian cells. The general operators provide the reference,
    // including their wedge normalization and circumcentric-dual conventions.
    ddc::parallel_for_each(
            Exec(),
            halo(),
            KOKKOS_LAMBDA(Domain::discrete_element_type element) {
                position.mem(element, ddc::DiscreteElement<PositionIndex>(0))
                        = origin + spacing * element.uid<GridX>();
                position.mem(element, ddc::DiscreteElement<PositionIndex>(1))
                        = origin + spacing * element.uid<GridY>();
                position.mem(element, ddc::DiscreteElement<PositionIndex>(2))
                        = origin + spacing * element.uid<GridZ>();
            });
    fill_form(input);
    sil::exterior::StagedOperator<Reduction> const
            reduction(Exec(), cached, input, metric, position);
    sil::exterior::StagedOperator<Reconstruction> const
            reconstruction(Exec(), cached, input, position);
    Kokkos::View<double*, Memory> errors("scaling_coefficient_errors", domain().size());
    Domain const nodes = domain();
    ddc::parallel_for_each(
            Exec(),
            nodes,
            KOKKOS_LAMBDA(Domain::discrete_element_type element) {
                ReferenceReduction::
                operator()(expected[element], input[element], metric, position, element);
                Reduction::operator()(direct[element], input[element], metric, position, element);
                // value() must expand the shared coefficient onto every diagonal
                // entry, including components other than the stored representative.
                std::array const reference
                        = ReferenceReduction::value(metric, position, element).weights;
                std::array const actual = reduction.value(metric, position, element).weights;
                double error = 0.;
                for (std::size_t i = 0; i < reference.size(); ++i)
                    error = Kokkos::
                            max(error,
                                Kokkos::abs(reference[i] - actual[i])
                                        / (1. + Kokkos::abs(reference[i])));
                sil::tensor::tensor_accessor_for_domain_t<
                        sil::exterior::reconstruction_domain_t<Indices>> const accessor;
                ddc::device_for_each(accessor.domain(), [&](auto component) {
                    typename sil::tensor::tensor_accessor_for_domain_t<
                            sil::exterior::reconstruction_domain_t<Indices>>::natural_domain_t::
                            discrete_element_type const natural
                            = accessor.canonical_natural_element(component);
                    double const reference_coefficient
                            = ReferenceReconstruction::value(position, element, natural);
                    error = Kokkos::
                            max(error,
                                Kokkos::abs(
                                        reference_coefficient
                                        - reconstruction.value(position, element, natural))
                                        / (1. + Kokkos::abs(reference_coefficient)));
                });
                errors(sil::exterior::detail::stencil_node(nodes, element)) = error;
            });
    expect_equal(direct, expected);
    reduction(cached, input);
    expect_equal(cached, expected);
    Kokkos::View<double*, Kokkos::HostSpace> const errors_host
            = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), errors);
    for (std::size_t i = 0; i < errors_host.size(); ++i)
        EXPECT_LT(errors_host(i), 3.e-12);
    ddc::parallel_for_each(
            Exec(),
            nodes,
            KOKKOS_LAMBDA(Domain::discrete_element_type element) {
                ReferenceReconstruction::
                operator()(expected[element], input[element], position, element);
                Reconstruction::operator()(direct[element], input[element], position, element);
            });
    expect_equal(direct, expected);
    reconstruction(cached, input);
    expect_equal(cached, expected);
}

TEST(ScalingLocalOperators, PrimalAllDegrees)
{
    check_scaling<Scalar, sil::exterior::CellComplex::Primal>();
    check_scaling<One, sil::exterior::CellComplex::Primal>();
    check_scaling<Two, sil::exterior::CellComplex::Primal>();
    check_scaling<Three, sil::exterior::CellComplex::Primal>();
}

TEST(ScalingLocalOperators, UniformCoordinateRoundoff)
{
    check_scaling<One, sil::exterior::CellComplex::Primal>(0.005, -5.);
}

TEST(ScalingLocalOperators, DualAllDegrees)
{
#if defined(KOKKOS_ENABLE_CUDA)
    ASSERT_EQ(cudaDeviceSetLimit(cudaLimitStackSize, 32768), cudaSuccess);
#endif
    check_scaling<Scalar, sil::exterior::CellComplex::CircumcentricDual>();
    check_scaling<One, sil::exterior::CellComplex::CircumcentricDual>();
    check_scaling<Two, sil::exterior::CellComplex::CircumcentricDual>();
    check_scaling<Three, sil::exterior::CellComplex::CircumcentricDual>();
}
} // namespace
