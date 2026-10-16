// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#include <gtest/gtest.h>

#include "local_staging_fixture.hpp"

namespace {
using namespace local_staging_test;

template <class Form>
void expect_equal(Field<Form> a, Field<Form> b)
{
    Kokkos::View<double*, Memory, Kokkos::MemoryUnmanaged> a_view(a.data_handle(), a.size());
    Kokkos::View<double*, Memory, Kokkos::MemoryUnmanaged> b_view(b.data_handle(), b.size());
    auto first = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), a_view);
    auto second = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), b_view);
    for (std::size_t i = 0; i < a.size(); ++i)
        EXPECT_NEAR(first(i), second(i), 3.e-12 * (1.0 + std::abs(second(i))));
}

template <class Form, template <class...> class Coefficients>
void check_local(bool curved)
{
    using Indices = ddc::to_type_seq_t<sil::tensor::natural_domain_t<Form>>;
    auto positions = allocate<PositionIndex>(halo());
    auto input_alloc = allocate<Form>(domain());
    auto direct_alloc = allocate<Form>(domain());
    auto cached_alloc = allocate<Form>(domain());
    Field<PositionIndex> position(positions);
    Field<Form> input(input_alloc), reference(direct_alloc), cached(cached_alloc);
    fill_geometry(position, curved);
    using Reduction = sil::exterior::
            Reduction<Indices, Field<PositionIndex>, Domain::discrete_element_type>;
    using Reconstruction = sil::exterior::
            Reconstruction<Indices, Field<PositionIndex>, Domain::discrete_element_type>;
    sil::exterior::StagedReduction<
            Indices,
            Field<Form>,
            Field<Form>,
            Field<PositionIndex>,
            Exec,
            sil::exterior::CellComplex::Primal,
            Coefficients>
            reduction(Exec(), cached, input, position);
    sil::exterior::StagedReconstruction<
            Indices,
            Field<Form>,
            Field<Form>,
            Field<PositionIndex>,
            Exec,
            sil::exterior::CellComplex::Primal,
            Coefficients>
            reconstruction(Exec(), cached, input, position);
    for (int trial = 0; trial < 2; ++trial) {
        fill_form(input, 0.7 * trial);
        ddc::parallel_for_each(
                Exec(),
                domain(),
                KOKKOS_LAMBDA(Domain::discrete_element_type element) {
                    Reduction::operator()(reference[element], input[element], position, element);
                });
        reduction(cached, input);
        expect_equal(cached, reference);
        ddc::parallel_for_each(
                Exec(),
                domain(),
                KOKKOS_LAMBDA(Domain::discrete_element_type element) {
                    Reconstruction::
                    operator()(reference[element], input[element], position, element);
                });
        reconstruction(cached, input);
        expect_equal(cached, reference);
    }
    // Geometry changes do not alter either cache.
    reconstruction(reference, input);
    fill_geometry(position, curved, 1.7);
    reconstruction(cached, input);
    expect_equal(cached, reference);
    reduction(reference, input);
    fill_geometry(position, curved, 2.1);
    reduction(cached, input);
    expect_equal(cached, reference);
}

TEST(StagedLocalOperators, DenseCurvedGeometry)
{
    check_local<Scalar, sil::tensor::TensorFullIndex>(true);
    check_local<One, sil::tensor::TensorFullIndex>(true);
    check_local<Two, sil::tensor::TensorFullIndex>(true);
    check_local<Three, sil::tensor::TensorFullIndex>(true);
}
TEST(StagedLocalOperators, DiagonalCartesianGeometry)
{
    check_local<Scalar, sil::tensor::TensorDiagonalIndex>(false);
    check_local<One, sil::tensor::TensorDiagonalIndex>(false);
    check_local<Two, sil::tensor::TensorDiagonalIndex>(false);
    check_local<Three, sil::tensor::TensorDiagonalIndex>(false);
}
TEST(StagedLocalOperators, DiagonalStorageSize)
{
    using Full
            = sil::exterior::detail::LocalCoefficientIndex<Two, Two, sil::tensor::TensorFullIndex>;
    using Diagonal = sil::exterior::detail::
            LocalCoefficientIndex<Two, Two, sil::tensor::TensorDiagonalIndex>;
    static_assert(Full::mem_size() == 9);
    static_assert(Diagonal::mem_size() == 3);
    EXPECT_EQ(Full::mem_size(), 9u);
    EXPECT_EQ(Diagonal::mem_size(), 3u);
}

template <class Indices, class Position, class Element, sil::exterior::CellComplex Complex>
using CachedReconstruction
        = sil::exterior::detail::CachedReconstruction<Indices, Position, Element, Complex>;
template <class Indices, class Position, class Element, sil::exterior::CellComplex Complex>
using CachedReduction = sil::exterior::detail::CachedReduction<Indices, Position, Element, Complex>;

void check_continuous_and_composed()
{
    using SymmetricMetric = sil::tensor::TensorSymmetricIndex<
            sil::tensor::Covariant<sil::tensor::MetricIndex1<X, Y, Z>>,
            sil::tensor::Covariant<sil::tensor::MetricIndex2<X, Y, Z>>>;
    using SourceIndices
            = sil::tensor::upper_t<ddc::to_type_seq_t<sil::tensor::natural_domain_t<Two>>>;
    using TargetIndices = sil::exterior::codifferential_hodge_output_indices_t<1, One>;
    using Target = sil::exterior::reduction_index_t<TargetIndices>;
    auto positions = allocate<PositionIndex>(halo());
    auto metrics = allocate<SymmetricMetric>(domain());
    auto inputs = allocate<Two>(domain());
    auto references = allocate<Target>(domain());
    auto results = allocate<Target>(domain());
    Field<PositionIndex> position(positions);
    Field<SymmetricMetric> metric(metrics);
    Field<Two> input(inputs);
    Field<Target> reference(references), result(results);
    fill_geometry(position, true);
    fill_form(input);
    ddc::parallel_for_each(
            Exec(),
            metric.domain(),
            KOKKOS_LAMBDA(typename FullDomain<SymmetricMetric>::discrete_element_type element) {
                auto const ids = SymmetricMetric::mem_id_to_canonical_natural_ids(
                        element.template uid<SymmetricMetric>());
                metric.mem(element) = ids[0] == ids[1] ? 1.5 + 0.2 * ids[0] : 0.12;
            });
    using Direct = sil::exterior::ContinuousHodgeStar<
            SourceIndices,
            TargetIndices,
            Field<SymmetricMetric>,
            Domain::discrete_element_type>;
    sil::exterior::StagedContinuousHodgeStar<
            SourceIndices,
            TargetIndices,
            Field<SymmetricMetric>,
            Domain::discrete_element_type>
            continuous(Exec(), metric);
    ddc::parallel_for_each(
            Exec(),
            domain(),
            KOKKOS_LAMBDA(Domain::discrete_element_type element) {
                Direct::operator()(reference[element], input[element], metric, element);
            });
    continuous(result, input);
    expect_equal(result, reference);

    using DirectDiscrete = sil::exterior::DiscreteHodgeStar<
            sil::exterior::CellComplex::Primal,
            SourceIndices,
            TargetIndices,
            Field<SymmetricMetric>,
            Field<PositionIndex>,
            Domain::discrete_element_type>;
    sil::exterior::ComposedDiscreteHodgeStar<
            sil::exterior::CellComplex::Primal,
            SourceIndices,
            TargetIndices,
            Field<SymmetricMetric>,
            Field<PositionIndex>,
            Domain::discrete_element_type,
            CachedReconstruction,
            CachedReduction,
            sil::exterior::StagedContinuousHodgeStar>
            composed(Exec(), metric, position);
    ddc::parallel_for_each(
            Exec(),
            domain(),
            KOKKOS_LAMBDA(Domain::discrete_element_type element) {
                DirectDiscrete::
                operator()(reference[element], input[element], metric, position, element);
                composed(result[element], input[element], metric, position, element);
            });
    expect_equal(result, reference);
    // Both metric and positions are frozen by the composed cache.
    fill_geometry(position, true, 1.3);
    ddc::parallel_for_each(
            Exec(),
            metric.domain(),
            KOKKOS_LAMBDA(typename FullDomain<SymmetricMetric>::discrete_element_type element) {
                metric.mem(element) *= 1.7;
            });
    ddc::parallel_for_each(
            Exec(),
            domain(),
            KOKKOS_LAMBDA(Domain::discrete_element_type element) {
                composed(result[element], input[element], metric, position, element);
            });
    expect_equal(result, reference);
}
TEST(StagedLocalOperators, ContinuousAndComposedHodgeWithGeneralMetric)
{
    check_continuous_and_composed();
}
} // namespace
