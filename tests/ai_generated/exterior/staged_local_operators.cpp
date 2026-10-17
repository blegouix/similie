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

// Component views must preserve both the cached matrix stride and the field
// component stride. Use const input views and nonsymmetric dense coefficients
// to detect accidental transposition or a lost mapping.
template <class Source, class Target, template <class...> class Coefficients>
void check_strided_component_product()
{
    using Matrix = sil::exterior::detail::LocalCoefficientIndex<Source, Target, Coefficients>;
    constexpr bool diagonal = sil::misc::Specialization<Matrix, sil::tensor::TensorDiagonalIndex>;
    constexpr std::size_t nodes = 4;
    Kokkos::View<double**, Kokkos::LayoutLeft, Memory>
            input("strided_input", nodes, Source::mem_size());
    Kokkos::View<double**, Kokkos::LayoutLeft, Memory>
            output("strided_output", nodes, Target::mem_size());
    Kokkos::View<double**, Kokkos::LayoutLeft, Memory>
            weights("strided_weights", nodes, Matrix::mem_size());
    Kokkos::View<double const**, Kokkos::LayoutLeft, Memory> const_input = input;
    Kokkos::parallel_for(
            "strided_component_product",
            Kokkos::RangePolicy<Exec>(0, nodes),
            KOKKOS_LAMBDA(std::size_t node) {
                for (std::size_t source = 0; source < Source::mem_size(); ++source)
                    input(node, source) = 0.5 + node + 0.25 * source;
                for (std::size_t entry = 0; entry < Matrix::mem_size(); ++entry) {
                    auto const ids = Matrix::mem_id_to_canonical_natural_ids(entry);
                    weights(node, entry) = 1.0 + node + 2.0 * ids[0] - 0.3 * ids[1];
                }
                sil::tensor::Tensor source(
                        ddc::ChunkSpan<
                                double const,
                                ddc::DiscreteDomain<Source>,
                                Kokkos::layout_stride,
                                Memory>(
                                Kokkos::subview(const_input, node, Kokkos::ALL()),
                                sil::tensor::TensorAccessor<Source>::domain()));
                sil::tensor::Tensor target(
                        ddc::ChunkSpan<
                                double,
                                ddc::DiscreteDomain<Target>,
                                Kokkos::layout_stride,
                                Memory>(
                                Kokkos::subview(output, node, Kokkos::ALL()),
                                sil::tensor::TensorAccessor<Target>::domain()));
                sil::exterior::detail::apply_local_stencil<
                        Source,
                        Target,
                        Coefficients>(target, source, weights, node);
            });
    auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), output);
    for (std::size_t node = 0; node < nodes; ++node)
        for (std::size_t target = 0; target < Target::mem_size(); ++target) {
            double expected = 0.0;
            for (std::size_t source = 0; source < Source::mem_size(); ++source)
                if (!diagonal || source == target)
                    expected += (1.0 + node + 2.0 * target - 0.3 * source)
                                * (0.5 + node + 0.25 * source);
            EXPECT_NEAR(host(node, target), expected, 2.e-12 * (1.0 + std::abs(expected)));
        }
}

TEST(StagedLocalOperators, StridedDenseComponentProduct)
{
    check_strided_component_product<Two, Two, sil::tensor::TensorFullIndex>();
    check_strided_component_product<One, Scalar, sil::tensor::TensorFullIndex>();
    check_strided_component_product<Scalar, One, sil::tensor::TensorFullIndex>();
}
TEST(StagedLocalOperators, StridedDiagonalComponentProduct)
{
    check_strided_component_product<Two, Two, sil::tensor::TensorDiagonalIndex>();
    check_strided_component_product<Scalar, Scalar, sil::tensor::TensorDiagonalIndex>();
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
