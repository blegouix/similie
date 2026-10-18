// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#include <gtest/gtest.h>

#include "local_staging_fixture.hpp"

namespace {
using namespace local_staging_test;

template <class SourceIndices, class TargetIndices, sil::exterior::CellComplex Complex>
void check_weighted_hodge()
{
#if defined(KOKKOS_ENABLE_CUDA)
    // The general dual reference requires a larger CUDA stack, as in the benchmark.
    ASSERT_EQ(cudaDeviceSetLimit(cudaLimitStackSize, 32768), cudaSuccess);
#endif
    using Source = sil::tensor::lower_t<sil::exterior::reduction_index_t<SourceIndices>>;
    using Target = sil::exterior::reduction_index_t<TargetIndices>;
    using Weighted = sil::exterior::WeightedLeviCivitaDiscreteHodgeStar<
            Complex,
            SourceIndices,
            TargetIndices,
            Field<MetricIndex>,
            Field<PositionIndex>,
            Domain::discrete_element_type>;
    using Reference = sil::exterior::ComposedDiscreteHodgeStar<
            Complex,
            SourceIndices,
            TargetIndices,
            Field<MetricIndex>,
            Field<PositionIndex>,
            Domain::discrete_element_type,
            sil::exterior::Reconstruction,
            sil::exterior::Reduction,
            sil::exterior::LeviCivitaContinuousHodgeStar>;
    static_assert(sil::exterior::DiscreteHodgeStarConcept<
                  Weighted,
                  Field<Target>,
                  Field<Source>,
                  Field<MetricIndex>,
                  Field<PositionIndex>,
                  Domain::discrete_element_type,
                  typename sil::misc::convert_type_seq_to_t<
                          ddc::DiscreteElement,
                          ddc::type_seq_merge_t<SourceIndices, TargetIndices>>>);
    auto positions = allocate<PositionIndex>(halo());
    auto metrics = allocate<MetricIndex>(domain());
    auto inputs = allocate<Source>(domain());
    auto references = allocate<Target>(domain());
    auto outputs = allocate<Target>(domain());
    auto staged_outputs = allocate<Target>(domain());
    Field<PositionIndex> position(positions);
    Field<MetricIndex> metric(metrics);
    Field<Source> input(inputs);
    Field<Target> reference(references), output(outputs), staged_output(staged_outputs);
    fill_geometry(position, false);
    fill_form(input);
    Reference const reference_operator(Exec(), metric, position);
    sil::exterior::StagedOperator<Weighted> const staged(Exec(), metric, position);
    static_assert(
            sil::exterior::StagedOperatorConcept<decltype(staged), Field<Target>, Field<Source>>);
    Kokkos::View<double*, Memory> errors("weighted_hodge_errors", domain().size());
    Domain const nodes = domain();
    ddc::parallel_for_each(
            Exec(),
            nodes,
            KOKKOS_LAMBDA(Domain::discrete_element_type element) {
                reference_operator(reference[element], input[element], metric, position, element);
                Weighted::operator()(output[element], input[element], metric, position, element);
                staged(staged_output[element], input[element], metric, position, element);
                auto const full = reference_operator.value(metric, position, element);
                auto const direct = Weighted::value(metric, position, element);
                auto const cached = staged.value(metric, position, element);
                double error = 0.;
                for (std::size_t i = 0; i < full.weights.size(); ++i) {
                    error = Kokkos::max(error, Kokkos::abs(full.weights[i] - direct.weights[i]));
                    error = Kokkos::max(error, Kokkos::abs(full.weights[i] - cached.weights[i]));
                }
                sil::tensor::tensor_accessor_for_domain_t<
                        sil::exterior::hodge_star_domain_t<SourceIndices, TargetIndices>> const
                        accessor;
                ddc::device_for_each(accessor.natural_domain(), [&](auto natural) {
                    double const expected
                            = reference_operator.value(metric, position, element, natural);
                    error = Kokkos::
                            max(error,
                                Kokkos::abs(
                                        expected
                                        - Weighted::value(metric, position, element, natural)));
                    error = Kokkos::
                            max(error,
                                Kokkos::abs(
                                        expected
                                        - staged.value(metric, position, element, natural)));
                });
                ddc::device_for_each(reference[element].domain(), [&](auto component) {
                    error = Kokkos::
                            max(error,
                                Kokkos::abs(
                                        reference[element].mem(component)
                                        - output[element].mem(component)));
                    error = Kokkos::
                            max(error,
                                Kokkos::abs(
                                        reference[element].mem(component)
                                        - staged_output[element].mem(component)));
                });
                errors(sil::exterior::detail::stencil_node(nodes, element)) = error;
            });
    auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), errors);
    for (std::size_t i = 0; i < host.size(); ++i)
        EXPECT_LT(host(i), 1.e-11);
    // Cached coefficients remain tied to their construction geometry.
    fill_geometry(position, false, 1.3);
    staged(output, input);
    auto host_output = ddc::
            create_mirror_view_and_copy(Kokkos::DefaultHostExecutionSpace(), outputs.span_view());
    auto host_staged = ddc::create_mirror_view_and_copy(
            Kokkos::DefaultHostExecutionSpace(),
            staged_outputs.span_view());
    ddc::host_for_each(host_output.domain(), [&](auto element) {
        EXPECT_DOUBLE_EQ(host_output(element), host_staged(element));
    });
}

template <sil::exterior::CellComplex Complex>
void all_degrees()
{
    using namespace sil::tensor;
    check_weighted_hodge<
            ddc::TypeSeq<>,
            ddc::TypeSeq<One, Covariant<Other>, Covariant<Third>>,
            Complex>();
    check_weighted_hodge<
            ddc::TypeSeq<Contravariant<Third>>,
            ddc::TypeSeq<One, Covariant<Other>>,
            Complex>();
    check_weighted_hodge<
            ddc::TypeSeq<Contravariant<Other>, Contravariant<Third>>,
            ddc::TypeSeq<One>,
            Complex>();
    check_weighted_hodge<
            ddc::TypeSeq<Contravariant<Direction>, Contravariant<Other>, Contravariant<Third>>,
            ddc::TypeSeq<>,
            Complex>();
}
TEST(WeightedLeviCivitaDiscreteHodgeStar, AnisotropicPrimalAllDegrees)
{
    all_degrees<sil::exterior::CellComplex::Primal>();
}
TEST(WeightedLeviCivitaDiscreteHodgeStar, AnisotropicDualAllDegrees)
{
    all_degrees<sil::exterior::CellComplex::CircumcentricDual>();
}
} // namespace
