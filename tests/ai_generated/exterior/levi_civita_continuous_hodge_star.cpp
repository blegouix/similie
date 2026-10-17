// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#include <gtest/gtest.h>

#include "local_staging_fixture.hpp"

namespace {
using namespace local_staging_test;

template <class Source, class Target>
KOKKOS_FUNCTION double canonical_sign(std::size_t source, std::size_t target)
{
    std::array<std::size_t, Source::rank() + Target::rank()> ids {};
    std::array const source_ids = Source::mem_id_to_canonical_natural_ids(source);
    std::array const target_ids = Target::mem_id_to_canonical_natural_ids(target);
    for (std::size_t i = 0; i < source_ids.size(); ++i)
        ids[i] = source_ids[i];
    for (std::size_t i = 0; i < target_ids.size(); ++i)
        ids[source_ids.size() + i] = target_ids[i];
    double sign = 1.0;
    for (std::size_t i = 0; i < ids.size(); ++i)
        for (std::size_t j = i + 1; j < ids.size(); ++j) {
            if (ids[i] == ids[j])
                return 0.0;
            if (ids[i] > ids[j])
                sign = -sign;
        }
    return sign;
}

template <class SourceIndices, class TargetIndices, class Metric>
void check_hodge()
{
    using Source = sil::tensor::lower_t<sil::exterior::reduction_index_t<SourceIndices>>;
    using Target = sil::exterior::reduction_index_t<TargetIndices>;
    using HostMetric = sil::tensor::
            Tensor<double, FullDomain<Metric>, Kokkos::layout_right, Kokkos::HostSpace>;
    using Direct = sil::exterior::ContinuousHodgeStar<
            SourceIndices,
            TargetIndices,
            HostMetric,
            Domain::discrete_element_type>;
    using Levi = sil::exterior::LeviCivitaContinuousHodgeStar<
            SourceIndices,
            TargetIndices,
            Field<Metric>,
            Domain::discrete_element_type>;
    using Inverse = sil::exterior::LeviCivitaContinuousHodgeStar<
            sil::tensor::upper_t<TargetIndices>,
            sil::tensor::lower_t<SourceIndices>,
            Field<Metric>,
            Domain::discrete_element_type>;
    auto inputs = allocate<Source>(domain());
    auto references = allocate<Target>(domain());
    auto results = allocate<Target>(domain());
    auto inverses = allocate<Source>(domain());
    auto metrics = allocate<Metric>(domain());
    Field<Source> input(inputs), inverse(inverses);
    Field<Target> reference(references), result(results);
    Field<Metric> metric(metrics);
    fill_form(input);
    Domain const nodes = domain();
    auto host_inputs = ddc::
            create_mirror_view_and_copy(Kokkos::DefaultHostExecutionSpace(), inputs.span_view());
    ddc::Chunk<double, FullDomain<Target>, ddc::HostAllocator<double>>
            host_references(references.domain(), ddc::HostAllocator<double>());
    sil::tensor::Tensor host_input(host_inputs);
    sil::tensor::Tensor host_reference(host_references);
    HostMetric host_metric(
            ddc::ChunkSpan<
                    double,
                    FullDomain<Metric>,
                    Kokkos::layout_right,
                    Kokkos::HostSpace>(nullptr, metric.domain()));
    // Keep the general reference on the CPU: its 4D coefficient array and
    // recursive component indexing exceed the default CUDA stack limit.
    ddc::host_for_each(nodes, [&](Domain::discrete_element_type element) {
        if constexpr (Source::rank() == 0) {
            sil::tensor::Tensor scalar(
                    ddc::ChunkSpan<
                            double,
                            ddc::DiscreteDomain<>,
                            Kokkos::layout_right,
                            Kokkos::HostSpace>(
                            host_input[element].data_handle(),
                            ddc::DiscreteDomain<>()));
            Direct::operator()(host_reference[element], scalar, host_metric, element);
        } else {
            Direct::operator()(host_reference[element], host_input[element], host_metric, element);
        }
        sil::tensor::tensor_accessor_for_domain_t<
                sil::exterior::hodge_star_domain_t<SourceIndices, TargetIndices>>
                accessor;
        ddc::host_for_each(accessor.domain(), [&](auto component) {
            auto const natural = accessor.canonical_natural_element(component);
            EXPECT_DOUBLE_EQ(
                    (sil::exterior::LeviCivitaContinuousHodgeStar<
                            SourceIndices,
                            TargetIndices,
                            HostMetric,
                            Domain::discrete_element_type>::value(host_metric, element, natural)),
                    Direct::value(host_metric, element, natural));
        });
    });
    ddc::parallel_deepcopy(references.span_view(), host_references.span_view());
    Kokkos::View<double*, Memory> errors("levi_civita_errors", nodes.size());
    ddc::parallel_for_each(
            Exec(),
            domain(),
            KOKKOS_LAMBDA(Domain::discrete_element_type element) {
                Levi::operator()(result[element], input[element], metric, element);
            });
    ddc::parallel_for_each(
            Exec(),
            domain(),
            KOKKOS_LAMBDA(Domain::discrete_element_type element) {
                Inverse::operator()(inverse[element], result[element], metric, element);
            });
    ddc::parallel_for_each(
            Exec(),
            domain(),
            KOKKOS_LAMBDA(Domain::discrete_element_type element) {
                double error = 0.0;
                auto const stencil = Levi::value(metric, element);
                for (std::size_t target = 0; target < Target::mem_size(); ++target) {
                    double expected = 0.0;
                    for (std::size_t source = 0; source < Source::mem_size(); ++source) {
                        double const sign = canonical_sign<Source, Target>(source, target);
                        expected += sign * input.mem(element, ddc::DiscreteElement<Source>(source));
                        error += Kokkos::abs(
                                stencil.weights[target * Source::mem_size() + source] - sign);
                    }
                    error += Kokkos::abs(
                            result.mem(element, ddc::DiscreteElement<Target>(target)) - expected);
                    error += Kokkos::abs(
                            result.mem(element, ddc::DiscreteElement<Target>(target))
                            - reference.mem(element, ddc::DiscreteElement<Target>(target)));
                }
                double const square_sign = (Source::rank() * Target::rank()) % 2 ? -1.0 : 1.0;
                for (std::size_t source = 0; source < Source::mem_size(); ++source)
                    error += Kokkos::abs(
                            inverse.mem(element, ddc::DiscreteElement<Source>(source))
                            - square_sign
                                      * input.mem(element, ddc::DiscreteElement<Source>(source)));
                errors(sil::exterior::detail::stencil_node(nodes, element)) = error;
            });
    auto const host_errors = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), errors);
    for (std::size_t i = 0; i < domain().size(); ++i)
        EXPECT_LT(host_errors(i), 2.e-12);
}

template <class Form>
void check_three_dimensional()
{
    using SourceIndices = std::conditional_t<
            Form::rank() == 0,
            ddc::TypeSeq<>,
            sil::tensor::upper_t<ddc::to_type_seq_t<sil::tensor::natural_domain_t<Form>>>>;
    check_hodge<
            SourceIndices,
            sil::exterior::codifferential_hodge_output_indices_t<3 - Form::rank(), One>,
            MetricIndex>();
}

TEST(LeviCivitaContinuousHodgeStar, Scalar3D)
{
    check_three_dimensional<Scalar>();
}
TEST(LeviCivitaContinuousHodgeStar, OneForm3D)
{
    check_three_dimensional<One>();
}
TEST(LeviCivitaContinuousHodgeStar, TwoForm3D)
{
    check_three_dimensional<Two>();
}
TEST(LeviCivitaContinuousHodgeStar, ThreeForm3D)
{
    check_three_dimensional<Three>();
}

template <std::size_t Id, class... Coordinates>
struct Index : sil::tensor::TensorNaturalIndex<Coordinates...>
{
};

TEST(LeviCivitaContinuousHodgeStar, OneForm2D)
{
    check_hodge<
            ddc::TypeSeq<sil::tensor::Contravariant<Index<0, X, Y>>>,
            ddc::TypeSeq<sil::tensor::Covariant<Index<1, X, Y>>>,
            sil::tensor::TensorIdentityIndex<
                    sil::tensor::Covariant<sil::tensor::MetricIndex1<X, Y>>,
                    sil::tensor::Covariant<sil::tensor::MetricIndex2<X, Y>>>>();
}

struct T
{
};

TEST(LeviCivitaContinuousHodgeStar, RepeatedIndices4D)
{
    // These separated repetitions escaped the previous parity helper's
    // adjacent-duplicate check and introduced spurious Hodge coefficients.
    using Levi = sil::tensor::TensorLeviCivitaIndex<
            Index<0, T, X, Y, Z>,
            Index<1, T, X, Y, Z>,
            Index<2, T, X, Y, Z>,
            Index<3, T, X, Y, Z>>;
    static_assert(Levi::access_id({0, 1, 0, 2}) == 0);
    static_assert(Levi::access_id({0, 1, 2, 1}) == 0);
    static_assert(sil::misc::permutation_parity(std::array<std::size_t, 0> {}) == 1);
    static_assert(sil::misc::permutation_parity(std::array<std::size_t, 1> {0}) == 1);
    EXPECT_EQ(Levi::access_id({0, 1, 0, 2}), 0u);
    EXPECT_EQ(Levi::access_id({0, 1, 2, 1}), 0u);
}

TEST(LeviCivitaContinuousHodgeStar, TwoForm4D)
{
    check_hodge<
            ddc::TypeSeq<
                    sil::tensor::Contravariant<Index<0, T, X, Y, Z>>,
                    sil::tensor::Contravariant<Index<1, T, X, Y, Z>>>,
            ddc::TypeSeq<
                    sil::tensor::Covariant<Index<2, T, X, Y, Z>>,
                    sil::tensor::Covariant<Index<3, T, X, Y, Z>>>,
            sil::tensor::TensorIdentityIndex<
                    sil::tensor::Covariant<sil::tensor::MetricIndex1<T, X, Y, Z>>,
                    sil::tensor::Covariant<sil::tensor::MetricIndex2<T, X, Y, Z>>>>();
}
} // namespace
