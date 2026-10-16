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

struct Periodic
{
    template <class Tensor, class Element>
    KOKKOS_FUNCTION auto operator()(Tensor const& field, Element element) const
    {
        typename Tensor::non_indices_domain_t::discrete_element_type wrapped;
        for (std::size_t axis = 0; axis < Tensor::non_indices_domain_t::rank(); ++axis) {
            std::ptrdiff_t const front
                    = ddc::detail::array(field.non_indices_domain().front())[axis];
            std::ptrdiff_t const count
                    = ddc::detail::array(field.non_indices_domain().extents())[axis];
            std::ptrdiff_t const index
                    = static_cast<std::ptrdiff_t>(ddc::detail::array(element)[axis]) - front;
            ddc::detail::array(wrapped)[axis] = front + (index % count + count) % count;
        }
        return sil::tensor::detail::pointwise_tensor<Tensor>(
                [&](auto component) { return field.mem(wrapped, component); });
    }
};

template <class Form, sil::exterior::CellComplex Complex>
void check_hodge(Field<Form> input, Field<MetricIndex> metric, Field<PositionIndex> position)
{
    using Exec = Kokkos::DefaultExecutionSpace;
    using Memory = typename Exec::memory_space;
    using SourceIndices
            = sil::tensor::upper_t<ddc::to_type_seq_t<sil::tensor::natural_domain_t<Form>>>;
    using TargetIndices
            = sil::exterior::codifferential_hodge_output_indices_t<3 - Form::rank(), OneForm>;
    using Target = sil::misc::
            convert_type_seq_to_t<sil::tensor::TensorAntisymmetricIndex, TargetIndices>;
    sil::exterior::StagedHodgeStar<
            Complex,
            SourceIndices,
            TargetIndices,
            Field<MetricIndex>,
            Field<PositionIndex>,
            SpatialDomain::discrete_element_type>
            staged(Exec(), metric, position);
    using Direct = sil::exterior::DiscreteHodgeStar<
            Complex,
            SourceIndices,
            TargetIndices,
            Field<MetricIndex>,
            Field<PositionIndex>,
            SpatialDomain::discrete_element_type>;
    ddc::DiscreteDomain<GridX, GridY, GridZ, Target> const
            domain(input.non_indices_domain(), sil::tensor::TensorAccessor<Target>::domain());
    ddc::Chunk staged_alloc(domain, ddc::KokkosAllocator<double, Memory>());
    ddc::Chunk direct_alloc(domain, ddc::KokkosAllocator<double, Memory>());
    Field<Target> staged_output(staged_alloc), direct_output(direct_alloc);
    staged(staged_output, input);
    Kokkos::View<double*, Memory>
            errors("hodge_coefficient_error", input.non_indices_domain().size());
    ddc::parallel_for_each(
            Exec(),
            input.non_indices_domain(),
            KOKKOS_LAMBDA(SpatialDomain::discrete_element_type element) {
                Direct::
                operator()(direct_output[element], input[element], metric, position, element);
                sil::tensor::tensor_accessor_for_domain_t<
                        sil::exterior::hodge_star_domain_t<SourceIndices, TargetIndices>> const
                        accessor;
                double error = 0.0;
                ddc::device_for_each(accessor.domain(), [&](auto component) {
                    auto const natural = accessor.canonical_natural_element(component);
                    double const reference = Direct::value(metric, position, element, natural);
                    double const difference
                            = Kokkos::abs(
                                      staged.value(metric, position, element, natural) - reference)
                              / (1.0 + Kokkos::abs(reference));
                    error = Kokkos::max(error, difference);
                });
                errors(sil::exterior::detail::
                               laplacian_node_id(input.non_indices_domain(), element))
                        = error;
            });
    auto a = Kokkos::
            create_mirror_view_and_copy(Kokkos::HostSpace(), staged_alloc.allocation_kokkos_view());
    auto b = Kokkos::
            create_mirror_view_and_copy(Kokkos::HostSpace(), direct_alloc.allocation_kokkos_view());
    auto e = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), errors);
    for (std::size_t i = 0; i < a.size(); ++i)
        EXPECT_NEAR(a.data()[i], b.data()[i], 2.e-11 * (1.0 + std::abs(b.data()[i])));
    for (std::size_t i = 0; i < e.size(); ++i)
        EXPECT_LT(e(i), 2.e-11);
}

template <class Form, class Primal, class Dual>
void compare(bool curved, Primal primal, Dual dual)
{
    using Memory = Kokkos::DefaultExecutionSpace::memory_space;
    using Exec = Kokkos::DefaultExecutionSpace;
    using Operator = sil::exterior::StagedLaplacian<
            MetricIndex,
            OneForm,
            Form,
            Field<Form>,
            Field<MetricIndex>,
            Field<PositionIndex>,
            Exec>;
    static_assert(sil::exterior::StagedOperatorConcept<Operator, Field<Form>>);
    SpatialDomain const
            domain(ddc::DiscreteElement<GridX, GridY, GridZ>(7, 7, 7),
                   ddc::DiscreteVector<GridX, GridY, GridZ>(3, 3, 3));
    SpatialDomain const
            halo(ddc::DiscreteElement<GridX, GridY, GridZ>(5, 5, 5),
                 ddc::DiscreteVector<GridX, GridY, GridZ>(7, 7, 7));
    ddc::Chunk position_alloc(
            ddc::DiscreteDomain<
                    GridX,
                    GridY,
                    GridZ,
                    PositionIndex>(halo, sil::tensor::TensorAccessor<PositionIndex>::domain()),
            ddc::KokkosAllocator<double, Memory>());
    ddc::Chunk metric_alloc(
            ddc::DiscreteDomain<
                    GridX,
                    GridY,
                    GridZ,
                    MetricIndex>(domain, sil::tensor::TensorAccessor<MetricIndex>::domain()),
            ddc::KokkosAllocator<double, Memory>());
    Field<PositionIndex> position(position_alloc);
    Field<MetricIndex> metric(metric_alloc);
    ddc::parallel_for_each(
            Exec(),
            halo,
            KOKKOS_LAMBDA(SpatialDomain::discrete_element_type element) {
                double const x = element.uid<GridX>(), y = element.uid<GridY>(),
                             z = element.uid<GridZ>();
                position.mem(element, ddc::DiscreteElement<PositionIndex>(0))
                        = 0.7 * x + (curved ? 0.01 * y * z : 0.0);
                position.mem(element, ddc::DiscreteElement<PositionIndex>(1))
                        = 0.8 * y + (curved ? 0.02 * x * z : 0.0);
                position.mem(element, ddc::DiscreteElement<PositionIndex>(2))
                        = 0.9 * z + (curved ? 0.01 * x * y : 0.0);
            });
    ddc::parallel_fill(Exec(), metric_alloc, 1.0);
    ddc::DiscreteDomain<GridX, GridY, GridZ, Form> const
            form_domain(domain, sil::tensor::TensorAccessor<Form>::domain());
    ddc::Chunk input_alloc(form_domain, ddc::KokkosAllocator<double, Memory>());
    ddc::Chunk expected_alloc(form_domain, ddc::KokkosAllocator<double, Memory>());
    ddc::Chunk actual_alloc(form_domain, ddc::KokkosAllocator<double, Memory>());
    ddc::Chunk dynamic_alloc(form_domain, ddc::KokkosAllocator<double, Memory>());
    Field<Form> input(input_alloc), expected(expected_alloc), actual(actual_alloc);
    sil::exterior::Laplacian<
            MetricIndex,
            OneForm,
            Form,
            Field<Form>,
            Field<MetricIndex>,
            Field<PositionIndex>,
            Exec,
            sil::exterior::StagedHodgeStar>
            reference(Exec(), expected, input, metric, position);
    Operator operation(Exec(), actual, input, metric, position, primal, dual);
    sil::exterior::Laplacian<
            MetricIndex,
            OneForm,
            Form,
            Field<Form>,
            Field<MetricIndex>,
            Field<PositionIndex>,
            Exec>
            dynamic(Exec(), Field<Form>(dynamic_alloc), input, metric, position);
    for (int trial = 0; trial < 4; ++trial) {
        // Staged Hodge coefficients and assembled weights keep the old geometry;
        // the default Hodge policy must observe these changed coordinates.
        if (trial == 3)
            ddc::parallel_for_each(
                    Exec(),
                    position.domain(),
                    KOKKOS_LAMBDA(typename Field<PositionIndex>::discrete_element_type element) {
                        position.mem(element) *= 1.1;
                    });
        ddc::parallel_for_each(
                Exec(),
                form_domain,
                KOKKOS_LAMBDA(typename decltype(form_domain)::discrete_element_type element) {
                    double const x = element.template uid<GridX>(),
                                 y = element.template uid<GridY>(),
                                 z = element.template uid<GridZ>();
                    double const component = element.template uid<Form>();
                    input.mem(element) = trial == 2
                                                 ? 1.0
                                                 : Kokkos::sin(0.7 * x + 0.2 * y * z + component)
                                                           + trial
                                                                     * Kokkos::cos(
                                                                             0.3 * z * x + 0.5 * y
                                                                             + 2.0 * component);
                });
        Kokkos::fence();
        if (trial == 0) {
            check_hodge<Form, sil::exterior::CellComplex::Primal>(input, metric, position);
            check_hodge<
                    Form,
                    sil::exterior::CellComplex::CircumcentricDual>(input, metric, position);
        }
        reference(expected, input, primal, dual);
        operation(actual, input);
        dynamic(Field<Form>(dynamic_alloc), input, primal, dual);
        auto a = Kokkos::create_mirror_view_and_copy(
                Kokkos::HostSpace(),
                actual_alloc.allocation_kokkos_view());
        auto e = Kokkos::create_mirror_view_and_copy(
                Kokkos::HostSpace(),
                expected_alloc.allocation_kokkos_view());
        auto d = Kokkos::create_mirror_view_and_copy(
                Kokkos::HostSpace(),
                dynamic_alloc.allocation_kokkos_view());
        for (std::size_t i = 0; i < a.size(); ++i) {
            EXPECT_NEAR(
                    d.data()[i],
                    e.data()[i] / (trial == 3 ? 1.21 : 1.0),
                    2.e-10 * (1.0 + std::abs(e.data()[i])));
            EXPECT_NEAR(a.data()[i], e.data()[i], 2.e-10 * (1.0 + std::abs(e.data()[i])));
            if (trial == 2 && Form::rank() == 0 && e.data()[i] == 0.0)
                EXPECT_EQ(a.data()[i], 0.0);
        }
    }
}

TEST(StagedLaplacian, Scalar)
{
    compare<Scalar>(false, Periodic {}, Periodic {});
    compare<Scalar>(
            true,
            sil::tensor::ClampExtrapolationRule {},
            sil::tensor::ZeroExtrapolationRule {});
    compare<Scalar>(
            true,
            sil::tensor::ConstantScalarExtrapolationRule {0.7},
            sil::tensor::ZeroExtrapolationRule {});
}
TEST(StagedLaplacian, OneForm)
{
    compare<OneForm>(true, Periodic {}, Periodic {});
    compare<OneForm>(
            true,
            sil::tensor::ClampExtrapolationRule {},
            sil::tensor::ZeroExtrapolationRule {});
}
TEST(StagedLaplacian, TwoForm)
{
    compare<TwoForm>(true, Periodic {}, Periodic {});
    compare<TwoForm>(
            false,
            sil::tensor::ClampExtrapolationRule {},
            sil::tensor::ZeroExtrapolationRule {});
}
TEST(StagedLaplacian, TopForm)
{
    compare<ThreeForm>(true, Periodic {}, Periodic {});
    compare<ThreeForm>(
            true,
            sil::tensor::ClampExtrapolationRule {},
            sil::tensor::ZeroExtrapolationRule {});
}
} // namespace
