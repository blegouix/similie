// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#include <cmath>
#include <numbers>
#include <tuple>

#include <gtest/gtest.h>
#include <similie/exterior/reduction_and_reconstruction/vertex_form_transfer.hpp>
#include <similie/tensor/periodic_extrapolation_rule.hpp>

#include "local_staging_fixture.hpp"

namespace {
using namespace local_staging_test;

using FormView = sil::exterior::Form<Field<One>>;
using CochainView = sil::exterior::Cochain<Field<One>>;
static_assert(!std::is_convertible_v<Field<One>, FormView>);
static_assert(!std::is_convertible_v<Field<One>, CochainView>);
static_assert(!sil::exterior::FormType<CochainView>);
static_assert(!sil::exterior::CochainType<FormView>);
static_assert(sil::exterior::FormType<
              decltype(std::declval<FormView>()[std::declval<Domain::discrete_element_type>()])>);
static_assert(sil::exterior::CochainType<decltype(std::declval<CochainView>()[std::declval<
                      Domain::discrete_element_type>()])>);

template <class Output, class Input>
concept CanInsertTwoForm = requires(
        Output const& output,
        Field<sil::tensor::upper_t<One>> const& vector,
        Input const& input) {
    sil::exterior::InteriorProduct<sil::tensor::upper_t<One>, Two> {}(output, vector, input);
};
static_assert(!CanInsertTwoForm<sil::exterior::Form<Field<Two>>, sil::exterior::Form<Field<Two>>>);
static_assert(!CanInsertTwoForm<sil::exterior::Form<Field<One>>, sil::exterior::Form<Field<One>>>);

template <class InputIndex, class OutputIndex>
void apply_cochain_interior(
        Field<OutputIndex> const& output,
        Field<One> const& velocity,
        Field<InputIndex> const& input,
        std::array<double, 3> const& spacing)
{
    using SourceIndices = ddc::to_type_seq_t<sil::tensor::natural_domain_t<InputIndex>>;
    using First = ddc::type_seq_element_t<0, SourceIndices>;
    using Tail = ddc::type_seq_remove_t<SourceIndices, ddc::TypeSeq<First>>;
    using ResultIndex = sil::exterior::reduction_index_t<Tail>;
    using InverseMetric = sil::tensor::upper_t<MetricIndex>;
    Domain const mesh = output.non_indices_domain();
    Domain const vertices(
            mesh.front(),
            mesh.extents() + ddc::DiscreteVector<GridX, GridY, GridZ>(1, 1, 1));
    Domain const geometry(
            ddc::DiscreteElement<GridX, GridY, GridZ>(0, 0, 0),
            mesh.extents() + ddc::DiscreteVector<GridX, GridY, GridZ>(3, 3, 3));
    Allocation<PositionIndex> positions = allocate<PositionIndex>(geometry);
    Allocation<One> velocity_forms = allocate<One>(vertices);
    Allocation<InputIndex> input_forms = allocate<InputIndex>(vertices);
    Allocation<ResultIndex> output_forms = allocate<ResultIndex>(vertices);
    Field<PositionIndex> const position(positions.span_view());
    sil::exterior::Form const reconstructed_velocity(Field<One>(velocity_forms.span_view()));
    sil::exterior::Form const reconstructed_input(Field<InputIndex>(input_forms.span_view()));
    sil::exterior::Form const transformed(Field<ResultIndex>(output_forms.span_view()));
    sil::tensor::Tensor const inverse_metric(
            ddc::ChunkSpan<double, FullDomain<InverseMetric>, Kokkos::layout_right, Memory>(
                    nullptr,
                    FullDomain<InverseMetric>(
                            vertices,
                            sil::tensor::TensorAccessor<InverseMetric>::domain())));
    ddc::parallel_for_each(
            Exec(),
            geometry,
            KOKKOS_LAMBDA(Domain::discrete_element_type element) {
                for (std::size_t axis = 0; axis < 3; ++axis)
                    position.mem(element, ddc::DiscreteElement<PositionIndex>(axis))
                            = spacing[axis] * ddc::detail::array(element)[axis];
            });
    sil::exterior::OperatorActingOnCochainInPlaceOfForm<
            sil::exterior::InteriorProduct<sil::tensor::upper_t<First>, InputIndex>> {}(
            Exec(),
            sil::exterior::Cochain(output),
            std::tuple {sil::exterior::Cochain(velocity), sil::exterior::Cochain(input)},
            std::tuple {reconstructed_velocity, reconstructed_input},
            transformed,
            sil::exterior::VertexFormReconstructionField(
                    position,
                    sil::tensor::PeriodicExtrapolationRule {}),
            sil::exterior::VertexFormProjectionField(position),
            inverse_metric);
}

template <class InputIndex, class OutputIndex>
void constant_contraction()
{
    Domain const
            mesh(ddc::DiscreteElement<GridX, GridY, GridZ>(1, 1, 1),
                 ddc::DiscreteVector<GridX, GridY, GridZ>(4, 4, 4));
    std::array<double, 3> const spacing {0.3, 0.5, 0.7};
    Allocation<One> velocity_alloc = allocate<One>(mesh);
    Allocation<InputIndex> form_alloc = allocate<InputIndex>(mesh);
    Allocation<OutputIndex> output_alloc = allocate<OutputIndex>(mesh);
    Field<One> const velocity(velocity_alloc.span_view());
    Field<InputIndex> const form(form_alloc.span_view());
    Field<OutputIndex> const output(output_alloc.span_view());
    ddc::parallel_for_each(
            Exec(),
            velocity.domain(),
            KOKKOS_LAMBDA(FullDomain<One>::discrete_element_type element) {
                int const axis = element.template uid<One>();
                velocity.mem(element) = spacing[axis] * (axis + 1);
            });
    ddc::parallel_for_each(
            Exec(),
            form.domain(),
            KOKKOS_LAMBDA(typename FullDomain<InputIndex>::discrete_element_type element) {
                std::array<std::size_t, InputIndex::rank()> const axes
                        = sil::exterior::detail::flat_natural_elem_ids(
                                form.accessor().canonical_natural_element(
                                        ddc::DiscreteElement<InputIndex>(element)));
                double volume = 1;
                for (std::size_t axis : axes)
                    volume *= spacing[axis];
                form.mem(element) = volume;
            });
    apply_cochain_interior<InputIndex>(output, velocity, form, spacing);
    double error = 0;
    error = ddc::parallel_transform_reduce(
            Exec(),
            output.domain(),
            0.0,
            ddc::reducer::sum<double>(),
            KOKKOS_LAMBDA(typename FullDomain<OutputIndex>::discrete_element_type element) {
                auto const natural_element = output.accessor().canonical_natural_element(
                        ddc::DiscreteElement<OutputIndex>(element));
                std::array<std::size_t, OutputIndex::rank()> axes {};
                if constexpr (OutputIndex::rank() > 0)
                    axes = sil::exterior::detail::flat_natural_elem_ids(natural_element);
                double volume = 1;
                for (std::size_t axis : axes)
                    volume *= spacing[axis];
                double expected = 0;
                for (std::size_t b = 0; b < 3; ++b) {
                    bool repeated = false;
                    int sign = 1;
                    for (std::size_t axis : axes) {
                        repeated = repeated || axis == b;
                        if (axis < b)
                            sign *= -1;
                    }
                    if (!repeated)
                        expected += sign * double(b + 1) * volume;
                }
                return std::pow(output.mem(element) - expected, 2);
            });
    EXPECT_LT(error, 1e-25);
}

void skew_pairing()
{
    Domain const
            mesh(ddc::DiscreteElement<GridX, GridY, GridZ>(1, 1, 1),
                 ddc::DiscreteVector<GridX, GridY, GridZ>(8, 8, 8));
    std::array<double, 3> const spacing {0.3, 0.5, 0.7};
    Allocation<One> velocity_alloc = allocate<One>(mesh), output_alloc = allocate<One>(mesh);
    Allocation<Two> form_alloc = allocate<Two>(mesh);
    Field<One> const velocity(velocity_alloc.span_view()), output(output_alloc.span_view());
    Field<Two> const form(form_alloc.span_view());
    fill_form(velocity);
    fill_form(form, 1.2);
    apply_cochain_interior<Two>(output, velocity, form, spacing);
    double pairing = 0;
    pairing = ddc::parallel_transform_reduce(
            Exec(),
            output.domain(),
            0.0,
            ddc::reducer::sum<double>(),
            KOKKOS_LAMBDA(FullDomain<One>::discrete_element_type element) {
                double const h = spacing[element.template uid<One>()];
                return velocity.mem(element) * output.mem(element) / (h * h);
            });
    EXPECT_NEAR(pairing, 0, 1e-10);
}

using ShearTwo = sil::tensor::TensorAntisymmetricIndex<sil::tensor::Covariant<Other>, One>;

double shear_error(int n, double stream)
{
    Domain const
            mesh(ddc::DiscreteElement<GridX, GridY, GridZ>(1, 1, 1),
                 ddc::DiscreteVector<GridX, GridY, GridZ>(n, 2, 2));
    double const h = 2 * std::numbers::pi / n;
    Allocation<One> velocity_alloc = allocate<One>(mesh), transport_alloc = allocate<One>(mesh);
    Allocation<ShearTwo> form_alloc = allocate<ShearTwo>(mesh),
                         derivative_alloc = allocate<ShearTwo>(mesh);
    Field<One> const velocity(velocity_alloc.span_view()), transport(transport_alloc.span_view());
    Field<ShearTwo> const form(form_alloc.span_view()), derivative(derivative_alloc.span_view());
    ddc::parallel_for_each(
            Exec(),
            velocity.domain(),
            KOKKOS_LAMBDA(FullDomain<One>::discrete_element_type element) {
                velocity.mem(element) = element.template uid<One>() == 0 ? stream * h : 0;
            });
    ddc::parallel_for_each(
            Exec(),
            form.domain(),
            KOKKOS_LAMBDA(FullDomain<ShearTwo>::discrete_element_type element) {
                form.mem(element) = element.template uid<ShearTwo>() == 0
                                            ? h * h * std::sin(h * element.template uid<GridX>())
                                            : 0;
            });
    apply_cochain_interior<ShearTwo>(transport, velocity, form, std::array<double, 3> {h, h, h});
    sil::exterior::deriv<sil::tensor::Covariant<Other>, One>(
            Exec(),
            sil::exterior::Cochain(derivative),
            sil::exterior::Cochain(transport),
            sil::tensor::PeriodicExtrapolationRule {});
    double error = 0;
    error = ddc::parallel_transform_reduce(
            Exec(),
            derivative.domain(),
            0.0,
            ddc::reducer::sum<double>(),
            KOKKOS_LAMBDA(FullDomain<ShearTwo>::discrete_element_type element) {
                double const expected
                        = element.template uid<ShearTwo>() == 0
                                  ? stream * std::cos(h * element.template uid<GridX>())
                                  : 0;
                return std::pow(derivative.mem(element) / (h * h) - expected, 2);
            });
    // No advective energy growth: centered advection is skew for either sign.
    double production = 0;
    production = ddc::parallel_transform_reduce(
            Exec(),
            derivative.domain(),
            0.0,
            ddc::reducer::sum<double>(),
            KOKKOS_LAMBDA(FullDomain<ShearTwo>::discrete_element_type element) {
                return form.mem(element) * derivative.mem(element);
            });
    EXPECT_NEAR(production, 0, 1e-12);
    return std::sqrt(error / mesh.size());
}

struct SkewGeometryCheck
{
    static void run()
    {
        Domain const geometry(
                ddc::DiscreteElement<GridX, GridY, GridZ>(0, 0, 0),
                ddc::DiscreteVector<GridX, GridY, GridZ>(6, 6, 6));
        Domain const
                mesh(ddc::DiscreteElement<GridX, GridY, GridZ>(1, 1, 1),
                     ddc::DiscreteVector<GridX, GridY, GridZ>(3, 3, 3));
        Domain const vertices(mesh.front(), ddc::DiscreteVector<GridX, GridY, GridZ>(4, 4, 4));
        using ResultIndex = sil::tensor::Covariant<Other>;
        using InverseMetric = sil::tensor::upper_t<MetricIndex>;
        Allocation<PositionIndex> position_alloc = allocate<PositionIndex>(geometry);
        Allocation<One> physical_velocity_alloc = allocate<One>(geometry),
                        velocity_alloc = allocate<One>(mesh),
                        reconstructed_velocity_alloc = allocate<One>(vertices),
                        output_alloc = allocate<One>(mesh);
        Allocation<Two> physical_input_alloc = allocate<Two>(geometry),
                        input_alloc = allocate<Two>(mesh),
                        reconstructed_input_alloc = allocate<Two>(vertices);
        Allocation<ResultIndex> transformed_alloc = allocate<ResultIndex>(vertices);
        Field<PositionIndex> const position(position_alloc.span_view());
        sil::exterior::Form const physical_velocity(
                Field<One>(physical_velocity_alloc.span_view()));
        sil::exterior::Form const physical_input(Field<Two>(physical_input_alloc.span_view()));
        sil::exterior::Cochain const velocity(Field<One>(velocity_alloc.span_view()));
        sil::exterior::Cochain const input(Field<Two>(input_alloc.span_view()));
        sil::exterior::Cochain const output(Field<One>(output_alloc.span_view()));
        sil::exterior::Form const reconstructed_velocity(
                Field<One>(reconstructed_velocity_alloc.span_view()));
        sil::exterior::Form const reconstructed_input(
                Field<Two>(reconstructed_input_alloc.span_view()));
        sil::exterior::Form const transformed(Field<ResultIndex>(transformed_alloc.span_view()));
        sil::tensor::Tensor const inverse_metric(
                ddc::ChunkSpan<double, FullDomain<InverseMetric>, Kokkos::layout_right, Memory>(
                        nullptr,
                        FullDomain<InverseMetric>(
                                vertices,
                                sil::tensor::TensorAccessor<InverseMetric>::domain())));
        ddc::parallel_for_each(
                Exec(),
                geometry,
                KOKKOS_LAMBDA(Domain::discrete_element_type element) {
                    double const x = element.uid<GridX>(), y = element.uid<GridY>(),
                                 z = element.uid<GridZ>();
                    position.mem(element, ddc::DiscreteElement<PositionIndex>(0)) = 2 * x + 0.3 * y;
                    position.mem(element, ddc::DiscreteElement<PositionIndex>(1)) = 3 * y + 0.2 * z;
                    position.mem(element, ddc::DiscreteElement<PositionIndex>(2)) = 0.1 * x + 4 * z;
                    for (std::size_t component = 0; component < 3; ++component) {
                        physical_velocity.mem(element, ddc::DiscreteElement<One>(component))
                                = component + 1;
                        physical_input.mem(element, ddc::DiscreteElement<Two>(component))
                                = component + 4;
                    }
                });
        sil::exterior::VertexFormProjectionField const project(position);
        project(Exec(), velocity, physical_velocity);
        project(Exec(), input, physical_input);
        sil::exterior::OperatorActingOnCochainInPlaceOfForm<
                sil::exterior::InteriorProduct<sil::tensor::upper_t<One>, Two>> {}(
                Exec(),
                output,
                std::tuple {velocity, input},
                std::tuple {reconstructed_velocity, reconstructed_input},
                transformed,
                sil::exterior::VertexFormReconstructionField(
                        position,
                        sil::tensor::PeriodicExtrapolationRule {}),
                project,
                inverse_metric);
        double const error = ddc::parallel_transform_reduce(
                Exec(),
                output.domain(),
                0.0,
                ddc::reducer::sum<double>(),
                KOKKOS_LAMBDA(FullDomain<One>::discrete_element_type element) {
                    // i_u alpha=(-23,-14,17), evaluated on the three skew edges.
                    double const expected = element.uid<One>() == 0
                                                    ? -44.3
                                                    : (element.uid<One>() == 1 ? -48.9 : 65.2);
                    double const difference = output.mem(element) - expected;
                    return difference * difference;
                });
        EXPECT_LT(error, 1.e-20);
    }
};

TEST(InteriorProduct, ConstantPhysicalFormsOnSkewGeometry)
{
    SkewGeometryCheck::run();
}

struct DirectFormCheck
{
    static void run()
    {
        Kokkos::View<double*, Memory> result("direct_form_contraction", 3);
        Kokkos::parallel_for(
                "direct_form_contraction",
                1,
                KOKKOS_LAMBDA(int) {
                    std::array<double, 3> vector_storage {2.0, -3.0, 5.0};
                    std::array<double, 3> form_storage {7.0, 11.0, 13.0};
                    using Insertion = sil::tensor::upper_t<One>;
                    using Output = sil::tensor::Covariant<Other>;
                    sil::tensor::Tensor const vector(
                            ddc::ChunkSpan<
                                    double,
                                    ddc::DiscreteDomain<Insertion>,
                                    Kokkos::layout_right,
                                    Memory>(
                                    vector_storage.data(),
                                    sil::tensor::TensorAccessor<Insertion>::domain()));
                    sil::exterior::Form const form(
                            sil::tensor::Tensor(
                                    ddc::ChunkSpan<
                                            double,
                                            ddc::DiscreteDomain<Two>,
                                            Kokkos::layout_right,
                                            Memory>(
                                            form_storage.data(),
                                            sil::tensor::TensorAccessor<Two>::domain())));
                    sil::exterior::Form const output(
                            sil::tensor::Tensor(
                                    ddc::ChunkSpan<
                                            double,
                                            ddc::DiscreteDomain<Output>,
                                            Kokkos::layout_right,
                                            Memory>(
                                            result.data(),
                                            sil::tensor::TensorAccessor<Output>::domain())));
                    sil::exterior::InteriorProduct<Insertion, Two> {}(output, vector, form);
                });
        Kokkos::View<double*, Kokkos::HostSpace> const host
                = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), result);
        EXPECT_DOUBLE_EQ(host(0), -34.0);
        EXPECT_DOUBLE_EQ(host(1), -51.0);
        EXPECT_DOUBLE_EQ(host(2), -17.0);
    }
};

struct OwningFormCheck
{
    static void run()
    {
        Allocation<One> allocation = allocate<One>(domain());
        Field<One> const source(allocation.span_view());
        fill_form(source);
        Kokkos::View<double*, Memory> result("owning_form_copy", 2);
        Domain::discrete_element_type const element = domain().front();
        Kokkos::parallel_for(
                "owning_form_copy",
                1,
                KOKKOS_LAMBDA(int) {
                    // A sampled tensor owns its coefficients; copying a Form must
                    // retain that ownership and rebind its view to the new storage.
                    sil::exterior::Form const sample(
                            sil::tensor::PeriodicExtrapolationRule {}(source, element));
                    auto copy = sample;
                    ddc::DiscreteElement<One> const component(0);
                    double const original = sample.mem(component);
                    copy.mem(component) = original + 7.0;
                    result(0) = sample.mem(component) - original;
                    result(1) = copy.mem(component) - original;
                });
        Kokkos::View<double*, Kokkos::HostSpace> const host
                = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), result);
        EXPECT_DOUBLE_EQ(host(0), 0.0);
        EXPECT_DOUBLE_EQ(host(1), 7.0);
    }
};

TEST(Form, SampleCopyOwnsItsCoefficients)
{
    OwningFormCheck::run();
}

TEST(Form, RejectsTooManyArrayCoefficients)
{
    EXPECT_DEATH(
            ([] {
                sil::exterior::Form<std::array<double, 1>> const value {1.0, 2.0};
                (void)value;
            }()),
            "Too many differential-form coefficients");
}

TEST(InteriorProduct, PointwiseFormsUseContravariantInsertion)
{
    DirectFormCheck::run();
}

TEST(InteriorProduct, ConstantOneForm)
{
    constant_contraction<One, Scalar>();
}
TEST(InteriorProduct, ConstantTwoForm)
{
    constant_contraction<Two, One>();
}
TEST(InteriorProduct, ConstantThreeForm)
{
    constant_contraction<Three, Two>();
}
TEST(InteriorProduct, SkewPairingAtPeriodicSeam)
{
    skew_pairing();
}
TEST(InteriorProduct, AdvectedShearConvergesForBothDirections)
{
    for (double stream : {-1.0, 1.0}) {
        double const coarse = shear_error(16, stream), fine = shear_error(32, stream);
        EXPECT_GT(coarse, 0);
        EXPECT_LT(fine, 0.26 * coarse);
    }
}
} // namespace
