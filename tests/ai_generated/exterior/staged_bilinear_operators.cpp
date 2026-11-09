// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#include <gtest/gtest.h>

#include "local_staging_fixture.hpp"

namespace {
using namespace local_staging_test;

template <std::size_t Radius>
struct Product
{
    using staged_storage_type = sil::exterior::BilinearSpatialOperatorStorage<
            Product,
            Direction,
            One,
            Scalar,
            One,
            Radius,
            Radius,
            Domain,
            Memory>;
    using Coefficients
            = sil::exterior::BilinearStencil<Direction, One, Scalar, One, Radius, Radius>;

    template <class A, class B>
    KOKKOS_FUNCTION static Coefficients value(
            double factor,
            A const&,
            B const&,
            Domain::discrete_element_type element,
            bool complete = true)
    {
        Coefficients result;
        result.complete = complete;
        for (std::size_t target = 0; target < 3; ++target) {
            // First: +x (point 2), second: -y (point 3); local: both center.
            std::size_t const a = (Radius == 0 ? 0 : 2) * 3 + (target + 1) % 3;
            std::size_t const b = Radius == 0 ? 0 : 3;
            result.weights[(target * Coefficients::first_size + a) * Coefficients::second_size + b]
                    = factor + element.template uid<GridZ>();
        }
        return result;
    }
};
using A = sil::exterior::Cochain<Field<One>>;
using B = sil::exterior::Cochain<Field<Scalar>>;
static_assert(
        sil::exterior::
                BilinearStagedOperatorConcept<sil::exterior::StagedOperator<Product<1>>, A, A, B>);
static_assert(
        !sil::exterior::LinearStagedOperatorConcept<sil::exterior::StagedOperator<Product<1>>, A>);
static_assert(!sil::exterior::BilinearStagedOperatorConcept<
              sil::exterior::StagedOperator<Product<1>>,
              A,
              A,
              int>);

template <std::size_t Radius>
void check_product()
{
    Allocation<One> aa = allocate<One>(domain()), oo = allocate<One>(domain());
    Allocation<Scalar> bb = allocate<Scalar>(domain());
    Field<One> a(aa), out(oo);
    Field<Scalar> b(bb);
    double factor = 2.0;
    sil::exterior::StagedOperator<Product<Radius>> const staged(Exec(), A(a), B(b), factor, true);
    factor = 17.0; // Cached geometry must remain unchanged.
    for (int pass = 0; pass < 2; ++pass) {
        fill_form(a, 0.4 + pass);
        fill_form(b, 0.7 - pass);
        staged(A(out), A(a), B(b));
        if constexpr (Radius == 0) {
            auto point_alloc = allocate<One>(domain());
            Field<One> point(point_alloc);
            ddc::parallel_for_each(
                    Exec(),
                    domain(),
                    KOKKOS_LAMBDA(Domain::discrete_element_type element) {
                        staged(sil::exterior::Cochain(point[element]),
                               sil::exterior::Cochain(a[element]),
                               sil::exterior::Cochain(b[element]),
                               element);
                    });
            auto const host_point = Kokkos::create_mirror_view_and_copy(
                    Kokkos::HostSpace(),
                    point_alloc.allocation_kokkos_view());
            auto const host_field = Kokkos::
                    create_mirror_view_and_copy(Kokkos::HostSpace(), oo.allocation_kokkos_view());
            for (int i = 0; i < 2; ++i)
                for (int j = 0; j < 2; ++j)
                    for (int k = 0; k < 2; ++k)
                        for (int c = 0; c < 3; ++c)
                            EXPECT_NEAR(host_point(i, j, k, c), host_field(i, j, k, c), 1e-12);
        }
        auto host_a = Kokkos::
                create_mirror_view_and_copy(Kokkos::HostSpace(), aa.allocation_kokkos_view());
        auto host_b = Kokkos::
                create_mirror_view_and_copy(Kokkos::HostSpace(), bb.allocation_kokkos_view());
        auto host_out = Kokkos::
                create_mirror_view_and_copy(Kokkos::HostSpace(), oo.allocation_kokkos_view());
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j)
                for (int k = 0; k < 2; ++k)
                    for (int c = 0; c < 3; ++c)
                        EXPECT_NEAR(
                                host_out(i, j, k, c),
                                (3.0 + k) * host_a(Radius == 0 ? i : (i + 1) % 2, j, k, (c + 1) % 3)
                                        * host_b(i, Radius == 0 ? j : (j + 1) % 2, k, 0),
                                1e-12);
    }
    EXPECT_THROW(
            (sil::exterior::StagedOperator<Product<Radius>>(Exec(), A(a), B(b), factor, false)),
            std::invalid_argument);
    auto wrong_alloc = allocate<Scalar>(halo());
    EXPECT_THROW(staged(A(out), A(a), B(Field<Scalar>(wrong_alloc))), std::invalid_argument);
}
TEST(StagedBilinearOperators, LocalDynamicOperands)
{
    check_product<0>();
}
TEST(StagedBilinearOperators, IndependentSpatialSampling)
{
    check_product<1>();
}
} // namespace

#include <similie/exterior/interior_product/staged_vertex_form_composition.hpp>

namespace {
struct ComposedInteriorChecks
{
    static void check()
    {
        auto positions = allocate<PositionIndex>(halo());
        auto metrics = allocate<MetricIndex>(domain());
        auto velocities = allocate<One>(domain());
        auto inputs = allocate<Two>(domain());
        auto result_alloc = allocate<One>(domain()), reference_alloc = allocate<One>(domain()),
             velocity_form_alloc = allocate<One>(domain()),
             transformed_alloc = allocate<One>(domain());
        auto input_form_alloc = allocate<Two>(domain());
        Field<PositionIndex> position(positions);
        Field<MetricIndex> metric(metrics);
        Field<One> velocity(velocities), result(result_alloc), reference(reference_alloc),
                velocity_form(velocity_form_alloc), transformed(transformed_alloc);
        Field<Two> input(inputs), input_form(input_form_alloc);
        fill_geometry(position, true);
        auto const inverse = sil::tensor::relabelize_indices_of<
                ddc::to_type_seq_t<typename Field<MetricIndex>::accessor_t::natural_domain_t>,
                sil::tensor::upper_t<ddc::to_type_seq_t<
                        typename Field<MetricIndex>::accessor_t::natural_domain_t>>>(metric);
        using Operation = sil::exterior::InteriorProduct<sil::tensor::Contravariant<Other>, Two>;
        using Composed = sil::exterior::VertexFormBilinearComposition<
                Operation,
                One,
                Two,
                One,
                Domain,
                Memory,
                decltype(inverse)>;
        sil::exterior::StagedOperator<Composed> const
                staged(Exec(),
                       A(result),
                       A(velocity),
                       sil::exterior::Cochain<Field<Two>>(input),
                       position,
                       inverse);
        for (int pass = 0; pass < 2; ++pass) {
            fill_form(velocity, 0.2 + pass);
            fill_form(input, 0.8 - 0.1 * pass);
            sil::exterior::OperatorActingOnCochainInPlaceOfForm<Operation> {}(
                    Exec(),
                    A(reference),
                    std::tuple {A(velocity), sil::exterior::Cochain<Field<Two>>(input)},
                    std::
                            tuple {sil::exterior::Form(velocity_form),
                                   sil::exterior::Form(input_form)},
                    sil::exterior::Form(transformed),
                    sil::exterior::VertexFormReconstructionField(
                            position,
                            sil::tensor::PeriodicExtrapolationRule {}),
                    sil::exterior::VertexFormProjectionField(
                            position,
                            sil::tensor::PeriodicExtrapolationRule {}),
                    inverse);
            staged(A(result), A(velocity), sil::exterior::Cochain<Field<Two>>(input));
            auto host = Kokkos::create_mirror_view_and_copy(
                    Kokkos::HostSpace(),
                    result_alloc.allocation_kokkos_view());
            auto expected = Kokkos::create_mirror_view_and_copy(
                    Kokkos::HostSpace(),
                    reference_alloc.allocation_kokkos_view());
            for (int i = 0; i < 2; ++i)
                for (int j = 0; j < 2; ++j)
                    for (int k = 0; k < 2; ++k)
                        for (int c = 0; c < 3; ++c)
                            EXPECT_NEAR(host(i, j, k, c), expected(i, j, k, c), 1e-11);
        }
    }
};
TEST(StagedBilinearOperators, ComposedInteriorProduct)
{
    ComposedInteriorChecks::check();
}
} // namespace
