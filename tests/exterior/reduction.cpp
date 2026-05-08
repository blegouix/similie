// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: MIT

#include <ddc/ddc.hpp>

#include <gtest/gtest.h>

#include "exterior.hpp"

struct X
{
};

struct Y
{
};

struct DDimX : ddc::UniformPointSampling<X>
{
};

struct DDimY : ddc::UniformPointSampling<Y>
{
};

struct Mu2 : sil::tensor::TensorNaturalIndex<X, Y>
{
};

struct Nu2 : sil::tensor::TensorNaturalIndex<X, Y>
{
};

using PositionIndex = sil::tensor::Contravariant<Mu2>;
using ScalarIndex = sil::tensor::Covariant<sil::tensor::ScalarIndex>;
using OneFormIndex = sil::tensor::Covariant<Mu2>;
using TwoFormIndex = sil::tensor::
        TensorAntisymmetricIndex<sil::tensor::Covariant<Mu2>, sil::tensor::Covariant<Nu2>>;

template <class FormIndex, class SetupForm, class CheckValue>
void run_reduction_test(SetupForm&& setup_form, CheckValue&& check_value)
{
    ddc::Coordinate<X, Y> lower_bounds(0., 0.);
    ddc::Coordinate<X, Y> upper_bounds(2., 2.);
    ddc::DiscreteVector<DDimX, DDimY> nb_cells(2, 2);
    ddc::DiscreteDomain<DDimX> mesh_x = ddc::init_discrete_space<DDimX>(DDimX::init<DDimX>(
            ddc::Coordinate<X>(lower_bounds),
            ddc::Coordinate<X>(upper_bounds),
            ddc::DiscreteVector<DDimX>(nb_cells)));
    ddc::DiscreteDomain<DDimY> mesh_y = ddc::init_discrete_space<DDimY>(DDimY::init<DDimY>(
            ddc::Coordinate<Y>(lower_bounds),
            ddc::Coordinate<Y>(upper_bounds),
            ddc::DiscreteVector<DDimY>(nb_cells)));
    ddc::DiscreteDomain<DDimX, DDimY> mesh_xy(mesh_x, mesh_y);

    [[maybe_unused]] sil::tensor::TensorAccessor<PositionIndex> position_accessor;
    ddc::DiscreteDomain<DDimX, DDimY, PositionIndex>
            position_dom(mesh_xy, position_accessor.domain());
    ddc::Chunk position_alloc(position_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor position(position_alloc);

    ddc::host_for_each(mesh_xy, [&](ddc::DiscreteElement<DDimX, DDimY> elem) {
        double const xi = static_cast<double>(ddc::coordinate(ddc::DiscreteElement<DDimX>(elem)));
        double const eta = static_cast<double>(ddc::coordinate(ddc::DiscreteElement<DDimY>(elem)));
        position(elem, position.accessor().template access_element<X>()) = xi + eta;
        position(elem, position.accessor().template access_element<Y>()) = 2. * eta;
    });

    [[maybe_unused]] sil::tensor::TensorAccessor<FormIndex> form_accessor;
    ddc::DiscreteDomain<DDimX, DDimY, FormIndex> form_dom(mesh_xy, form_accessor.domain());
    ddc::Chunk form_alloc(form_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor form(form_alloc);
    setup_form(form);

    [[maybe_unused]] sil::tensor::TensorAccessor<FormIndex> reduced_accessor;
    ddc::DiscreteDomain<DDimX, DDimY, FormIndex> reduced_dom(mesh_xy, reduced_accessor.domain());
    ddc::Chunk reduced_alloc(reduced_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor reduced(reduced_alloc);

    using IndexSeq
            = ddc::to_type_seq_t<typename sil::tensor::TensorAccessor<FormIndex>::natural_domain_t>;
    ddc::host_for_each(mesh_xy, [&](ddc::DiscreteElement<DDimX, DDimY> elem) {
        sil::exterior::Reduction<IndexSeq, decltype(position), ddc::DiscreteElement<DDimX, DDimY>>::
                run(reduced[elem], form[elem], position, elem);
    });

    [[maybe_unused]] sil::tensor::tensor_accessor_for_domain_t<
            sil::exterior::reduction_domain_t<IndexSeq>> reduction_accessor;
    ddc::cartesian_prod_t<decltype(mesh_xy), sil::exterior::reduction_domain_t<IndexSeq>>
            reduction_dom(mesh_xy, reduction_accessor.domain());
    ddc::Chunk reduction_alloc(reduction_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor reduction_operator(reduction_alloc);
    sil::exterior::fill_reduction_operator<
            IndexSeq>(Kokkos::DefaultHostExecutionSpace(), reduction_operator, position);

    ddc::host_for_each(
            mesh_xy.remove_last(ddc::DiscreteVector<DDimX, DDimY>(1, 1)),
            [&](ddc::DiscreteElement<DDimX, DDimY> elem) {
                check_value(reduced, elem);
                ddc::device_for_each(reduction_operator.accessor().domain(), [&](auto mem_elem) {
                    double const expected_operator_value = sil::exterior::Reduction<
                            IndexSeq,
                            decltype(position),
                            ddc::DiscreteElement<DDimX, DDimY>>::
                            value(position,
                                  elem,
                                  reduction_operator.accessor().canonical_natural_element(
                                          mem_elem));
                    EXPECT_DOUBLE_EQ(
                            reduction_operator.mem(elem, mem_elem),
                            expected_operator_value);
                });
            });

    ddc::detail::g_discrete_space_dual<DDimX>.reset();
    ddc::detail::g_discrete_space_dual<DDimY>.reset();
}

TEST(Reduction, Scalar)
{
    run_reduction_test<ScalarIndex>(
            [](auto form) { ddc::parallel_fill(form, 3.5); },
            [](auto reduced, auto elem) {
                EXPECT_DOUBLE_EQ(reduced(elem, ddc::DiscreteElement<ScalarIndex>(0)), 3.5);
            });
}

TEST(Reduction, OneForm)
{
    run_reduction_test<OneFormIndex>(
            [](auto form) {
                ddc::host_for_each(form.non_indices_domain(), [&](auto elem) {
                    form(elem, form.accessor().template access_element<X>()) = 3.;
                    form(elem, form.accessor().template access_element<Y>()) = -1.;
                });
            },
            [](auto reduced, auto elem) {
                EXPECT_DOUBLE_EQ(
                        reduced(elem, reduced.accessor().template access_element<X>()),
                        6.);
                EXPECT_DOUBLE_EQ(
                        reduced(elem, reduced.accessor().template access_element<Y>()),
                        2.);
            });
}

TEST(Reduction, TwoForm)
{
    run_reduction_test<TwoFormIndex>(
            [](auto form) {
                ddc::host_for_each(form.non_indices_domain(), [&](auto elem) {
                    form(elem, form.accessor().template access_element<X, Y>()) = 4.;
                });
            },
            [](auto reduced, auto elem) {
                EXPECT_DOUBLE_EQ(
                        reduced(elem, reduced.accessor().template access_element<X, Y>()),
                        32.);
            });
}
