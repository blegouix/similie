// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: MIT

#include <ddc/ddc.hpp>

#include <gtest/gtest.h>
#include <similie/tensor/symmetric_tensor.hpp>

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

template <class... CDim>
using MetricIndex = sil::tensor::TensorSymmetricIndex<
        sil::tensor::Covariant<sil::tensor::MetricIndex1<CDim...>>,
        sil::tensor::Covariant<sil::tensor::MetricIndex2<CDim...>>>;

template <class CodifferentialCallable>
void run_codifferential_test(CodifferentialCallable&& codifferential_callable)
{
    using PositionIndex = sil::tensor::Contravariant<sil::tensor::TensorNaturalIndex<X, Y>>;
    using TensorIndex = sil::tensor::Covariant<Mu2>;
    using OutputIndex = sil::tensor::Covariant<sil::tensor::ScalarIndex>;

    ddc::Coordinate<X, Y> lower_bounds(-5., -5.);
    ddc::Coordinate<X, Y> upper_bounds(5., 5.);
    ddc::DiscreteVector<DDimX, DDimY> nb_cells(31, 31);
    ddc::DiscreteDomain<DDimX> mesh_x = ddc::init_discrete_space<DDimX>(DDimX::init<DDimX>(
            ddc::Coordinate<X>(lower_bounds),
            ddc::Coordinate<X>(upper_bounds),
            ddc::DiscreteVector<DDimX>(nb_cells)));
    ddc::DiscreteDomain<DDimY> mesh_y = ddc::init_discrete_space<DDimY>(DDimY::init<DDimY>(
            ddc::Coordinate<Y>(lower_bounds),
            ddc::Coordinate<Y>(upper_bounds),
            ddc::DiscreteVector<DDimY>(nb_cells)));
    ddc::DiscreteDomain<DDimX, DDimY> mesh_xy(mesh_x, mesh_y);
    double const dx = (ddc::get<X>(upper_bounds) - ddc::get<X>(lower_bounds))
                      / static_cast<double>(ddc::get<DDimX>(nb_cells));
    double const dy = (ddc::get<Y>(upper_bounds) - ddc::get<Y>(lower_bounds))
                      / static_cast<double>(ddc::get<DDimY>(nb_cells));
    double const expected_codifferential
            = -(static_cast<double>(ddc::get<DDimX>(nb_cells) - 1))
                      / static_cast<double>(ddc::get<DDimX>(nb_cells))
              - (static_cast<double>(ddc::get<DDimY>(nb_cells) - 1))
                        / static_cast<double>(ddc::get<DDimY>(nb_cells));

    [[maybe_unused]] sil::tensor::TensorAccessor<TensorIndex> tensor_accessor;
    ddc::DiscreteDomain<DDimX, DDimY, TensorIndex> tensor_dom(mesh_xy, tensor_accessor.domain());
    ddc::Chunk tensor_alloc(tensor_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor tensor(tensor_alloc);

    [[maybe_unused]] sil::tensor::TensorAccessor<MetricIndex<X, Y>> metric_accessor;
    ddc::DiscreteDomain<DDimX, DDimY, MetricIndex<X, Y>>
            metric_dom(mesh_xy, metric_accessor.domain());
    ddc::Chunk metric_alloc(metric_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor metric(metric_alloc);

    [[maybe_unused]] sil::tensor::TensorAccessor<PositionIndex> position_accessor;
    ddc::DiscreteDomain<DDimX, DDimY, PositionIndex>
            position_dom(mesh_xy, position_accessor.domain());
    ddc::Chunk position_alloc(position_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor position(position_alloc);

    [[maybe_unused]] sil::tensor::TensorAccessor<OutputIndex> codifferential_accessor;
    ddc::DiscreteDomain<DDimX, DDimY, OutputIndex>
            codifferential_dom(mesh_xy, codifferential_accessor.domain());
    ddc::Chunk codifferential_alloc(codifferential_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor codifferential_tensor(codifferential_alloc);

    ddc::host_for_each(mesh_xy, [&](ddc::DiscreteElement<DDimX, DDimY> elem) {
        double const x = static_cast<double>(ddc::coordinate(ddc::DiscreteElement<DDimX>(elem)));
        double const y = static_cast<double>(ddc::coordinate(ddc::DiscreteElement<DDimY>(elem)));
        tensor(elem, tensor.accessor().template access_element<X>()) = x * dx;
        tensor(elem, tensor.accessor().template access_element<Y>()) = y * dy;
        position(elem, position.accessor().template access_element<X>()) = x;
        position(elem, position.accessor().template access_element<Y>()) = y;
        metric(elem, metric.accessor().template access_element<X, X>()) = 1.;
        metric(elem, metric.accessor().template access_element<X, Y>()) = 0.;
        metric(elem, metric.accessor().template access_element<Y, Y>()) = 1.;
    });

    codifferential_callable(codifferential_tensor, tensor, metric, position);

    ddc::host_for_each(
            mesh_xy.remove_first(ddc::DiscreteVector<DDimX, DDimY>(1, 1))
                    .remove_last(ddc::DiscreteVector<DDimX, DDimY>(1, 1)),
            [&](ddc::DiscreteElement<DDimX, DDimY> elem) {
                // The current evaluator clamps outside samples on the domain boundary.
                // TODO this is an assumption on boundary condition (free boundary), needs to be generalized.
                EXPECT_NEAR(
                        codifferential_tensor(elem, ddc::DiscreteElement<OutputIndex>(0)),
                        expected_codifferential,
                        1e-12);
            });

    ddc::detail::g_discrete_space_dual<DDimX>.reset();
    ddc::detail::g_discrete_space_dual<DDimY>.reset();
}

TEST(Codifferential, GeometryBasedImplementation)
{
    run_codifferential_test(
            [](auto codifferential_tensor, auto tensor, auto metric, auto position) {
                sil::exterior::codifferential<
                        MetricIndex<X, Y>,
                        sil::tensor::Covariant<Mu2>,
                        sil::tensor::Covariant<Mu2>>(
                        Kokkos::DefaultHostExecutionSpace(),
                        codifferential_tensor,
                        tensor,
                        metric,
                        position);
            });
}

TEST(Codifferential, PrefilledHodgeStarImplementation)
{
    run_codifferential_test([](auto codifferential_tensor,
                               auto tensor,
                               auto metric,
                               auto position) {
        using TensorType = decltype(tensor);
        using MetricType = decltype(metric);
        using MuUpSeq = sil::tensor::upper_t<
                ddc::to_type_seq_t<sil::tensor::natural_domain_t<sil::tensor::Covariant<Mu2>>>>;
        using NuLowSeq = ddc::detail::TypeSeq<sil::tensor::Covariant<
                sil::exterior::detail::CodifferentialDummyIndex<0, sil::tensor::Covariant<Mu2>>>>;
        using RhoLowSeq = ddc::
                type_seq_merge_t<ddc::detail::TypeSeq<sil::tensor::Covariant<Mu2>>, NuLowSeq>;
        using RhoUpSeq = sil::tensor::upper_t<RhoLowSeq>;
        using SigmaLowSeq = ddc::type_seq_remove_t<
                sil::tensor::lower_t<MuUpSeq>,
                ddc::detail::TypeSeq<sil::tensor::Covariant<Mu2>>>;
        using DualIndex
                = sil::misc::convert_type_seq_to_t<sil::tensor::TensorAntisymmetricIndex, NuLowSeq>;
        using DualCodifferentialIndex = sil::misc::
                convert_type_seq_to_t<sil::tensor::TensorAntisymmetricIndex, RhoLowSeq>;

        [[maybe_unused]] sil::tensor::tensor_accessor_for_domain_t<
                sil::exterior::hodge_star_domain_t<MuUpSeq, NuLowSeq>> hodge_star_accessor;
        ddc::cartesian_prod_t<
                typename MetricType::non_indices_domain_t,
                sil::exterior::hodge_star_domain_t<MuUpSeq, NuLowSeq>>
                hodge_star_dom(metric.non_indices_domain(), hodge_star_accessor.domain());
        ddc::Chunk hodge_star_alloc(hodge_star_dom, ddc::HostAllocator<double>());
        sil::tensor::Tensor hodge_star(hodge_star_alloc);

        [[maybe_unused]] sil::tensor::tensor_accessor_for_domain_t<
                sil::exterior::hodge_star_domain_t<RhoUpSeq, SigmaLowSeq>> dual_hodge_star_accessor;
        ddc::cartesian_prod_t<
                typename MetricType::non_indices_domain_t,
                sil::exterior::hodge_star_domain_t<RhoUpSeq, SigmaLowSeq>>
                dual_hodge_star_dom(metric.non_indices_domain(), dual_hodge_star_accessor.domain());
        ddc::Chunk dual_hodge_star_alloc(dual_hodge_star_dom, ddc::HostAllocator<double>());
        sil::tensor::Tensor dual_hodge_star(dual_hodge_star_alloc);

        [[maybe_unused]] sil::tensor::TensorAccessor<DualIndex> dual_tensor_accessor;
        ddc::cartesian_prod_t<
                typename TensorType::non_indices_domain_t,
                ddc::DiscreteDomain<DualIndex>>
                dual_tensor_dom(tensor.non_indices_domain(), dual_tensor_accessor.domain());
        ddc::Chunk dual_tensor_alloc(dual_tensor_dom, ddc::HostAllocator<double>());
        sil::tensor::Tensor dual_tensor(dual_tensor_alloc);

        [[maybe_unused]] sil::tensor::TensorAccessor<DualCodifferentialIndex>
                dual_codifferential_accessor;
        ddc::cartesian_prod_t<
                typename TensorType::non_indices_domain_t,
                ddc::DiscreteDomain<DualCodifferentialIndex>>
                dual_codifferential_dom(
                        tensor.non_indices_domain(),
                        dual_codifferential_accessor.domain());
        ddc::Chunk dual_codifferential_alloc(dual_codifferential_dom, ddc::HostAllocator<double>());
        sil::tensor::Tensor dual_codifferential(dual_codifferential_alloc);

        sil::exterior::fill_discrete_hodge_star<
                MuUpSeq,
                NuLowSeq>(Kokkos::DefaultHostExecutionSpace(), hodge_star, metric, position);
        sil::exterior::fill_discrete_hodge_star<RhoUpSeq, SigmaLowSeq>(
                Kokkos::DefaultHostExecutionSpace(),
                dual_hodge_star,
                metric,
                position);

        sil::exterior::codifferential<
                MetricIndex<X, Y>,
                sil::tensor::Covariant<Mu2>,
                sil::tensor::Covariant<Mu2>>(
                Kokkos::DefaultHostExecutionSpace(),
                codifferential_tensor,
                tensor,
                hodge_star,
                dual_hodge_star,
                dual_tensor,
                dual_codifferential);
    });
}
