// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

#include <ddc/ddc.hpp>

#include <gtest/gtest.h>
#include <similie/tensor/identity_tensor.hpp>

#include "exterior.hpp"

template <class... CDim>
using MetricIndex = sil::tensor::TensorIdentityIndex<
        sil::tensor::Covariant<sil::tensor::MetricIndex1<CDim...>>,
        sil::tensor::Covariant<sil::tensor::MetricIndex2<CDim...>>>;

template <class InterestIndex, class Index, class... DDim>
static auto test_derivative(auto potential)
{
    using PositionIndex = sil::tensor::Contravariant<
            sil::tensor::TensorNaturalIndex<typename DDim::continuous_dimension_type...>>;
    using MetricTensorIndex = MetricIndex<typename DDim::continuous_dimension_type...>;
    using TensorType = decltype(potential);
    using LaplacianDummyIndex2 = sil::tensor::Covariant<
            sil::exterior::LaplacianDummy2<sil::tensor::uncharacterize_t<InterestIndex>>>;

    // Allocate and instantiate an identity metric tensor field.
    [[maybe_unused]] sil::tensor::TensorAccessor<MetricTensorIndex> metric_accessor;
    ddc::DiscreteDomain<DDim..., MetricTensorIndex>
            metric_dom(potential.non_indices_domain(), metric_accessor.domain());
    ddc::Chunk metric_alloc(metric_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor metric(metric_alloc);

    // Allocate and instantiate the position field from the mesh coordinates.
    [[maybe_unused]] sil::tensor::TensorAccessor<PositionIndex> position_accessor;
    ddc::DiscreteDomain<DDim..., PositionIndex>
            position_dom(potential.non_indices_domain(), position_accessor.domain());
    ddc::Chunk position_alloc(position_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor position(position_alloc);

    ddc::host_for_each(
            potential.non_indices_domain(),
            [&](typename decltype(potential)::non_indices_domain_t::discrete_element_type elem) {
                ((position
                          .mem(elem,
                               position_accessor.template access_element<
                                       typename DDim::continuous_dimension_type>())
                  = static_cast<double>(ddc::coordinate(ddc::DiscreteElement<DDim>(elem)))),
                 ...);
            });

    // Allocate and compute Laplacian
    [[maybe_unused]] sil::tensor::TensorAccessor<Index> laplacian_accessor;
    ddc::DiscreteDomain<DDim..., Index> laplacian_dom(
            potential.non_indices_domain().remove_last(
                    ddc::DiscreteVector<DDim...>(ddc::DiscreteVector<DDim>(1)...)),
            laplacian_accessor.domain());
    ddc::Chunk laplacian_alloc(laplacian_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor laplacian(laplacian_alloc);

    if constexpr (Index::rank() == 0) {
        using DerivativeIndex = sil::exterior::coboundary_index_t<LaplacianDummyIndex2, Index>;
        using DerivativeMuUpSeq = sil::tensor::upper_t<
                ddc::to_type_seq_t<sil::tensor::natural_domain_t<DerivativeIndex>>>;
        using DerivativeNuLowSeq = typename sil::exterior::detail::CodifferentialDummyIndexSeq<
                LaplacianDummyIndex2::size() - DerivativeIndex::rank(),
                LaplacianDummyIndex2>::type;
        using DerivativeRhoLowSeq = ddc::
                type_seq_merge_t<ddc::detail::TypeSeq<LaplacianDummyIndex2>, DerivativeNuLowSeq>;
        using DerivativeRhoUpSeq = sil::tensor::upper_t<DerivativeRhoLowSeq>;
        using DerivativeSigmaLowSeq = ddc::type_seq_remove_t<
                sil::tensor::lower_t<DerivativeMuUpSeq>,
                ddc::detail::TypeSeq<LaplacianDummyIndex2>>;
        using DerivativeDualIndex = sil::misc::
                convert_type_seq_to_t<sil::tensor::TensorAntisymmetricIndex, DerivativeNuLowSeq>;

        [[maybe_unused]] sil::tensor::tensor_accessor_for_domain_t<
                sil::exterior::hodge_star_domain_t<DerivativeMuUpSeq, DerivativeNuLowSeq>>
                derivative_hodge_star_accessor;
        ddc::cartesian_prod_t<
                typename TensorType::non_indices_domain_t,
                sil::exterior::hodge_star_domain_t<DerivativeMuUpSeq, DerivativeNuLowSeq>>
                derivative_hodge_star_dom(
                        metric.non_indices_domain(),
                        derivative_hodge_star_accessor.domain());
        ddc::Chunk derivative_hodge_star_alloc(
                derivative_hodge_star_dom,
                ddc::HostAllocator<double>());
        sil::tensor::Tensor derivative_hodge_star(derivative_hodge_star_alloc);

        [[maybe_unused]] sil::tensor::tensor_accessor_for_domain_t<
                sil::exterior::hodge_star_domain_t<DerivativeRhoUpSeq, DerivativeSigmaLowSeq>>
                dual_derivative_hodge_star_accessor;
        ddc::cartesian_prod_t<
                typename TensorType::non_indices_domain_t,
                sil::exterior::hodge_star_domain_t<DerivativeRhoUpSeq, DerivativeSigmaLowSeq>>
                dual_derivative_hodge_star_dom(
                        metric.non_indices_domain(),
                        dual_derivative_hodge_star_accessor.domain());
        ddc::Chunk dual_derivative_hodge_star_alloc(
                dual_derivative_hodge_star_dom,
                ddc::HostAllocator<double>());
        sil::tensor::Tensor dual_derivative_hodge_star(dual_derivative_hodge_star_alloc);

        [[maybe_unused]] sil::tensor::TensorAccessor<DerivativeDualIndex>
                derivative_dual_tensor_accessor;
        ddc::cartesian_prod_t<
                typename TensorType::non_indices_domain_t,
                ddc::DiscreteDomain<DerivativeDualIndex>>
                derivative_dual_tensor_dom(
                        potential.non_indices_domain(),
                        derivative_dual_tensor_accessor.domain());
        ddc::Chunk derivative_dual_tensor_alloc(
                derivative_dual_tensor_dom,
                ddc::HostAllocator<double>());
        sil::tensor::Tensor derivative_dual_tensor_buffer(derivative_dual_tensor_alloc);

        sil::exterior::fill_discrete_hodge_star<DerivativeMuUpSeq, DerivativeNuLowSeq>(
                Kokkos::DefaultHostExecutionSpace(),
                derivative_hodge_star,
                metric,
                position);
        sil::exterior::fill_discrete_hodge_star<DerivativeRhoUpSeq, DerivativeSigmaLowSeq>(
                Kokkos::DefaultHostExecutionSpace(),
                dual_derivative_hodge_star,
                metric,
                position);

        sil::exterior::laplacian<MetricTensorIndex, InterestIndex, Index>(
                Kokkos::DefaultHostExecutionSpace(),
                laplacian,
                potential,
                derivative_hodge_star,
                dual_derivative_hodge_star,
                derivative_dual_tensor_buffer);
    } else if constexpr (Index::rank() < InterestIndex::size()) {
        using DerivativeIndex = sil::exterior::coboundary_index_t<LaplacianDummyIndex2, Index>;
        using DerivativeMuUpSeq = sil::tensor::upper_t<
                ddc::to_type_seq_t<sil::tensor::natural_domain_t<DerivativeIndex>>>;
        using DerivativeNuLowSeq = typename sil::exterior::detail::CodifferentialDummyIndexSeq<
                LaplacianDummyIndex2::size() - DerivativeIndex::rank(),
                LaplacianDummyIndex2>::type;
        using DerivativeRhoLowSeq = ddc::
                type_seq_merge_t<ddc::detail::TypeSeq<LaplacianDummyIndex2>, DerivativeNuLowSeq>;
        using DerivativeRhoUpSeq = sil::tensor::upper_t<DerivativeRhoLowSeq>;
        using DerivativeSigmaLowSeq = ddc::type_seq_remove_t<
                sil::tensor::lower_t<DerivativeMuUpSeq>,
                ddc::detail::TypeSeq<LaplacianDummyIndex2>>;
        using DerivativeDualIndex = sil::misc::
                convert_type_seq_to_t<sil::tensor::TensorAntisymmetricIndex, DerivativeNuLowSeq>;
        using MuUpSeq
                = sil::tensor::upper_t<ddc::to_type_seq_t<sil::tensor::natural_domain_t<Index>>>;
        using NuLowSeq = typename sil::exterior::detail::CodifferentialDummyIndexSeq<
                InterestIndex::size() - Index::rank(),
                InterestIndex>::type;
        using RhoLowSeq = ddc::type_seq_merge_t<ddc::detail::TypeSeq<InterestIndex>, NuLowSeq>;
        using RhoUpSeq = sil::tensor::upper_t<RhoLowSeq>;
        using SigmaLowSeq = ddc::type_seq_remove_t<
                sil::tensor::lower_t<MuUpSeq>,
                ddc::detail::TypeSeq<InterestIndex>>;
        using DualIndex
                = sil::misc::convert_type_seq_to_t<sil::tensor::TensorAntisymmetricIndex, NuLowSeq>;
        using CodifferentialIndex = sil::exterior::codifferential_index_t<InterestIndex, Index>;

        [[maybe_unused]] sil::tensor::tensor_accessor_for_domain_t<
                sil::exterior::hodge_star_domain_t<DerivativeMuUpSeq, DerivativeNuLowSeq>>
                derivative_hodge_star_accessor;
        ddc::cartesian_prod_t<
                typename TensorType::non_indices_domain_t,
                sil::exterior::hodge_star_domain_t<DerivativeMuUpSeq, DerivativeNuLowSeq>>
                derivative_hodge_star_dom(
                        metric.non_indices_domain(),
                        derivative_hodge_star_accessor.domain());
        ddc::Chunk derivative_hodge_star_alloc(
                derivative_hodge_star_dom,
                ddc::HostAllocator<double>());
        sil::tensor::Tensor derivative_hodge_star(derivative_hodge_star_alloc);

        [[maybe_unused]] sil::tensor::tensor_accessor_for_domain_t<
                sil::exterior::hodge_star_domain_t<DerivativeRhoUpSeq, DerivativeSigmaLowSeq>>
                dual_derivative_hodge_star_accessor;
        ddc::cartesian_prod_t<
                typename TensorType::non_indices_domain_t,
                sil::exterior::hodge_star_domain_t<DerivativeRhoUpSeq, DerivativeSigmaLowSeq>>
                dual_derivative_hodge_star_dom(
                        metric.non_indices_domain(),
                        dual_derivative_hodge_star_accessor.domain());
        ddc::Chunk dual_derivative_hodge_star_alloc(
                dual_derivative_hodge_star_dom,
                ddc::HostAllocator<double>());
        sil::tensor::Tensor dual_derivative_hodge_star(dual_derivative_hodge_star_alloc);

        [[maybe_unused]] sil::tensor::TensorAccessor<DerivativeDualIndex>
                derivative_dual_tensor_accessor;
        ddc::cartesian_prod_t<
                typename TensorType::non_indices_domain_t,
                ddc::DiscreteDomain<DerivativeDualIndex>>
                derivative_dual_tensor_dom(
                        potential.non_indices_domain(),
                        derivative_dual_tensor_accessor.domain());
        ddc::Chunk derivative_dual_tensor_alloc(
                derivative_dual_tensor_dom,
                ddc::HostAllocator<double>());
        sil::tensor::Tensor derivative_dual_tensor_buffer(derivative_dual_tensor_alloc);

        [[maybe_unused]] sil::tensor::tensor_accessor_for_domain_t<
                sil::exterior::hodge_star_domain_t<MuUpSeq, NuLowSeq>> hodge_star_accessor;
        ddc::cartesian_prod_t<
                typename TensorType::non_indices_domain_t,
                sil::exterior::hodge_star_domain_t<MuUpSeq, NuLowSeq>>
                hodge_star_dom(metric.non_indices_domain(), hodge_star_accessor.domain());
        ddc::Chunk hodge_star_alloc(hodge_star_dom, ddc::HostAllocator<double>());
        sil::tensor::Tensor hodge_star(hodge_star_alloc);

        [[maybe_unused]] sil::tensor::tensor_accessor_for_domain_t<
                sil::exterior::hodge_star_domain_t<RhoUpSeq, SigmaLowSeq>> dual_hodge_star_accessor;
        ddc::cartesian_prod_t<
                typename TensorType::non_indices_domain_t,
                sil::exterior::hodge_star_domain_t<RhoUpSeq, SigmaLowSeq>>
                dual_hodge_star_dom(metric.non_indices_domain(), dual_hodge_star_accessor.domain());
        ddc::Chunk dual_hodge_star_alloc(dual_hodge_star_dom, ddc::HostAllocator<double>());
        sil::tensor::Tensor dual_hodge_star(dual_hodge_star_alloc);

        [[maybe_unused]] sil::tensor::TensorAccessor<DualIndex> dual_tensor_accessor;
        ddc::cartesian_prod_t<
                typename TensorType::non_indices_domain_t,
                ddc::DiscreteDomain<DualIndex>>
                dual_tensor_dom(potential.non_indices_domain(), dual_tensor_accessor.domain());
        ddc::Chunk dual_tensor_alloc(dual_tensor_dom, ddc::HostAllocator<double>());
        sil::tensor::Tensor dual_tensor_buffer(dual_tensor_alloc);

        [[maybe_unused]] sil::tensor::TensorAccessor<CodifferentialIndex> codifferential_accessor;
        ddc::DiscreteDomain<DDim..., CodifferentialIndex> codifferential_dom(
                potential.non_indices_domain(),
                codifferential_accessor.domain());
        ddc::Chunk codifferential_alloc(codifferential_dom, ddc::HostAllocator<double>());
        sil::tensor::Tensor codifferential_tensor_buffer(codifferential_alloc);

        ddc::Chunk coboundary_of_codifferential_alloc(laplacian_dom, ddc::HostAllocator<double>());
        sil::tensor::Tensor coboundary_of_codifferential_buffer(coboundary_of_codifferential_alloc);

        sil::exterior::fill_discrete_hodge_star<DerivativeMuUpSeq, DerivativeNuLowSeq>(
                Kokkos::DefaultHostExecutionSpace(),
                derivative_hodge_star,
                metric,
                position);
        sil::exterior::fill_discrete_hodge_star<DerivativeRhoUpSeq, DerivativeSigmaLowSeq>(
                Kokkos::DefaultHostExecutionSpace(),
                dual_derivative_hodge_star,
                metric,
                position);
        sil::exterior::fill_discrete_hodge_star<
                MuUpSeq,
                NuLowSeq>(Kokkos::DefaultHostExecutionSpace(), hodge_star, metric, position);
        sil::exterior::fill_discrete_hodge_star<RhoUpSeq, SigmaLowSeq>(
                Kokkos::DefaultHostExecutionSpace(),
                dual_hodge_star,
                metric,
                position);

        sil::exterior::laplacian<MetricTensorIndex, InterestIndex, Index>(
                Kokkos::DefaultHostExecutionSpace(),
                laplacian,
                potential,
                derivative_hodge_star,
                dual_derivative_hodge_star,
                derivative_dual_tensor_buffer,
                hodge_star,
                dual_hodge_star,
                dual_tensor_buffer,
                codifferential_tensor_buffer,
                coboundary_of_codifferential_buffer);
    } else if constexpr (Index::rank() == InterestIndex::size()) {
        using MuUpSeq
                = sil::tensor::upper_t<ddc::to_type_seq_t<sil::tensor::natural_domain_t<Index>>>;
        using NuLowSeq = typename sil::exterior::detail::CodifferentialDummyIndexSeq<
                InterestIndex::size() - Index::rank(),
                InterestIndex>::type;
        using RhoLowSeq = ddc::type_seq_merge_t<ddc::detail::TypeSeq<InterestIndex>, NuLowSeq>;
        using RhoUpSeq = sil::tensor::upper_t<RhoLowSeq>;
        using SigmaLowSeq = ddc::type_seq_remove_t<
                sil::tensor::lower_t<MuUpSeq>,
                ddc::detail::TypeSeq<InterestIndex>>;
        using DualIndex
                = sil::misc::convert_type_seq_to_t<sil::tensor::TensorAntisymmetricIndex, NuLowSeq>;
        using CodifferentialIndex = sil::exterior::codifferential_index_t<InterestIndex, Index>;

        [[maybe_unused]] sil::tensor::tensor_accessor_for_domain_t<
                sil::exterior::hodge_star_domain_t<MuUpSeq, NuLowSeq>> hodge_star_accessor;
        ddc::cartesian_prod_t<
                typename TensorType::non_indices_domain_t,
                sil::exterior::hodge_star_domain_t<MuUpSeq, NuLowSeq>>
                hodge_star_dom(metric.non_indices_domain(), hodge_star_accessor.domain());
        ddc::Chunk hodge_star_alloc(hodge_star_dom, ddc::HostAllocator<double>());
        sil::tensor::Tensor hodge_star(hodge_star_alloc);

        [[maybe_unused]] sil::tensor::tensor_accessor_for_domain_t<
                sil::exterior::hodge_star_domain_t<RhoUpSeq, SigmaLowSeq>> dual_hodge_star_accessor;
        ddc::cartesian_prod_t<
                typename TensorType::non_indices_domain_t,
                sil::exterior::hodge_star_domain_t<RhoUpSeq, SigmaLowSeq>>
                dual_hodge_star_dom(metric.non_indices_domain(), dual_hodge_star_accessor.domain());
        ddc::Chunk dual_hodge_star_alloc(dual_hodge_star_dom, ddc::HostAllocator<double>());
        sil::tensor::Tensor dual_hodge_star(dual_hodge_star_alloc);

        [[maybe_unused]] sil::tensor::TensorAccessor<DualIndex> dual_tensor_accessor;
        ddc::cartesian_prod_t<
                typename TensorType::non_indices_domain_t,
                ddc::DiscreteDomain<DualIndex>>
                dual_tensor_dom(potential.non_indices_domain(), dual_tensor_accessor.domain());
        ddc::Chunk dual_tensor_alloc(dual_tensor_dom, ddc::HostAllocator<double>());
        sil::tensor::Tensor dual_tensor_buffer(dual_tensor_alloc);

        [[maybe_unused]] sil::tensor::TensorAccessor<CodifferentialIndex> codifferential_accessor;
        ddc::DiscreteDomain<DDim..., CodifferentialIndex> codifferential_dom(
                potential.non_indices_domain(),
                codifferential_accessor.domain());
        ddc::Chunk codifferential_alloc(codifferential_dom, ddc::HostAllocator<double>());
        sil::tensor::Tensor codifferential_tensor_buffer(codifferential_alloc);

        sil::exterior::fill_discrete_hodge_star<
                MuUpSeq,
                NuLowSeq>(Kokkos::DefaultHostExecutionSpace(), hodge_star, metric, position);
        sil::exterior::fill_discrete_hodge_star<RhoUpSeq, SigmaLowSeq>(
                Kokkos::DefaultHostExecutionSpace(),
                dual_hodge_star,
                metric,
                position);

        sil::exterior::laplacian<MetricTensorIndex, InterestIndex, Index>(
                Kokkos::DefaultHostExecutionSpace(),
                laplacian,
                potential,
                hodge_star,
                dual_hodge_star,
                dual_tensor_buffer,
                codifferential_tensor_buffer);
    }
    Kokkos::fence();

    return std::make_pair(std::move(laplacian_alloc), laplacian);
}

struct X
{
};

struct Y
{
};

struct Z
{
};

struct DDimX : ddc::UniformPointSampling<X>
{
};

struct DDimY : ddc::UniformPointSampling<Y>
{
};

struct DDimZ : ddc::UniformPointSampling<Z>
{
};

struct Mu1 : sil::tensor::TensorNaturalIndex<X>
{
};

TEST(Laplacian, 1D0Form)
{
    ddc::Coordinate<X> lower_bounds(-5.);
    ddc::Coordinate<X> upper_bounds(5.);
    ddc::DiscreteVector<DDimX> nb_cells(1000);
    ddc::DiscreteDomain<DDimX> mesh_x = ddc::init_discrete_space<DDimX>(
            DDimX::init<DDimX>(lower_bounds, upper_bounds, nb_cells));

    // Potential
    [[maybe_unused]] sil::tensor::TensorAccessor<sil::tensor::Covariant<sil::tensor::ScalarIndex>>
            potential_accessor;
    ddc::DiscreteDomain<DDimX, sil::tensor::Covariant<sil::tensor::ScalarIndex>>
            potential_dom(mesh_x, potential_accessor.domain());
    ddc::Chunk potential_alloc(potential_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor potential(potential_alloc);

    double const R = 2.;
    double const alpha = 0.5;
    ddc::host_for_each(
            potential.domain(),
            [&](ddc::DiscreteElement<DDimX, sil::tensor::Covariant<sil::tensor::ScalarIndex>>
                        elem) {
                double const r = Kokkos::abs(
                        static_cast<double>(ddc::coordinate(ddc::DiscreteElement<DDimX>(elem))));
                if (r <= R) {
                    potential.mem(elem) = -alpha * r * r;
                } else {
                    potential.mem(elem) = -alpha * R * (2 * r - R);
                }
            });


    auto [alloc, laplacian] = test_derivative<
            sil::tensor::Covariant<Mu1>,
            sil::tensor::Covariant<sil::tensor::ScalarIndex>,
            DDimX>(potential);

    ddc::host_for_each(
            laplacian.template domain<DDimX>()
                    .remove_first(ddc::DiscreteVector<DDimX>(1))
                    .remove_last(ddc::DiscreteVector<DDimX>(1)),
            [&](ddc::DiscreteElement<DDimX> elem) {
                double const value = laplacian(
                        elem,
                        ddc::DiscreteElement<sil::tensor::Covariant<sil::tensor::ScalarIndex>> {0});
                if (ddc::coordinate(elem) < -1.2 * R || ddc::coordinate(elem) > 1.2 * R) {
                    EXPECT_NEAR(value, 0., 1e-2);
                } else if (ddc::coordinate(elem) > -.8 * R && ddc::coordinate(elem) < .8 * R) {
                    EXPECT_NEAR(value, 1., 1e-2);
                }
            });
    ddc::detail::g_discrete_space_dual<DDimX>.reset();
}

TEST(Laplacian, 1D1Form)
{
    ddc::Coordinate<X> lower_bounds(-5.);
    ddc::Coordinate<X> upper_bounds(5.);
    ddc::DiscreteVector<DDimX> nb_cells(1000);
    ddc::DiscreteDomain<DDimX> mesh_x = ddc::init_discrete_space<DDimX>(
            DDimX::init<DDimX>(lower_bounds, upper_bounds, nb_cells));

    // Potential
    [[maybe_unused]] sil::tensor::TensorAccessor<sil::tensor::Covariant<Mu1>> potential_accessor;
    ddc::DiscreteDomain<DDimX, sil::tensor::Covariant<Mu1>>
            potential_dom(mesh_x, potential_accessor.domain());
    ddc::Chunk potential_alloc(potential_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor potential(potential_alloc);

    double const R = 2.;
    double const alpha = 0.5;
    ddc::host_for_each(
            potential.domain(),
            [&](ddc::DiscreteElement<DDimX, sil::tensor::Covariant<Mu1>> elem) {
                double const r = Kokkos::abs(
                        static_cast<double>(ddc::coordinate(ddc::DiscreteElement<DDimX>(elem))));
                if (r <= R) {
                    potential.mem(elem) = -alpha * r * r;
                } else {
                    potential.mem(elem) = -alpha * R * (2 * r - R);
                }
            });


    auto [alloc, laplacian]
            = test_derivative<sil::tensor::Covariant<Mu1>, sil::tensor::Covariant<Mu1>, DDimX>(
                    potential);

    ddc::host_for_each(
            laplacian.template domain<DDimX>()
                    .remove_first(ddc::DiscreteVector<DDimX>(1))
                    .remove_last(ddc::DiscreteVector<DDimX>(1)),
            [&](ddc::DiscreteElement<DDimX> elem) {
                double const value = laplacian(elem, laplacian.accessor().access_element<X>());
                if (ddc::coordinate(elem) < -1.2 * R || ddc::coordinate(elem) > 1.2 * R) {
                    EXPECT_NEAR(value, 0., 1e-2);
                } else if (ddc::coordinate(elem) > -.8 * R && ddc::coordinate(elem) < .8 * R) {
                    EXPECT_NEAR(value, 1., 1e-2);
                }
            });
    ddc::detail::g_discrete_space_dual<DDimX>.reset();
}

struct Mu2 : sil::tensor::TensorNaturalIndex<X, Y>
{
};

TEST(Laplacian, 2D0Form)
{
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

    // Potential
    [[maybe_unused]] sil::tensor::TensorAccessor<sil::tensor::Covariant<sil::tensor::ScalarIndex>>
            potential_accessor;
    ddc::DiscreteDomain<DDimX, DDimY, sil::tensor::Covariant<sil::tensor::ScalarIndex>>
            potential_dom(mesh_xy, potential_accessor.domain());
    ddc::Chunk potential_alloc(potential_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor potential(potential_alloc);

    double const R = 2.;
    double const alpha = 0.25;
    ddc::host_for_each(
            potential.domain(),
            [&](ddc::DiscreteElement<DDimX, DDimY, sil::tensor::Covariant<sil::tensor::ScalarIndex>>
                        elem) {
                double const r = Kokkos::sqrt(
                        static_cast<double>(
                                ddc::coordinate(ddc::DiscreteElement<DDimX>(elem))
                                * ddc::coordinate(ddc::DiscreteElement<DDimX>(elem)))
                        + static_cast<double>(
                                ddc::coordinate(ddc::DiscreteElement<DDimY>(elem))
                                * ddc::coordinate(ddc::DiscreteElement<DDimY>(elem))));
                if (r <= R) {
                    potential.mem(elem) = -alpha * r * r;
                } else {
                    potential.mem(elem) = alpha * R * R * (2 * Kokkos::log(R / r) - 1);
                }
            });


    auto [alloc, laplacian] = test_derivative<
            sil::tensor::Covariant<Mu2>,
            sil::tensor::Covariant<sil::tensor::ScalarIndex>,
            DDimX,
            DDimY>(potential);

    ddc::host_for_each(
            laplacian.template domain<DDimX>()
                    .remove_first(ddc::DiscreteVector<DDimX>(1))
                    .remove_last(ddc::DiscreteVector<DDimX>(1)),
            [&](ddc::DiscreteElement<DDimX> elem) {
                double const value = laplacian(
                        elem,
                        ddc::DiscreteElement<DDimY> {
                                static_cast<std::size_t>(nb_cells.template get<DDimY>()) / 2},
                        ddc::DiscreteElement<sil::tensor::Covariant<sil::tensor::ScalarIndex>> {0});
                if (ddc::coordinate(elem) < -1.2 * R || ddc::coordinate(elem) > 1.2 * R) {
                    EXPECT_NEAR(value, 0., .5);
                } else if (ddc::coordinate(elem) > -.8 * R && ddc::coordinate(elem) < .8 * R) {
                    EXPECT_NEAR(value, 1., .5);
                }
            });
    ddc::detail::g_discrete_space_dual<DDimX>.reset();
    ddc::detail::g_discrete_space_dual<DDimY>.reset();
    ddc::detail::g_discrete_space_dual<DDimZ>.reset();
}

TEST(Laplacian, 2D1Form)
{
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

    // Potential
    [[maybe_unused]] sil::tensor::TensorAccessor<sil::tensor::Covariant<Mu2>> potential_accessor;
    ddc::DiscreteDomain<DDimX, DDimY, sil::tensor::Covariant<Mu2>>
            potential_dom(mesh_xy, potential_accessor.domain());
    ddc::Chunk potential_alloc(potential_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor potential(potential_alloc);

    double const R = 2.;
    ddc::host_for_each(
            potential.non_indices_domain(),
            [&](ddc::DiscreteElement<DDimX, DDimY> elem) {
                double const x_coord = ddc::coordinate(ddc::DiscreteElement<DDimX>(elem));
                double const y_coord = ddc::coordinate(ddc::DiscreteElement<DDimY>(elem));
                double const r = Kokkos::sqrt(
                        static_cast<double>(x_coord * x_coord)
                        + static_cast<double>(y_coord * y_coord));
                if (r <= R) {
                    double const factor = r / 3. - R / 2.;
                    potential.mem(elem, potential_accessor.access_element<X>()) = y_coord * factor;
                    potential.mem(elem, potential_accessor.access_element<Y>()) = -x_coord * factor;
                } else {
                    double const factor = -R * R * R / (6. * r * r);
                    potential.mem(elem, potential_accessor.access_element<X>()) = y_coord * factor;
                    potential.mem(elem, potential_accessor.access_element<Y>()) = -x_coord * factor;
                }
            });


    auto [alloc, laplacian] = test_derivative<
            sil::tensor::Covariant<Mu2>,
            sil::tensor::Covariant<Mu2>,
            DDimX,
            DDimY>(potential);

    ddc::host_for_each(
            laplacian.template domain<DDimX>()
                    .remove_first(ddc::DiscreteVector<DDimX>(1))
                    .remove_last(ddc::DiscreteVector<DDimX>(1)),
            [&](ddc::DiscreteElement<DDimX> elem) {
                double const value = laplacian(
                        elem,
                        ddc::DiscreteElement<DDimY> {
                                static_cast<std::size_t>(nb_cells.template get<DDimY>()) / 2},
                        laplacian.accessor().access_element<Y>());
                if (ddc::coordinate(elem) < -1.2 * R || ddc::coordinate(elem) > 1.2 * R) {
                    EXPECT_NEAR(value, 0., .5);
                } else if (ddc::coordinate(elem) > -.8 * R && ddc::coordinate(elem) < -.2 * R) {
                    EXPECT_NEAR(value, -1., .5);
                } else if (ddc::coordinate(elem) > .2 * R && ddc::coordinate(elem) < .8 * R) {
                    EXPECT_NEAR(value, 1., .5);
                }
            });
    ddc::detail::g_discrete_space_dual<DDimX>.reset();
    ddc::detail::g_discrete_space_dual<DDimY>.reset();
    ddc::detail::g_discrete_space_dual<DDimZ>.reset();
}

struct Mu3 : sil::tensor::TensorNaturalIndex<X, Y, Z>
{
};

struct Nu3 : sil::tensor::TensorNaturalIndex<X, Y, Z>
{
};

using TwoForm3 = sil::tensor::
        TensorAntisymmetricIndex<sil::tensor::Covariant<Nu3>, sil::tensor::Covariant<Mu3>>;

TEST(Laplacian, 3D1Form)
{
    ddc::Coordinate<X, Y, Z> lower_bounds(-5., -5., -5.);
    ddc::Coordinate<X, Y, Z> upper_bounds(5., 5., 5.);
    ddc::DiscreteVector<DDimX, DDimY, DDimZ> nb_cells(21, 21, 21);
    ddc::DiscreteDomain<DDimX> mesh_x = ddc::init_discrete_space<DDimX>(DDimX::init<DDimX>(
            ddc::Coordinate<X>(lower_bounds),
            ddc::Coordinate<X>(upper_bounds),
            ddc::DiscreteVector<DDimX>(nb_cells)));
    ddc::DiscreteDomain<DDimY> mesh_y = ddc::init_discrete_space<DDimY>(DDimY::init<DDimY>(
            ddc::Coordinate<Y>(lower_bounds),
            ddc::Coordinate<Y>(upper_bounds),
            ddc::DiscreteVector<DDimY>(nb_cells)));
    ddc::DiscreteDomain<DDimZ> mesh_z = ddc::init_discrete_space<DDimZ>(DDimZ::init<DDimZ>(
            ddc::Coordinate<Z>(lower_bounds),
            ddc::Coordinate<Z>(upper_bounds),
            ddc::DiscreteVector<DDimZ>(nb_cells)));
    ddc::DiscreteDomain<DDimX, DDimY, DDimZ> mesh_xyz(mesh_x, mesh_y, mesh_z);

    // Potential
    [[maybe_unused]] sil::tensor::TensorAccessor<sil::tensor::Covariant<Mu3>> potential_accessor;
    ddc::DiscreteDomain<DDimX, DDimY, DDimZ, sil::tensor::Covariant<Mu3>>
            potential_dom(mesh_xyz, potential_accessor.domain());
    ddc::Chunk potential_alloc(potential_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor potential(potential_alloc);

    double const R = 2.;
    double const alpha = 0.2;
    ddc::host_for_each(
            potential.non_indices_domain(),
            [&](ddc::DiscreteElement<DDimX, DDimY, DDimZ> elem) {
                double const r = Kokkos::sqrt(
                        static_cast<double>(
                                ddc::coordinate(ddc::DiscreteElement<DDimX>(elem))
                                * ddc::coordinate(ddc::DiscreteElement<DDimX>(elem)))
                        + static_cast<double>(
                                ddc::coordinate(ddc::DiscreteElement<DDimY>(elem))
                                * ddc::coordinate(ddc::DiscreteElement<DDimY>(elem)))
                        + static_cast<double>(
                                ddc::coordinate(ddc::DiscreteElement<DDimZ>(elem))
                                * ddc::coordinate(ddc::DiscreteElement<DDimZ>(elem))));
                double const theta = Kokkos::
                        atan2(ddc::coordinate(ddc::DiscreteElement<DDimY>(elem)),
                              ddc::coordinate(ddc::DiscreteElement<DDimX>(elem)));
                if (r <= R) {
                    potential.mem(elem, potential_accessor.access_element<X>())
                            = alpha * r * r * Kokkos::sin(theta);
                    potential.mem(elem, potential_accessor.access_element<Y>())
                            = -alpha * r * r * Kokkos::cos(theta);
                } else {
                    potential.mem(elem, potential_accessor.access_element<X>())
                            = alpha * R * R * R * R / (r * r) * Kokkos::sin(theta);
                    potential.mem(elem, potential_accessor.access_element<Y>())
                            = -alpha * R * R * R * R / (r * r) * Kokkos::cos(theta);
                }
                potential.mem(elem, potential_accessor.access_element<Z>()) = 0.;
            });


    auto [alloc, laplacian] = test_derivative<
            sil::tensor::Covariant<Mu3>,
            sil::tensor::Covariant<Mu3>,
            DDimX,
            DDimY,
            DDimZ>(potential);

    ddc::host_for_each(
            laplacian.template domain<DDimX>()
                    .remove_first(ddc::DiscreteVector<DDimX>(1))
                    .remove_last(ddc::DiscreteVector<DDimX>(1)),
            [&](ddc::DiscreteElement<DDimX> elem) {
                double const value = laplacian(
                        elem,
                        ddc::DiscreteElement<DDimY> {
                                static_cast<std::size_t>(nb_cells.template get<DDimY>()) / 2},
                        ddc::DiscreteElement<DDimZ> {
                                static_cast<std::size_t>(nb_cells.template get<DDimZ>()) / 2},

                        laplacian.accessor().access_element<Y>());
                if (ddc::coordinate(elem) < -1.3 * R || ddc::coordinate(elem) > 1.3 * R) {
                    EXPECT_NEAR(value, 0., .5);
                } else if (ddc::coordinate(elem) > -.7 * R && ddc::coordinate(elem) < -.3 * R) {
                    EXPECT_NEAR(value, -1., .5);
                } else if (ddc::coordinate(elem) > .3 * R && ddc::coordinate(elem) < .7 * R) {
                    EXPECT_NEAR(value, 1., .5);
                }
            });
    ddc::detail::g_discrete_space_dual<DDimX>.reset();
    ddc::detail::g_discrete_space_dual<DDimY>.reset();
    ddc::detail::g_discrete_space_dual<DDimZ>.reset();
}

TEST(Laplacian, 3D2Form)
{
    ddc::Coordinate<X, Y, Z> lower_bounds(-5., -5., -5.);
    ddc::Coordinate<X, Y, Z> upper_bounds(5., 5., 5.);
    ddc::DiscreteVector<DDimX, DDimY, DDimZ> nb_cells(21, 21, 21);
    ddc::DiscreteDomain<DDimX> mesh_x = ddc::init_discrete_space<DDimX>(DDimX::init<DDimX>(
            ddc::Coordinate<X>(lower_bounds),
            ddc::Coordinate<X>(upper_bounds),
            ddc::DiscreteVector<DDimX>(nb_cells)));
    ddc::DiscreteDomain<DDimY> mesh_y = ddc::init_discrete_space<DDimY>(DDimY::init<DDimY>(
            ddc::Coordinate<Y>(lower_bounds),
            ddc::Coordinate<Y>(upper_bounds),
            ddc::DiscreteVector<DDimY>(nb_cells)));
    ddc::DiscreteDomain<DDimZ> mesh_z = ddc::init_discrete_space<DDimZ>(DDimZ::init<DDimZ>(
            ddc::Coordinate<Z>(lower_bounds),
            ddc::Coordinate<Z>(upper_bounds),
            ddc::DiscreteVector<DDimZ>(nb_cells)));
    ddc::DiscreteDomain<DDimX, DDimY, DDimZ> mesh_xyz(mesh_x, mesh_y, mesh_z);

    [[maybe_unused]] sil::tensor::TensorAccessor<TwoForm3> potential_accessor;
    ddc::DiscreteDomain<DDimX, DDimY, DDimZ, TwoForm3>
            potential_dom(mesh_xyz, potential_accessor.domain());
    ddc::Chunk potential_alloc(potential_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor potential(potential_alloc);

    double const R = 2.;
    double const alpha = 0.2;
    ddc::host_for_each(
            potential.non_indices_domain(),
            [&](ddc::DiscreteElement<DDimX, DDimY, DDimZ> elem) {
                double const r = Kokkos::sqrt(
                        static_cast<double>(
                                ddc::coordinate(ddc::DiscreteElement<DDimX>(elem))
                                * ddc::coordinate(ddc::DiscreteElement<DDimX>(elem)))
                        + static_cast<double>(
                                ddc::coordinate(ddc::DiscreteElement<DDimY>(elem))
                                * ddc::coordinate(ddc::DiscreteElement<DDimY>(elem)))
                        + static_cast<double>(
                                ddc::coordinate(ddc::DiscreteElement<DDimZ>(elem))
                                * ddc::coordinate(ddc::DiscreteElement<DDimZ>(elem))));
                double const theta = Kokkos::
                        atan2(ddc::coordinate(ddc::DiscreteElement<DDimY>(elem)),
                              ddc::coordinate(ddc::DiscreteElement<DDimX>(elem)));
                if (r <= R) {
                    potential(elem, potential_accessor.access_element<Y, Z>())
                            = alpha * r * r * Kokkos::sin(theta);
                    potential(elem, potential_accessor.access_element<X, Z>())
                            = alpha * r * r * Kokkos::cos(theta);
                } else {
                    potential(elem, potential_accessor.access_element<Y, Z>())
                            = alpha * R * R * R * R / (r * r) * Kokkos::sin(theta);
                    potential(elem, potential_accessor.access_element<X, Z>())
                            = alpha * R * R * R * R / (r * r) * Kokkos::cos(theta);
                }
                potential(elem, potential_accessor.access_element<X, Y>()) = 0.;
            });


    auto [alloc, laplacian]
            = test_derivative<sil::tensor::Covariant<Nu3>, TwoForm3, DDimX, DDimY, DDimZ>(
                    potential);

    ddc::host_for_each(
            laplacian.template domain<DDimX>()
                    .remove_first(ddc::DiscreteVector<DDimX>(1))
                    .remove_last(ddc::DiscreteVector<DDimX>(1)),
            [&](ddc::DiscreteElement<DDimX> elem) {
                double const value = laplacian(
                        elem,
                        ddc::DiscreteElement<DDimY> {
                                static_cast<std::size_t>(nb_cells.template get<DDimY>()) / 2},
                        ddc::DiscreteElement<DDimZ> {
                                static_cast<std::size_t>(nb_cells.template get<DDimZ>()) / 2},

                        laplacian.accessor().access_element<X, Z>());
                if (ddc::coordinate(elem) < -1.3 * R || ddc::coordinate(elem) > 1.3 * R) {
                    EXPECT_NEAR(value, 0., .5);
                } else if (ddc::coordinate(elem) > -.7 * R && ddc::coordinate(elem) < -.3 * R) {
                    EXPECT_NEAR(value, 1., .5);
                } else if (ddc::coordinate(elem) > .3 * R && ddc::coordinate(elem) < .7 * R) {
                    EXPECT_NEAR(value, -1., .5);
                }
            });
    ddc::detail::g_discrete_space_dual<DDimX>.reset();
    ddc::detail::g_discrete_space_dual<DDimY>.reset();
    ddc::detail::g_discrete_space_dual<DDimZ>.reset();
}
