// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#include <cmath>
#include <concepts>

#include <ddc/ddc.hpp>

#include <gtest/gtest.h>
#include <similie/tensor/external_domain_extrapolation_rule.hpp>
#include <similie/tensor/scalar_extrapolation_rules.hpp>
#include <similie/tensor/tensor_with_extrapolation_rules.hpp>

#include "tensor.hpp"

struct T
{
};

struct X
{
};

struct Y
{
};

struct Z
{
};

struct Alpha : sil::tensor::TensorNaturalIndex<X, Y, Z>
{
};

struct Beta : sil::tensor::TensorNaturalIndex<X, Y, Z>
{
};

struct Gamma : sil::tensor::TensorNaturalIndex<X, Y, Z>
{
};

struct Delta : sil::tensor::TensorNaturalIndex<X, Y, Z>
{
};

struct Mu : sil::tensor::TensorNaturalIndex<T, X, Y, Z>
{
};

struct Nu : sil::tensor::TensorNaturalIndex<T, X, Y, Z>
{
};

using AntisymIndex = sil::tensor::TensorAntisymmetricIndex<Mu, Nu>;

using FullIndex = sil::tensor::TensorFullIndex<Alpha, Nu>;
using DiagIndex = sil::tensor::TensorDiagonalIndex<Mu, Nu>;
using SymIndex = sil::tensor::TensorSymmetricIndex<Mu, Nu>;
using IdIndex = sil::tensor::TensorIdentityIndex<Mu, Nu>;
using LorentzianSignIndex
        = sil::tensor::TensorLorentzianSignIndex<std::integral_constant<std::size_t, 2>, Mu, Nu>;
using LeviCivitaIndex = sil::tensor::TensorLeviCivitaIndex<Alpha, Beta, Gamma>;

namespace {
struct SignedGridX
{
    using continuous_dimension_type = X;
};
struct SignedGridY
{
    using continuous_dimension_type = Y;
};
struct SignedSample
{
};

void test_signed_tensor_sampling(std::size_t origin)
{
    using Scalar = sil::tensor::Covariant<sil::tensor::ScalarIndex>;
    [[maybe_unused]] sil::tensor::TensorAccessor<Scalar> accessor;
    ddc::DiscreteDomain<SignedGridX, SignedGridY> const
            grid(ddc::DiscreteElement<SignedGridX, SignedGridY>(origin, origin),
                 ddc::DiscreteVector<SignedGridX, SignedGridY>(4, 4));
    ddc::DiscreteDomain<SignedGridX, SignedGridY> const donor_grid(
            ddc::DiscreteElement<SignedGridX, SignedGridY>(20, 30),
            ddc::DiscreteVector<SignedGridX, SignedGridY>(4, 4));
    ddc::Chunk field_alloc(
            ddc::DiscreteDomain<SignedGridX, SignedGridY, Scalar>(grid, accessor.domain()),
            ddc::DeviceAllocator<double>());
    ddc::Chunk donor_alloc(
            ddc::DiscreteDomain<SignedGridX, SignedGridY, Scalar>(donor_grid, accessor.domain()),
            ddc::DeviceAllocator<double>());
    sil::tensor::Tensor field(field_alloc);
    sil::tensor::Tensor donor(donor_alloc);
    ddc::parallel_for_each(
            Kokkos::DefaultExecutionSpace(),
            grid,
            KOKKOS_LAMBDA(ddc::DiscreteElement<SignedGridX, SignedGridY> elem) {
                field.mem(elem, ddc::DiscreteElement<Scalar>(0))
                        = 1.0 + 2.0 * elem.uid<SignedGridX>() + 3.0 * elem.uid<SignedGridY>();
            });
    ddc::parallel_for_each(
            Kokkos::DefaultExecutionSpace(),
            donor_grid,
            KOKKOS_LAMBDA(ddc::DiscreteElement<SignedGridX, SignedGridY> elem) {
                donor.mem(elem, ddc::DiscreteElement<Scalar>(0))
                        = 1.0 + 2.0 * elem.uid<SignedGridX>() + 3.0 * elem.uid<SignedGridY>();
            });
    auto const natural = sil::tensor::make_tensor_with_extrapolation_rules(
            field,
            sil::tensor::NaturalScalarExtrapolationRule {});
    auto const clamp = sil::tensor::
            make_tensor_with_extrapolation_rules(field, sil::tensor::ClampExtrapolationRule {});
    auto const zero = sil::tensor::
            make_tensor_with_extrapolation_rules(field, sil::tensor::ZeroExtrapolationRule {});
    auto const prescribed = sil::tensor::make_tensor_with_extrapolation_rules(
            field,
            sil::tensor::PrescribedScalarExtrapolationRule {13.0});
    auto const blended = sil::tensor::make_tensor_with_extrapolation_rules(
            field,
            sil::tensor::ExtrapolationRules(
                    std::
                            pair {sil::tensor::PrescribedScalarExtrapolationRule {1.0},
                                  sil::tensor::PrescribedScalarExtrapolationRule {2.0}},
                    std::
                            pair {sil::tensor::PrescribedScalarExtrapolationRule {3.0},
                                  sil::tensor::PrescribedScalarExtrapolationRule {4.0}}));
    sil::tensor::ExternalDomainExtrapolationRule const external {
            donor,
            sil::tensor::ExternalDomainBoundaryMap<SignedGridX> {
                    sil::tensor::BoundarySide::Lower,
                    sil::tensor::BoundarySide::Upper}};
    auto const connected = sil::tensor::make_tensor_with_extrapolation_rules(
            field,
            sil::tensor::ExtrapolationRules(
                    std::pair {external, sil::tensor::ClampExtrapolationRule {}},
                    std::
                            pair {sil::tensor::PrescribedScalarExtrapolationRule {50.0},
                                  sil::tensor::ClampExtrapolationRule {}}));
    ddc::DiscreteDomain<SignedSample> const
            samples(ddc::DiscreteElement<SignedSample>(0), ddc::DiscreteVector<SignedSample>(11));
    ddc::Chunk results(samples, ddc::DeviceAllocator<double>());
    ddc::ChunkSpan const output = results.span_view();
    ddc::parallel_for_each(
            Kokkos::DefaultExecutionSpace(),
            samples,
            KOKKOS_LAMBDA(ddc::DiscreteElement<SignedSample> sample) {
                sil::misc::SignedDiscreteElement<SignedGridY, SignedGridX> const negative(-2, -1);
                ddc::DiscreteElement<Scalar> const component(0);
                sil::misc::SignedDiscreteElement<SignedGridX, SignedGridY> const inside(
                        grid.front());
                sil::misc::SignedDiscreteElement<SignedGridX, SignedGridY> const
                        outside(static_cast<std::ptrdiff_t>(origin) - 1,
                                static_cast<std::ptrdiff_t>(origin));
                switch (sample.uid<SignedSample>()) {
                case 0:
                    output(sample) = natural(negative, component);
                    break;
                case 1:
                    output(sample) = clamp.mem(negative, component);
                    break;
                case 2:
                    output(sample) = zero(negative, component);
                    break;
                case 3:
                    output(sample) = prescribed(negative, component);
                    break;
                case 4:
                    output(sample) = blended(
                            sil::misc::SignedDiscreteElement<SignedGridY, SignedGridX>(
                                    static_cast<std::ptrdiff_t>(origin) - 1,
                                    static_cast<std::ptrdiff_t>(origin) - 2),
                            component);
                    break;
                case 5:
                    output(sample) = connected(outside, component);
                    break;
                case 6:
                    output(sample) = sil::tensor::detail::contract_extrapolation_basis(
                            connected.value(field.non_indices_domain(), outside),
                            [&](auto entry) { return field.mem(entry, component); },
                            [&](std::size_t) { return connected(outside, component); });
                    break;
                case 7:
                    output(sample) = connected(
                            sil::misc::SignedDiscreteElement<SignedGridX, SignedGridY>(
                                    static_cast<std::ptrdiff_t>(origin) - 1,
                                    static_cast<std::ptrdiff_t>(origin) - 1),
                            component);
                    break;
                case 8:
                    output(sample) = natural(inside, component);
                    break;
                case 9:
                    output(sample) = sil::tensor::detail::contract_extrapolation_basis(
                            natural.value(field.non_indices_domain(), negative),
                            [](auto entry) {
                                return 10.0 * entry.template uid<SignedGridX>()
                                       + 100.0 * entry.template uid<SignedGridY>();
                            },
                            [](std::size_t) { return 0.0; });
                    break;
                case 10:
                    output(sample) = natural(
                            sil::misc::SignedDiscreteElement<
                                    SignedGridY,
                                    Scalar,
                                    SignedGridX>(-2, 0, -1));
                    break;
                }
            });
    auto host = ddc::
            create_mirror_view_and_copy(Kokkos::DefaultHostExecutionSpace(), results.span_view());
    EXPECT_DOUBLE_EQ(host(ddc::DiscreteElement<SignedSample>(0)), -7.0);
    EXPECT_DOUBLE_EQ(host(ddc::DiscreteElement<SignedSample>(1)), 1.0 + 5.0 * origin);
    EXPECT_DOUBLE_EQ(host(ddc::DiscreteElement<SignedSample>(2)), 0.0);
    EXPECT_DOUBLE_EQ(host(ddc::DiscreteElement<SignedSample>(3)), 13.0);
    EXPECT_DOUBLE_EQ(host(ddc::DiscreteElement<SignedSample>(4)), 5.0 / 3.0);
    EXPECT_DOUBLE_EQ(host(ddc::DiscreteElement<SignedSample>(5)), 135.0);
    EXPECT_DOUBLE_EQ(host(ddc::DiscreteElement<SignedSample>(6)), 135.0);
    EXPECT_DOUBLE_EQ(host(ddc::DiscreteElement<SignedSample>(7)), 92.5);
    EXPECT_DOUBLE_EQ(host(ddc::DiscreteElement<SignedSample>(8)), 1.0 + 5.0 * origin);
    EXPECT_DOUBLE_EQ(host(ddc::DiscreteElement<SignedSample>(9)), -210.0);
    EXPECT_DOUBLE_EQ(host(ddc::DiscreteElement<SignedSample>(10)), -7.0);
}
} // namespace

TEST(Tensor, SignedExtrapolationAtZeroOrigin)
{
    test_signed_tensor_sampling(0);
}

TEST(Tensor, SignedExtrapolationAtNonzeroOrigin)
{
    test_signed_tensor_sampling(7);
}

namespace {
struct NearestTensorView
{
    template <class Field, class Element>
    KOKKOS_FUNCTION auto operator()(Field const& field, Element element) const
    {
        return field[sil::misc::clamp_to_domain(field.non_indices_domain(), element)
                             .storage_element()];
    }
};
} // namespace

TEST(Tensor, WholeTensorExtrapolationAndSpatialStencil)
{
    using Vector = sil::tensor::Covariant<sil::tensor::TensorNaturalIndex<X, Y>>;
    [[maybe_unused]] sil::tensor::TensorAccessor<Vector> accessor;
    ddc::DiscreteDomain<SignedGridX, SignedGridY> const
            grid(ddc::DiscreteElement<SignedGridX, SignedGridY>(0, 0),
                 ddc::DiscreteVector<SignedGridX, SignedGridY>(3, 3));
    ddc::Chunk allocation(
            ddc::DiscreteDomain<SignedGridX, SignedGridY, Vector>(grid, accessor.domain()),
            ddc::HostAllocator<double>());
    sil::tensor::Tensor field(allocation);
    ddc::host_for_each(field.domain(), [&](auto elem) {
        double const component = elem.template uid<Vector>();
        field.mem(elem) = 5.0 * component + (component + 1.0) * elem.template uid<SignedGridX>()
                          - 3.0 * elem.template uid<SignedGridY>();
    });
    auto const view = sil::tensor::make_tensor_with_extrapolation_rules(
            field,
            sil::tensor::NaturalScalarExtrapolationRule {});
    sil::misc::SignedDiscreteElement<SignedGridX, SignedGridY> const exterior(-2, -1);
    auto const point = view[exterior];
    auto const stencil = view.value(grid, exterior);
    static_assert(
            sil::misc::
                    Specialization<std::remove_cv_t<decltype(stencil)>, sil::tensor::OwningTensor>);
    sil::misc::SignedDiscreteElement<SignedGridY, SignedGridX> const reversed(-1, -2);
    auto const reversed_point = sil::tensor::NaturalScalarExtrapolationRule {}(field, reversed);
    EXPECT_DOUBLE_EQ(reversed_point.mem(ddc::DiscreteElement<Vector>(1)), 4.0);
    sil::tensor::ExtrapolationRules const
            mixed(std::pair {NearestTensorView {}, sil::tensor::NaturalScalarExtrapolationRule {}},
                  std::
                          pair {sil::tensor::NaturalScalarExtrapolationRule {},
                                sil::tensor::NaturalScalarExtrapolationRule {}});
    EXPECT_DOUBLE_EQ(mixed(field, exterior).mem(ddc::DiscreteElement<Vector>(1)), 14.0 / 3.0);

    EXPECT_EQ(point.domain(), accessor.domain());
    EXPECT_EQ(view.domain(), field.domain());
    EXPECT_EQ(view.access_domain(), field.access_domain());
    EXPECT_EQ(view.natural_domain(), field.natural_domain());
    for (std::size_t component = 0; component < 2; ++component) {
        ddc::DiscreteElement<Vector> const index(component);
        EXPECT_DOUBLE_EQ(point.mem(index), 3.0 * component + 1.0);
        EXPECT_DOUBLE_EQ(view.mem(exterior, index), point.mem(index));
        EXPECT_DOUBLE_EQ(
                sil::tensor::detail::contract_extrapolation_basis(
                        stencil,
                        [&](auto entry) { return field.mem(entry, index); },
                        [](std::size_t) { return 0.0; }),
                point.mem(index));
    }
    EXPECT_DOUBLE_EQ(
            view
                    .get(ddc::DiscreteElement<SignedGridX, SignedGridY>(1, 2),
                         accessor.access_element<Y>()),
            1.0);
    auto const component_slice
            = view[ddc::DiscreteElement<SignedGridX, SignedGridY, Vector>(1, 2, 1)];
    EXPECT_DOUBLE_EQ(component_slice.mem(ddc::DiscreteElement<>()), 1.0);
    auto const component_field = view[ddc::DiscreteElement<Vector>(1)];
    EXPECT_EQ(component_field.domain(), grid);
    EXPECT_DOUBLE_EQ(component_field.mem(exterior), 4.0);
    auto const row = view[sil::misc::SignedDiscreteElement<SignedGridX>(-2)];
    EXPECT_DOUBLE_EQ(
            row.mem(sil::misc::SignedDiscreteElement<SignedGridY>(-1),
                    ddc::DiscreteElement<Vector>(1)),
            4.0);
    EXPECT_DOUBLE_EQ(
            row[sil::misc::SignedDiscreteElement<SignedGridY>(-1)].mem(
                    ddc::DiscreteElement<Vector>(1)),
            4.0);

    // The same domain-only weights remain valid after changing every component.
    ddc::host_for_each(field.domain(), [&](auto elem) { field.mem(elem) += 11.0; });
    EXPECT_DOUBLE_EQ(view[exterior].mem(ddc::DiscreteElement<Vector>(1)), 15.0);
    EXPECT_DOUBLE_EQ(
            sil::tensor::detail::contract_extrapolation_basis(
                    stencil,
                    [&](auto entry) { return field.mem(entry, ddc::DiscreteElement<Vector>(1)); },
                    [](std::size_t) { return 0.0; }),
            15.0);
}

TEST(Tensor, ExtrapolatedAntisymmetricTensorAccessAndSlices)
{
    [[maybe_unused]] sil::tensor::TensorAccessor<AntisymIndex> accessor;
    ddc::DiscreteDomain<SignedGridX> const
            grid(ddc::DiscreteElement<SignedGridX>(0), ddc::DiscreteVector<SignedGridX>(3));
    ddc::Chunk allocation(
            ddc::DiscreteDomain<SignedGridX, AntisymIndex>(grid, accessor.domain()),
            ddc::HostAllocator<double>());
    sil::tensor::Tensor field(allocation);
    ddc::host_for_each(field.domain(), [&](auto elem) {
        field.mem(elem)
                = 10.0 * elem.template uid<AntisymIndex>() + elem.template uid<SignedGridX>();
    });
    auto const view = sil::tensor::make_tensor_with_extrapolation_rules(
            field,
            sil::tensor::NaturalScalarExtrapolationRule {});
    sil::misc::SignedDiscreteElement<SignedGridX> const exterior(-1);
    ddc::DiscreteElement<AntisymIndex> const xy = accessor.access_element<X, Y>();
    EXPECT_DOUBLE_EQ(view.mem(exterior, ddc::DiscreteElement<AntisymIndex>(3)), 29.0);
    EXPECT_DOUBLE_EQ(view(exterior, xy), 29.0);
    EXPECT_DOUBLE_EQ(view.get(exterior, xy), 29.0);
    EXPECT_DOUBLE_EQ(view.get(exterior, accessor.access_element<Y, X>()), -29.0);
    EXPECT_DOUBLE_EQ(view.get(exterior, accessor.access_element<T, T>()), 0.0);
    auto const slice = view[xy];
    EXPECT_DOUBLE_EQ(slice.mem(exterior), 29.0);
    EXPECT_DOUBLE_EQ(slice(exterior), 29.0);
    EXPECT_DOUBLE_EQ(slice.get(exterior), 29.0);
    auto const point = view[ddc::DiscreteElement<SignedGridX, AntisymIndex>(grid.front(), xy)];
    EXPECT_DOUBLE_EQ(point.mem(ddc::DiscreteElement<>()), 30.0);
}

TEST(Tensor, ScalarStructuredIndexContract)
{
    EXPECT_EQ(sil::tensor::TensorFullIndex<>::rank(), 0);
    EXPECT_EQ(sil::tensor::TensorFullIndex<>::size(), 1);
    EXPECT_EQ(sil::tensor::TensorFullIndex<>::mem_size(), 1);
    EXPECT_EQ(sil::tensor::TensorFullIndex<>::access_size(), 1);
    EXPECT_EQ(sil::tensor::TensorFullIndex<>::access_id({}), 0);
    EXPECT_EQ(sil::tensor::TensorSymmetricIndex<>::rank(), 0);
    EXPECT_EQ(sil::tensor::TensorSymmetricIndex<>::size(), 1);
    EXPECT_EQ(sil::tensor::TensorSymmetricIndex<>::mem_size(), 1);
    EXPECT_EQ(sil::tensor::TensorSymmetricIndex<>::access_size(), 1);
    EXPECT_EQ(sil::tensor::TensorSymmetricIndex<>::mem_id({}), 0);
    EXPECT_EQ(sil::tensor::TensorSymmetricIndex<>::access_id({}), 0);
    EXPECT_EQ(sil::tensor::TensorAntisymmetricIndex<>::size(), 1);
    EXPECT_EQ(sil::tensor::TensorAntisymmetricIndex<>::mem_size(), 1);
    EXPECT_EQ(sil::tensor::TensorAntisymmetricIndex<>::access_size(), 1);
    EXPECT_EQ(sil::tensor::TensorAntisymmetricIndex<>::mem_id({}), 0);
}

TEST(Tensor, AntisymmetricAccessDomainIncludesBothOrientations)
{
    using Index = sil::tensor::TensorAntisymmetricIndex<Alpha, Beta, Gamma>;
    ddc::Chunk
            allocation(sil::tensor::TensorAccessor<Index>::domain(), ddc::HostAllocator<double>());
    sil::tensor::Tensor field(allocation);
    field.mem(ddc::DiscreteElement<Index>(0)) = 7.0;
    EXPECT_EQ(Index::size(), 27);
    EXPECT_EQ(Index::mem_size(), 1);
    EXPECT_EQ(Index::access_size(), 3);
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            for (std::size_t k = 0; k < 3; ++k) {
                std::size_t const access = Index::access_id({i, j, k});
                EXPECT_LT(access, Index::access_size());
                if (i == j || i == k || j == k) {
                    EXPECT_EQ(access, 0);
                    EXPECT_DOUBLE_EQ(field.get(ddc::DiscreteElement<Index>(access)), 0.0);
                } else {
                    bool const odd = ((i > j) + (i > k) + (j > k)) % 2 != 0;
                    EXPECT_EQ(Index::access_id_to_mem_id(access), 0);
                    EXPECT_DOUBLE_EQ(
                            field.get(ddc::DiscreteElement<Index>(access)),
                            odd ? -7.0 : 7.0);
                }
            }
        }
    }
}

TEST(Tensor, AntisymmetricCanonicalStorageRoundTrip)
{
    using Index = sil::tensor::TensorAntisymmetricIndex<Mu, Nu, sil::tensor::Covariant<Mu>>;
    for (std::size_t stored = 0; stored < Index::mem_size(); ++stored) {
        std::array<std::size_t, 3> const ids = Index::mem_id_to_canonical_natural_ids(stored);
        EXPECT_LT(ids[0], ids[1]);
        EXPECT_LT(ids[1], ids[2]);
        EXPECT_EQ(Index::mem_id(ids), stored);
        EXPECT_EQ(Index::access_id_to_mem_id(Index::access_id(ids)), stored);
    }
}

namespace {

template <class Index>
concept DirectTensorIndexAPI = requires(std::size_t id) {
    requires Index::is_tensor_index;
    requires Index::is_explicitely_stored_tensor;
    { Index::subindices_domain() } -> std::same_as<typename Index::subindices_domain_t>;
    { Index::rank() } -> std::same_as<std::size_t>;
    { Index::size() } -> std::same_as<std::size_t>;
    { Index::mem_size() } -> std::same_as<std::size_t>;
    { Index::access_size() } -> std::same_as<std::size_t>;
    { Index::access_id_to_mem_id(id) } -> std::same_as<std::size_t>;
    {
        Index::mem_id_to_canonical_natural_ids(id)
    } -> std::same_as<std::array<std::size_t, Index::rank()>>;
};

static_assert(DirectTensorIndexAPI<sil::tensor::ScalarIndex>);
static_assert(DirectTensorIndexAPI<Alpha>);
static_assert(DirectTensorIndexAPI<FullIndex>);
static_assert(DirectTensorIndexAPI<DiagIndex>);
static_assert(DirectTensorIndexAPI<SymIndex>);
static_assert(DirectTensorIndexAPI<AntisymIndex>);
static_assert(DirectTensorIndexAPI<IdIndex>);
static_assert(DirectTensorIndexAPI<LorentzianSignIndex>);
static_assert(DirectTensorIndexAPI<LeviCivitaIndex>);

} // namespace
