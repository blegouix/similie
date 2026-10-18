// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#include <cmath>
#include <limits>

#include <gtest/gtest.h>

#include "tensor.hpp"

namespace {
using namespace sil::tensor;
struct X
{
};
struct Y
{
};
struct Z
{
};
struct T
{
};
template <int Id, class... Coordinates>
struct Index : TensorNaturalIndex<Coordinates...>
{
};
using I = Covariant<Index<0, X, Y, Z>>;
using J = Covariant<Index<1, X, Y, Z>>;
using K = Covariant<Index<2, X, Y, Z>>;
using L = Covariant<Index<3, X, Y, Z>>;
using A = Covariant<Index<0, T, X, Y, Z>>;
using B = Covariant<Index<1, T, X, Y, Z>>;
using C = Covariant<Index<2, T, X, Y, Z>>;
using D = Covariant<Index<3, T, X, Y, Z>>;

template <class Domain>
struct Full;
template <class... Tags>
struct Full<ddc::DiscreteDomain<Tags...>>
{
    using type = TensorFullIndex<Tags...>;
};

template <
        class LeftTensor,
        class RightTensor,
        class DenseLeftTensor,
        class DenseRightTensor,
        class OutputTensor>
struct CompareKernel
{
    LeftTensor left;
    RightTensor right;
    DenseLeftTensor dense_left;
    DenseRightTensor dense_right;
    OutputTensor output;
    OutputTensor reverse;
    OutputTensor reference;

    KOKKOS_FUNCTION void operator()(int) const
    {
        std::size_t counter = 0;
        ddc::device_for_each(left.domain(), [&](auto element) {
            ++counter;
            left.mem(element) = Kokkos::sin(0.37 * counter) + 0.13 * counter;
        });
        counter = 0;
        ddc::device_for_each(right.domain(), [&](auto element) {
            ++counter;
            right.mem(element) = Kokkos::cos(0.63 * counter) - 0.17 * counter;
        });
        ddc::device_for_each(dense_left.domain(), [&](auto element) {
            dense_left.mem(element)
                    = left.get(left.access_element(dense_left.canonical_natural_element(element)));
        });
        ddc::device_for_each(dense_right.domain(), [&](auto element) {
            dense_right.mem(element) = right.get(
                    right.access_element(dense_right.canonical_natural_element(element)));
        });
        tensor_prod(output, left, right);
        tensor_prod(reverse, right, left);
        tensor_prod(reference, dense_left, dense_right);
    }
};

// Materialize both operands, then use the unchanged dense contraction as the
// reference. Every case exercises the structured tensor on both sides.
template <class Output, class... Left, class... Right>
void compare(
        ddc::TypeSeq<Left...>,
        ddc::TypeSeq<Right...>,
        ddc::DiscreteDomain<Output> const& output_domain = TensorAccessor<Output>().domain())
{
    using DenseLeft = typename Full<typename TensorAccessor<Left...>::natural_domain_t>::type;
    using DenseRight = typename Full<typename TensorAccessor<Right...>::natural_domain_t>::type;
    ddc::Chunk left_alloc(TensorAccessor<Left...>().domain(), ddc::DeviceAllocator<double>());
    ddc::Chunk right_alloc(TensorAccessor<Right...>().domain(), ddc::DeviceAllocator<double>());
    ddc::Chunk
            dense_left_alloc(TensorAccessor<DenseLeft>().domain(), ddc::DeviceAllocator<double>());
    ddc::Chunk dense_right_alloc(
            TensorAccessor<DenseRight>().domain(),
            ddc::DeviceAllocator<double>());
    ddc::Chunk output_alloc(output_domain, ddc::DeviceAllocator<double>());
    ddc::Chunk reverse_alloc(output_domain, ddc::DeviceAllocator<double>());
    ddc::Chunk reference_alloc(output_domain, ddc::DeviceAllocator<double>());
    Tensor left(left_alloc);
    Tensor right(right_alloc);
    Tensor dense_left(dense_left_alloc);
    Tensor dense_right(dense_right_alloc);
    Tensor output(output_alloc);
    Tensor reverse(reverse_alloc);
    Tensor reference(reference_alloc);
    Kokkos::parallel_for(
            "structured_tensor_product",
            Kokkos::RangePolicy<>(0, 1),
            CompareKernel {left, right, dense_left, dense_right, output, reverse, reference});
    auto host_output = ddc::create_mirror_view_and_copy(
            Kokkos::DefaultHostExecutionSpace(),
            output_alloc.span_view());
    auto host_reverse = ddc::create_mirror_view_and_copy(
            Kokkos::DefaultHostExecutionSpace(),
            reverse_alloc.span_view());
    auto host_reference = ddc::create_mirror_view_and_copy(
            Kokkos::DefaultHostExecutionSpace(),
            reference_alloc.span_view());
    ddc::host_for_each(output_domain, [&](auto element) {
        double const expected = host_reference(element);
        EXPECT_NEAR(host_output(element), expected, 2.e-12 * (1. + std::abs(expected)));
        EXPECT_NEAR(host_reverse(element), expected, 2.e-12 * (1. + std::abs(expected)));
    });
}

template <template <class...> class StructuredIndex>
void equal_index_cases()
{
    compare<I>(ddc::TypeSeq<StructuredIndex<I, upper_t<J>>>(), ddc::TypeSeq<J>());
    compare<TensorFullIndex<I, upper_t<K>>>(
            ddc::TypeSeq<StructuredIndex<I, upper_t<J>>>(),
            ddc::TypeSeq<TensorFullIndex<J, upper_t<K>>>());
    compare<ScalarIndex>(
            ddc::TypeSeq<StructuredIndex<upper_t<I>, upper_t<J>>>(),
            ddc::TypeSeq<TensorSymmetricIndex<I, J>>());
    compare<J>(
            ddc::TypeSeq<StructuredIndex<upper_t<I>, J, upper_t<K>>>(),
            ddc::TypeSeq<TensorFullIndex<I, K>>());
    compare<TensorFullIndex<I, J, K>>(ddc::TypeSeq<StructuredIndex<I, J>>(), ddc::TypeSeq<K>());
    compare<I>(
            ddc::TypeSeq<StructuredIndex<I, upper_t<J>>>(),
            ddc::TypeSeq<J>(),
            ddc::DiscreteDomain<I>(ddc::DiscreteElement<I>(1), ddc::DiscreteVector<I>(1)));
}

TEST(TensorProdStructured, IdentityBothSides)
{
    equal_index_cases<TensorIdentityIndex>();
}
TEST(TensorProdStructured, DiagonalBothSides)
{
    equal_index_cases<TensorDiagonalIndex>();
}
TEST(TensorProdStructured, LeviCivitaBothSides)
{
    compare<TensorAntisymmetricIndex<
            J,
            K>>(ddc::TypeSeq<TensorLeviCivitaIndex<upper_t<I>, J, K>>(), ddc::TypeSeq<I>());
    compare<I>(
            ddc::TypeSeq<TensorLeviCivitaIndex<I, upper_t<J>, upper_t<K>>>(),
            ddc::TypeSeq<J, K>());
    compare<I>(
            ddc::TypeSeq<TensorLeviCivitaIndex<I, upper_t<J>, upper_t<K>>>(),
            ddc::TypeSeq<TensorAntisymmetricIndex<J, K>>());
    compare<J>(
            ddc::TypeSeq<TensorLeviCivitaIndex<upper_t<I>, J, upper_t<K>>>(),
            ddc::TypeSeq<TensorFullIndex<I, K>>());
    compare<ScalarIndex>(
            ddc::TypeSeq<TensorLeviCivitaIndex<upper_t<I>, upper_t<J>, upper_t<K>>>(),
            ddc::TypeSeq<TensorFullIndex<I, J, K>>());
    compare<TensorAntisymmetricIndex<C, D>>(
            ddc::TypeSeq<TensorLeviCivitaIndex<upper_t<A>, upper_t<B>, C, D>>(),
            ddc::TypeSeq<TensorAntisymmetricIndex<A, B>>());
}
// A true identity copy must neither evaluate zero terms against unrelated NaNs
// nor add a positive zero to the retained component.
void check_identity_exceptional_values()
{
    using Identity = TensorIdentityIndex<I, upper_t<J>>;
    ddc::Chunk input_alloc(TensorAccessor<J>().domain(), ddc::DeviceAllocator<double>());
    ddc::Chunk identity_alloc(TensorAccessor<Identity>().domain(), ddc::DeviceAllocator<double>());
    ddc::Chunk output_alloc(TensorAccessor<I>().domain(), ddc::DeviceAllocator<double>());
    ddc::Chunk reverse_alloc(TensorAccessor<I>().domain(), ddc::DeviceAllocator<double>());
    Tensor input(input_alloc);
    Tensor identity(identity_alloc);
    Tensor output(output_alloc);
    Tensor reverse(reverse_alloc);
    Kokkos::parallel_for(
            "identity_copy",
            Kokkos::RangePolicy<>(0, 1),
            KOKKOS_LAMBDA(int) {
                input.mem(ddc::DiscreteElement<J>(0)) = std::numeric_limits<double>::quiet_NaN();
                input.mem(ddc::DiscreteElement<J>(1)) = -0.;
                input.mem(ddc::DiscreteElement<J>(2)) = std::numeric_limits<double>::infinity();
                tensor_prod(output, identity, input);
                tensor_prod(reverse, input, identity);
            });
    auto host_output = ddc::create_mirror_view_and_copy(
            Kokkos::DefaultHostExecutionSpace(),
            output_alloc.span_view());
    auto host_reverse = ddc::create_mirror_view_and_copy(
            Kokkos::DefaultHostExecutionSpace(),
            reverse_alloc.span_view());
    EXPECT_TRUE(std::isnan(host_output(ddc::DiscreteElement<I>(0))));
    EXPECT_TRUE(std::isnan(host_reverse(ddc::DiscreteElement<I>(0))));
    EXPECT_EQ(host_output(ddc::DiscreteElement<I>(1)), 0.);
    EXPECT_EQ(host_reverse(ddc::DiscreteElement<I>(1)), 0.);
    EXPECT_TRUE(std::signbit(host_output(ddc::DiscreteElement<I>(1))));
    EXPECT_TRUE(std::signbit(host_reverse(ddc::DiscreteElement<I>(1))));
    EXPECT_EQ(host_output(ddc::DiscreteElement<I>(2)), std::numeric_limits<double>::infinity());
    EXPECT_EQ(host_reverse(ddc::DiscreteElement<I>(2)), std::numeric_limits<double>::infinity());
}
TEST(TensorProdStructured, IdentityPreservesExceptionalValues)
{
    check_identity_exceptional_values();
}
TEST(TensorProdStructured, TwoStructuredOperands)
{
    compare<TensorFullIndex<I, upper_t<K>>>(
            ddc::TypeSeq<TensorIdentityIndex<I, upper_t<J>>>(),
            ddc::TypeSeq<TensorDiagonalIndex<J, upper_t<K>>>());
    compare<TensorFullIndex<I, upper_t<K>>>(
            ddc::TypeSeq<TensorDiagonalIndex<I, upper_t<J>>>(),
            ddc::TypeSeq<TensorDiagonalIndex<J, upper_t<K>>>());
    compare<I>(
            ddc::TypeSeq<TensorLeviCivitaIndex<I, upper_t<J>, upper_t<K>>>(),
            ddc::TypeSeq<TensorIdentityIndex<J, K>>());
}
TEST(TensorProdStructured, IndependentContraction)
{
    compare<TensorFullIndex<I, L>>(
            ddc::TypeSeq<upper_t<K>, TensorDiagonalIndex<I, upper_t<J>>>(),
            ddc::TypeSeq<K, J, L>());
}
} // namespace
