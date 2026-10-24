// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#include <ddc/ddc.hpp>

#include <gtest/gtest.h>
#include <similie/tensor/extrapolation_rules.hpp>
#include <similie/tensor/periodic_extrapolation_rule.hpp>
#include <similie/tensor/tensor_indices/character.hpp>

namespace {
struct X
{
};
struct Y
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
using Vector = sil::tensor::Covariant<sil::tensor::TensorNaturalIndex<X, Y>>;

void check_rectangular_corners()
{
    ddc::DiscreteDomain<GridX, GridY> const
            domain(ddc::DiscreteElement<GridX, GridY>(5, 9),
                   ddc::DiscreteVector<GridX, GridY>(3, 4));
    ddc::DiscreteDomain<GridX, GridY, Vector> const
            field_domain(domain, sil::tensor::TensorAccessor<Vector>::domain());
    ddc::Chunk allocation(field_domain, ddc::DeviceAllocator<double>());
    sil::tensor::Tensor field(allocation);
    ddc::parallel_for_each(
            Kokkos::DefaultExecutionSpace(),
            field_domain,
            KOKKOS_LAMBDA(ddc::DiscreteElement<GridX, GridY, Vector> element) {
                field.mem(element) = 100 * element.uid<GridX>() + 10 * element.uid<GridY>()
                                     + element.uid<Vector>();
            });
    double error;
    Kokkos::parallel_reduce(
            "periodic_rectangular_corner_samples",
            1,
            KOKKOS_LAMBDA(int, double& sum) {
                // Interior, either seam, and corners several periods away.
                std::array<std::array<int, 4>, 5> const samples {{
                        {6, 10, 6, 10},
                        {4, 13, 7, 9},
                        {8, 8, 5, 12},
                        {-8, -10, 7, 10},
                        {20, 29, 5, 9},
                }};
                sil::tensor::PeriodicExtrapolationRule const periodic;
                sil::tensor::ExtrapolationRules const
                        rules(std::pair {periodic, periodic}, std::pair {periodic, periodic});
                for (std::array<int, 4> const& sample : samples) {
                    sil::misc::SignedDiscreteElement<GridX, GridY> const
                            element(sample[0], sample[1]);
                    auto const direct = periodic(field, element);
                    auto const combined = rules(field, element);
                    auto const basis = rules.value(domain, element);
                    ddc::device_for_each(direct.domain(), [&](auto component) {
                        double const expected = 100 * sample[2] + 10 * sample[3]
                                                + component.template uid<Vector>();
                        double const contracted = sil::tensor::detail::contract_extrapolation_basis(
                                basis,
                                [&](auto stored) { return field.mem(stored, component); },
                                [&](std::size_t) { return direct.mem(component); });
                        sum += Kokkos::abs(direct.mem(component) - expected)
                               + Kokkos::abs(combined.mem(component) - expected)
                               + Kokkos::abs(contracted - expected);
                    });
                }
            },
            error);
    EXPECT_DOUBLE_EQ(error, 0.0);
}

void check_singleton_axis()
{
    ddc::DiscreteDomain<GridX, GridY> const
            domain(ddc::DiscreteElement<GridX, GridY>(0, 0),
                   ddc::DiscreteVector<GridX, GridY>(1, 3));
    ddc::DiscreteDomain<GridX, GridY, Vector> const
            field_domain(domain, sil::tensor::TensorAccessor<Vector>::domain());
    ddc::Chunk allocation(field_domain, ddc::DeviceAllocator<double>());
    sil::tensor::Tensor field(allocation);
    ddc::parallel_for_each(
            Kokkos::DefaultExecutionSpace(),
            field_domain,
            KOKKOS_LAMBDA(ddc::DiscreteElement<GridX, GridY, Vector> element) {
                field.mem(element) = 10 * element.uid<GridY>() + element.uid<Vector>();
            });
    double error;
    Kokkos::parallel_reduce(
            "periodic_unsigned_lower_boundary",
            1,
            KOKKOS_LAMBDA(int, double& sum) {
                ddc::DiscreteElement<GridX, GridY> const element
                        = domain.front() - ddc::DiscreteVector<GridX, GridY>(1, 1);
                auto const sampled = sil::tensor::PeriodicExtrapolationRule {}(field, element);
                ddc::DiscreteElement<GridY, Vector, GridX> const
                        reordered(element, ddc::DiscreteElement<Vector>(0));
                auto const reordered_sample
                        = sil::tensor::PeriodicExtrapolationRule {}(field, reordered);
                ddc::device_for_each(sampled.domain(), [&](auto component) {
                    sum += Kokkos::abs(
                            sampled.mem(component) - (20 + component.template uid<Vector>()));
                    sum += Kokkos::abs(reordered_sample.mem(component) - sampled.mem(component));
                });
            },
            error);
    EXPECT_DOUBLE_EQ(error, 0.0);
}
TEST(PeriodicExtrapolation, WholeVectorAcrossRectangularCorners)
{
    check_rectangular_corners();
}

TEST(PeriodicExtrapolation, SingletonAxisAndUnsignedLowerBoundary)
{
    check_singleton_axis();
}
} // namespace
