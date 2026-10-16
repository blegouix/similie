// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

#include "coboundary.hpp"
#include "concepts.hpp"
#include "stencil_indices.hpp"

namespace {
template <std::size_t Dimensions>
struct Direction
{
    static constexpr std::size_t size()
    {
        return Dimensions;
    }
};
template <std::size_t Count>
struct Components
{
    static constexpr std::size_t mem_size()
    {
        return Count;
    }
};
using Input = Components<3>;
using Output = Components<2>;

struct ValidOperator
{
    [[maybe_unused]] static constexpr auto stencil_nonzero_indices()
    {
        return sil::exterior::detail::dense_stencil_indices<3>();
    }
};
struct WrongOperator
{
    [[maybe_unused]] static constexpr std::array<int, 3> stencil_nonzero_indices()
    {
        return {};
    }
};
struct DynamicOperator
{
    [[maybe_unused]] static std::array<std::size_t, 3> stencil_nonzero_indices()
    {
        return {};
    }
};
static_assert(sil::exterior::StencilOperatorConcept<ValidOperator>);
static_assert(!sil::exterior::StencilOperatorConcept<WrongOperator>);
static_assert(!sil::exterior::StencilOperatorConcept<DynamicOperator>);

struct X
{
};
struct Y
{
};
struct Z
{
};
struct Axes : sil::tensor::TensorNaturalIndex<X, Y, Z>
{
};
struct OtherAxes : sil::tensor::TensorNaturalIndex<X, Y, Z>
{
};
using Scalar = sil::tensor::Covariant<sil::tensor::ScalarIndex>;
using One = sil::tensor::Covariant<Axes>;
using Added = sil::tensor::Covariant<OtherAxes>;
using Two = sil::tensor::TensorAntisymmetricIndex<Added, One>;

TEST(StencilIndices, ExteriorDerivativePattern)
{
    constexpr auto scalar = sil::exterior::Coboundary<One, Scalar>::stencil_nonzero_indices();
    constexpr auto vector = sil::exterior::Coboundary<Added, One>::stencil_nonzero_indices();
    constexpr auto transposed
            = sil::exterior::TransposedCoboundary<Added, One>::stencil_nonzero_indices();
    static_assert(scalar.size() == 6);
    static_assert(vector.size() == 12);
    static_assert(transposed.size() == 12);
    for (std::size_t flat : vector) {
        auto const entry = sil::exterior::stencil_reindex<Added, One, Two, 1>(flat);
        auto const target = Two::mem_id_to_canonical_natural_ids(entry.output_component);
        EXPECT_TRUE(entry.input_component == target[0] || entry.input_component == target[1]);
        for (int offset : entry.spatial_offset)
            EXPECT_GE(offset, 0);
    }
    for (std::size_t flat : transposed) {
        auto const entry = sil::exterior::stencil_reindex<Added, One, Two, 1>(flat);
        for (int offset : entry.spatial_offset)
            EXPECT_LE(offset, 0);
    }
}

TEST(StencilIndices, StandardOrdering)
{
    EXPECT_EQ(ValidOperator::stencil_nonzero_indices()[0], 0u);
    EXPECT_EQ(WrongOperator::stencil_nonzero_indices()[0], 0);
    EXPECT_EQ(DynamicOperator::stencil_nonzero_indices()[0], 0u);
    constexpr auto center = sil::exterior::stencil_reindex<Direction<3>, Input, Output, 2>(75 + 2);
    static_assert(center.output_component == 1);
    static_assert(center.input_component == 2);
    static_assert(center.spatial_offset == std::array<int, 3> {0, 0, 0});
    constexpr auto neighbor
            = sil::exterior::stencil_reindex<Direction<3>, Input, Output, 2>(75 + 3 + 2);
    static_assert(neighbor.output_component == 1 && neighbor.input_component == 2);
    EXPECT_EQ(neighbor.spatial_offset, (std::array<int, 3> {-1, 0, 0}));
    constexpr auto diagonal
            = sil::exterior::stencil_reindex<Direction<3>, Input, Output, 2>(13 * 3);
    EXPECT_EQ(diagonal.spatial_offset, (std::array<int, 3> {-1, -1, 0}));
}

template <std::size_t Dimensions, std::size_t Radius>
void round_trip()
{
    constexpr std::size_t size
            = Output::mem_size() * Input::mem_size()
              * sil::exterior::detail::StencilShape<Direction<Dimensions>, Radius>::size;
    int errors = 0;
    Kokkos::parallel_reduce(
            "stencil_reindex_round_trip",
            size,
            KOKKOS_LAMBDA(int flat, int& count) {
                auto const index = sil::exterior::
                        stencil_reindex<Direction<Dimensions>, Input, Output, Radius>(flat);
                if (sil::exterior::stencil_flatten<Direction<Dimensions>, Input, Output, Radius>(
                            index)
                    != std::size_t(flat))
                    ++count;
            },
            errors);
    EXPECT_EQ(errors, 0);
}
TEST(StencilIndices, DeviceRoundTrip)
{
    round_trip<1, 0>();
    round_trip<2, 0>();
    round_trip<3, 0>();
    round_trip<1, 1>();
    round_trip<1, 2>();
    round_trip<2, 1>();
    round_trip<2, 2>();
    round_trip<3, 1>();
    round_trip<3, 2>();
}
TEST(StencilIndices, CartesianPattern)
{
    constexpr auto indices = sil::exterior::detail::laplacian_nonzero_indices<
            Direction<3>,
            Input,
            sil::exterior::CartesianPeriodicStencil>();
    static_assert(indices.size() == 21);
    for (std::size_t i = 0; i < indices.size(); ++i) {
        if (i)
            EXPECT_LT(indices[i - 1], indices[i]);
        auto const index
                = sil::exterior::stencil_reindex<Direction<3>, Input, Input, 2>(indices[i]);
        EXPECT_EQ(index.input_component, index.output_component);
        int distance = 0;
        for (int coordinate : index.spatial_offset)
            distance += std::abs(coordinate);
        EXPECT_LE(distance, 1);
    }
}
} // namespace
