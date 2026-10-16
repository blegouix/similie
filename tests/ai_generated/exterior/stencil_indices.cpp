// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

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

TEST(StencilIndices, StandardOrdering)
{
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
} // namespace
