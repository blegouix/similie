// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#pragma once

#include <array>

#include <ddc/ddc.hpp>

#include "coboundary.hpp"

namespace sil::exterior {

/** Parallel transport from the second vertex's fibre to the first one's.
 * This is the flat connection in a fixed frame; the fibre rank is independent
 * of the spatial dimension (scalar, vector and flattened tensor bundles).
 */
template <std::size_t Rank>
struct IdentityTransport
{
    [[nodiscard]] KOKKOS_FUNCTION auto operator()(std::size_t, std::size_t) const
    {
        std::array<std::array<double, Rank>, Rank> result {};
        for (std::size_t i = 0; i < Rank; ++i)
            result[i][i] = 1;
        return result;
    }
};

template <class... SpatialIndex>
class CovariantDerivative
{
public:
    /** Exterior covariant coboundary of a bundle-valued k-cochain on a cube.
     * Cochains are stored in the fibre at their lower vertex. Each opposite
     * face is transported to that fibre before taking the oriented difference.
     * With identity transport this is the ordinary cubical coboundary; with
     * curvature its square need not vanish. directions must be increasing.
     * Incidence is supplied by boundary(Simplex), and transported components
     * are paired with that boundary by Cochain::integrate(), as in Coboundary.
     * sampler(mask, vertex) returns the Rank components of a face cochain.
     */
    template <
            std::size_t Degree,
            std::size_t Rank,
            class Sampler,
            class Transport = IdentityTransport<Rank>>
    [[nodiscard]] KOKKOS_FUNCTION static auto cochain_value(
            std::array<std::size_t, Degree + 1> const& directions,
            std::size_t vertex,
            Sampler sampler,
            Transport transport = {})
    {
        static_assert(Degree < sizeof...(SpatialIndex));
        static_assert(Rank > 0);
        std::array<double, Rank> result {};
        std::size_t mask = 0;
        for (std::size_t j = 0; j <= Degree; ++j) {
            assert(directions[j] < sizeof...(SpatialIndex));
            assert(j == 0 || directions[j - 1] < directions[j]);
            mask |= std::size_t(1) << directions[j];
        }
        assert((vertex & mask) == 0);
        // Vertex masks only adapt the public cell-local API to the existing
        // simplex/boundary algebra. Boundary owns all incidence signs.
        ddc::DiscreteElement<SpatialIndex...> base;
        ddc::DiscreteVector<SpatialIndex...> vector;
        for (std::size_t d = 0; d < sizeof...(SpatialIndex); ++d) {
            ddc::detail::array(base)[d] = (vertex >> d) & 1;
            ddc::detail::array(vector)[d] = (mask >> d) & 1;
        }
        Simplex<Degree + 1, SpatialIndex...> const
                cell(std::integral_constant<std::size_t, Degree + 1> {}, base, vector);
        auto const faces = boundary<typename Kokkos::DefaultExecutionSpace::memory_space>(cell);
        std::array<std::array<double, Rank>, 2 * (Degree + 1)> values {};
        std::size_t face_id = 0;
        for (auto face = faces.begin(); face < faces.end(); ++face, ++face_id) {
            std::size_t face_mask = 0;
            std::size_t face_vertex = 0;
            for (std::size_t d = 0; d < sizeof...(SpatialIndex); ++d) {
                face_mask |= std::size_t(ddc::detail::array(face->discrete_vector())[d]) << d;
                face_vertex |= ddc::detail::array(face->discrete_element())[d] << d;
            }
            auto const sampled = sampler(face_mask, face_vertex);
            if (face_vertex == vertex) {
                values[face_id] = sampled;
            } else {
                auto const parallel = transport(vertex, face_vertex);
                for (std::size_t a = 0; a < Rank; ++a)
                    for (std::size_t b = 0; b < Rank; ++b)
                        values[face_id][a] += parallel[a][b] * sampled[b];
            }
        }
        for (std::size_t a = 0; a < Rank; ++a) {
            std::array<double, 2 * (Degree + 1)> component {};
            for (std::size_t f = 0; f < component.size(); ++f)
                component[f] = values[f][a];
            Kokkos::View<
                    double*,
                    Kokkos::LayoutRight,
                    typename Kokkos::DefaultExecutionSpace::memory_space,
                    Kokkos::MemoryTraits<Kokkos::Unmanaged>>
                    span(component.data(), component.size());
            Cochain<LocalChain<
                    Simplex<Degree, SpatialIndex...>,
                    Kokkos::LayoutRight,
                    typename Kokkos::DefaultExecutionSpace::memory_space>>
                    cochain(faces, span);
            result[a] = cochain.integrate();
        }
        return result;
    }
};

} // namespace sil::exterior
