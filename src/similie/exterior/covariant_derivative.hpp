// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#pragma once

#include <array>
#include <type_traits>

#include <ddc/ddc.hpp>

#include <similie/tensor/character.hpp>
#include <similie/tensor/tensor.hpp>

#include "coboundary.hpp"
#include "cubical_reconstruction.hpp"

namespace sil::exterior {

struct ZeroConnection
{
    template <class OutputComponentIndex, class DerivativeIndex, class InputComponentIndex>
    [[nodiscard]] KOKKOS_FUNCTION double value(auto, auto) const
    {
        return 0.0;
    }
};

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
    using spatial_index_seq = ddc::detail::TypeSeq<SpatialIndex...>;
    using scalar_component_index = tensor::ScalarIndex;

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

    /** Reconstruct d^nabla of a 0-cochain in the fibre at cell vertex 0.
     * Result [vertex][physical direction][output fibre][input fibre] is a
     * linear stencil. All edges participate through Whitney 1-form bases.
     * For non-flat connections the supplied transport also defines paths from
     * each edge base to vertex 0. Identity transport yields exactly R_1 d,
     * and d R_0 = R_1 d. No strain, material or quadrature rule enters here.
     */
    template <std::size_t Rank, class Transport = IdentityTransport<Rank>>
    [[nodiscard]] KOKKOS_FUNCTION static auto cell_stencil(
            CubicalReconstruction<sizeof...(SpatialIndex)> const& reconstruction,
            Transport transport = {})
    {
        constexpr std::size_t dimension = sizeof...(SpatialIndex);
        constexpr std::size_t vertices = std::size_t(1) << dimension;
        std::array<std::array<std::array<std::array<double, Rank>, Rank>, dimension>, vertices>
                result {};
        for (std::size_t d = 0; d < dimension; ++d) {
            for (std::size_t edge = 0; edge < vertices; ++edge) {
                if (edge & (std::size_t(1) << d))
                    continue;
                auto const to_root = transport(0, edge);
                for (std::size_t node = 0; node < vertices; ++node) {
                    for (std::size_t b = 0; b < Rank; ++b) {
                        auto const difference = cochain_value<0, Rank>(
                                std::array<std::size_t, 1> {d},
                                edge,
                                [=](std::size_t, std::size_t vertex) {
                                    std::array<double, Rank> value {};
                                    if (vertex == node)
                                        value[b] = 1;
                                    return value;
                                },
                                transport);
                        for (std::size_t i = 0; i < dimension; ++i) {
                            double const weight = reconstruction.template basis<1>({d}, edge, {i});
                            for (std::size_t a = 0; a < Rank; ++a) {
                                for (std::size_t c = 0; c < Rank; ++c) {
                                    result[node][i][a][b] += weight * to_root[a][c] * difference[c];
                                }
                            }
                        }
                    }
                }
            }
        }
        return result;
    }

    template <
            class OutputComponentIndex,
            class DerivativeIndex,
            class InputComponentIndex = OutputComponentIndex,
            class Elem,
            class PositionType,
            class Connection = ZeroConnection>
    [[nodiscard]] KOKKOS_FUNCTION static auto value(
            Elem elem,
            PositionType position,
            Connection connection = ZeroConnection {},
            std::array<double, sizeof...(SpatialIndex)> const& point = {})
    {
        // Differential-connection adapter for the existing tensor API. The
        // connection callback returns A_i at the requested point (captured by
        // the callback if it varies within the cell). Nodal values use R_0.
        constexpr std::size_t dimension = sizeof...(SpatialIndex);
        static_assert(ddc::type_seq_size_v<ddc::to_type_seq_t<Elem>> == dimension);
        static_assert([]<class... Tags>(ddc::detail::TypeSeq<Tags...>*) {
            return std::is_same_v<
                    ddc::detail::TypeSeq<typename Tags::continuous_dimension_type...>,
                    spatial_index_seq>;
        }(static_cast<ddc::to_type_seq_t<Elem>*>(nullptr)));
        std::array<std::array<double, dimension>, std::size_t(1) << dimension> positions {};
        for (std::size_t v = 0; v < positions.size(); ++v) {
            auto corner = elem;
            for (std::size_t d = 0; d < dimension; ++d)
                ddc::detail::array(corner)[d] += (v >> d) & 1;
            positions[v] = {position(
                    corner,
                    position.accessor().template access_element<SpatialIndex>())...};
        }
        CubicalReconstruction<dimension> const reconstruction(positions, point);
        auto const derivative = cell_stencil<1>(reconstruction);
        auto stencil = detail::make_stencil<Kokkos::HostSpace, scalar_component_index>(elem);
        double const connection_coefficient = connection.template value<
                OutputComponentIndex,
                DerivativeIndex,
                InputComponentIndex>(elem, position);
        for (std::size_t v = 0; v < positions.size(); ++v) {
            auto corner = elem;
            for (std::size_t d = 0; d < dimension; ++d)
                ddc::detail::array(corner)[d] += (v >> d) & 1;
            double coefficient = connection_coefficient * reconstruction.reference_basis(0, v);
            if constexpr (std::is_same_v<OutputComponentIndex, InputComponentIndex>) {
                coefficient
                        += derivative[v][ddc::type_seq_rank_v<DerivativeIndex, spatial_index_seq>]
                                     [0][0];
            }
            stencil.mem(
                    typename decltype(stencil)::discrete_element_type(
                            corner,
                            ddc::DiscreteElement<scalar_component_index>(0)))
                    = coefficient;
        }
        return stencil;
    }

    template <
            class OutputComponentIndex,
            class DerivativeIndex,
            class InputComponentIndex = OutputComponentIndex,
            class TensorType,
            class Elem,
            class PositionType,
            class Connection = ZeroConnection>
    [[nodiscard]] KOKKOS_FUNCTION double operator()(
            TensorType tensor,
            Elem elem,
            PositionType position,
            Connection connection = ZeroConnection {},
            std::array<double, sizeof...(SpatialIndex)> const& point = {}) const
    {
        auto const stencil
                = value<OutputComponentIndex,
                        DerivativeIndex,
                        InputComponentIndex>(elem, position, connection, point);
        auto const input_component
                = tensor.accessor().template access_element<InputComponentIndex>();
        double result = 0.0;
        ddc::device_for_each(stencil.domain(), [&](auto stencil_elem) {
            result += stencil.mem(stencil_elem)
                      * tensor
                                .mem(typename TensorType::non_indices_domain_t::
                                             discrete_element_type(stencil_elem),
                                     input_component);
        });
        return result;
    }
};

} // namespace sil::exterior
