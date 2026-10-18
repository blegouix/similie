// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#pragma once

#include <cstddef>

#include <Kokkos_Core.hpp>

#include "concepts.hpp"

namespace similie::physics {
/** Device-compatible composition of physics objects with independent state laws.
 * Coupled models derive from this class and supply shared-variable mappings,
 * interaction fluxes, and sources. Independent operator() calls alone cannot
 * introduce momentum or energy exchange between physical sectors.
 * \important This operator and documentation is fully AI-generated.
 */
template <PhysicsConcept... Physics>
class CombinePhysics;

template <>
class CombinePhysics<>
{
public:
    static constexpr std::size_t size = 0;
    CombinePhysics() = default;
};

template <PhysicsConcept First, PhysicsConcept... Rest>
class CombinePhysics<First, Rest...>
{
    First m_first;
    CombinePhysics<Rest...> m_rest;

public:
    static constexpr std::size_t size = 1 + sizeof...(Rest);
    KOKKOS_FUNCTION explicit CombinePhysics(First const& first, Rest const&... rest)
        : m_first(first)
        , m_rest(rest...)
    {
    }
    template <std::size_t Index>
    KOKKOS_FUNCTION decltype(auto) get()
    {
        static_assert(Index < size);
        if constexpr (Index == 0)
            return (m_first);
        else
            return m_rest.template get<Index - 1>();
    }
    template <std::size_t Index>
    KOKKOS_FUNCTION decltype(auto) get() const
    {
        static_assert(Index < size);
        if constexpr (Index == 0)
            return (m_first);
        else
            return m_rest.template get<Index - 1>();
    }
};
} // namespace similie::physics
