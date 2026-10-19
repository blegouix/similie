// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#pragma once

#include <cstddef>

#include <similie/physics/scalar_field/scalar_field_with_power_coupling.hpp>

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
    static constexpr std::size_t N = 0;
    static constexpr bool IS_LINEAR = true;
    struct State
    {
    };
    CombinePhysics() = default;
};

template <PhysicsConcept First, PhysicsConcept... Rest>
class CombinePhysics<First, Rest...>
{
    First m_first;
    CombinePhysics<Rest...> m_rest;

public:
    static constexpr std::size_t size = 1 + sizeof...(Rest);
    static constexpr std::size_t N = First::N + CombinePhysics<Rest...>::N;
    static constexpr bool IS_LINEAR = First::IS_LINEAR && CombinePhysics<Rest...>::IS_LINEAR;
    struct State
    {
        typename First::State first;
        typename CombinePhysics<Rest...>::State rest;
    };
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
static_assert(
        PhysicsConcept<CombinePhysics<
                scalar_field::ScalarFieldWithPowerCouplingHamiltonian<detail::GridConcept<int>>,
                scalar_field::ScalarFieldWithPowerCouplingHamiltonian<detail::GridConcept<int>>>>);

} // namespace similie::physics
