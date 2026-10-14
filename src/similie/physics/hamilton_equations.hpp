// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#pragma once

#include <type_traits>
#include <utility>

#include <ddc/ddc.hpp>

#include <Kokkos_Core.hpp>

#include "concepts.hpp"

namespace similie::physics {

template <class Hamiltonian>
class HamiltonEquations
{
    Hamiltonian m_hamiltonian;

public:
    using State = typename Hamiltonian::State;
    static constexpr bool IS_LINEAR = Hamiltonian::IS_LINEAR;

    KOKKOS_FUNCTION constexpr explicit HamiltonEquations(Hamiltonian hamiltonian)
        : m_hamiltonian(std::move(hamiltonian))
    {
    }

    template <class Index, class Elem>
    [[nodiscard]] KOKKOS_FUNCTION constexpr double dpotential_dt(State const& state, Elem elem)
            const
    {
        return m_hamiltonian.template dhamiltonian_dmoments<Index>(state, elem);
    }

    template <class Index, class Elem>
    [[nodiscard]] KOKKOS_FUNCTION constexpr double dmoments_dt(State const& state, Elem elem) const
    {
        return -m_hamiltonian.template dhamiltonian_dpotential<Index>(state, elem);
    }

    template <class RowIndex, class ColumnIndex, class Elem>
    [[nodiscard]] KOKKOS_FUNCTION constexpr double jacobian(State const& state, Elem elem) const
    {
        return m_hamiltonian.template jacobian<RowIndex, ColumnIndex>(state, elem);
    }

    template <class Index, class Elem>
        requires detail::HasMomentCoefficientConcept<Hamiltonian, Index, Elem>
    [[nodiscard]] KOKKOS_FUNCTION constexpr auto dpotential_dt_value(State const& state, Elem elem)
            const
    {
        return m_hamiltonian.template dhamiltonian_dmoments_value<Index>(state, elem);
    }

    template <class Index, class Elem>
        requires detail::HasPotentialCoefficientConcept<Hamiltonian, Index, Elem>
    [[nodiscard]] KOKKOS_FUNCTION constexpr auto dmoments_dt_value(State const& state, Elem elem)
            const
    {
        auto value = m_hamiltonian.template dhamiltonian_dpotential_value<Index>(state, elem);
        if constexpr (std::is_same_v<std::remove_cvref_t<decltype(value)>, double>) {
            return -value;
        } else {
            ddc::device_for_each(value.domain(), [&](auto element) { value(element) *= -1.0; });
            return value;
        }
    }
};

static_assert(HamiltonEquationsConcept<
              HamiltonEquations<detail::HamiltonianPhysicsConcept>,
              detail::ElementConcept>);

} // namespace similie::physics
