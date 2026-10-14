// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#pragma once

#include <utility>

#include <Kokkos_Core.hpp>

#include "concepts.hpp"

namespace similie::physics {

template <class Hamiltonian>
class DeDonderWeylEquations
{
    Hamiltonian m_hamiltonian;

public:
    using State = typename Hamiltonian::State;
    static constexpr bool IS_LINEAR = Hamiltonian::IS_LINEAR;

    KOKKOS_FUNCTION constexpr explicit DeDonderWeylEquations(Hamiltonian hamiltonian)
        : m_hamiltonian(std::move(hamiltonian))
    {
    }

    template <class Index, class Elem>
    [[nodiscard]] KOKKOS_FUNCTION constexpr double potential_grad(State const& state, Elem elem)
            const
    {
        return m_hamiltonian.template dhamiltonian_dmoments<Index>(state, elem);
    }

    template <class Index, class Elem>
    [[nodiscard]] KOKKOS_FUNCTION constexpr double moments_div(State const& state, Elem elem) const
    {
        return -m_hamiltonian.template dhamiltonian_dpotential<Index>(state, elem);
    }
};

static_assert(DeDonderWeylEquationsConcept<
              DeDonderWeylEquations<detail::HamiltonianPhysicsConcept>,
              detail::ElementConcept>);

} // namespace similie::physics
