// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

#include <cstddef>
#include <span>
#include <utility>

namespace similie::physics {

template <class Hamiltonian>
class DeDonderWeylEquations
{
    Hamiltonian m_hamiltonian;

public:
    constexpr explicit DeDonderWeylEquations(Hamiltonian hamiltonian)
        : m_hamiltonian(std::move(hamiltonian))
    {
    }

    template <class Index>
    [[nodiscard]] constexpr double potential_grad(
            std::span<double const, Hamiltonian::N> moments) const
    {
        return m_hamiltonian.template dhamiltonian_dmoments<Index>(moments);
    }

    template <class Index>
    [[nodiscard]] constexpr double potential_grad(double moments_component) const
    {
        return m_hamiltonian.template dhamiltonian_dmoments<Index>(moments_component);
    }

    template <class Index>
    [[nodiscard]] constexpr double moments_div(std::span<double const, 1> potential) const
    {
        static_cast<void>(sizeof(Index));
        return -m_hamiltonian.dhamiltonian_dpotential(potential[0]);
    }

    [[nodiscard]] constexpr double moments_div(double potential) const
    {
        return -m_hamiltonian.dhamiltonian_dpotential(potential);
    }
};

} // namespace similie::physics
