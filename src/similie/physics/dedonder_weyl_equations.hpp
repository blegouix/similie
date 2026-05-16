// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
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

    template <std::size_t I>
    [[nodiscard]] constexpr double potential_grad(double momentum_component) const
    {
        return m_hamiltonian.template dH_dpi<I>(momentum_component);
    }

    [[nodiscard]] constexpr double momentum_div(double potential) const
    {
        return -m_hamiltonian.dH_dphi(potential);
    }
};

} // namespace similie::physics
