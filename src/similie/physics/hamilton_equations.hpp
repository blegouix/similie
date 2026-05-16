// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <utility>

namespace similie::physics {

template <class Hamiltonian>
class HamiltonEquations
{
    Hamiltonian m_hamiltonian;

public:
    constexpr explicit HamiltonEquations(Hamiltonian hamiltonian)
        : m_hamiltonian(std::move(hamiltonian))
    {
    }

    template <std::size_t I>
    [[nodiscard]] constexpr double dpotential_dt(double spatial_momentum_component) const
    {
        return m_hamiltonian.template dH_dpi<I>(spatial_momentum_component);
    }

    template <std::size_t I>
    [[nodiscard]] constexpr double dmomentum_dt(double potential) const
    {
        static_cast<void>(I);
        return -m_hamiltonian.dH_dphi(potential);
    }
};

} // namespace similie::physics
