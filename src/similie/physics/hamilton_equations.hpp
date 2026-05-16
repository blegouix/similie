// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: MIT

#pragma once

#include <stdexcept>
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

    [[nodiscard]] constexpr Hamiltonian const& hamiltonian() const
    {
        return m_hamiltonian;
    }

    template <std::size_t I>
    [[nodiscard]] static constexpr double dpotential_dt(
            Hamiltonian const& hamiltonian,
            double spatial_momentum_component)
    {
        return hamiltonian.template dH_dpi<I>(spatial_momentum_component);
    }

    template <std::size_t I>
    [[nodiscard]] static constexpr double dmomentum_dt(
            Hamiltonian const& hamiltonian,
            double potential)
    {
        static_cast<void>(I);
        if constexpr (requires(Hamiltonian const& h) { h.dH_dphi(potential); }) {
            return -hamiltonian.dH_dphi(potential);
        } else {
            throw std::logic_error(
                    "HamiltonEquations::dmomentum_dt requires a Hamiltonian exposing dH_dphi");
        }
    }
};

} // namespace similie::physics
