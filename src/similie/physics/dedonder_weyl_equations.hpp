// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: MIT

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

    template <std::size_t I>
    [[nodiscard]] constexpr double potential_grad(
            std::span<double const, Hamiltonian::N> moments) const
    {
        if constexpr (requires(Hamiltonian const& h) { h.template dH_dmoments<I>(moments[I]); }) {
            return m_hamiltonian.template dH_dmoments<I>(moments[I]);
        } else {
            return m_hamiltonian.template dH_dpi<I>(moments[I]);
        }
    }

    template <std::size_t I>
    [[nodiscard]] constexpr double potential_grad(double moments_component) const
    {
        return m_hamiltonian.template dH_dpi<I>(moments_component);
    }

    template <std::size_t I = 0>
    [[nodiscard]] constexpr double moments_div(std::span<double const, 1> potential) const
    {
        static_cast<void>(I);
        return -m_hamiltonian.dH_dphi(potential[0]);
    }

    [[nodiscard]] constexpr double moments_div(double potential) const
    {
        return -m_hamiltonian.dH_dphi(potential);
    }
};

} // namespace similie::physics
