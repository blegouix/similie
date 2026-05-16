// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <type_traits>
#include <utility>

namespace similie::physics {

struct PotentialTimeDerivative
{
};

struct MomentumTimeDerivative
{
};

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
    [[nodiscard]] constexpr double dpotential_dt(double spatial_moments_component) const
    {
        return m_hamiltonian.template dH_dpi<I>(spatial_moments_component);
    }

    template <std::size_t I>
    [[nodiscard]] constexpr double dmoments_dt(double potential) const
    {
        static_cast<void>(I);
        return -m_hamiltonian.dH_dphi(potential);
    }

    template <class EquationTerm, std::size_t I = 0>
    [[nodiscard]] constexpr double value(double variable) const
    {
        if constexpr (std::is_same_v<EquationTerm, PotentialTimeDerivative>) {
            return m_hamiltonian.template dH_dpi_value<I>(variable);
        } else {
            static_cast<void>(I);
            return -m_hamiltonian.dH_dphi_value(variable);
        }
    }
};

} // namespace similie::physics
