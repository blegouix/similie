// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <type_traits>
#include <utility>
#include <variant>

namespace similie::physics {

struct PotentialTimeDerivative
{
};

struct MomentumTimeDerivative
{
};

namespace detail {

struct NoPiComputerValue
{
};

} // namespace detail

template <class Hamiltonian, class PiComputerValue = detail::NoPiComputerValue>
class HamiltonEquations
{
    Hamiltonian m_hamiltonian;
    PiComputerValue m_pi_computer_value;

public:
    constexpr explicit HamiltonEquations(Hamiltonian hamiltonian)
        : m_hamiltonian(std::move(hamiltonian))
        , m_pi_computer_value()
    {
    }

    constexpr explicit HamiltonEquations(Hamiltonian hamiltonian, PiComputerValue pi_computer_value)
        : m_hamiltonian(std::move(hamiltonian))
        , m_pi_computer_value(std::move(pi_computer_value))
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

    template <class EquationTerm, std::size_t I = 0, class ChainType, class LowerChainType, class Elem>
    [[nodiscard]] constexpr auto
    value(ChainType chain, LowerChainType lower_chain, Elem elem) const
    {
        if constexpr (std::is_same_v<EquationTerm, PotentialTimeDerivative>) {
            return m_hamiltonian.template dH_dpi_value<I>(
                    chain,
                    lower_chain,
                    elem,
                    m_pi_computer_value);
        } else {
            static_cast<void>(I);
            return -m_hamiltonian.dH_dphi_value(
                    chain,
                    lower_chain,
                    elem,
                    m_pi_computer_value);
        }
    }
};

} // namespace similie::physics
