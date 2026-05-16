// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <type_traits>
#include <utility>

namespace similie::physics {

namespace detail {

struct NoPiComputerValue
{
};

template <class Hamiltonian, class = void>
struct HamiltonianValueComputerType
{
    using type = NoPiComputerValue;
};

template <class Hamiltonian>
struct HamiltonianValueComputerType<Hamiltonian, std::void_t<typename Hamiltonian::value_computer_type>>
{
    using type = typename Hamiltonian::value_computer_type;
};

} // namespace detail

template <
        class Hamiltonian,
        class PiComputerValue = typename detail::HamiltonianValueComputerType<Hamiltonian>::type>
class HamiltonEquations
{
    Hamiltonian m_hamiltonian;
    PiComputerValue m_pi_computer_value;

public:
    constexpr explicit HamiltonEquations(Hamiltonian hamiltonian)
        : m_hamiltonian(std::move(hamiltonian))
        , m_pi_computer_value(PiComputerValue())
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

    template <std::size_t I, class ChainType, class LowerChainType, class Elem>
    [[nodiscard]] constexpr auto
    dpotential_dt_value(ChainType chain, LowerChainType lower_chain, Elem elem) const
    {
        return m_hamiltonian.template dH_dpi_value<I>(chain, lower_chain, elem, m_pi_computer_value);
    }

    template <std::size_t I = 0, class ChainType, class LowerChainType, class Elem>
    [[nodiscard]] constexpr auto
    dmoments_dt_value(ChainType chain, LowerChainType lower_chain, Elem elem) const
    {
        static_cast<void>(I);
        return -m_hamiltonian.dH_dphi_value(chain, lower_chain, elem, m_pi_computer_value);
    }
};

} // namespace similie::physics
