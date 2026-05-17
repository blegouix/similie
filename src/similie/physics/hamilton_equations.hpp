// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <type_traits>
#include <utility>

#include <ddc/ddc.hpp>

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
    [[nodiscard]] constexpr double dpotential_dt(double spatial_moments_component) const
    {
        return m_hamiltonian.template dH_dmoments<I>(spatial_moments_component);
    }

    template <std::size_t I, class Elem>
    [[nodiscard]] constexpr double dpotential_dt(double spatial_moments_component, Elem elem) const
    {
        if constexpr (requires(Hamiltonian const& h) {
                          h.template dH_dmoments<I>(spatial_moments_component, elem);
                      }) {
            return m_hamiltonian.template dH_dmoments<I>(spatial_moments_component, elem);
        } else {
            static_cast<void>(elem);
            return m_hamiltonian.template dH_dmoments<I>(spatial_moments_component);
        }
    }

    template <std::size_t I>
    [[nodiscard]] constexpr double dmoments_dt(double potential) const
    {
        static_cast<void>(I);
        return -m_hamiltonian.dH_dpotential(potential);
    }

    template <std::size_t I, class Elem>
    [[nodiscard]] constexpr double dmoments_dt(double potential, Elem elem) const
    {
        static_cast<void>(I);
        if constexpr (requires(Hamiltonian const& h) { h.dH_dpotential(potential, elem); }) {
            return -m_hamiltonian.dH_dpotential(potential, elem);
        } else {
            static_cast<void>(elem);
            return -m_hamiltonian.dH_dpotential(potential);
        }
    }

    template <std::size_t I, class Elem>
    [[nodiscard]] constexpr auto dpotential_dt_value(Elem elem) const
    {
        return m_hamiltonian.template dH_dmoments_value<I>(elem);
    }

    template <std::size_t I = 0, class Elem>
        requires requires(Hamiltonian const& h, Elem e) { h.dH_dpotential_value(e); }
    [[nodiscard]] constexpr auto dmoments_dt_value(Elem elem) const
    {
        static_cast<void>(I);
        auto value = m_hamiltonian.dH_dpotential_value(elem);
        if constexpr (std::is_same_v<std::remove_cvref_t<decltype(value)>, double>) {
            return -value;
        } else {
            ddc::device_for_each(value.domain(), [&](auto mem_elem) { value(mem_elem) *= -1.0; });
            return value;
        }
    }
};

} // namespace similie::physics
