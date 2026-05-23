// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <span>
#include <type_traits>
#include <utility>

#include <ddc/ddc.hpp>

#include <Kokkos_Core.hpp>

namespace similie::physics {

namespace detail {

template <class Hamiltonian, std::size_t I>
inline constexpr bool has_component_dmoments_v
        = requires(Hamiltonian const& h) { h.template dhamiltonian_dmoments<I>(0.0); };

template <class Hamiltonian, std::size_t I>
inline constexpr bool has_span_dmoments_v
        = requires(Hamiltonian const& h, std::span<double const, Hamiltonian::N> moments) {
              h.template dhamiltonian_dmoments<I>(moments);
          };

template <class Hamiltonian, std::size_t I, class Elem>
inline constexpr bool has_elem_dmoments_v = requires(Hamiltonian const& h, Elem elem) {
    h.template dhamiltonian_dmoments<I>(0.0, elem);
};

template <class Hamiltonian, std::size_t I, class Elem>
inline constexpr bool has_elem_span_dmoments_v = requires(
        Hamiltonian const& h,
        std::span<double const, Hamiltonian::N> moments,
        Elem elem) { h.template dhamiltonian_dmoments<I>(moments, elem); };

template <class Hamiltonian, class Elem>
inline constexpr bool has_elem_dpotential_v
        = requires(Hamiltonian const& h, Elem elem) { h.dhamiltonian_dpotential(0.0, elem); };

} // namespace detail

template <class Hamiltonian>
class HamiltonEquations
{
    Hamiltonian m_hamiltonian;

public:
    static constexpr bool IS_LINEAR = Hamiltonian::IS_LINEAR;

    KOKKOS_FUNCTION constexpr explicit HamiltonEquations(Hamiltonian hamiltonian)
        : m_hamiltonian(std::move(hamiltonian))
    {
    }

    template <std::size_t I>
    [[nodiscard]] KOKKOS_FUNCTION constexpr double dpotential_dt(
            std::span<double const, Hamiltonian::N> spatial_moments) const
    {
        if constexpr (detail::has_component_dmoments_v<Hamiltonian, I>) {
            return m_hamiltonian.template dhamiltonian_dmoments<I>(spatial_moments[I]);
        } else {
            return m_hamiltonian.template dhamiltonian_dmoments<I>(spatial_moments);
        }
    }

    template <std::size_t I, class Elem>
    [[nodiscard]] KOKKOS_FUNCTION constexpr double dpotential_dt(
            std::span<double const, Hamiltonian::N> spatial_moments,
            Elem elem) const
    {
        if constexpr (detail::has_elem_dmoments_v<Hamiltonian, I, Elem>) {
            return m_hamiltonian.template dhamiltonian_dmoments<I>(spatial_moments[I], elem);
        } else {
            return m_hamiltonian.template dhamiltonian_dmoments<I>(spatial_moments, elem);
        }
    }

    template <std::size_t I>
    [[nodiscard]] KOKKOS_FUNCTION constexpr double dpotential_dt(
            double spatial_moments_component) const
    {
        return m_hamiltonian.template dhamiltonian_dmoments<I>(spatial_moments_component);
    }

    template <std::size_t I, class Elem>
    [[nodiscard]] KOKKOS_FUNCTION constexpr double dpotential_dt(
            double spatial_moments_component,
            Elem elem) const
    {
        if constexpr (detail::has_elem_dmoments_v<Hamiltonian, I, Elem>) {
            return m_hamiltonian.template dhamiltonian_dmoments<I>(spatial_moments_component, elem);
        } else {
            return m_hamiltonian.template dhamiltonian_dmoments<I>(spatial_moments_component);
        }
    }

    template <std::size_t I>
    [[nodiscard]] KOKKOS_FUNCTION constexpr double dmoments_dt(
            std::span<double const, 1> potential) const
    {
        static_cast<void>(I);
        return -m_hamiltonian.dhamiltonian_dpotential(potential[0]);
    }

    template <std::size_t I, class Elem>
    [[nodiscard]] KOKKOS_FUNCTION constexpr double dmoments_dt(
            std::span<double const, 1> potential,
            Elem elem) const
    {
        static_cast<void>(I);
        if constexpr (detail::has_elem_dpotential_v<Hamiltonian, Elem>) {
            return -m_hamiltonian.dhamiltonian_dpotential(potential[0], elem);
        } else {
            static_cast<void>(elem);
            return dmoments_dt<I>(potential);
        }
    }

    template <std::size_t I>
    [[nodiscard]] KOKKOS_FUNCTION constexpr double dmoments_dt(double potential) const
    {
        static_cast<void>(I);
        return -m_hamiltonian.dhamiltonian_dpotential(potential);
    }

    template <std::size_t I, class Elem>
    [[nodiscard]] KOKKOS_FUNCTION constexpr double dmoments_dt(double potential, Elem elem) const
    {
        static_cast<void>(I);
        if constexpr (detail::has_elem_dpotential_v<Hamiltonian, Elem>) {
            return -m_hamiltonian.dhamiltonian_dpotential(potential, elem);
        } else {
            static_cast<void>(elem);
            return -m_hamiltonian.dhamiltonian_dpotential(potential);
        }
    }

    template <std::size_t I, class Elem>
    [[nodiscard]] KOKKOS_FUNCTION constexpr auto dpotential_dt_value(Elem elem) const
    {
        return m_hamiltonian.template dhamiltonian_dmoments_value<I>(elem);
    }

    template <std::size_t I, std::size_t J, class Elem>
    [[nodiscard]] KOKKOS_FUNCTION constexpr double dpotential_dt_tangent(
            std::span<double const, Hamiltonian::N> moments,
            Elem elem) const
    {
        return m_hamiltonian.template d2hamiltonian_dmoments2<I, J>(moments, elem);
    }

    template <std::size_t I = 0, class Elem>
        requires requires(Hamiltonian const& h, Elem e) { h.dhamiltonian_dpotential_value(e); }
    [[nodiscard]] KOKKOS_FUNCTION constexpr auto dmoments_dt_value(Elem elem) const
    {
        static_cast<void>(I);
        auto value = m_hamiltonian.dhamiltonian_dpotential_value(elem);
        if constexpr (std::is_same_v<std::remove_cvref_t<decltype(value)>, double>) {
            return -value;
        } else {
            ddc::device_for_each(value.domain(), [&](auto mem_elem) { value(mem_elem) *= -1.0; });
            return value;
        }
    }
};

} // namespace similie::physics
