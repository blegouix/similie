// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#pragma once

#include <array>
#include <concepts>
#include <cstddef>
#include <type_traits>

#include <Kokkos_Core.hpp>

namespace similie::physics {

namespace detail {

struct ConceptIndex
{
};

struct ConceptElement
{
};

template <class Dimension>
struct ConceptGrid
{
    using continuous_dimension_type = Dimension;
};

struct ConceptParameter
{
    template <class Element, class Component>
    constexpr double operator()(Element, Component) const
    {
        return 1;
    }
};

// A complete state keeps coupled components available to every operation.
struct ConceptHamiltonian
{
    static constexpr std::size_t N = 1;
    static constexpr bool IS_LINEAR = true;
    struct State
    {
        std::array<double, 1> potential {};
        std::array<double, N> moments {};
    };

    template <class Element>
    KOKKOS_FUNCTION constexpr double hamiltonian(State const& state, Element) const
    {
        return (state.potential[0] * state.potential[0] + state.moments[0] * state.moments[0]) / 2;
    }
    template <class Index, class Element>
    KOKKOS_FUNCTION constexpr double dhamiltonian_dmoments(State const& state, Element) const
    {
        return state.moments[Index::value];
    }
    template <class Index, class Element>
    KOKKOS_FUNCTION constexpr double dhamiltonian_dpotential(State const& state, Element) const
    {
        return state.potential[Index::value];
    }
    template <class Row, class Column, class Element>
    KOKKOS_FUNCTION constexpr double jacobian(State const&, Element) const
    {
        return Row::value == Column::value ? 1 : 0;
    }
    template <class Index, class Element>
    KOKKOS_FUNCTION constexpr double dhamiltonian_dmoments_value(State const&, Element) const
    {
        return 1;
    }
    template <class Index, class Element>
    KOKKOS_FUNCTION constexpr double dhamiltonian_dpotential_value(State const&, Element) const
    {
        return 1;
    }
};

// Namespace-level concepts avoid nvcc rejecting dependent member-template calls
// in requires-expressions declared inside a class template.
template <class Hamiltonian, class Index, class Element>
concept HasMomentCoefficient = requires(
        Hamiltonian const& h,
        typename Hamiltonian::State const& state,
        Element element) { h.template dhamiltonian_dmoments_value<Index>(state, element); };

template <class Hamiltonian, class Index, class Element>
concept HasPotentialCoefficient = requires(
        Hamiltonian const& h,
        typename Hamiltonian::State const& state,
        Element element) { h.template dhamiltonian_dpotential_value<Index>(state, element); };

} // namespace detail

// Every Hamiltonian has the same arguments; quantities belong to its State.
template <class Hamiltonian, class Element>
concept HamiltonianConcept = requires(
        Hamiltonian const& h,
        typename Hamiltonian::State const& state,
        Element element) {
    requires std::same_as<std::remove_cv_t<decltype(Hamiltonian::N)>, std::size_t>;
    requires std::same_as<std::remove_cv_t<decltype(Hamiltonian::IS_LINEAR)>, bool>;
    typename std::integral_constant<std::size_t, Hamiltonian::N>;
    typename std::bool_constant<Hamiltonian::IS_LINEAR>;
    { h.hamiltonian(state, element) } -> std::same_as<double>;
    {
        h.template dhamiltonian_dmoments<std::integral_constant<std::size_t, 0>>(state, element)
    } -> std::same_as<double>;
    {
        h.template dhamiltonian_dpotential<std::integral_constant<std::size_t, 0>>(state, element)
    } -> std::same_as<double>;
    {
        h.template jacobian<
                std::integral_constant<std::size_t, 0>,
                std::integral_constant<std::size_t, 0>>(state, element)
    } -> std::same_as<double>;
};

template <class Hamiltonian, class Element>
concept NonlocalHamiltonianConcept
        = HamiltonianConcept<Hamiltonian, Element>
          && requires(
                  Hamiltonian const& h,
                  typename Hamiltonian::State const& state,
                  Element element) {
                 h.template dhamiltonian_dmoments_value<
                         std::integral_constant<std::size_t, 0>>(state, element);
             };

template <class Equations, class Element>
concept HamiltonEquationsConcept = requires(
        Equations const& equations,
        typename Equations::State const& state,
        Element element) {
    {
        equations.template dpotential_dt<std::integral_constant<std::size_t, 0>>(state, element)
    } -> std::same_as<double>;
    {
        equations.template dmoments_dt<std::integral_constant<std::size_t, 0>>(state, element)
    } -> std::same_as<double>;
    {
        equations.template jacobian<
                std::integral_constant<std::size_t, 0>,
                std::integral_constant<std::size_t, 0>>(state, element)
    } -> std::same_as<double>;
};

template <class Equations, class Element>
concept MomentCoefficientHamiltonEquationsConcept
        = HamiltonEquationsConcept<Equations, Element>
          && requires(
                  Equations const& equations,
                  typename Equations::State const& state,
                  Element element) {
                 equations.template dpotential_dt_value<
                         std::integral_constant<std::size_t, 0>>(state, element);
             };

template <class Equations, class Element>
concept PotentialCoefficientHamiltonEquationsConcept
        = HamiltonEquationsConcept<Equations, Element>
          && requires(
                  Equations const& equations,
                  typename Equations::State const& state,
                  Element element) {
                 equations.template dmoments_dt_value<
                         std::integral_constant<std::size_t, 0>>(state, element);
             };

template <class Equations, class Element>
concept DeDonderWeylEquationsConcept = requires(
        Equations const& equations,
        typename Equations::State const& state,
        Element element) {
    {
        equations.template potential_grad<std::integral_constant<std::size_t, 0>>(state, element)
    } -> std::same_as<double>;
    {
        equations.template moments_div<std::integral_constant<std::size_t, 0>>(state, element)
    } -> std::same_as<double>;
};

// Scalar and vector laws use arrays of the same form, with N components.
template <class Law, class Element>
concept ConstitutiveLawConcept
        = requires(Law const& law, typename Law::State const& state, Element element) {
              typename std::integral_constant<std::size_t, Law::N>;
              { law(state, element) } -> std::same_as<std::array<double, Law::N>>;
              { law.value(state, element) } -> std::same_as<std::array<double, Law::N * Law::N>>;
              { law.jacobian(state, element) } -> std::same_as<std::array<double, Law::N * Law::N>>;
              { law.inverse(state, element) } -> std::same_as<std::array<double, Law::N>>;
              {
                  law.inverse_value(state, element)
              } -> std::same_as<std::array<double, Law::N * Law::N>>;
          };

} // namespace similie::physics
