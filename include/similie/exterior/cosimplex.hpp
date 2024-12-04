// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <ddc/ddc.hpp>

#include <similie/misc/are_all_same.hpp>
#include <similie/misc/specialization.hpp>

#include "chain.hpp"

namespace sil {

namespace exterior {

/// Cosimplex class
template <class SimplexType, class ElementType = double>
class Cosimplex
{
public:
    using simplex_type = SimplexType;
    using element_type = ElementType;
    using discrete_element_type = SimplexType::discrete_element_type;
    using discrete_vector_type = SimplexType::discrete_vector_type;

private:
    SimplexType m_simplex;
    ElementType m_value;

public:
    KOKKOS_DEFAULTED_FUNCTION constexpr Cosimplex() = default;

    KOKKOS_DEFAULTED_FUNCTION constexpr Cosimplex(Cosimplex const&) = default;

    KOKKOS_FUNCTION constexpr explicit Cosimplex(SimplexType simplex, ElementType value) noexcept
        : m_simplex(simplex)
        , m_value(value)
    {
    }

    KOKKOS_DEFAULTED_FUNCTION ~Cosimplex() = default;

    KOKKOS_DEFAULTED_FUNCTION Cosimplex& operator=(Cosimplex const& other) = default;

    static KOKKOS_FUNCTION constexpr std::size_t dimension() noexcept
    {
        return SimplexType::dimension();
    }

    KOKKOS_FUNCTION SimplexType simplex() noexcept
    {
        return m_simplex;
    }

    KOKKOS_FUNCTION SimplexType const simplex() const noexcept
    {
        return m_simplex;
    }

    KOKKOS_FUNCTION discrete_element_type discrete_element() noexcept
    {
        return m_simplex.discrete_element();
    }

    KOKKOS_FUNCTION discrete_element_type discrete_element() const noexcept
    {
        return m_simplex.discrete_element();
    }

    KOKKOS_FUNCTION discrete_vector_type discrete_vector() noexcept
    {
        return m_simplex.discrete_vector();
    }

    KOKKOS_FUNCTION discrete_vector_type discrete_vector() const noexcept
    {
        return m_simplex.discrete_vector();
    }

    KOKKOS_FUNCTION bool negative() noexcept
    {
        return m_simplex.negative();
    }

    KOKKOS_FUNCTION bool negative() const noexcept
    {
        return m_simplex.negative();
    }

    KOKKOS_FUNCTION element_type value() noexcept
    {
        return m_value;
    }

    KOKKOS_FUNCTION element_type const value() const noexcept
    {
        return m_value;
    }
};

template <misc::Specialization<Cosimplex> CosimplexType>
std::ostream& operator<<(std::ostream& out, CosimplexType const& cosimplex)
{
    out << " " << cosimplex.simplex() << ": " << cosimplex.value() << "\n";
    return out;
}

} // namespace exterior

} // namespace sil
