// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <ddc/ddc.hpp>

#include "are_all_same.hpp"
#include "chain.hpp"
#include "specialization.hpp"

namespace sil {

namespace form {

/// Cochain class
template <
        misc::Specialization<Chain> ChainType,
        class ElementType = double,
        class Allocator = std::allocator<ElementType>>
class Cochain : public std::vector<ElementType, Allocator>
{
public:
    using chain_type = ChainType;
    using element_type = ElementType;
    using base_type = std::vector<ElementType, Allocator>;

private:
    ChainType const& m_chain;

public:
    KOKKOS_FUNCTION constexpr explicit Cochain(ChainType& chain) noexcept
        : base_type {}
        , m_chain(chain)
    {
    }

    template <class... T>
        requires misc::are_all_same<T...>
    KOKKOS_FUNCTION constexpr explicit Cochain(ChainType& chain, T... value) noexcept
        : base_type {value...}
        , m_chain(chain)
    {
    }

    KOKKOS_FUNCTION constexpr explicit Cochain(
            ChainType& chain,
            std::vector<ElementType> simplices) noexcept
        : base_type(simplices)
        , m_chain(chain)
    {
    }

    static KOKKOS_FUNCTION constexpr std::size_t dimension() noexcept
    {
        return ChainType::dimension();
    }

    KOKKOS_FUNCTION ChainType& chain() noexcept
    {
        return m_chain;
    }

    KOKKOS_FUNCTION ChainType const& chain() const noexcept
    {
        return m_chain;
    }

    KOKKOS_FUNCTION auto const simplex_it(auto it) const noexcept
    {
        return m_chain.begin() + std::distance(this->begin(), it);
    }

    KOKKOS_FUNCTION auto simplex_it(auto it) noexcept
    {
        return m_chain.begin() + std::distance(this->begin(), it);
    }

    KOKKOS_FUNCTION element_type operator()()
    {
        element_type out = 0;
        for (auto i = this->begin(); i < this->end(); ++i) {
            out += (simplex_it(i)->negative() ? -1 : 1) * *i;
        }
        return out;
    }
};

template <misc::Specialization<Cochain> CochainType>
std::ostream& operator<<(std::ostream& out, CochainType const& cochain)
{
    out << "[\n";
    for (auto i = cochain.begin(); i < cochain.end(); ++i) {
        out << " " << *cochain.simplex_it(i) << ": " << *i << "\n";
    }
    out << "]";
    return out;
}

template <
        misc::Specialization<Chain> ChainType,
        class ElementType = double,
        class Allocator = std::allocator<ElementType>>
using Form = Cochain<ChainType, ElementType, Allocator>;

} // namespace form

} // namespace sil