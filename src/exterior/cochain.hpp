// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <ddc/ddc.hpp>

#include "are_all_same.hpp"
#include "chain.hpp"
#include "specialization.hpp"

namespace sil {

namespace exterior {

/// Cochain class
template <
        misc::Specialization<Chain> ChainType,
        class ElementType = double,
        class Allocator = std::allocator<ElementType>>
class Cochain : public std::vector<ElementType, Allocator>
{
public:
    using simplex_type = typename ChainType::simplex_type;
    using chain_type = ChainType;
    using element_type = ElementType;
    using elem_type = ChainType::elem_type;
    using vect_type = ChainType::vect_type;
    using base_type = std::vector<ElementType, Allocator>;

private:
    ChainType const& m_chain;

public:
    KOKKOS_FUNCTION constexpr explicit Cochain(ChainType& chain) noexcept
        : base_type(ChainType::size())
        , m_chain(chain)
    {
    }

    KOKKOS_FUNCTION constexpr explicit Cochain(ChainType&& chain) noexcept
        : base_type(ChainType::size())
        , m_chain(std::move(chain))
    {
    }

    template <class... T>
        requires(sizeof...(T) >= 1)
    KOKKOS_FUNCTION constexpr explicit Cochain(ChainType& chain, T... value) noexcept
        : base_type {value...}
        , m_chain(chain)
    {
        assert(sizeof...(T) == chain.size()
               && "cochain constructor must get as much values as the chain contains simplices");
    }

    template <class... T>
        requires(sizeof...(T) >= 1)
    KOKKOS_FUNCTION constexpr explicit Cochain(ChainType&& chain, T... value) noexcept
        : base_type {value...}
        , m_chain(std::move(chain))
    {
        assert(sizeof...(T) == chain.size()
               && "cochain constructor must get as much values as the chain contains simplices");
    }

    KOKKOS_FUNCTION constexpr explicit Cochain(
            ChainType& chain,
            std::vector<ElementType>& values) noexcept
        : base_type(values)
        , m_chain(chain)
    {
        static_assert(values.size() == chain.size());
    }

    static KOKKOS_FUNCTION constexpr std::size_t dimension() noexcept
    {
        return ChainType::dimension();
    }

    KOKKOS_FUNCTION ChainType const& chain() const noexcept
    {
        return m_chain;
    }

    KOKKOS_FUNCTION auto const chain_it(auto it) const noexcept
    {
        return m_chain.begin() + std::distance(this->begin(), it);
    }

    KOKKOS_FUNCTION auto chain_it(auto it) noexcept
    {
        return m_chain.begin() + std::distance(this->begin(), it);
    }

    KOKKOS_FUNCTION element_type integrate() noexcept
    {
        element_type out = 0;
        for (auto i = this->begin(); i < this->end(); ++i) {
            out += (chain_it(i)->negative() ? -1 : 1) * *i;
        }
        return out;
    }

    KOKKOS_FUNCTION element_type const integrate() const noexcept
    {
        element_type out = 0;
        for (auto i = this->begin(); i < this->end(); ++i) {
            out += (chain_it(i)->negative() ? -1 : 1) * *i;
        }
        return out;
    }
};

template <misc::Specialization<Cochain> CochainType>
std::ostream& operator<<(std::ostream& out, CochainType const& cochain)
{
    out << "[\n";
    for (auto i = cochain.begin(); i < cochain.end(); ++i) {
        out << " " << *cochain.chain_it(i) << ": " << *i << "\n";
    }
    out << "]";
    return out;
}

} // namespace exterior

} // namespace sil
