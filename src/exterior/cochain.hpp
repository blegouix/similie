// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <ddc/ddc.hpp>

#include "antisymmetric_tensor.hpp"
#include "are_all_same.hpp"
#include "chain.hpp"
#include "cosimplex.hpp"
#include "local_chain.hpp"
#include "specialization.hpp"

namespace sil {

namespace exterior {

/// Cochain class
template <
        class ChainType,
        class ElementType = double,
        class Allocator = std::allocator<ElementType>>
    requires(misc::Specialization<ChainType, Chain> || misc::Specialization<ChainType, LocalChain>)
class Cochain
{
public:
    using simplex_type = typename ChainType::simplex_type;
    using chain_type = ChainType;
    using element_type = ElementType;
    using elem_type = ChainType::elem_type;
    using vect_type = ChainType::vect_type;
    using values_type = std::vector<ElementType, Allocator>;

private:
    // ChainType const& m_chain;
    ChainType m_chain;
    values_type m_values;

public:
    KOKKOS_FUNCTION constexpr explicit Cochain(ChainType chain) noexcept
        : m_chain(chain)
        , m_values(chain.size())
    {
    }

    template <class... T>
        requires(sizeof...(T) >= 1)
    KOKKOS_FUNCTION constexpr explicit Cochain(ChainType chain, T... value) noexcept
        : m_chain(chain)
        , m_values {value...}
    {
        assert(sizeof...(T) == chain.size()
               && "cochain constructor must get as much values as the chain contains simplices");
    }

    KOKKOS_FUNCTION constexpr explicit Cochain(
            ChainType& chain,
            std::vector<ElementType>& values) noexcept
        : m_chain(chain)
        , m_values(values)
    {
        assert(values.size() == chain.size()
               && "cochain constructor must get as much values as the chain contains simplices");
    }

    template <
            class OElementType,
            misc::Specialization<tensor::TensorAntisymmetricIndex> AntisymmetricIndex,
            class LayoutStridedPolicy,
            class MemorySpace>
    KOKKOS_FUNCTION constexpr explicit Cochain(
            ChainType& chain,
            tensor::Tensor<
                    OElementType,
                    ddc::DiscreteDomain<AntisymmetricIndex>,
                    LayoutStridedPolicy,
                    MemorySpace> tensor) noexcept
        : m_chain(chain)
        , m_values(tensor.domain().size())
    {
        assert(m_values.size() == chain.size()
               && "cochain constructor must get as much values as the chain contains simplices");
        // TODO replace std::vectors with Kokkos::View to replace the pointer (avoid copy)
        for (auto i = m_values.begin(); i < m_values.end(); ++i) {
            *i = tensor.mem(
                    ddc::DiscreteElement<AntisymmetricIndex>(std::distance(m_values.begin(), i)));
        }
    }

    static KOKKOS_FUNCTION constexpr std::size_t dimension() noexcept
    {
        return ChainType::dimension();
    }

    KOKKOS_FUNCTION std::size_t size() noexcept
    {
        return m_chain.size();
    }

    KOKKOS_FUNCTION std::size_t const size() const noexcept
    {
        return m_chain.size();
    }

    KOKKOS_FUNCTION ChainType const& chain() const noexcept
    {
        return m_chain;
    }

    KOKKOS_FUNCTION ChainType const& values() const noexcept
    {
        return m_values;
    }

    KOKKOS_FUNCTION auto begin()
    {
        return m_values.begin();
    }

    KOKKOS_FUNCTION auto begin() const
    {
        return m_values.begin();
    }

    KOKKOS_FUNCTION auto end()
    {
        return m_values.end();
    }

    KOKKOS_FUNCTION auto end() const
    {
        return m_values.end();
    }

    KOKKOS_FUNCTION auto cbegin() const
    {
        return m_values.begin();
    }

    KOKKOS_FUNCTION auto cend() const
    {
        return m_values.end();
    }

    KOKKOS_FUNCTION auto const chain_it(auto it) const noexcept
    {
        return m_chain.begin() + std::distance(m_values.begin(), it);
    }

    KOKKOS_FUNCTION auto chain_it(auto it) noexcept
    {
        return m_chain.begin() + std::distance(m_values.begin(), it);
    }

    KOKKOS_FUNCTION Cosimplex<simplex_type, element_type>& operator[](std::size_t i) noexcept
    {
        return Cosimplex<simplex_type, element_type>(m_chain[i], m_values[i]);
    }

    KOKKOS_FUNCTION Cosimplex<simplex_type, element_type> const& operator[](
            std::size_t i) const noexcept
    {
        return Cosimplex<simplex_type, element_type>(m_chain[i], m_values[i]);
    }

    KOKKOS_FUNCTION element_type integrate() noexcept
    {
        element_type out = 0;
        for (auto i = m_values.begin(); i < m_values.end(); ++i) {
            if constexpr (misc::Specialization<chain_type, Chain>) {
                out += (chain_it(i)->negative() ? -1 : 1) * *i;
            } else if (misc::Specialization<chain_type, LocalChain>) {
                out += (m_chain.negative() ? -1 : 1) * *i; // always false
            }
        }
        return out;
    }

    KOKKOS_FUNCTION element_type const integrate() const noexcept
    {
        element_type out = 0;
        for (auto i = m_values.begin(); i < m_values.end(); ++i) {
            if constexpr (misc::Specialization<chain_type, Chain>) {
                out += (chain_it(i)->negative() ? -1 : 1) * *i;
            } else if (misc::Specialization<chain_type, LocalChain>) {
                out += (m_chain.negative() ? -1 : 1) * *i; // always false
            }
        }
        return out;
    }
};

/*
template <
        class ChainType, misc::Specialization<tensor::Tensor> TensorType>
Cochain(ChainType&, TensorType) -> Cochain<ChainType, typename TensorType::element_type>;
*/

template <misc::Specialization<Cochain> CochainType>
std::ostream& operator<<(std::ostream& out, CochainType const& cochain)
{
    out << "[\n";
    for (auto i = cochain.begin(); i < cochain.end(); ++i) {
        if constexpr (misc::Specialization<typename CochainType::chain_type, Chain>) {
            out << " " << *cochain.chain_it(i) << " : " << *i << "\n";
        } else if (misc::Specialization<typename CochainType::chain_type, LocalChain>) {
            out << " -> " << *cochain.chain_it(i) << " : " << *i << "\n";
        }
    }

    out << "]";

    return out;
}

} // namespace exterior

} // namespace sil
