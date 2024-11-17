// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <ddc/ddc.hpp>

#include "are_all_same.hpp"
#include "chain.hpp"
#include "local_chain.hpp"
#include "specialization.hpp"

namespace sil {

namespace exterior {

template <class CochainType, class SupportType, class LayoutStridedPolicy, class MemorySpace>
class StructuredCochain;

} // namespace exterior

} // namespace sil

// @cond

namespace ddc {

template <class CochainType, class SupportType, class LayoutStridedPolicy, class MemorySpace>
inline constexpr bool enable_chunk<
        sil::exterior::
                StructuredCochain<CochainType, SupportType, LayoutStridedPolicy, MemorySpace>>
        = true;

template <class CochainType, class SupportType, class LayoutStridedPolicy, class MemorySpace>
inline constexpr bool enable_borrowed_chunk<
        sil::exterior::
                StructuredCochain<CochainType, SupportType, LayoutStridedPolicy, MemorySpace>>
        = true;

} // namespace ddc

// @endcond

namespace sil {

namespace exterior {

/// Cochain class
template <class CochainType, class... DDim, class LayoutStridedPolicy, class MemorySpace>
class StructuredCochain<CochainType, ddc::DiscreteDomain<DDim...>, LayoutStridedPolicy, MemorySpace>
    : public ddc::
              ChunkSpan<CochainType, ddc::DiscreteDomain<DDim...>, LayoutStridedPolicy, MemorySpace>
{
protected:
    using base_type = ddc::
            ChunkSpan<CochainType, ddc::DiscreteDomain<DDim...>, LayoutStridedPolicy, MemorySpace>;

public:
    using ddc::
            ChunkSpan<CochainType, ddc::DiscreteDomain<DDim...>, LayoutStridedPolicy, MemorySpace>::
                    ChunkSpan;
    using reference = ddc::
            ChunkSpan<CochainType, ddc::DiscreteDomain<DDim...>, LayoutStridedPolicy, MemorySpace>::
                    reference;
    using discrete_domain_type = ddc::
            ChunkSpan<CochainType, ddc::DiscreteDomain<DDim...>, LayoutStridedPolicy, MemorySpace>::
                    discrete_domain_type;
    using discrete_element_type = ddc::
            ChunkSpan<CochainType, ddc::DiscreteDomain<DDim...>, LayoutStridedPolicy, MemorySpace>::
                    discrete_element_type;

    using ddc::
            ChunkSpan<CochainType, ddc::DiscreteDomain<DDim...>, LayoutStridedPolicy, MemorySpace>::
                    domain;
    using ddc::
            ChunkSpan<CochainType, ddc::DiscreteDomain<DDim...>, LayoutStridedPolicy, MemorySpace>::
            operator();

    KOKKOS_FUNCTION constexpr explicit StructuredCochain(ddc::ChunkSpan<
                                                         CochainType,
                                                         ddc::DiscreteDomain<DDim...>,
                                                         LayoutStridedPolicy,
                                                         MemorySpace> other) noexcept
        : base_type(other)
    {
    }
    /*
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
        for (auto i = this->begin(); i < this->end(); ++i) {
            if constexpr (misc::Specialization<chain_type, Chain>) {
                out += (chain_it(i)->negative() ? -1 : 1) * *i;
            } else if (misc::Specialization<chain_type, LocalChain>) {
                out += (m_chain.negative() ? -1 : 1) * *i; // always false
            }
        }
        return out;
    }
    */
};

/*
template <misc::Specialization<Cochain> CochainType>
std::ostream& operator<<(std::ostream& out, CochainType const& cochain)
{
    out << "[\n";
    for (auto i = cochain.begin(); i < cochain.end(); ++i) {
        if constexpr (misc::Specialization<typename CochainType::chain_type, Chain>) {
            out << " " << *cochain.chain_it(i) << " : " << *i << "\n";
        } else if (misc::Specialization<typename CochainType::chain_type, LocalChain>) {
            out << " " << cochain.chain().discrete_element() << " -> "
                << cochain.chain().discrete_element() + *cochain.chain_it(i) << " : " << *i << "\n";
        }
    }

    out << "]";

    return out;
}
*/

} // namespace exterior

} // namespace sil
