// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <ddc/ddc.hpp>

#include "are_all_same.hpp"
#include "cochain.hpp"
#include "specialization.hpp"
#include "tensor.hpp"

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
template <
        misc::Specialization<Cochain> CochainType,
        class... DDim,
        class LayoutStridedPolicy,
        class MemorySpace>
class StructuredCochain<CochainType, ddc::DiscreteDomain<DDim...>, LayoutStridedPolicy, MemorySpace>
    : public ddc::
              ChunkSpan<CochainType, ddc::DiscreteDomain<DDim...>, LayoutStridedPolicy, MemorySpace>
{
    // TODO static_assert local cochain
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

    using element_type
            = CochainType; // for consistency with ChunkSpan, but cochain_type is prefered
    using cochain_type = CochainType;
    using simplex_type = typename CochainType::simplex_type;
    using chain_type = CochainType::chain_type;
    using scalar_type = CochainType::element_type;
    using elem_type = CochainType::elem_type;
    using vect_type = CochainType::vect_type;

    KOKKOS_FUNCTION constexpr explicit StructuredCochain(ddc::ChunkSpan<
                                                         CochainType,
                                                         ddc::DiscreteDomain<DDim...>,
                                                         LayoutStridedPolicy,
                                                         MemorySpace> other) noexcept
        : base_type(other)
    {
    }

    KOKKOS_FUNCTION scalar_type& operator()(elem_type elem, vect_type vect) noexcept
    {
        cochain_type& cochain = ddc::ChunkSpan<
                CochainType,
                ddc::DiscreteDomain<DDim...>,
                LayoutStridedPolicy,
                MemorySpace>::operator()(elem);
        for (auto i = cochain.begin(); i < cochain.end(); ++i) {
            if (*cochain.chain_it(i) == vect) {
                return *i;
            }
        }
        assert(false && "vector not found in the local chain");
    }
} // namespace exterior

} // namespace sil
