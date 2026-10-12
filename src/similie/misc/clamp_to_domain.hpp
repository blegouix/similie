// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

#include <algorithm>

#include <ddc/ddc.hpp>

#include "select_from_type_seq.hpp"
#include "signed_discrete_element.hpp"
#include "specialization.hpp"
#include "type_seq_conversion.hpp"
#include "type_seq_ext.hpp"

namespace sil {

namespace misc {

namespace detail {

template <class CommonSeq>
struct ClampToDomain;

template <class... CommonDDim>
struct ClampToDomain<ddc::TypeSeq<CommonDDim...>>
{
    template <class BatchDomain, class Elem>
    KOKKOS_FUNCTION static Elem operator()(BatchDomain const& batch_domain, Elem elem)
    {
        ddc::DiscreteDomain<CommonDDim...> const common_domain
                = select_from_type_seq<ddc::TypeSeq<CommonDDim...>>(batch_domain);
        ddc::DiscreteElement<CommonDDim...> const front = common_domain.front();
        ddc::DiscreteElement<CommonDDim...> const back = common_domain.back();

        if constexpr (misc::Specialization<Elem, SignedDiscreteElement>) {
            ((elem.template uid<CommonDDim>() = std::
                      max(static_cast<std::ptrdiff_t>(front.template uid<CommonDDim>()),
                          std::
                                  min(elem.template uid<CommonDDim>(),
                                      static_cast<std::ptrdiff_t>(
                                              back.template uid<CommonDDim>())))),
             ...);
        } else {
            // Signed offsets distinguish a sample below zero from a large upper index.
            ((elem.template uid<CommonDDim>()
              = (ddc::DiscreteElement<CommonDDim>(elem) - ddc::DiscreteElement<CommonDDim>(front))
                                        .template get<CommonDDim>()
                                < 0
                        ? front.template uid<CommonDDim>()
                        : std::
                                  min(elem.template uid<CommonDDim>(),
                                      back.template uid<CommonDDim>())),
             ...);
        }
        return elem;
    }
};

} // namespace detail

template <class BatchDomain, class Elem>
KOKKOS_FUNCTION Elem clamp_to_domain(BatchDomain const& batch_domain, Elem const& elem)
{
    using CommonSeq = misc::
            type_seq_intersect_t<misc::to_type_seq_t<Elem>, ddc::to_type_seq_t<BatchDomain>>;
    return detail::ClampToDomain<CommonSeq>::operator()(batch_domain, elem);
}

} // namespace misc

} // namespace sil
