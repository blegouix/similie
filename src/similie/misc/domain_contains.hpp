// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

#include <ddc/ddc.hpp>

#include "select_from_type_seq.hpp"
#include "type_seq_ext.hpp"

namespace sil {

namespace misc {

namespace detail {

template <class Seq>
struct IsInDomain;

template <class... DDim>
struct IsInDomain<ddc::TypeSeq<DDim...>>
{
    static KOKKOS_FUNCTION bool operator()(
            ddc::DiscreteDomain<DDim...> dom,
            ddc::DiscreteElement<DDim...> elem)
    {
        return ((elem.template uid<DDim>() >= dom.front().template uid<DDim>()) && ...)
               && ((elem.template uid<DDim>() <= dom.back().template uid<DDim>()) && ...);
    }
};

} // namespace detail

template <class... DDim, class... ODDim>
KOKKOS_FUNCTION bool domain_contains(
        ddc::DiscreteDomain<DDim...> dom,
        ddc::DiscreteElement<ODDim...> elem)
{
    return detail::IsInDomain<
            misc::type_seq_intersect_t<ddc::TypeSeq<DDim...>, ddc::TypeSeq<ODDim...>>>::
    operator()(
            select_from_type_seq<
                    misc::type_seq_intersect_t<ddc::TypeSeq<ODDim...>, ddc::TypeSeq<DDim...>>>(dom),
            select_from_type_seq<
                    misc::type_seq_intersect_t<ddc::TypeSeq<ODDim...>, ddc::TypeSeq<DDim...>>>(
                    elem));
}

} // namespace misc

} // namespace sil
