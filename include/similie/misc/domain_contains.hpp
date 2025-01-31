// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: MIT

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
struct IsInDomain<ddc::detail::TypeSeq<DDim...>>
{
    static KOKKOS_FUNCTION bool run(
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
    return detail::IsInDomain<misc::type_seq_intersect_t<
            ddc::detail::TypeSeq<DDim...>,
            ddc::detail::TypeSeq<ODDim...>>>::
            run(select_from_type_seq<misc::type_seq_intersect_t<
                        ddc::detail::TypeSeq<ODDim...>,
                        ddc::detail::TypeSeq<DDim...>>>(dom),
                select_from_type_seq<misc::type_seq_intersect_t<
                        ddc::detail::TypeSeq<ODDim...>,
                        ddc::detail::TypeSeq<DDim...>>>(elem));
}

} // namespace misc

} // namespace sil
