// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: MIT

#pragma once

#include <ddc/ddc.hpp>

#include "select_from_type_seq.hpp"
#include "type_seq_ext.hpp"

namespace sil {

namespace misc {

namespace detail {

template <class CommonSeq>
struct ClampToDomain;

template <class... CommonDDim>
struct ClampToDomain<ddc::detail::TypeSeq<CommonDDim...>>
{
    template <class BatchDomain, class Elem>
    KOKKOS_FUNCTION static Elem run(BatchDomain const& batch_domain, Elem elem)
    {
        ddc::DiscreteDomain<CommonDDim...> const common_domain
                = select_from_type_seq<ddc::detail::TypeSeq<CommonDDim...>>(batch_domain);
        ddc::DiscreteElement<CommonDDim...> const front = common_domain.front();
        ddc::DiscreteElement<CommonDDim...> const back = common_domain.back();

        ((elem.template uid<CommonDDim>() = std::
                  min(std::max(elem.template uid<CommonDDim>(), front.template uid<CommonDDim>()),
                      back.template uid<CommonDDim>())),
         ...);
        return elem;
    }
};

} // namespace detail

template <class BatchDomain, class Elem>
KOKKOS_FUNCTION Elem clamp_to_domain(BatchDomain const& batch_domain, Elem const& elem)
{
    using CommonSeq
            = misc::type_seq_intersect_t<ddc::to_type_seq_t<Elem>, ddc::to_type_seq_t<BatchDomain>>;
    return detail::ClampToDomain<CommonSeq>::run(batch_domain, elem);
}

} // namespace misc

} // namespace sil
