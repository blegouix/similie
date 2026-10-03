// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

#include <ddc/ddc.hpp>

#include <similie/misc/type_seq_conversion.hpp>

namespace sil {

namespace misc {

template <class TagSeqA, class TagSeqB>
using type_seq_intersect_t
        = ddc::type_seq_remove_t<TagSeqA, ddc::type_seq_remove_t<TagSeqA, TagSeqB>>;

namespace detail {

template <class... Domains>
struct CartesianProd
{
};

template <>
struct CartesianProd<>
{
    using type = ddc::TypeSeq<>;
};

template <class Domain, class... Domains>
struct CartesianProd<Domain, Domains...>
{
    using type = ddc::type_seq_cat_t<
            ddc::to_type_seq_t<Domain>,
            typename CartesianProd<Domains...>::type>;
};

} // namespace detail

template <class... Domains>
using cartesian_prod_t
        = convert_type_seq_to_t<ddc::DiscreteDomain, typename detail::CartesianProd<Domains...>::type>;

} // namespace misc

} // namespace sil
