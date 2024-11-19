// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <ddc/ddc.hpp>

namespace sil {

namespace misc {

namespace detail {

template <class Seq>
struct SelectFromTypeSeq;

template <class... DDim>
struct SelectFromTypeSeq<ddc::detail::TypeSeq<DDim...>>
{
    template <class T>
    static KOKKOS_FUNCTION auto run(T t)
    {
        return ddc::select<DDim...>(t);
    }
};

} // namespace detail

template <class Seq, class T>
KOKKOS_FUNCTION auto select_from_type_seq(T t)
{
    return detail::SelectFromTypeSeq<Seq>::run(t);
}

} // namespace misc

} // namespace sil
