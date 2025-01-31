// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: MIT

#pragma once

#include <ddc/ddc.hpp>

namespace sil {

namespace misc {

namespace detail {

template <class T>
struct ToTypeSeq
{
    using type = ddc::detail::TypeSeq<>;
};

template <template <class...> class T, class... Arg>
struct ToTypeSeq<T<Arg...>>
{
    using type = ddc::detail::TypeSeq<Arg...>;
};

} // namespace detail

template <class T>
using to_type_seq_t = typename detail::ToTypeSeq<T>::type;

namespace detail {

template <template <class...> class T, class Seq>
struct ConvertTypeSeqTo;

template <template <class...> class T, class... Arg>
struct ConvertTypeSeqTo<T, ddc::detail::TypeSeq<Arg...>>
{
    using type = T<Arg...>;
};

} // namespace detail

template <template <class...> class T, class Seq>
using convert_type_seq_to_t = typename detail::ConvertTypeSeqTo<T, Seq>::type;

} // namespace misc

} // namespace sil
