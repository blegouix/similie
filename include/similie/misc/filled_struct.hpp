// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <ddc/ddc.hpp>

namespace sil {

namespace misc {

namespace detail {

template <class T>
struct FilledStruct;

template <template <class...> class T, class... Arg>
struct FilledStruct<T<Arg...>>
{
    static constexpr T<Arg...> run(auto const n)
    {
        return T<Arg...> {
                n * (ddc::type_seq_rank_v<Arg, ddc::detail::TypeSeq<Arg...>> + 42)
                / (ddc::type_seq_rank_v<Arg, ddc::detail::TypeSeq<Arg...>> + 42)...};
    }
};

} // namespace detail

template <class T, class ElementType = std::size_t>
inline constexpr T filled_struct(ElementType const n = 0)
{
    return detail::FilledStruct<T>::run(n);
}

} // namespace misc

} // namespace sil
