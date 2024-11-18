// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <ddc/ddc.hpp>

namespace sil {

namespace misc {

namespace detail {

template <class T>
struct NullStruct;

template <template <class...> class T, class... Arg>
struct NullStruct<T<Arg...>>
{
    static constexpr T<Arg...> run()
    {
        return T<Arg...> {0 * ddc::type_seq_rank_v<Arg, ddc::detail::TypeSeq<Arg...>>...};
    }
};

} // namespace detail

template <class T>
inline constexpr T null_struct()
{
    return detail::NullStruct<T>::run();
}

} // namespace misc

} // namespace sil
