// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "specialization.hpp"

namespace sil {

namespace misc {

template <class Head, class... Tail>
inline constexpr bool are_all_same = (std::is_same_v<Head, Tail> && ...);

template <class Head, class... Tail>
inline constexpr bool are_all_equal(Head head, Tail... tail)
{
    return ((head == tail) && ...);
}

template <misc::Specialization<std::vector> T>
inline constexpr bool are_all_equal(T t)
{
    return std::all_of(t.begin(), t.end(), [&](const std::size_t i) { return i == *t.begin(); });
}

} // namespace misc

} // namespace sil
