// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

namespace sil {

namespace misc {

template <class Head, class... Tail>
inline constexpr bool are_all_same = (std::is_same_v<Head, Tail> && ...);

template <class Head, class... Tail>
inline constexpr bool are_all_equal(Head head, Tail... tail)
{
    return ((head == tail) && ...);
}

} // namespace misc

} // namespace sil
