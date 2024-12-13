// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <ddc/ddc.hpp>

namespace sil {

namespace misc {

/*
 Given a permutation of unique digits 0...N, 
 returns its parity: +1 for even parity; -1 for odd.
 If lst contains duplicates, returns 0. If it contains no
 duplicate but is not a permutation of 0...N, this is undetermined.
 */
template <std::size_t N>
inline constexpr int permutation_parity(std::array<std::size_t, N> lst)
{
    int parity = 1;
    for (std::size_t i = 0; i < lst.size() - 1; ++i) {
        if (lst[i] != i) {
            parity *= -1;
            std::size_t mn
                    = std::distance(lst.begin(), std::min_element(lst.begin() + i, lst.end()));
            std::swap(lst[i], lst[mn]);
        }
    }
    if (std::adjacent_find(lst.begin(), lst.end()) != lst.end()) {
        return 0;
    }
    if (lst[lst.size() - 1] == lst[0]) {
        return 0;
    }

    return parity;
}

} // namespace misc

} // namespace sil
