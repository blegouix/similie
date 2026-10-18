// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later

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
KOKKOS_FUNCTION inline constexpr int permutation_parity(std::array<std::size_t, N> const& lst)
{
    int parity = 1;
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t j = i + 1; j < N; ++j) {
            if (lst[i] == lst[j])
                return 0;
            if (lst[i] > lst[j])
                parity = -parity;
        }
    return parity;
}

} // namespace misc

} // namespace sil
