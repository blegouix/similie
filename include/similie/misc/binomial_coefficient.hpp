// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: MIT

#pragma once

#include <ddc/ddc.hpp>

namespace sil {

namespace misc {

// From https://stackoverflow.com/a/44719219
constexpr inline std::size_t binomial_coefficient(std::size_t n, std::size_t k) noexcept
{
    return (k > n) ? 0 : // out of range
                   (k == 0 || k == n) ? 1
                                      : // edge
                   (k == 1 || k == n - 1) ? n
                                          : // first
                   binomial_coefficient(n - 1, k - 1) * n / k; // recursive
}

} // namespace misc

} // namespace sil
