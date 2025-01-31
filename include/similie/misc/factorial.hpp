// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: MIT

#pragma once

#include <ddc/ddc.hpp>

namespace sil {

namespace misc {

// From https://stackoverflow.com/a/44719219
constexpr inline std::size_t factorial(std::size_t k) noexcept
{
    return (k <= 1) ? 1 : k * factorial(k - 1);
}

} // namespace misc

} // namespace sil
