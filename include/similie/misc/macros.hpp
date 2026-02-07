// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: MIT

#pragma once

#include <string_view>

#ifdef SIMILIE_DEBUG_LOG_
#include <iostream>
#endif

namespace sil::misc::detail {

inline void debug_log_launch([[maybe_unused]] std::string_view name)
{
#if defined(SIMILIE_DEBUG_LOG_)
    std::cout << "Launch " << name << std::endl;
#endif
}

} // namespace sil::misc::detail

#define SIMILIE_DEBUG_LOG(name) ::sil::misc::detail::debug_log_launch(name)
