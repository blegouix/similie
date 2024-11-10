// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <concepts>

namespace sil {

namespace misc {

template <template <typename...> class T, typename U>
struct is_specialization_of : std::false_type
{
};

template <template <typename...> class T, typename... Us>
struct is_specialization_of<T, T<Us...>> : std::true_type
{
};

template <typename U, template <typename...> class T>
concept Specialization = is_specialization_of<T, U>::value;

} // namespace misc

} // namespace sil
