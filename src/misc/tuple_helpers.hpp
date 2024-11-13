// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <ddc/ddc.hpp>

namespace sil {

namespace misc {

// Helper to remove the nth element of a tuple at runtime
template <typename Tuple, std::size_t... Is>
auto remove_impl(const Tuple& tuple, std::size_t N, std::index_sequence<Is...>)
{
    return std::make_tuple((Is < N ? std::get<Is>(tuple) : std::get<Is + 1>(tuple))...);
}

template <typename... Ts>
auto remove(const std::tuple<Ts...>& tuple, std::size_t N)
{
    const std::size_t tuple_size = sizeof...(Ts);
    if (N >= tuple_size) {
        throw std::out_of_range("Index N is out of bounds");
    }
    return remove_impl(tuple, N, std::make_index_sequence<tuple_size - 1> {});
}

} // namespace misc

} // namespace sil
