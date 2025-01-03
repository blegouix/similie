// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <Kokkos_Core.hpp>

namespace sil {

namespace misc {

namespace detail {

// Not sure why KOKKOS_FUNCTION is required despite constexpr
template <class InputIt, class T = typename std::iterator_traits<InputIt>::value_type>
KOKKOS_FUNCTION constexpr InputIt find(InputIt first, InputIt last, const T& value)
{
    for (; first != last; ++first)
        if (*first == value)
            return first;

    return last;
}

/*
template <class InputIt, class UnaryPred>
KOKKOS_FUNCTION constexpr InputIt find_if(InputIt first, InputIt last, UnaryPred p)
{
    for (; first != last; ++first)
        if (p(*first))
            return first;

    return last;
}

template <class InputIt, class UnaryPred>
KOKKOS_FUNCTION constexpr InputIt find_if_not(InputIt first, InputIt last, UnaryPred q)
{
    for (; first != last; ++first)
        if (!q(*first))
            return first;

    return last;
}

template <class InputIt, class UnaryPred>
KOKKOS_FUNCTION constexpr bool all_of(InputIt first, InputIt last, UnaryPred p)
{
    return find_if_not(first, last, p) == last;
}
*/

template <class T>
constexpr std::remove_reference_t<T>&& move(T&& t) noexcept
{
    return static_cast<typename std::remove_reference<T>::type&&>(t);
}

template <class InputIt, class OutputIt>
KOKKOS_FUNCTION OutputIt move(InputIt first, InputIt last, OutputIt d_first)
{
    for (; first != last; ++d_first, ++first)
        *d_first = move(*first);

    return d_first;
}

template <class I>
KOKKOS_FUNCTION constexpr std::size_t bounded_advance(I& i, std::size_t n, I const bound)
{
    for (; n > 0 && i != bound; --n, void(++i)) {
        ;
    }

    return n;
}

template <class ForwardIt>
KOKKOS_FUNCTION ForwardIt shift_left(ForwardIt first, ForwardIt last, std::size_t n)
{
    if (n <= 0) {
        return last;
    }

    auto mid = first;
    if (bounded_advance(mid, n, last)) {
        return first;
    }

    return move(move(mid), move(last), move(first));
}

template <class InputIt, class OutputIt>
constexpr OutputIt copy(InputIt first, InputIt last, OutputIt d_first)
{
    for (; first != last; (void)++first, (void)++d_first)
        *d_first = *first;

    return d_first;
}


template <typename It, typename Compare = std::less<>>
constexpr void sort(It begin, It end, Compare comp = Compare())
{
    for (It i = begin; i != end; ++i) {
        for (It j = begin; j < end - 1; ++j) {
            if (comp(*(j + 1), *j)) {
                Kokkos::kokkos_swap(*j, *(j + 1));
            }
        }
    }
}

} // namespace detail

} // namespace misc

} // namespace sil
