// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: MIT

#pragma once

#include <ddc/ddc.hpp>

#include <similie/misc/are_all_same.hpp>
#include <similie/misc/portable_stl.hpp>
#include <similie/misc/specialization.hpp>

#include <Kokkos_StdAlgorithms.hpp>

#include "simplex.hpp"

namespace sil {

namespace exterior {

/// Chain class
template <
        class SimplexType,
        class LayoutStridedPolicy = Kokkos::LayoutRight,
        class MemorySpace = Kokkos::HostSpace>
class Chain
{
public:
    using memory_space = MemorySpace;

    using simplex_type = SimplexType;
    using simplices_type = Kokkos::View<SimplexType*, LayoutStridedPolicy, memory_space>;
    using discrete_element_type = typename simplex_type::discrete_element_type;
    using discrete_vector_type = typename simplex_type::discrete_vector_type;

    using iterator_type = Kokkos::Experimental::Impl::RandomAccessIterator<simplices_type>;

private:
    static constexpr bool s_is_local = false;
    static constexpr std::size_t s_k = simplex_type::dimension();
    simplices_type m_simplices;
    std::size_t
            m_size; // Effective size, m_simplices elements between m_size and m_simplices.size() are undefined

public:
    KOKKOS_DEFAULTED_FUNCTION constexpr Chain() = default;

    KOKKOS_DEFAULTED_FUNCTION constexpr Chain(Chain const&) = default;

    KOKKOS_DEFAULTED_FUNCTION constexpr Chain(Chain&&) = default;

    template <class... T>
        requires(!std::is_convertible_v<T, std::size_t> && ...)
    KOKKOS_FUNCTION constexpr explicit Chain(simplices_type allocation, T... simplex) noexcept
        : m_simplices(std::move(allocation))
        , m_size(sizeof...(T))
    {
        std::size_t i = 0;
        ((m_simplices(i++) = simplex), ...);
        assert(check() == 0 && "there are duplicate simplices in the chain");
    }

    KOKKOS_FUNCTION constexpr explicit Chain(simplices_type allocation, std::size_t size) noexcept
        : m_simplices(std::move(allocation))
        , m_size(size)
    {
        assert(check() == 0 && "there are duplicate simplices in the chain");
    }

    KOKKOS_DEFAULTED_FUNCTION ~Chain() = default;

    KOKKOS_DEFAULTED_FUNCTION Chain& operator=(Chain const& other) = default;

    KOKKOS_DEFAULTED_FUNCTION Chain& operator=(Chain&& other) = default;

    static KOKKOS_FUNCTION constexpr bool is_local() noexcept
    {
        return s_is_local;
    }

    static KOKKOS_FUNCTION constexpr std::size_t dimension() noexcept
    {
        return s_k;
    }

    KOKKOS_FUNCTION simplices_type& allocation() noexcept
    {
        return m_simplices;
    }

    KOKKOS_FUNCTION std::size_t size() noexcept
    {
        return m_size;
    }

    KOKKOS_FUNCTION std::size_t size() const noexcept
    {
        return m_size;
    }

    KOKKOS_FUNCTION std::size_t allocation_size() noexcept
    {
        return m_simplices.size();
    }

    KOKKOS_FUNCTION std::size_t allocation_size() const noexcept
    {
        return m_simplices.size();
    }

    void resize()
    {
        Kokkos::resize(m_simplices, size());
    }

    void resize(std::size_t size)
    {
        Kokkos::resize(m_simplices, size);
    }

    KOKKOS_FUNCTION int check()
    {
        for (auto i = begin(); i < end() - 1; ++i) {
            for (auto j = i + 1; j < end(); ++j) {
                if (*i == *j) {
                    return -1;
                }
            }
        }
        return 0;
    }

    KOKKOS_FUNCTION void optimize()
    {
        auto i = begin();
        auto stop = end();
        while (i < stop - 1) {
            auto k = i;
            for (auto j = i + 1; k == i && j < stop; ++j) {
                if (*i == -*j) {
                    k = j;
                }
            }
            if (k != i) {
                misc::detail::shift_left(k, stop, 1);
                misc::detail::shift_left(i, stop, 1);
                m_size -= 2;
                stop = end();
            } else {
                i++;
            }
        }
        assert(check() == 0 && "there are duplicate simplices in the chain");
    }

    KOKKOS_FUNCTION auto begin()
    {
        return Kokkos::Experimental::begin(m_simplices);
    }

    KOKKOS_FUNCTION auto begin() const
    {
        return Kokkos::Experimental::begin(m_simplices);
    }

    KOKKOS_FUNCTION auto end()
    {
        return Kokkos::Experimental::begin(m_simplices) + size();
    }

    KOKKOS_FUNCTION auto end() const
    {
        return Kokkos::Experimental::begin(m_simplices) + size();
    }

    KOKKOS_FUNCTION auto cbegin() const
    {
        return Kokkos::Experimental::begin(m_simplices);
    }

    KOKKOS_FUNCTION auto cend() const
    {
        return Kokkos::Experimental::begin(m_simplices) + size();
    }

    KOKKOS_FUNCTION simplex_type& operator[](std::size_t i) noexcept
    {
        assert(i < size());
        return m_simplices(i);
    }

    KOKKOS_FUNCTION simplex_type const& operator[](std::size_t i) const noexcept
    {
        assert(i < size());
        return m_simplices(i);
    }

    KOKKOS_FUNCTION Chain& operator++()
    {
        assert(size() < allocation_size());
        m_size++;
        return *this;
    }

    KOKKOS_FUNCTION Chain& operator+=(const std::size_t n)
    {
        assert(size() + n <= allocation_size());
        m_size += n;
        return *this;
    }

    KOKKOS_FUNCTION Chain& operator+=(const simplex_type& simplex)
    {
        assert(size() < allocation_size());
        m_simplices(m_size) = simplex;
        m_size++;
        return *this;
    }

    KOKKOS_FUNCTION Chain& operator+=(const Chain& simplices_to_add)
    {
        assert(size() + simplices_to_add.size() <= allocation_size());
        for (auto i = simplices_to_add.begin(); i < simplices_to_add.end(); ++i) {
            m_simplices(m_size + Kokkos::Experimental::distance(simplices_to_add.begin(), i)) = *i;
        }
        m_size += simplices_to_add.size();
        return *this;
    }

    KOKKOS_FUNCTION Chain operator+(simplex_type simplex)
    {
        Chain chain = *this;
        chain += simplex;
        return chain;
    }

    KOKKOS_FUNCTION Chain operator+(Chain simplices_to_add)
    {
        Chain chain = *this;
        chain += simplices_to_add;
        return chain;
    }

    KOKKOS_FUNCTION Chain& revert()
    {
        for (auto i = Kokkos::Experimental::begin(m_simplices);
             i < Kokkos::Experimental::begin(m_simplices) + size();
             ++i) {
            *i = -*i;
        }
        return *this;
    }

    KOKKOS_FUNCTION Chain operator-()
    {
        Chain chain = *this;
        chain.revert();
        return chain;
    }

    template <class T>
    KOKKOS_FUNCTION auto operator-(T t)
    {
        return *this + (-t);
    }

    template <class T>
    KOKKOS_FUNCTION Chain& operator*=(T t)
    {
        if (t == 1) {
        } else if (t == -1) {
            revert();
        } else {
            assert(false && "chain must be multiplied  by 1 or -1");
        }
        return *this;
    }

    template <class T>
    KOKKOS_FUNCTION auto operator*(T t)
    {
        Chain chain = *this;
        chain *= t;
        return chain;
    }

    KOKKOS_FUNCTION bool operator==(Chain simplices)
    {
        for (auto i = simplices.begin(); i < simplices.end(); ++i) {
            if (*i != m_simplices(Kokkos::Experimental::distance(simplices.begin(), i))) {
                return false;
            }
        }
        return true;
    }
};

template <class Head, class... Tail>
Chain(Head, Tail...) -> Chain<
                             typename Head::value_type,
                             typename Head::array_layout,
                             typename Head::memory_space>;

template <misc::Specialization<Chain> ChainType>
std::ostream& operator<<(std::ostream& out, ChainType const& chain)
{
    out << "[\n";
    for (typename ChainType::simplex_type const& simplex : chain) {
        out << " " << simplex << "\n";
    }
    out << "]";
    return out;
}

} // namespace exterior

} // namespace sil
