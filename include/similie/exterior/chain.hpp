// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <ddc/ddc.hpp>

#include <similie/misc/are_all_same.hpp>
#include <similie/misc/specialization.hpp>

#include <Kokkos_StdAlgorithms.hpp>

#include "simplex.hpp"

namespace sil {

namespace exterior {

/// Chain class
template <class SimplexType, class ExecSpace = Kokkos::DefaultHostExecutionSpace>
class Chain
{
public:
    using execution_space = ExecSpace;
    using memory_space = typename ExecSpace::memory_space;

    using simplex_type = SimplexType;
    using simplices_type = Kokkos::View<SimplexType*, memory_space>;
    using discrete_element_type = typename simplex_type::discrete_element_type;
    using discrete_vector_type = typename simplex_type::discrete_vector_type;

    using iterator_type = Kokkos::Experimental::Impl::RandomAccessIterator<simplices_type>;

private:
    static constexpr bool s_is_local = false;
    static constexpr std::size_t s_k = simplex_type::dimension();
    simplices_type m_simplices;

public:
    KOKKOS_DEFAULTED_FUNCTION constexpr Chain() = default;

    KOKKOS_DEFAULTED_FUNCTION constexpr Chain(Chain const&) = default;

    template <class... T>
        requires misc::are_all_same<T...>
    KOKKOS_FUNCTION constexpr explicit Chain(T... simplex) noexcept
        : m_simplices("chain_simplices", sizeof...(T))
    {
        int i = 0;
        ((m_simplices(i++) = simplex), ...);
        assert(check() == 0 && "there are duplicate simplices in the chain");
    }

    KOKKOS_FUNCTION constexpr explicit Chain(simplices_type simplices) noexcept
        : m_simplices(simplices)
    {
        assert(check() == 0 && "there are duplicate simplices in the chain");
    }

    KOKKOS_DEFAULTED_FUNCTION ~Chain() = default;

    KOKKOS_DEFAULTED_FUNCTION Chain& operator=(Chain const& other) = default;

    static KOKKOS_FUNCTION constexpr bool is_local() noexcept
    {
        return s_is_local;
    }

    static KOKKOS_FUNCTION constexpr std::size_t dimension() noexcept
    {
        return s_k;
    }

    KOKKOS_FUNCTION std::size_t size() noexcept
    {
        return m_simplices.size();
    }

    KOKKOS_FUNCTION std::size_t const size() const noexcept
    {
        return m_simplices.size();
    }

    KOKKOS_FUNCTION int check()
    {
        for (auto i = Kokkos::Experimental::begin(m_simplices);
             i < Kokkos::Experimental::end(m_simplices) - 1;
             ++i) {
            for (auto j = i + 1; j < Kokkos::Experimental::end(m_simplices); ++j) {
                if (*i == *j) {
                    return -1;
                }
            }
        }
        return 0;
    }

    KOKKOS_FUNCTION void optimize()
    {
        auto simplices_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), m_simplices);
        std::vector<simplex_type> simplices_vect(
                simplices_host.data(),
                simplices_host.data() + simplices_host.extent(0));
        for (auto i = simplices_vect.begin(); i < simplices_vect.end() - 1; ++i) {
            auto k = i;
            for (auto j = i + 1; k == i && j < simplices_vect.end(); ++j) {
                if (*i == -*j) {
                    k = j;
                }
            }
            if (k != i) {
                simplices_vect.erase(k);
                simplices_vect.erase(i--);
            }
        }
        Kokkos::View<simplex_type*, Kokkos::HostSpace>
                new_simplices_host(simplices_vect.data(), simplices_vect.size());
        m_simplices = Kokkos::create_mirror_view_and_copy(memory_space(), new_simplices_host);
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
        return Kokkos::Experimental::end(m_simplices);
    }

    KOKKOS_FUNCTION auto end() const
    {
        return Kokkos::Experimental::end(m_simplices);
    }

    KOKKOS_FUNCTION auto cbegin() const
    {
        return Kokkos::Experimental::cbegin(m_simplices);
    }

    KOKKOS_FUNCTION auto cend() const
    {
        return Kokkos::Experimental::cend(m_simplices);
    }

    KOKKOS_FUNCTION simplex_type& operator[](std::size_t i) noexcept
    {
        return m_simplices(i);
    }

    KOKKOS_FUNCTION simplex_type const& operator[](std::size_t i) const noexcept
    {
        return m_simplices(i);
    }

    KOKKOS_FUNCTION void push_back(const simplex_type& simplex)
    {
        Kokkos::resize(m_simplices, m_simplices.size() + 1);
        m_simplices(m_simplices.size()) = simplex;
    }

    KOKKOS_FUNCTION void push_back(const Chain<simplex_type>& simplices_to_add)
    {
        std::size_t old_size = m_simplices.size();
        Kokkos::resize(m_simplices, old_size + simplices_to_add.size());
        for (auto i = simplices_to_add.begin(); i < simplices_to_add.end(); ++i) {
            m_simplices(old_size + Kokkos::Experimental::distance(simplices_to_add.begin(), i))
                    = *i;
        }
    }

    KOKKOS_FUNCTION Chain<simplex_type> operator-()
    {
        simplices_type simplices = m_simplices;
        for (auto i = Kokkos::Experimental::begin(simplices);
             i < Kokkos::Experimental::end(simplices) - 1;
             ++i) {
            *i = -*i;
        }
        return Chain<simplex_type>(simplices);
    }

    KOKKOS_FUNCTION Chain<simplex_type> operator+(simplex_type simplex)
    {
        Chain<simplex_type> chain = *this;
        chain.push_back(simplex);
        return chain;
    }

    KOKKOS_FUNCTION Chain<simplex_type> operator+(Chain<simplex_type> simplices_to_add)
    {
        Chain<simplex_type> chain = *this;
        chain.push_back(simplices_to_add);
        return chain;
    }

    template <class T>
    KOKKOS_FUNCTION auto operator-(T t)
    {
        return *this + (-t);
    }

    template <class T>
    KOKKOS_FUNCTION auto operator*(T t)
    {
        if (t == 1) {
            return *this;
        } else if (t == -1) {
            return -*this;
        } else {
            assert(false && "chain must be multiplied  by 1 or -1");
            return *this;
        }
    }

    KOKKOS_FUNCTION bool operator==(Chain<simplex_type> simplices)
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
Chain(Head, Tail...) -> Chain<Head>;

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
