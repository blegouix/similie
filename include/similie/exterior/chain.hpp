// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <ddc/ddc.hpp>

#include <similie/misc/are_all_same.hpp>
#include <similie/misc/specialization.hpp>

#include "simplex.hpp"

namespace sil {

namespace exterior {

/// Chain class
template <class SimplexType, class Allocator = std::allocator<SimplexType>>
class Chain
{
public:
    using simplex_type = SimplexType;
    using simplices_type = std::vector<SimplexType, Allocator>;
    using discrete_element_type = typename simplex_type::discrete_element_type;
    using discrete_vector_type = typename simplex_type::discrete_vector_type;

private:
    static constexpr bool s_is_local = false;
    static constexpr std::size_t s_k = SimplexType::dimension();
    simplices_type m_simplices;

public:
    KOKKOS_FUNCTION constexpr explicit Chain() noexcept : m_simplices {} {}

    template <class... T>
        requires misc::are_all_same<T...>
    KOKKOS_FUNCTION constexpr explicit Chain(T... simplex) noexcept : m_simplices {simplex...}
    {
        assert(check() == 0 && "there are duplicate simplices in the chain");
    }

    KOKKOS_FUNCTION constexpr explicit Chain(std::vector<SimplexType> simplices) noexcept
        : m_simplices(simplices)
    {
        assert(check() == 0 && "there are duplicate simplices in the chain");
    }

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
        for (auto i = m_simplices.begin(); i < m_simplices.end() - 1; ++i) {
            for (auto j = i + 1; j < m_simplices.end(); ++j) {
                if (*i == *j) {
                    return -1;
                }
            }
        }
        return 0;
    }

    KOKKOS_FUNCTION void optimize()
    {
        for (auto i = m_simplices.begin(); i < m_simplices.end() - 1; ++i) {
            auto k = i;
            for (auto j = i + 1; k == i && j < m_simplices.end(); ++j) {
                if (*i == -*j) {
                    k = j;
                }
            }
            if (k != i) {
                m_simplices.erase(k);
                m_simplices.erase(i--);
            }
        }
        assert(check() == 0 && "there are duplicate simplices in the chain");
    }

    KOKKOS_FUNCTION auto begin()
    {
        return m_simplices.begin();
    }

    KOKKOS_FUNCTION auto begin() const
    {
        return m_simplices.begin();
    }

    KOKKOS_FUNCTION auto end()
    {
        return m_simplices.end();
    }

    KOKKOS_FUNCTION auto end() const
    {
        return m_simplices.end();
    }

    KOKKOS_FUNCTION auto cbegin() const
    {
        return m_simplices.begin();
    }

    KOKKOS_FUNCTION auto cend() const
    {
        return m_simplices.end();
    }

    KOKKOS_FUNCTION SimplexType& operator[](std::size_t i) noexcept
    {
        return m_simplices[i];
    }

    KOKKOS_FUNCTION SimplexType const& operator[](std::size_t i) const noexcept
    {
        return m_simplices[i];
    }

    void push_back(const simplex_type& simplex)
    {
        m_simplices.push_back(simplex);
    };

    KOKKOS_FUNCTION Chain<SimplexType> operator-()
    {
        std::vector<SimplexType> simplices = m_simplices;
        for (SimplexType& simplex : simplices) {
            simplex = -simplex;
        }
        return Chain<SimplexType>(simplices);
    }

    KOKKOS_FUNCTION Chain<SimplexType> operator+(SimplexType simplex)
    {
        simplices_type simplices = m_simplices;
        simplices.push_back(simplex);
        return Chain<SimplexType>(simplices);
    }

    KOKKOS_FUNCTION Chain<SimplexType> operator+(Chain<SimplexType> simplices_to_add)
    {
        simplices_type simplices = m_simplices;
        simplices.insert(simplices.end(), simplices_to_add.begin(), simplices_to_add.end());
        return Chain<SimplexType>(simplices);
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

    KOKKOS_FUNCTION bool operator==(Chain<SimplexType> simplices)
    {
        for (auto i = simplices.begin(); i < simplices.end(); ++i) {
            if (*i != m_simplices[std::distance(simplices.begin(), i)]) {
                return false;
            }
        }
        return true;
    }
};

template <class... T>
Chain(T...) -> Chain<ddc::type_seq_element_t<0, ddc::detail::TypeSeq<T...>>>;

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
