// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <ddc/ddc.hpp>

#include "are_all_same.hpp"
#include "simplex.hpp"
#include "specialization.hpp"

namespace sil {

namespace form {

/// Chain class
template <class SimplexType>
class Chain
{
    template <class T>
    friend class Chain;

public:
    using simplex_type = SimplexType;

private:
    static constexpr std::size_t s_k = SimplexType::dimension();
    std::vector<SimplexType> m_simplices;

public:
    template <class... T>
        requires misc::are_all_same<T...>
    KOKKOS_FUNCTION constexpr explicit Chain(T... simplex) noexcept : m_simplices {simplex...}
    {
        assert(check() == 0 && "there are duplicate simplices in the chain");
    }

    KOKKOS_FUNCTION constexpr explicit Chain(std::vector<SimplexType> simplices) noexcept
        : m_simplices(simplices)
    {
    }

    static KOKKOS_FUNCTION constexpr std::size_t dimension() noexcept
    {
        return s_k;
    }

    KOKKOS_FUNCTION std::vector<SimplexType>& simplices() noexcept
    {
        return m_simplices;
    }

    KOKKOS_FUNCTION std::vector<SimplexType> const& simplices() const noexcept
    {
        return m_simplices;
    }

    KOKKOS_FUNCTION std::size_t extent() noexcept
    {
        return m_simplices.size();
    }

    KOKKOS_FUNCTION std::size_t const extent() const noexcept
    {
        return m_simplices.size();
    }

    KOKKOS_FUNCTION SimplexType& operator[](std::size_t const i) noexcept
    {
        return m_simplices[i];
    }

    KOKKOS_FUNCTION SimplexType const& operator[](std::size_t const i) const noexcept
    {
        return m_simplices[i];
    }

    KOKKOS_FUNCTION SimplexType& front() noexcept
    {
        return m_simplices[0];
    }

    KOKKOS_FUNCTION SimplexType const& front() const noexcept
    {
        return m_simplices[0];
    }

    KOKKOS_FUNCTION SimplexType& back() noexcept
    {
        return m_simplices[extent() - 1];
    }

    KOKKOS_FUNCTION SimplexType const& back() const noexcept
    {
        return m_simplices[extent() - 1];
    }

    KOKKOS_FUNCTION int check()
    {
#ifdef NDEBUG
        for (auto i = m_simplices.begin(); i < m_simplices.end(); ++i) {
            for (auto j = i + 1; k == i && j < m_simplices.end(); ++j) {
                if (*i == *j) {
                    return -1;
                }
            }
        }
#endif
        return 0;
    }

    KOKKOS_FUNCTION void optimize()
    {
        for (auto i = m_simplices.begin(); i < m_simplices.end(); ++i) {
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
        std::vector<SimplexType> simplices = m_simplices;
        simplices.push_back(simplex);
        return Chain<SimplexType>(simplices);
    }

    KOKKOS_FUNCTION Chain<SimplexType> operator+(Chain<SimplexType> simplices_to_add)
    {
        std::vector<SimplexType> simplices = m_simplices;
        simplices
                .insert(simplices.end(),
                        simplices_to_add.m_simplices.begin(),
                        simplices_to_add.m_simplices.end());
        return Chain<SimplexType>(simplices);
    }

    template <class T>
    KOKKOS_FUNCTION auto operator-(T t)
    {
        return *this + (-t);
    }

    KOKKOS_FUNCTION bool operator==(Chain<SimplexType> simplices)
    {
        for (auto i = simplices.m_simplices.begin(); i < simplices.m_simplices.end(); ++i) {
            if (*i != m_simplices[std::distance(simplices.m_simplices.begin(), i)]) {
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
    for (typename ChainType::simplex_type const& simplex : chain.simplices()) {
        out << " " << simplex << "\n";
    }
    out << "]";
    return out;
}

} // namespace form

} // namespace sil
