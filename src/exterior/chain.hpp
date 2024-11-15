// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <ddc/ddc.hpp>

#include "are_all_same.hpp"
#include "simplex.hpp"
#include "specialization.hpp"

namespace sil {

namespace exterior {

/// Chain class
template <class SimplexType, class Allocator = std::allocator<SimplexType>>
class Chain : public std::vector<SimplexType, Allocator>
{
public:
    using simplex_type = SimplexType;
    using base_type = std::vector<SimplexType, Allocator>;
    using elem_type = typename simplex_type::elem_type;
    using vect_type = typename simplex_type::vect_type;

private:
    static constexpr std::size_t s_k = SimplexType::dimension();

public:
    KOKKOS_FUNCTION constexpr explicit Chain() noexcept : base_type {} {}

    template <class... T>
        requires misc::are_all_same<T...>
    KOKKOS_FUNCTION constexpr explicit Chain(T... simplex) noexcept : base_type {simplex...}
    {
        assert(check() == 0 && "there are duplicate simplices in the chain");
    }

    KOKKOS_FUNCTION constexpr explicit Chain(std::vector<SimplexType> simplices) noexcept
        : base_type(simplices)
    {
        assert(check() == 0 && "there are duplicate simplices in the chain");
    }

    static KOKKOS_FUNCTION constexpr std::size_t dimension() noexcept
    {
        return s_k;
    }

    KOKKOS_FUNCTION int check()
    {
#ifdef NDEBUG
        for (auto i = this->begin(); i < this->end(); ++i) {
            for (auto j = i + 1; k == i && j < this->end(); ++j) {
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
        for (auto i = this->begin(); i < this->end(); ++i) {
            auto k = i;
            for (auto j = i + 1; k == i && j < this->end(); ++j) {
                if (*i == -*j) {
                    k = j;
                }
            }
            if (k != i) {
                this->erase(k);
                this->erase(i--);
            }
        }
        assert(check() == 0 && "there are duplicate simplices in the chain");
    }

    KOKKOS_FUNCTION Chain<SimplexType> operator-()
    {
        std::vector<SimplexType> simplices = *this;
        for (SimplexType& simplex : simplices) {
            simplex = -simplex;
        }
        return Chain<SimplexType>(simplices);
    }

    KOKKOS_FUNCTION Chain<SimplexType> operator+(SimplexType simplex)
    {
        std::vector<SimplexType> simplices = *this;
        simplices.push_back(simplex);
        return Chain<SimplexType>(simplices);
    }

    KOKKOS_FUNCTION Chain<SimplexType> operator+(Chain<SimplexType> simplices_to_add)
    {
        std::vector<SimplexType> simplices = *this;
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
        }
    }

    KOKKOS_FUNCTION bool operator==(Chain<SimplexType> simplices)
    {
        for (auto i = simplices.begin(); i < simplices.end(); ++i) {
            if (*i != (*this)[std::distance(simplices.begin(), i)]) {
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
