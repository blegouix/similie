// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <ddc/ddc.hpp>

#include "are_all_same.hpp"
#include "simplex.hpp"
#include "specialization.hpp"

namespace sil {

namespace exterior {

/// LocalChain class
template <class SimplexType, class Allocator = std::allocator<typename SimplexType::vect_type>>
class LocalChain : public std::vector<typename SimplexType::vect_type, Allocator>
{
public:
    using simplex_type = SimplexType;
    using base_type = std::vector<typename simplex_type::vect_type, Allocator>;
    using elem_type = typename simplex_type::elem_type;
    using vect_type = typename simplex_type::vect_type;

private:
    static constexpr std::size_t s_k = SimplexType::dimension();
    elem_type m_elem;

public:
    KOKKOS_FUNCTION constexpr explicit LocalChain() noexcept : elem_type {}, base_type {} {}

    // TODO Reorganize discrete vectors in all constructors ?

    template <class... T>
        requires misc::are_all_same<T...>
    KOKKOS_FUNCTION constexpr explicit LocalChain(T... simplex) noexcept
        : m_elem {std::array<elem_type, sizeof...(T)> {simplex.discrete_element()...}[0]}
        , base_type {simplex.discrete_vector()...}
    {
        assert(check() == 0 && "there are duplicate simplices in the chain");
        assert(misc::are_all_equal(simplex.discrete_element()...)
               && "LocalChain must contain simplices with same origin (if not, use Chain)");
        assert((!simplex.negative() && ...)
               && "negative simplices are not supported in LocalChain");
    }

    KOKKOS_FUNCTION constexpr explicit LocalChain(std::vector<SimplexType> simplices) noexcept
        : m_elem {simplices[0]}
        , base_type(simplices)
    {
        assert(check() == 0 && "there are duplicate simplices in the chain");
        std::function<bool()> check_common_elem = [&]() {
            std::vector<elem_type> elems(simplices.size());
            for (auto i = simplices.begin(); i < simplices.end(); ++i) {
                elems[std::distance(simplices.begin(), i)] = i->discrete_element();
            }
            return misc::are_all_equal(elems);
        };
        assert(check_common_elem()
               && "LocalChain must contain simplices with same origin (if not, use Chain)");
        assert(std::
                       all_of(simplices.begin(),
                              simplices.end(),
                              [&](const std::size_t i) { return !simplices[i].negative(); })
               && "LocalChain must contain simplices with same origin (if not, use Chain)");
    }

    template <class... T>
        requires misc::are_all_same<T...>
    KOKKOS_FUNCTION constexpr explicit LocalChain(elem_type elem, T... vect) noexcept
        : m_elem {elem}
        , base_type {vect...}
    {
        assert(check() == 0 && "there are duplicate simplices in the chain");
    }

    KOKKOS_FUNCTION constexpr explicit LocalChain(
            elem_type elem,
            std::vector<vect_type> vects) noexcept
        : m_elem {elem}
        , base_type(vects)
    {
        assert(check() == 0 && "there are duplicate simplices in the chain");
    }

    static KOKKOS_FUNCTION constexpr std::size_t dimension() noexcept
    {
        return s_k;
    }

    KOKKOS_FUNCTION elem_type discrete_element()
    {
        return m_elem;
    }

    KOKKOS_FUNCTION elem_type const discrete_element() const
    {
        return m_elem;
    }

    KOKKOS_FUNCTION int check()
    {
        for (auto i = this->begin(); i < this->end(); ++i) {
            for (auto j = i + 1; j < this->end(); ++j) {
                if (*i == *j) {
                    return -1;
                }
            }
        }
        return 0;
    }

    // TODO clarifiy if negative are supported
    /*
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
    */

    LocalChain<SimplexType> operator-() = delete;

    KOKKOS_FUNCTION LocalChain<SimplexType> operator+(SimplexType simplex)
    {
        base_type vects(*this);
        vects.push_back(simplex.discrete_vector());
        return LocalChain<SimplexType>(this->discrete_element(), vects);
    }

    KOKKOS_FUNCTION LocalChain<SimplexType> operator+(LocalChain<SimplexType> simplices_to_add)
    {
        base_type vects(*this);
        vects.insert(vects.end(), simplices_to_add.begin(), simplices_to_add.end());
        return LocalChain<SimplexType>(this->discrete_element(), vects);
    }

    LocalChain<SimplexType> operator-(LocalChain<SimplexType>) = delete;

    template <class T>
    KOKKOS_FUNCTION auto operator*(T t)
    {
        if (t == 1) {
            return *this;
        } else if (t == -1) {
            assert(false && "negative simplices are unsupported in LocalChain");
            return *this;
        } else {
            assert(false && "chain must be multiplied  by 1 or -1");
        }
    }

    KOKKOS_FUNCTION bool operator==(LocalChain<SimplexType> simplices)
    {
        assert(discrete_element() == simplices.discrete_element());
        for (auto i = simplices.begin(); i < simplices.end(); ++i) {
            if (*i != (*this)[std::distance(simplices.begin(), i)]) {
                return false;
            }
        }
        return true;
    }
};

template <class... T>
LocalChain(T...) -> LocalChain<ddc::type_seq_element_t<0, ddc::detail::TypeSeq<T...>>>;

template <misc::Specialization<LocalChain> ChainType>
std::ostream& operator<<(std::ostream& out, ChainType const& chain)
{
    out << "[\n";
    for (typename ChainType::vect_type const& vect : chain) {
        out << " " << chain.discrete_element() << " -> " << chain.discrete_element() + vect << "\n";
    }
    out << "]";
    return out;
}

} // namespace exterior

} // namespace sil
