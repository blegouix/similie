// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <ddc/ddc.hpp>

#include "are_all_same.hpp"
#include "binomial_coefficient.hpp"
#include "null_struct.hpp"
#include "simplex.hpp"
#include "specialization.hpp"

namespace sil {

namespace exterior {

namespace detail {

template <class SimplexType>
std::vector<typename SimplexType::vect_type> extract_vects(std::vector<SimplexType> simplices)
{
    std::vector<typename SimplexType::vect_type> vects;
    for (auto& simplex : simplices) {
        vects.push_back(simplex.discrete_vector());
    }
    return vects;
}

} // namespace detail


/// LocalChain class
template <class SimplexType, class Allocator = std::allocator<typename SimplexType::vect_type>>
class LocalChain
{
public:
    using simplex_type = SimplexType;
    using elem_type = typename simplex_type::elem_type;
    using vect_type = typename simplex_type::vect_type;
    using vects_type = std::vector<vect_type>;

private:
    static constexpr std::size_t s_k = SimplexType::dimension();
    vects_type m_vects;

public:
    KOKKOS_FUNCTION constexpr explicit LocalChain() noexcept : elem_type {}, m_vects {} {}

    // TODO Reorganize discrete vectors in all constructors ?

    template <misc::NotSpecialization<ddc::DiscreteVector>... T>
        requires misc::are_all_same<T...>
    KOKKOS_FUNCTION constexpr explicit LocalChain(T... simplex) noexcept
        : m_vects {simplex.discrete_vector()...}
    {
        assert(check() == 0 && "there are duplicate simplices in the chain");
        assert(misc::are_all_equal(simplex.discrete_element()...)
               && "LocalChain must contain simplices with same origin (if not, use Chain)");
        assert((!simplex.negative() && ...)
               && "negative simplices are not supported in LocalChain");
    }

    KOKKOS_FUNCTION constexpr explicit LocalChain(std::vector<SimplexType> simplices) noexcept
        : m_vects(detail::extract_vects(simplices))
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

    template <misc::Specialization<ddc::DiscreteVector>... T>
    KOKKOS_FUNCTION constexpr explicit LocalChain(T... vect) noexcept
        : m_vects {(misc::null_struct<vect_type>() + vect)...}
    {
        assert(check() == 0 && "there are duplicate simplices in the chain");
    }

    KOKKOS_FUNCTION constexpr explicit LocalChain(std::vector<vect_type> vects) noexcept
        : m_vects(vects)
    {
        assert(check() == 0 && "there are duplicate simplices in the chain");
    }

    static KOKKOS_FUNCTION constexpr std::size_t dimension() noexcept
    {
        return s_k;
    }

    KOKKOS_FUNCTION std::size_t size() noexcept
    {
        return m_vects.size();
    }

    KOKKOS_FUNCTION std::size_t const size() const noexcept
    {
        return m_vects.size();
    }

    static KOKKOS_FUNCTION constexpr bool negative()
    {
        return false;
    }

    KOKKOS_FUNCTION int check()
    {
        for (auto i = this->begin(); i < this->end() - 1; ++i) {
            for (auto j = i + 1; j < this->end(); ++j) {
                if (*i == *j) {
                    return -1;
                }
            }
        }
        return 0;
    }

    KOKKOS_FUNCTION auto begin() const
    {
        return m_vects.begin();
    }

    KOKKOS_FUNCTION auto end() const
    {
        return m_vects.end();
    }

    KOKKOS_FUNCTION auto cbegin() const
    {
        return m_vects.begin();
    }

    KOKKOS_FUNCTION auto cend() const
    {
        return m_vects.end();
    }

    KOKKOS_FUNCTION SimplexType& operator[](std::size_t i) noexcept
    {
        return m_vects[i];
    }

    KOKKOS_FUNCTION SimplexType const& operator[](std::size_t i) const noexcept
    {
        return m_vects[i];
    }

    LocalChain<SimplexType> operator-() = delete;

    KOKKOS_FUNCTION LocalChain<SimplexType> operator+(SimplexType simplex)
    {
        vects_type vects(m_vects);
        vects.push_back(simplex.discrete_vector());
        return LocalChain<SimplexType>(vects);
    }

    KOKKOS_FUNCTION LocalChain<SimplexType> operator+(LocalChain<SimplexType> simplices_to_add)
    {
        vects_type vects(m_vects);
        vects.insert(vects.end(), simplices_to_add.begin(), simplices_to_add.end());
        return LocalChain<SimplexType>(vects);
    }

    LocalChain<SimplexType> operator-(LocalChain<SimplexType>) = delete;

    template <class T>
    KOKKOS_FUNCTION auto operator*(T t)
    {
        if (t == 1) {
            return *this;
        } else if (t == -1) {
            assert(false && "negative simplices are unsupported in LocalChain");
        } else {
            assert(false && "chain must be multiplied  by 1 or -1");
        }
    }

    KOKKOS_FUNCTION bool operator==(LocalChain<SimplexType> simplices)
    {
        for (auto i = simplices.begin(); i < simplices.end(); ++i) {
            if (*i != m_vects[std::distance(simplices.begin(), i)]) {
                return false;
            }
        }
        return true;
    }
};

template <misc::NotSpecialization<ddc::DiscreteVector>... T>
LocalChain(T...) -> LocalChain<ddc::type_seq_element_t<0, ddc::detail::TypeSeq<T...>>>;

template <std::size_t K, misc::NotSpecialization<ddc::DiscreteDomain>... Tag>
KOKKOS_FUNCTION constexpr LocalChain<Simplex<K, Tag...>> tangent_basis()
{
    std::array<std::ptrdiff_t, sizeof...(Tag)> permutation
            = {0 * ddc::type_seq_rank_v<Tag, ddc::detail::TypeSeq<Tag...>>...};
    for (auto i = permutation.begin(); i < permutation.begin() + K; ++i) {
        *i = 1;
    }
    std::vector<ddc::DiscreteVector<Tag...>> basis {};
    std::size_t i = 0;
    do {
        basis.push_back(ddc::DiscreteVector<Tag...>());
        ddc::detail::array(basis[i++]) = permutation;
    } while (std::prev_permutation(permutation.begin(), permutation.end()));

    return LocalChain<Simplex<K, Tag...>>(basis);
}

namespace detail {

template <std::size_t K, misc::Specialization<ddc::DiscreteDomain> Dom>
struct TangentBasis;


template <std::size_t K, class... Tag>
struct TangentBasis<K, ddc::DiscreteDomain<Tag...>>
{
    static auto constexpr run()
    {
        return tangent_basis<K, Tag...>();
    }
};

} // namespace detail

template <std::size_t K, misc::Specialization<ddc::DiscreteDomain> Dom>
KOKKOS_FUNCTION constexpr auto tangent_basis()
{
    return detail::TangentBasis<K, Dom>::run();
}

template <misc::Specialization<LocalChain> ChainType>
std::ostream& operator<<(std::ostream& out, ChainType const& chain)
{
    out << "[\n";
    for (typename ChainType::vect_type const& vect : chain) {
        out << " -> " << vect << "\n";
    }
    out << "]";
    return out;
}

} // namespace exterior

} // namespace sil
