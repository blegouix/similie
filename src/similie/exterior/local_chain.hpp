// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

#include <array>
#include <cstddef>
#include <iostream>
#include <iterator>
#include <type_traits>

#include <ddc/ddc.hpp>

#include <similie/misc/are_all_same.hpp>
#include <similie/misc/binomial_coefficient.hpp>
#include <similie/misc/filled_struct.hpp>
#include <similie/misc/specialization.hpp>

#include "simplex.hpp"

namespace sil {

namespace exterior {

template <
        class SimplexType,
        class LayoutStridedPolicy = Kokkos::LayoutRight,
        class MemorySpace = Kokkos::HostSpace>
class LocalChain;

template <class SimplexType, class LayoutStridedPolicy, class MemorySpace>
class LocalChainIterator
{
    using chain_type = LocalChain<SimplexType, LayoutStridedPolicy, MemorySpace>;

    chain_type const* m_chain;
    std::size_t m_index;

public:
    using difference_type = std::ptrdiff_t;
    using value_type = SimplexType;
    using reference = value_type;
    using iterator_category = std::random_access_iterator_tag;

    KOKKOS_FUNCTION constexpr LocalChainIterator(
            chain_type const* chain,
            std::size_t index) noexcept
        : m_chain(chain)
        , m_index(index)
    {
    }

    KOKKOS_FUNCTION constexpr value_type operator*() const noexcept
    {
        return (*m_chain)[m_index];
    }

    KOKKOS_FUNCTION constexpr LocalChainIterator& operator++() noexcept
    {
        ++m_index;
        return *this;
    }

    KOKKOS_FUNCTION constexpr LocalChainIterator operator++(int) noexcept
    {
        LocalChainIterator tmp = *this;
        ++(*this);
        return tmp;
    }

    KOKKOS_FUNCTION constexpr LocalChainIterator& operator--() noexcept
    {
        --m_index;
        return *this;
    }

    KOKKOS_FUNCTION constexpr LocalChainIterator operator+(difference_type n) const noexcept
    {
        return LocalChainIterator(m_chain, m_index + n);
    }

    KOKKOS_FUNCTION constexpr LocalChainIterator& operator+=(difference_type n) noexcept
    {
        m_index += n;
        return *this;
    }

    KOKKOS_FUNCTION constexpr LocalChainIterator operator-(difference_type n) const noexcept
    {
        return LocalChainIterator(m_chain, m_index - n);
    }

    KOKKOS_FUNCTION constexpr difference_type operator-(
            LocalChainIterator const& other) const noexcept
    {
        return static_cast<difference_type>(m_index) - static_cast<difference_type>(other.m_index);
    }

    KOKKOS_FUNCTION constexpr bool operator==(LocalChainIterator const& other) const noexcept
    {
        return m_index == other.m_index;
    }

    KOKKOS_FUNCTION constexpr bool operator!=(LocalChainIterator const& other) const noexcept
    {
        return !(*this == other);
    }

    KOKKOS_FUNCTION constexpr bool operator<(LocalChainIterator const& other) const noexcept
    {
        return m_index < other.m_index;
    }
};

template <class SimplexType, class LayoutStridedPolicy, class MemorySpace>
class LocalChain
{
public:
    using memory_space = MemorySpace;
    using simplex_type = SimplexType;
    using discrete_element_type = typename simplex_type::discrete_element_type;
    using discrete_vector_type = typename simplex_type::discrete_vector_type;
    static constexpr std::size_t MAX_SIZE
            = 1UL << ddc::type_seq_size_v<ddc::to_type_seq_t<discrete_vector_type>>;
    using storage_type = std::array<simplex_type, MAX_SIZE>;

    using iterator_type = LocalChainIterator<SimplexType, LayoutStridedPolicy, MemorySpace>;
    using const_iterator_type = iterator_type;

private:
    static constexpr bool s_is_local = true;
    static constexpr std::size_t s_k = simplex_type::dimension();
    discrete_element_type m_origin = misc::filled_struct<discrete_element_type>();
    storage_type m_vects {};
    std::size_t m_size = 0;

public:
    KOKKOS_DEFAULTED_FUNCTION constexpr LocalChain() = default;
    KOKKOS_DEFAULTED_FUNCTION constexpr LocalChain(LocalChain const&) = default;
    KOKKOS_DEFAULTED_FUNCTION constexpr LocalChain(LocalChain&&) = default;
    KOKKOS_DEFAULTED_FUNCTION constexpr ~LocalChain() = default;
    KOKKOS_DEFAULTED_FUNCTION LocalChain& operator=(LocalChain const&) = default;
    KOKKOS_DEFAULTED_FUNCTION LocalChain& operator=(LocalChain&&) = default;

    KOKKOS_FUNCTION constexpr explicit LocalChain(
            discrete_element_type origin,
            std::size_t size = 0) noexcept
        : m_origin(origin)
        , m_size(size)
    {
        assert(size <= MAX_SIZE);
    }

    template <misc::Specialization<Kokkos::View> AllocationType>
    KOKKOS_FUNCTION constexpr explicit LocalChain(
            AllocationType const&,
            discrete_element_type origin)
        : LocalChain(origin)
    {
    }

    template <misc::NotSpecialization<ddc::DiscreteVector>... T>
    KOKKOS_FUNCTION constexpr explicit LocalChain(
            discrete_element_type origin,
            T... simplex) noexcept
        : m_origin(origin)
        , m_size(sizeof...(T))
    {
        static_assert(sizeof...(T) <= MAX_SIZE);
        std::size_t i = 0;
        ((m_vects[i++] = simplex), ...);
        assert(check() == 0 && "there are duplicate simplices in the chain");
        if constexpr (sizeof...(T) > 1) {
            assert(misc::are_all_equal(simplex.discrete_element()...)
                   && "LocalChain must contain simplices with same origin (if not, use Chain)");
        }
        assert((!simplex.negative() && ...)
               && "negative simplices are not supported in LocalChain");
    }

    template <
            misc::Specialization<Kokkos::View> AllocationType,
            misc::NotSpecialization<ddc::DiscreteVector> First,
            misc::NotSpecialization<ddc::DiscreteVector>... Rest>
    KOKKOS_FUNCTION constexpr explicit LocalChain(
            AllocationType const&,
            First first_simplex,
            Rest... simplices) noexcept
        : LocalChain(first_simplex.discrete_element(), first_simplex, simplices...)
    {
    }

    template <misc::Specialization<ddc::DiscreteVector>... T>
        requires(sizeof...(T) >= 1)
    KOKKOS_FUNCTION constexpr explicit LocalChain(discrete_element_type origin, T... vect) noexcept
        : m_origin(origin)
        , m_size(sizeof...(T))
    {
        static_assert(sizeof...(T) <= MAX_SIZE);
        std::size_t i = 0;
        ((m_vects[i++] = simplex_type(origin, vect)), ...);
        assert(check() == 0 && "there are duplicate simplices in the chain");
    }

    template <
            misc::Specialization<Kokkos::View> AllocationType,
            misc::Specialization<ddc::DiscreteVector>... T>
        requires(sizeof...(T) >= 1)
    KOKKOS_FUNCTION constexpr explicit LocalChain(
            AllocationType const&,
            discrete_element_type origin,
            T... vect) noexcept
        : LocalChain(origin, vect...)
    {
    }

    static KOKKOS_FUNCTION constexpr bool is_local() noexcept
    {
        return s_is_local;
    }

    static KOKKOS_FUNCTION constexpr std::size_t dimension() noexcept
    {
        return s_k;
    }

    KOKKOS_FUNCTION constexpr discrete_element_type origin() const noexcept
    {
        return m_origin;
    }

    KOKKOS_FUNCTION constexpr storage_type& allocation() noexcept
    {
        return m_vects;
    }

    KOKKOS_FUNCTION constexpr storage_type const& allocation() const noexcept
    {
        return m_vects;
    }

    KOKKOS_FUNCTION constexpr std::size_t size() const noexcept
    {
        return m_size;
    }

    KOKKOS_FUNCTION constexpr std::size_t allocation_size() const noexcept
    {
        return MAX_SIZE;
    }

    static KOKKOS_FUNCTION constexpr bool negative()
    {
        return false;
    }

    KOKKOS_FUNCTION constexpr void resize(std::size_t size) noexcept
    {
        assert(size <= MAX_SIZE);
        m_size = size;
    }

    KOKKOS_FUNCTION constexpr int check() const noexcept
    {
        for (auto i = begin(); i + 1 < end(); ++i) {
            for (auto j = i + 1; j < end(); ++j) {
                if (*i == *j) {
                    return -1;
                }
            }
        }
        return 0;
    }

    KOKKOS_FUNCTION constexpr iterator_type begin() noexcept
    {
        return iterator_type(this, 0);
    }

    KOKKOS_FUNCTION constexpr const_iterator_type begin() const noexcept
    {
        return const_iterator_type(this, 0);
    }

    KOKKOS_FUNCTION constexpr iterator_type end() noexcept
    {
        return iterator_type(this, m_size);
    }

    KOKKOS_FUNCTION constexpr const_iterator_type end() const noexcept
    {
        return const_iterator_type(this, m_size);
    }

    KOKKOS_FUNCTION constexpr const_iterator_type cbegin() const noexcept
    {
        return const_iterator_type(this, 0);
    }

    KOKKOS_FUNCTION constexpr const_iterator_type cend() const noexcept
    {
        return const_iterator_type(this, m_size);
    }

    KOKKOS_FUNCTION constexpr simplex_type operator[](std::size_t i) const noexcept
    {
        assert(i < m_size);
        return m_vects[i];
    }

    KOKKOS_FUNCTION constexpr LocalChain& operator++()
    {
        assert(m_size < MAX_SIZE);
        ++m_size;
        return *this;
    }

    KOKKOS_FUNCTION constexpr LocalChain& operator+=(std::size_t n)
    {
        assert(m_size + n <= MAX_SIZE);
        m_size += n;
        return *this;
    }

    KOKKOS_FUNCTION constexpr LocalChain& operator+=(discrete_vector_type const& vect)
    {
        assert(m_size < MAX_SIZE);
        m_vects[m_size++] = simplex_type(m_origin, vect);
        return *this;
    }

    KOKKOS_FUNCTION constexpr LocalChain& operator+=(simplex_type const& simplex)
    {
        assert(m_size < MAX_SIZE);
        m_vects[m_size++] = simplex;
        return *this;
    }

    KOKKOS_FUNCTION constexpr LocalChain& operator+=(LocalChain const& simplices_to_add)
    {
        assert(m_size + simplices_to_add.size() <= MAX_SIZE);
        std::size_t const old_size = m_size;
        for (auto i = simplices_to_add.begin(); i < simplices_to_add.end(); ++i) {
            m_vects[old_size + Kokkos::Experimental::distance(simplices_to_add.begin(), i)] = *i;
        }
        m_size += simplices_to_add.size();
        return *this;
    }

    KOKKOS_FUNCTION constexpr LocalChain operator+(simplex_type const& simplex) const
    {
        LocalChain local_chain = *this;
        local_chain += simplex;
        return local_chain;
    }

    KOKKOS_FUNCTION constexpr LocalChain operator+(LocalChain const& simplices_to_add) const
    {
        LocalChain local_chain = *this;
        local_chain += simplices_to_add;
        return local_chain;
    }

    template <class T>
    KOKKOS_FUNCTION constexpr LocalChain& operator*=(T t)
    {
        if (t == 1) {
        } else if (t == -1) {
            for (std::size_t i = 0; i < m_size; ++i) {
                m_vects[i] = -m_vects[i];
            }
        } else {
            assert(false && "chain must be multiplied by 1 or -1");
        }
        return *this;
    }

    template <class T>
    KOKKOS_FUNCTION constexpr auto operator*(T t) const
    {
        LocalChain chain = *this;
        chain *= t;
        return chain;
    }

    KOKKOS_FUNCTION constexpr bool operator==(LocalChain const& simplices) const
    {
        if (m_size != simplices.m_size) {
            return false;
        }
        auto simplex = begin();
        auto other_simplex = simplices.begin();
        for (; simplex < end(); ++simplex, ++other_simplex) {
            if (*simplex != *other_simplex) {
                return false;
            }
        }
        return true;
    }
};

namespace detail {

template <std::size_t K, misc::Specialization<ddc::DiscreteDomain> Dom>
struct TangentBasis;

template <std::size_t K, class... Tag>
struct TangentBasis<K, ddc::DiscreteDomain<Tag...>>
{
    template <class MemorySpace = Kokkos::HostSpace, class Elem>
    KOKKOS_FUNCTION static constexpr auto run(Elem elem)
    {
        using chain_type = LocalChain<Simplex<K, Tag...>, Kokkos::LayoutRight, MemorySpace>;
        std::array<std::ptrdiff_t, sizeof...(Tag)> permutation
                = {0 * ddc::type_seq_rank_v<Tag, ddc::detail::TypeSeq<Tag...>>...};
        for (auto i = permutation.begin(); i < permutation.begin() + K; ++i) {
            *i = 1;
        }
        chain_type basis {typename chain_type::discrete_element_type(elem)};
        do {
            typename chain_type::discrete_vector_type vect;
            ddc::detail::array(vect) = permutation;
            basis += vect;
        } while (std::prev_permutation(permutation.begin(), permutation.end()));
        return basis;
    }
};

} // namespace detail

template <
        std::size_t K,
        misc::Specialization<ddc::DiscreteDomain> Dom,
        misc::Specialization<ddc::DiscreteElement> Elem>
KOKKOS_FUNCTION constexpr auto tangent_basis(Elem elem)
{
    return detail::TangentBasis<K, Dom>::template run<Kokkos::HostSpace>(elem);
}

template <std::size_t K, misc::Specialization<ddc::DiscreteDomain> Dom, class ExecSpace>
    requires(misc::NotSpecialization<ExecSpace, ddc::DiscreteElement>)
constexpr auto tangent_basis([[maybe_unused]] ExecSpace const& exec_space)
{
    return detail::TangentBasis<K, Dom>::template run<typename ExecSpace::memory_space>(
            misc::filled_struct<typename Dom::discrete_element_type>());
}

template <misc::Specialization<LocalChain> ChainType>
std::ostream& operator<<(std::ostream& out, ChainType const& chain)
{
    out << "[\n";
    for (auto const& simplex : chain) {
        out << " -> " << simplex << "\n";
    }
    out << "]";
    return out;
}

template <misc::Specialization<Kokkos::View> AllocationType, class SimplexType, class... T>
LocalChain(AllocationType, SimplexType, T...) -> LocalChain<
        std::remove_cvref_t<SimplexType>,
        typename AllocationType::array_layout,
        typename AllocationType::memory_space>;

} // namespace exterior

} // namespace sil
