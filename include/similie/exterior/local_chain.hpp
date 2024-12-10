// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <ddc/ddc.hpp>

#include <similie/misc/are_all_same.hpp>
#include <similie/misc/binomial_coefficient.hpp>
#include <similie/misc/filled_struct.hpp>
#include <similie/misc/specialization.hpp>

#include "simplex.hpp"

namespace sil {

namespace exterior {

namespace detail {

template <
        class SimplexType,
        class LayoutStridedPolicy1,
        class LayoutStridedPolicy2,
        class MemorySpace>
Kokkos::View<typename SimplexType::discrete_vector_type*, MemorySpace> extract_vects(
        Kokkos::View<typename SimplexType::discrete_vector_type*, LayoutStridedPolicy1, MemorySpace>
                vects,
        Kokkos::View<SimplexType*, LayoutStridedPolicy2, MemorySpace> simplices)
{
    for (auto i = Kokkos::Experimental::begin(simplices); i < Kokkos::Experimental::end(simplices);
         ++i) {
        vects(i) = i->discrete_vector();
    }
    return vects;
}

} // namespace detail


/// LocalChain class
template <
        class SimplexType,
        class LayoutStridedPolicy = Kokkos::LayoutRight,
        class ExecSpace = Kokkos::DefaultHostExecutionSpace>
class LocalChain
{
public:
    using execution_space = ExecSpace;
    using memory_space = typename ExecSpace::memory_space;

    using simplex_type = SimplexType;
    using simplices_type = Kokkos::View<SimplexType*, LayoutStridedPolicy, memory_space>;
    using discrete_element_type = typename simplex_type::discrete_element_type;
    using discrete_vector_type = typename simplex_type::discrete_vector_type;
    using vects_type = Kokkos::View<discrete_vector_type*, LayoutStridedPolicy, memory_space>;

    using iterator_type = Kokkos::Experimental::Impl::RandomAccessIterator<vects_type>;

private:
    static constexpr bool s_is_local = true;
    static constexpr std::size_t s_k = simplex_type::dimension();
    vects_type m_vects;
    std::size_t
            m_size; // Effective size, m_simplices elements between m_size and m_simplices.size() are undefined

public:
    KOKKOS_DEFAULTED_FUNCTION constexpr LocalChain() = default;

    KOKKOS_DEFAULTED_FUNCTION constexpr LocalChain(LocalChain const&) = default;

    KOKKOS_DEFAULTED_FUNCTION constexpr LocalChain(LocalChain&&) = default;

    // TODO Reorganize discrete vectors in all constructors ?

    template <misc::NotSpecialization<ddc::DiscreteVector>... T>
    KOKKOS_FUNCTION constexpr explicit LocalChain(vects_type allocation, T... simplex) noexcept
        : m_vects(std::move(allocation))
        , m_size(sizeof...(T))
    {
        std::size_t i = 0;
        ((m_vects(i++) = misc::filled_struct<discrete_vector_type>() + simplex.discrete_vector()),
         ...);
        assert(check() == 0 && "there are duplicate simplices in the chain");
        if constexpr (sizeof...(T) > 1) {
            assert(misc::are_all_equal(simplex.discrete_element()...)
                   && "LocalChain must contain simplices with same origin (if not, use Chain)");
        }
        assert((!simplex.negative() && ...)
               && "negative simplices are not supported in LocalChain");
    }

    KOKKOS_FUNCTION constexpr explicit LocalChain(
            vects_type allocation,
            simplices_type simplices,
            std::size_t size) noexcept
        : m_vects(std::move(allocation))
        , m_size(size)
    {
        detail::extract_vects(m_vects, simplices)
                assert(check() == 0 && "there are duplicate simplices in the chain");
        assert((KOKKOS_LAMBDA() {
                   Kokkos::View<discrete_element_type*, memory_space> elems(simplices.size());
                   for (auto i = Kokkos::Experimental::begin(simplices);
                        i < Kokkos::Experimental::end(simplices);
                        ++i) {
                       elems[Kokkos::Experimental::
                                     distance(Kokkos::Experimental::begin(simplices), i)]
                               = i->discrete_element();
                   }
                   return misc::are_all_equal(elems);
               })()
               && "LocalChain must contain simplices with same origin (if not, use Chain)");
        // TODO Kokkosify
        assert(std::all_of(
                       Kokkos::Experimental::begin(simplices),
                       Kokkos::Experimental::end(simplices),
                       KOKKOS_LAMBDA(const std::size_t i) { return !simplices[i].negative(); })
               && "LocalChain must contain simplices with same origin (if not, use Chain)");
    }

    template <misc::Specialization<ddc::DiscreteVector>... T>
        requires(sizeof...(T) >= 1)
    KOKKOS_FUNCTION constexpr explicit LocalChain(vects_type allocation, T... vect) noexcept
        : m_vects(std::move(allocation))
        , m_size(sizeof...(T))
    {
        std::size_t i = 0;
        ((m_vects(i++) = misc::filled_struct<discrete_vector_type>() + vect), ...);
        assert(check() == 0 && "there are duplicate simplices in the chain");
    }

    KOKKOS_FUNCTION constexpr explicit LocalChain(vects_type allocation, std::size_t size) noexcept
        : m_vects(std::move(allocation))
        , m_size(size)
    {
        assert(check() == 0 && "there are duplicate simplices in the chain");
    }

    KOKKOS_DEFAULTED_FUNCTION ~LocalChain() = default;

    KOKKOS_DEFAULTED_FUNCTION LocalChain& operator=(LocalChain const& other) = default;

    KOKKOS_DEFAULTED_FUNCTION LocalChain& operator=(LocalChain&& other) = default;

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
        return m_vects;
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
        return m_vects.size();
    }

    KOKKOS_FUNCTION std::size_t allocation_size() const noexcept
    {
        return m_vects.size();
    }

    static KOKKOS_FUNCTION constexpr bool negative()
    {
        return false;
    }

    void resize()
    {
        Kokkos::resize(m_vects, size());
    }

    void resize(std::size_t size)
    {
        Kokkos::resize(m_vects, size);
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

    KOKKOS_FUNCTION auto begin()
    {
        return Kokkos::Experimental::begin(m_vects);
    }

    KOKKOS_FUNCTION auto begin() const
    {
        return Kokkos::Experimental::begin(m_vects);
    }

    KOKKOS_FUNCTION auto end()
    {
        return Kokkos::Experimental::begin(m_vects) + size();
    }

    KOKKOS_FUNCTION auto end() const
    {
        return Kokkos::Experimental::begin(m_vects) + size();
    }

    KOKKOS_FUNCTION auto cbegin() const
    {
        return Kokkos::Experimental::begin(m_vects);
    }

    KOKKOS_FUNCTION auto cend() const
    {
        return Kokkos::Experimental::begin(m_vects) + size();
    }

    KOKKOS_FUNCTION simplex_type operator[](std::size_t i) noexcept
    {
        assert(i < size());
        return simplex_type(misc::filled_struct<discrete_element_type>(), m_vects[i]);
    }

    KOKKOS_FUNCTION simplex_type const operator[](std::size_t i) const noexcept
    {
        assert(i < size());
        return simplex_type(misc::filled_struct<discrete_element_type>(), m_vects[i]);
    }

    KOKKOS_FUNCTION LocalChain<simplex_type>& operator++()
    {
        assert(size() < allocation_size());
        m_size++;
        return *this;
    }

    KOKKOS_FUNCTION LocalChain<simplex_type>& operator+=(const std::size_t n)
    {
        assert(size() + n <= allocation_size());
        m_size += n;
        return *this;
    }

    KOKKOS_FUNCTION LocalChain<simplex_type>& operator+=(const discrete_vector_type& vect)
    {
        assert(size() < allocation_size());
        m_vects(m_size) = vect;
        m_size++;
        return *this;
    }

    KOKKOS_FUNCTION LocalChain<simplex_type>& operator+=(const simplex_type& simplex)
    {
        assert(size() < allocation_size());
        m_vects(m_size) = simplex.discrete_vector();
        m_size++;
        return *this;
    }

    KOKKOS_FUNCTION LocalChain<simplex_type>& operator+=(const vects_type& vects_to_add)
    {
        assert(size() + vects_to_add.size() <= allocation_size());
        for (auto i = vects_to_add.begin(); i < vects_to_add.end(); ++i) {
            m_vects(m_size + Kokkos::Experimental::distance(vects_to_add.begin(), i)) = *i;
        }
        m_size += vects_to_add.size();
        return *this;
    }

    KOKKOS_FUNCTION LocalChain<simplex_type>& operator+=(
            const LocalChain<simplex_type>& simplices_to_add)
    {
        assert(size() + simplices_to_add.size() <= allocation_size());
        for (auto i = simplices_to_add.begin(); i < simplices_to_add.end(); ++i) {
            m_vects(m_size + Kokkos::Experimental::distance(simplices_to_add.begin(), i)) = *i;
        }
        m_size += simplices_to_add.size();
        return *this;
    }

    KOKKOS_FUNCTION LocalChain<simplex_type> operator+(simplex_type simplex)
    {
        LocalChain<simplex_type> local_chain = *this;
        local_chain += simplex;
        return local_chain;
    }

    KOKKOS_FUNCTION LocalChain<simplex_type> operator+(LocalChain<simplex_type> simplices_to_add)
    {
        LocalChain<simplex_type> local_chain = *this;
        local_chain += simplices_to_add;
        return local_chain;
    }

    LocalChain<simplex_type> operator-() = delete;

    LocalChain<simplex_type> operator-(LocalChain<simplex_type>) = delete;

    template <class T>
    KOKKOS_FUNCTION LocalChain<simplex_type>& operator*=(T t)
    {
        if (t == 1) {
        } else if (t == -1) {
            assert(false && "negative simplices are unsupported in LocalChain");
        } else {
            assert(false && "chain must be multiplied  by 1 or -1");
        }
        return *this;
    }

    template <class T>
    KOKKOS_FUNCTION auto operator*(T t)
    {
        Chain<simplex_type> chain = *this;
        chain *= t;
        return chain;
    }

    KOKKOS_FUNCTION bool operator==(LocalChain<simplex_type> simplices)
    {
        for (auto i = simplices.begin(); i < simplices.end(); ++i) {
            if (*i != m_vects(Kokkos::Experimental::distance(simplices.begin(), i))) {
                return false;
            }
        }
        return true;
    }
};

template <class Head, misc::NotSpecialization<ddc::DiscreteVector>... Tail>
LocalChain(Head, Tail...) -> LocalChain<ddc::type_seq_element_t<0, ddc::detail::TypeSeq<Tail...>>>;

template <class Head, misc::Specialization<ddc::DiscreteVector>... Tail>
LocalChain(Head, Tail...) -> LocalChain<decltype(Simplex(
                                  misc::convert_type_seq_to_t<
                                          ddc::DiscreteElement,
                                          ddc::to_type_seq_t<typename Head::value_type>>(),
                                  ddc::type_seq_element_t<0, ddc::detail::TypeSeq<Tail...>>()))>;

// TODO Kokkosify
template <std::size_t K, misc::NotSpecialization<ddc::DiscreteDomain>... Tag>
KOKKOS_FUNCTION constexpr LocalChain<Simplex<K, Tag...>> tangent_basis()
{
    std::array<std::ptrdiff_t, sizeof...(Tag)> permutation
            = {0 * ddc::type_seq_rank_v<Tag, ddc::detail::TypeSeq<Tag...>>...};
    for (auto i = permutation.begin(); i < permutation.begin() + K; ++i) {
        *i = 1;
    }
    Kokkos::View<ddc::DiscreteVector<Tag...>*, Kokkos::HostSpace>
            basis("", misc::binomial_coefficient(sizeof...(Tag), K));
    std::size_t i = 0;
    do {
        basis(i) = ddc::DiscreteVector<Tag...>();
        ddc::detail::array(basis(i++)) = permutation;
    } while (std::prev_permutation(permutation.begin(), permutation.end()));

    return LocalChain<Simplex<K, Tag...>>(basis, basis.size());
}

namespace detail {

template <std::size_t K, misc::Specialization<ddc::DiscreteDomain> Dom>
struct TangentBasis;


template <std::size_t K, class... Tag>
struct TangentBasis<K, ddc::DiscreteDomain<Tag...>>
{
    KOKKOS_FUNCTION static auto constexpr run()
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
    for (typename ChainType::discrete_vector_type const& vect : chain) {
        out << " -> " << vect << "\n";
    }
    out << "]";
    return out;
}

} // namespace exterior

} // namespace sil
