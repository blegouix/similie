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

template <class SimplexType, class MemorySpace>
Kokkos::View<typename SimplexType::discrete_vector_type*, MemorySpace> extract_vects(
        Kokkos::View<SimplexType*, MemorySpace> simplices)
{
    Kokkos::View<typename SimplexType::discrete_vector_type*, MemorySpace> vects(
            simplices.extent(0));
    for (auto i = Kokkos::Experimental::begin(simplices); i < Kokkos::Experimental::end(simplices);
         ++i) {
        vects[i] = i->discrete_vector();
    }
    return vects;
}

} // namespace detail


/// LocalChain class
template <class SimplexType, class ExecSpace = Kokkos::DefaultHostExecutionSpace>
class LocalChain
{
public:
    using execution_space = ExecSpace;
    using memory_space = typename ExecSpace::memory_space;

    using simplex_type = SimplexType;
    using simplices_type = Kokkos::View<SimplexType*, memory_space>;
    using discrete_element_type = typename simplex_type::discrete_element_type;
    using discrete_vector_type = typename simplex_type::discrete_vector_type;
    using vects_type = Kokkos::View<discrete_vector_type*, memory_space>;

    using iterator_type = Kokkos::Experimental::Impl::RandomAccessIterator<vects_type>;

private:
    static constexpr bool s_is_local = true;
    static constexpr std::size_t s_k = simplex_type::dimension();
    vects_type m_vects;

public:
    KOKKOS_FUNCTION constexpr explicit LocalChain() noexcept : discrete_element_type {}, m_vects {}
    {
    }

    // TODO Reorganize discrete vectors in all constructors ?

    template <misc::NotSpecialization<ddc::DiscreteVector>... T>
        requires misc::are_all_same<T...>
    KOKKOS_FUNCTION constexpr explicit LocalChain(T... simplex) noexcept
        : m_vects("local_chain_vects", sizeof...(T))
    {
        int i = 0;
        ((m_vects(i++) = simplex.discrete_vector()), ...);
        assert(check() == 0 && "there are duplicate simplices in the chain");
        assert(misc::are_all_equal(simplex.discrete_element()...)
               && "LocalChain must contain simplices with same origin (if not, use Chain)");
        assert((!simplex.negative() && ...)
               && "negative simplices are not supported in LocalChain");
    }

    KOKKOS_FUNCTION constexpr explicit LocalChain(simplices_type simplices) noexcept
        : m_vects(detail::extract_vects(simplices))
    {
        assert(check() == 0 && "there are duplicate simplices in the chain");
        std::function<bool()> check_common_elem = [&]() {
            Kokkos::View<discrete_element_type*, memory_space> elems(simplices.size());
            for (auto i = Kokkos::Experimental::begin(simplices);
                 i < Kokkos::Experimental::end(simplices);
                 ++i) {
                elems[Kokkos::Experimental::distance(Kokkos::Experimental::begin(simplices), i)]
                        = i->discrete_element();
            }
            return misc::are_all_equal(elems);
        };
        assert(check_common_elem()
               && "LocalChain must contain simplices with same origin (if not, use Chain)");
        assert(std::
                       all_of(Kokkos::Experimental::begin(simplices),
                              Kokkos::Experimental::end(simplices),
                              [&](const std::size_t i) { return !simplices[i].negative(); })
               && "LocalChain must contain simplices with same origin (if not, use Chain)");
    }

    template <misc::Specialization<ddc::DiscreteVector>... T>
    KOKKOS_FUNCTION constexpr explicit LocalChain(T... vect) noexcept
        : m_vects {(misc::filled_struct<discrete_vector_type>() + vect)...}
    {
        assert(check() == 0 && "there are duplicate simplices in the chain");
    }

    KOKKOS_FUNCTION constexpr explicit LocalChain(vects_type vects) noexcept : m_vects(vects)
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
        return Kokkos::Experimental::end(m_vects);
    }

    KOKKOS_FUNCTION auto end() const
    {
        return Kokkos::Experimental::end(m_vects);
    }

    KOKKOS_FUNCTION auto cbegin() const
    {
        return Kokkos::Experimental::begin(m_vects);
    }

    KOKKOS_FUNCTION auto cend() const
    {
        return Kokkos::Experimental::end(m_vects);
    }

    KOKKOS_FUNCTION simplex_type operator[](std::size_t i) noexcept
    {
        return simplex_type(misc::filled_struct<discrete_element_type>(), m_vects[i]);
    }

    KOKKOS_FUNCTION simplex_type const operator[](std::size_t i) const noexcept
    {
        return simplex_type(misc::filled_struct<discrete_element_type>(), m_vects[i]);
    }

    KOKKOS_FUNCTION void push_back(const discrete_vector_type& vect)
    {
        Kokkos::resize(m_vects, m_vects.size() + 1);
        m_vects[m_vects.size()] = vect;
    }

    KOKKOS_FUNCTION void push_back(const simplex_type& simplex)
    {
        Kokkos::resize(m_vects, m_vects.size() + 1);
        m_vects[m_vects.size()] = simplex.discrete_vector();
    }

    KOKKOS_FUNCTION void push_back(const vects_type& vects)
    {
        std::size_t old_size = m_vects.size();
        Kokkos::resize(m_vects, old_size + vects.size());
        for (auto i = Kokkos::Experimental::begin(vects); i < Kokkos::Experimental::end(vects);
             ++i) {
            m_vects[old_size
                    + Kokkos::Experimental::distance(Kokkos::Experimental::begin(vects), i)]
                    = *i;
        }
    }

    KOKKOS_FUNCTION void push_back(const LocalChain<simplex_type>& simplices_to_add)
    {
        std::size_t old_size = m_vects.size();
        Kokkos::resize(m_vects, old_size + simplices_to_add.size());
        for (auto i = simplices_to_add.begin(); i < simplices_to_add.end(); ++i) {
            m_vects[old_size + Kokkos::Experimental::distance(simplices_to_add.begin(), i)] = *i;
        }
    }

    LocalChain<simplex_type> operator-() = delete;

    KOKKOS_FUNCTION LocalChain<simplex_type> operator+(simplex_type simplex)
    {
        LocalChain<simplex_type> local_chain = *this;
        local_chain.push_back(simplex);
        return local_chain;
    }

    KOKKOS_FUNCTION LocalChain<simplex_type> operator+(LocalChain<simplex_type> simplices_to_add)
    {
        LocalChain<simplex_type> local_chain = *this;
        local_chain.push_back(simplices_to_add);
        return local_chain;
    }

    LocalChain<simplex_type> operator-(LocalChain<simplex_type>) = delete;

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

    KOKKOS_FUNCTION bool operator==(LocalChain<simplex_type> simplices)
    {
        for (auto i = simplices.begin(); i < simplices.end(); ++i) {
            if (*i != m_vects[Kokkos::Experimental::distance(simplices.begin(), i)]) {
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
    Kokkos::View<ddc::DiscreteVector<Tag...>*, Kokkos::HostSpace> basis();
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
