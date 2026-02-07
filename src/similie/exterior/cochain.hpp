// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: MIT

#pragma once

#include <ddc/ddc.hpp>

#include <similie/misc/are_all_same.hpp>
#include <similie/misc/specialization.hpp>
#include <similie/tensor/antisymmetric_tensor.hpp>

#include "chain.hpp"
#include "cosimplex.hpp"
#include "local_chain.hpp"

namespace sil {

namespace exterior {

template <typename CochainType>
class CochainIterator;

/// Cochain class
template <
        class ChainType,
        class ElementType = double,
        class LayoutStridedPolicy = Kokkos::LayoutRight>
    requires(misc::Specialization<ChainType, Chain> || misc::Specialization<ChainType, LocalChain>)
class Cochain
{
public:
    using memory_space = typename ChainType::memory_space;

    using simplex_type = typename ChainType::simplex_type;
    using chain_type = ChainType;
    using element_type = ElementType;
    using discrete_element_type = ChainType::discrete_element_type;
    using discrete_vector_type = ChainType::discrete_vector_type;
    using values_type = Kokkos::View<ElementType*, LayoutStridedPolicy, memory_space>;
    using cosimplex_type = Cosimplex<simplex_type, element_type>;

private:
    // chain_type const& m_chain;
    chain_type m_chain;
    values_type m_values;

public:
    KOKKOS_DEFAULTED_FUNCTION constexpr Cochain() = default;

    KOKKOS_DEFAULTED_FUNCTION constexpr Cochain(Cochain const&) = default;

    KOKKOS_DEFAULTED_FUNCTION constexpr Cochain(Cochain&&) = default;

    template <class... T>
        requires(sizeof...(T) >= 1 && (std::is_convertible_v<T, double> && ...))
    KOKKOS_FUNCTION constexpr explicit Cochain(
            chain_type chain,
            values_type allocation,
            T... value) noexcept
        : m_chain(std::move(chain))
        , m_values(std::move(allocation))
    {
        int i = 0;
        ((m_values(i++) = value), ...);
        assert(sizeof...(T) == chain.size()
               && "cochain constructor must get as much values as the chain contains simplices");
    }

    KOKKOS_FUNCTION constexpr explicit Cochain(
            chain_type const& chain,
            values_type const& values) noexcept
        : m_chain(std::move(chain))
        , m_values(std::move(values))
    {
        assert(values.size() == chain.size()
               && "cochain constructor must get as much values as the chain contains simplices");
    }

    template <tensor::TensorIndex Index>
        requires(misc::Specialization<Index, tensor::TensorAntisymmetricIndex>
                 || tensor::TensorNatIndex<Index>)
    KOKKOS_FUNCTION constexpr explicit Cochain(
            chain_type const& chain,
            tensor::Tensor<
                    element_type,
                    ddc::DiscreteDomain<Index>,
                    ddc::detail::kokkos_to_mdspan_layout_t<LayoutStridedPolicy>,
                    memory_space> tensor) noexcept
        : m_chain(std::move(chain))
        , m_values(tensor.allocation_kokkos_view())
    {
        assert(m_values.size() == chain.size()
               && "cochain constructor must get as much values as the chain contains simplices");
    }

    KOKKOS_DEFAULTED_FUNCTION ~Cochain() = default;

    KOKKOS_DEFAULTED_FUNCTION Cochain& operator=(Cochain const& other) = default;

    KOKKOS_DEFAULTED_FUNCTION Cochain& operator=(Cochain&& other) = default;

    static KOKKOS_FUNCTION constexpr bool is_local() noexcept
    {
        return chain_type::is_local();
    }

    static KOKKOS_FUNCTION constexpr std::size_t dimension() noexcept
    {
        return chain_type::dimension();
    }

    KOKKOS_FUNCTION std::size_t size() noexcept
    {
        return m_chain.size();
    }

    KOKKOS_FUNCTION std::size_t size() const noexcept
    {
        return m_chain.size();
    }

    KOKKOS_FUNCTION std::size_t allocation_size() noexcept
    {
        return m_chain.allocation_size();
    }

    KOKKOS_FUNCTION std::size_t allocation_size() const noexcept
    {
        return m_chain.allocation_size();
    }

    KOKKOS_FUNCTION chain_type const& chain() const noexcept
    {
        return m_chain;
    }

    KOKKOS_FUNCTION chain_type const& values() const noexcept
    {
        return m_values;
    }

    using iterator = CochainIterator<Cochain>;

    KOKKOS_FUNCTION auto begin()
    {
        return iterator(m_chain.begin(), Kokkos::Experimental::begin(m_values));
    }

    KOKKOS_FUNCTION auto begin() const
    {
        return iterator(m_chain.begin(), Kokkos::Experimental::begin(m_values));
    }

    KOKKOS_FUNCTION auto end()
    {
        return iterator(m_chain.end(), Kokkos::Experimental::end(m_values));
    }

    KOKKOS_FUNCTION auto end() const
    {
        return iterator(m_chain.end(), Kokkos::Experimental::end(m_values));
    }

    KOKKOS_FUNCTION auto cbegin() const
    {
        return iterator(m_chain.begin(), Kokkos::Experimental::begin(m_values));
    }

    KOKKOS_FUNCTION auto cend() const
    {
        return iterator(m_chain.end(), Kokkos::Experimental::end(m_values));
    }

    KOKKOS_FUNCTION Cosimplex<simplex_type, element_type>& operator[](std::size_t i) noexcept
    {
        assert(i < size());
        return Cosimplex<simplex_type, element_type>(m_chain(i), m_values(i));
    }

    KOKKOS_FUNCTION Cosimplex<simplex_type, element_type> const& operator[](
            std::size_t i) const noexcept
    {
        assert(i < size());
        return Cosimplex<simplex_type, element_type>(m_chain(i), m_values(i));
    }

    KOKKOS_FUNCTION element_type integrate() noexcept
    {
        element_type out = 0;
        for (auto i = begin(); i < end(); ++i) {
            if constexpr (misc::Specialization<chain_type, Chain>) {
                out += ((*i).negative() ? -1 : 1) * (*i).value();
            } else if constexpr (misc::Specialization<chain_type, LocalChain>) {
                out += (m_chain.negative() ? -1 : 1) * (*i).value(); // always false
            }
        }
        return out;
    }

    KOKKOS_FUNCTION element_type const integrate() const noexcept
    {
        element_type out = 0;
        for (auto i = begin(); i < end(); ++i) {
            if constexpr (misc::Specialization<chain_type, Chain>) {
                out += ((*i).negative() ? -1 : 1) * (*i).value();
            } else if constexpr (misc::Specialization<chain_type, LocalChain>) {
                out += (m_chain.negative() ? -1 : 1) * (*i).value(); // always false
            }
        }
        return out;
    }
};

template <class ChainType, misc::Specialization<tensor::Tensor> TensorType>
Cochain(ChainType, TensorType) -> Cochain<
        ChainType,
        typename TensorType::value_type,
        ddc::detail::mdspan_to_kokkos_layout_t<typename TensorType::layout_type>>;

template <typename CochainType>
class CochainIterator
{
public:
    using iterator_category = std::random_access_iterator_tag;
    using value_type = typename CochainType::cosimplex_type;
    using difference_type = std::ptrdiff_t;
    using pointer = value_type*;
    using reference = value_type&;

    using cosimplex_type = typename CochainType::cosimplex_type;

    using chain_iterator = typename CochainType::chain_type::iterator_type;
    using values_iterator
            = Kokkos::Experimental::Impl::RandomAccessIterator<typename CochainType::values_type>;

private:
    chain_iterator m_chain;
    values_iterator m_values;

public:
    KOKKOS_DEFAULTED_FUNCTION CochainIterator() = default;

    KOKKOS_FUNCTION constexpr explicit CochainIterator(
            chain_iterator chain_it,
            values_iterator values_it)
        : m_chain(chain_it)
        , m_values(values_it)
    {
    }

    KOKKOS_FUNCTION constexpr value_type operator*() const noexcept
    {
        if constexpr (!CochainType::chain_type::is_local()) {
            return cosimplex_type(*m_chain, *m_values);
        } else {
            return cosimplex_type(
                    Simplex(std::integral_constant<std::size_t, CochainType::dimension()> {},
                            misc::filled_struct<typename CochainType::discrete_element_type>(),
                            *m_chain),
                    *m_values);
        }
    }

    KOKKOS_FUNCTION constexpr CochainIterator& operator++()
    {
        ++m_chain;
        ++m_values;
        return *this;
    }

    KOKKOS_FUNCTION constexpr CochainIterator operator++(int)
    {
        auto tmp = *this;
        ++*this;
        return tmp;
    }

    KOKKOS_FUNCTION constexpr CochainIterator& operator--()
    {
        --m_chain;
        --m_values;
        return *this;
    }

    KOKKOS_FUNCTION constexpr CochainIterator operator--(int)
    {
        auto tmp = *this;
        --*this;
        return tmp;
    }

    KOKKOS_FUNCTION constexpr CochainIterator& operator+=(difference_type n)
    {
        m_chain += n;
        m_values += n;
        return *this;
    }

    KOKKOS_FUNCTION constexpr CochainIterator& operator-=(difference_type n)
    {
        m_chain -= n;
        m_values -= n;
        return *this;
    }

    /*
    KOKKOS_FUNCTION constexpr value_type operator[](difference_type n) const
    {
        return m_value + n;
    }
    */

    friend KOKKOS_FUNCTION constexpr bool operator==(
            CochainIterator const& xx,
            CochainIterator const& yy)
    {
        return xx.m_chain == yy.m_chain && xx.m_values == yy.m_values;
    }

    friend KOKKOS_FUNCTION constexpr bool operator<(
            CochainIterator const& xx,
            CochainIterator const& yy)
    {
        return xx.m_chain < yy.m_chain && xx.m_values < yy.m_values;
    }

    friend KOKKOS_FUNCTION constexpr bool operator>(
            CochainIterator const& xx,
            CochainIterator const& yy)
    {
        return xx.m_chain > yy.m_chain && xx.m_values > yy.m_values;
    }

    friend KOKKOS_FUNCTION constexpr bool operator<=(
            CochainIterator const& xx,
            CochainIterator const& yy)
    {
        return !(yy < xx);
    }

    friend KOKKOS_FUNCTION constexpr bool operator>=(
            CochainIterator const& xx,
            CochainIterator const& yy)
    {
        return !(xx < yy);
    }

    friend KOKKOS_FUNCTION constexpr CochainIterator operator+(CochainIterator i, difference_type n)
    {
        return i += n;
    }

    friend KOKKOS_FUNCTION constexpr CochainIterator operator+(difference_type n, CochainIterator i)
    {
        return i += n;
    }

    friend KOKKOS_FUNCTION constexpr CochainIterator operator-(CochainIterator i, difference_type n)
    {
        return i -= n;
    }

    friend KOKKOS_FUNCTION constexpr difference_type operator-(
            CochainIterator const& xx,
            CochainIterator const& yy)
    {
        return xx.m_chain - yy.m_chain;
    }
};

template <misc::Specialization<Cochain> CochainType>
std::ostream& operator<<(std::ostream& out, CochainType const& cochain)
{
    out << "[\n";
    for (auto i = cochain.begin(); i < cochain.end(); ++i) {
        if constexpr (!cochain.is_local()) {
            out << " " << (*i).simplex() << " : " << (*i).value() << "\n";
        } else {
            out << " -> " << (*i).discrete_vector() << " : " << (*i).value() << "\n";
        }
    }

    out << "]";

    return out;
}

} // namespace exterior

} // namespace sil
