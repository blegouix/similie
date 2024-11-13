// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <ddc/ddc.hpp>

#include "are_all_same.hpp"
#include "simplex.hpp"
#include "specialization.hpp"
#include "tuple_helpers.hpp"

namespace sil {

namespace form {

/// Chain class
template <misc::Specialization<Simplex>... SimplexType>
    requires(misc::are_all_equal(SimplexType::dimension()...))
class Chain
{
    template <misc::Specialization<Simplex>... T>
        requires(misc::are_all_equal(T::dimension()...))
    friend class Chain;

private:
    static constexpr std::size_t m_dimension
            = ddc::type_seq_element_t<0, ddc::detail::TypeSeq<SimplexType...>>::dimension();
    std::tuple<SimplexType...> m_simplices;

public:
    KOKKOS_FUNCTION constexpr explicit Chain(SimplexType... simplex) noexcept
        : m_simplices {simplex...}
    {
    }

    KOKKOS_FUNCTION constexpr explicit Chain(std::tuple<SimplexType...> simplices) noexcept
        : m_simplices(simplices)
    {
    }

    static KOKKOS_FUNCTION constexpr std::size_t dimension() noexcept
    {
        return m_dimension;
    }

    static KOKKOS_FUNCTION constexpr std::size_t extent() noexcept
    {
        return sizeof...(SimplexType);
    }

    template <std::size_t I>
    KOKKOS_FUNCTION auto& get() noexcept
    {
        return std::get<I>(m_simplices);
    }

    template <std::size_t I>
    KOKKOS_FUNCTION auto const& get() const noexcept
    {
        return std::get<I>(m_simplices);
    }

    KOKKOS_FUNCTION auto& front() noexcept
    {
        return get<0>();
    }

    KOKKOS_FUNCTION auto const& front() const noexcept
    {
        return get<0>();
    }

    KOKKOS_FUNCTION auto& back() noexcept
    {
        return get<extent() - 1>();
    }

    KOKKOS_FUNCTION auto const& back() const noexcept
    {
        return get<extent() - 1>();
    }

    template <std::size_t I = 0>
        requires(I < extent())
    Chain<SimplexType...> apply(auto f)
    {
        f(get<I>());
        return apply<I + 1>(f);
    }

    template <std::size_t I = 0>
        requires(I >= extent())
    Chain<SimplexType...> apply(auto f)
    {
        return *this;
    }

    template <std::size_t I = 0>
        requires(I < extent())
    Chain<SimplexType...> apply(auto f) const
    {
        f(get<I>());
        return apply<I + 1>(f);
    }

    template <std::size_t I = 0>
        requires(I >= extent())
    Chain<SimplexType...> apply(auto f) const
    {
        return *this;
    }

    KOKKOS_FUNCTION auto remove(std::size_t i) const
    {
        return *this;
    }

    KOKKOS_FUNCTION void check() const
    {
#ifdef NDEBUG
// TODO assert simplices unique
#endif
    }

private:
    // Helper to help the CTAD in optimize()
    template <misc::Specialization<Simplex>... T>
    auto make_chain(std::tuple<T...> t) const
    {
        return Chain<T...>(t);
    }

public:
    template <std::size_t I = 0>
    KOKKOS_FUNCTION auto optimize(auto chain) const
    {
        if constexpr (I < chain.extent()) {
            std::size_t j = 0;
            std::size_t k = std::numeric_limits<std::size_t>::quiet_NaN();
            chain.apply([&](auto simplex_) {
                if (std::get<I>(chain.m_simplices) == -simplex_) {
                    k = j;
                }
                ++j;
            });
            if (k != std::numeric_limits<std::size_t>::quiet_NaN()) {
                // TODO fix
                // return optimize<I>(make_chain(misc::remove(misc::remove(chain.m_simplices, k), I)));
                return optimize<I + 1>(chain);
            } else {
                return optimize<I + 1>(chain);
            }
        } else {
            return chain;
        }
    }

    template <std::size_t I = 0>
    KOKKOS_FUNCTION auto optimize() const
    {
        auto chain = optimize(*this);
        chain.check();
        return chain;
    }

    KOKKOS_FUNCTION auto operator-()
    {
        return apply([](auto& simplex) { simplex = -simplex; });
    }

    template <misc::Specialization<Simplex> SimplexToAdd>
    KOKKOS_FUNCTION Chain<SimplexType..., SimplexToAdd> operator+(SimplexToAdd simplex)
    {
        return Chain<SimplexType..., SimplexToAdd>(
                std::tuple_cat(m_simplices, std::tuple<SimplexToAdd> {simplex}));
    }

    template <misc::Specialization<Simplex>... SimplexToAdd>
    KOKKOS_FUNCTION auto operator+(Chain<SimplexToAdd...> chain)
    {
        return Chain<SimplexType..., SimplexToAdd...>(
                std::tuple_cat(m_simplices, chain.m_simplices));
    }

    template <class T>
    KOKKOS_FUNCTION auto operator-(T t)
    {
        return *this + (-t);
    }
};

// Deduction guide
template <misc::Specialization<Simplex>... SimplexType>
Chain(std::tuple<SimplexType...>) -> Chain<SimplexType...>;

template <misc::Specialization<Chain> ChainType>
std::ostream& operator<<(std::ostream& out, ChainType const& chain)
{
    out << "[ ";
    chain.apply([&](auto simplex) { out << simplex << " "; });
    out << "]";
    return out;
}

} // namespace form

} // namespace sil
