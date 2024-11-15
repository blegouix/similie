// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <ddc/ddc.hpp>

#include "specialization.hpp"

namespace sil {

namespace exterior {

/// Simplex class
template <std::size_t K, class... Tag>
class Simplex : public ddc::DiscreteElement<Tag...>
{
protected:
    using base_type = ddc::DiscreteElement<Tag...>;

public:
    using elem_type = base_type;
    using vect_type = ddc::DiscreteVector<Tag...>;

private:
    static constexpr std::size_t s_k = K;
    vect_type
            m_vect; // Only booleans are stored but ddc::DiscreteVector supports only std::ptrdiff_t
    bool m_negative;

    template <class Tag_, class... T>
        requires(!ddc::type_seq_contains_v<ddc::detail::TypeSeq<Tag_>, ddc::detail::TypeSeq<T...>>)
    static constexpr ddc::DiscreteVector<Tag_> add_eventually_null_dimensions_(
            ddc::DiscreteVector<T...> vect)
    {
        return ddc::DiscreteVector<Tag_> {0};
    }

    template <class Tag_, class... T>
        requires(ddc::type_seq_contains_v<ddc::detail::TypeSeq<Tag_>, ddc::detail::TypeSeq<T...>>)
    static constexpr ddc::DiscreteVector<Tag_> add_eventually_null_dimensions_(
            ddc::DiscreteVector<T...> vect)
    {
        return ddc::DiscreteVector<Tag_>(vect);
    }

    template <class... T>
    static constexpr vect_type add_null_dimensions(ddc::DiscreteVector<T...> vect)
    {
        return vect_type(add_eventually_null_dimensions_<Tag, T...>(vect)...);
    }

    template <class... T>
    struct Reorient;

    template <>
    struct Reorient<>
    {
        static constexpr base_type run_elem(base_type elem, vect_type)
        {
            return elem;
        }
        static constexpr vect_type run_vect(base_type, vect_type vect)
        {
            return vect;
        }
        static constexpr bool run_negative(vect_type vect, bool negative)
        {
            return negative;
        }
    };


    template <class HeadTag, class... TailTag>
    struct Reorient<HeadTag, TailTag...>
    {
        static constexpr base_type run_elem(base_type elem, vect_type vect)
        {
            if (vect.template get<HeadTag>() == -1) {
                elem.template uid<HeadTag>()--;
            }
            return Reorient<TailTag...>::run_elem(elem, vect);
        }

        static constexpr vect_type run_vect(base_type elem, vect_type vect)
        {
            if (vect.template get<HeadTag>() == -1) {
                vect.template get<HeadTag>() = 1;
            }
            return Reorient<TailTag...>::run_vect(elem, vect);
        }

        static constexpr bool run_negative(vect_type vect, bool negative)
        {
            return Reorient<TailTag...>::
                    run_negative(vect, (vect.template get<HeadTag>() == -1) != negative);
        }
    };

public:
    template <misc::Specialization<ddc::DiscreteVector> T>
    KOKKOS_FUNCTION constexpr explicit Simplex(
            elem_type elem,
            T vect,
            bool negative = false) noexcept
        : base_type(Reorient<Tag...>::run_elem(elem, add_null_dimensions(vect)))
        , m_vect(Reorient<Tag...>::run_vect(elem, add_null_dimensions(vect)))
        , m_negative(Reorient<Tag...>::run_negative(add_null_dimensions(vect), negative))
    {
        assert(((m_vect.template get<Tag>() == 0 || m_vect.template get<Tag>() == 1) && ...)
               && "simplex vector must contain only -1, 0 or 1");
    }

    template <misc::Specialization<ddc::DiscreteVector> T>
    KOKKOS_FUNCTION constexpr explicit Simplex(
            std::integral_constant<std::size_t, K>,
            elem_type elem,
            T vect,
            bool negative = false) noexcept
        : base_type(Reorient<Tag...>::run_elem(elem, add_null_dimensions(vect)))
        , m_vect(Reorient<Tag...>::run_vect(elem, add_null_dimensions(vect)))
        , m_negative(Reorient<Tag...>::run_negative(add_null_dimensions(vect), negative))
    {
        assert(((m_vect.template get<Tag>() == 0 || m_vect.template get<Tag>() == 1) && ...)
               && "simplex vector must contain only -1, 0 or 1");
    }

    static KOKKOS_FUNCTION constexpr std::size_t dimension() noexcept
    {
        return s_k;
    }

    KOKKOS_FUNCTION base_type discrete_element() noexcept // TODO base_type& ?
    {
        return base_type {this->template uid<Tag>()...};
    }

    KOKKOS_FUNCTION const base_type discrete_element() const noexcept // TODO base_type& ?
    {
        return base_type {this->template uid<Tag>()...};
    }

    KOKKOS_FUNCTION vect_type& discrete_vector() noexcept
    {
        return m_vect;
    }

    KOKKOS_FUNCTION vect_type const& discrete_vector() const noexcept
    {
        return m_vect;
    }

    KOKKOS_FUNCTION bool& negative() noexcept
    {
        return m_negative;
    }

    KOKKOS_FUNCTION bool const& negative() const noexcept
    {
        return m_negative;
    }

    KOKKOS_FUNCTION Simplex<s_k, Tag...> operator-()
    {
        return Simplex<s_k, Tag...>(discrete_element(), discrete_vector(), !negative());
    }

    KOKKOS_FUNCTION bool operator==(Simplex<s_k, Tag...> simplex)
    {
        return (discrete_element() == simplex.discrete_element()
                && discrete_vector() == simplex.discrete_vector()
                && negative() == simplex.negative());
    }
};

template <class... Tag, class... T>
Simplex(ddc::DiscreteElement<Tag...>, ddc::DiscreteVector<T...>) -> Simplex<sizeof...(T), Tag...>;

template <class... Tag, class... T>
Simplex(ddc::DiscreteElement<Tag...>,
        ddc::DiscreteVector<T...>,
        bool) -> Simplex<sizeof...(T), Tag...>;

template <std::size_t K, class... Tag>
std::ostream& operator<<(std::ostream& out, Simplex<K, Tag...> const& simplex)
{
    out << " ";
    out << simplex.discrete_element();
    out << (simplex.negative() ? " <- " : " -> ");
    out << simplex.discrete_element() + simplex.discrete_vector();
    out << " ";
    return out;
}

} // namespace exterior

} // namespace sil
