// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <ddc/ddc.hpp>

#include "specialization.hpp"

namespace sil {

namespace form {

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

public:
    template <misc::Specialization<ddc::DiscreteVector> T>
    KOKKOS_FUNCTION constexpr explicit Simplex(
            elem_type elem,
            T vect,
            bool negative = false) noexcept
        : base_type(elem)
        , m_vect(add_null_dimensions(vect))
        , m_negative(negative)
    {
        assert(((m_vect.template get<Tag>() == -1 || m_vect.template get<Tag>() == 0
                 || m_vect.template get<Tag>() == 1)
                && ...)
               && "simplex vector must contains only -1, 0 or 1"); // TODO only 0 and 1 actually
        // TODO reorient the simplex to remove every -1 in m_vect;
    }

    template <misc::Specialization<ddc::DiscreteVector> T>
    KOKKOS_FUNCTION constexpr explicit Simplex(
            std::integral_constant<std::size_t, K>,
            elem_type elem,
            T vect,
            bool negative = false) noexcept
        : base_type(elem)
        , m_vect(add_null_dimensions(vect))
        , m_negative(negative)
    {
        assert(((m_vect.template get<Tag>() == -1 || m_vect.template get<Tag>() == 0
                 || m_vect.template get<Tag>() == 1)
                && ...)
               && "simplex vector must contains only -1, 0 or 1"); // TODO only 0 and 1 actually
        // TODO reorient the simplex to remove every -1 in m_vect;
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
    out << (simplex.negative() ? "- " : "+ ");
    out << simplex.discrete_element();
    out << " -> ";
    out << simplex.discrete_element() + simplex.discrete_vector();
    out << " ";
    return out;
}

} // namespace form

} // namespace sil
