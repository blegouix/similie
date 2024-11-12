// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <ddc/ddc.hpp>

#include "specialization.hpp"

namespace sil {

namespace form {

/// Simplex class
template <misc::Specialization<ddc::DiscreteVector> Vect, class... Tag>
class Simplex;

template <class... Tag, class... Edge>
class Simplex<ddc::DiscreteVector<Edge...>, Tag...> : public ddc::DiscreteElement<Tag...>
{
protected:
    using base_type = ddc::DiscreteElement<Tag...>;

private:
    using vect_type = ddc::DiscreteVector<Edge...>;

    vect_type m_vect;

public:
    KOKKOS_FUNCTION constexpr explicit Simplex(
            ddc::DiscreteElement<Tag...> elem,
            vect_type vect) noexcept
        : base_type(elem)
        , m_vect(vect)
    {
        assert(((vect.template get<Edge>() == -1 || vect.template get<Edge>() == 1) && ...)
               && "simplex vector must contains only -1 or 1");
    }

    static KOKKOS_FUNCTION constexpr std::size_t dimension() noexcept
    {
        return sizeof...(Edge);
    }

    KOKKOS_FUNCTION base_type discrete_element() noexcept
    {
        return base_type {this->template uid<Tag>()...};
    }

    KOKKOS_FUNCTION vect_type discrete_vector() noexcept
    {
        return m_vect;
    }

    KOKKOS_FUNCTION Simplex<ddc::DiscreteVector<Edge...>, Tag...> operator-()
    {
        return Simplex(
                discrete_element(),
                ddc::DiscreteVector<Edge...> {-m_vect.template get<Edge>()...});
    }
};

// Deduction guide
template <class... Tag, class... Edge>
Simplex(ddc::DiscreteElement<Tag...>,
        ddc::DiscreteVector<Edge...>) -> Simplex<ddc::DiscreteVector<Edge...>, Tag...>;

} // namespace form

} // namespace sil
