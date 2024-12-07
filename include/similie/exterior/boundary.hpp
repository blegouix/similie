// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <ddc/ddc.hpp>

#include <similie/misc/specialization.hpp>

#include "chain.hpp"

namespace sil {

namespace exterior {

namespace detail {

template <class SimplexType>
struct BoundaryType;

template <std::size_t K, class... Tag>
struct BoundaryType<Simplex<K, Tag...>>
{
    using type = Simplex<K - 1, Tag...>;
};

template <class SimplexType>
struct BoundaryType<Chain<SimplexType>>
{
    using type = Chain<typename BoundaryType<SimplexType>::type>;
};

} // namespace detail

template <class T>
using boundary_t = typename detail::BoundaryType<T>::type;

namespace detail {

// TODO Kokkosify
template <class SimplexType, misc::Specialization<Kokkos::View> AllocationType>
KOKKOS_FUNCTION constexpr Chain<boundary_t<SimplexType>> generate_half_subchain(
        AllocationType allocation,
        typename SimplexType::discrete_element_type elem,
        typename SimplexType::discrete_vector_type vect,
        bool negative = false)
{
    auto array = ddc::detail::array(vect);
    Chain<boundary_t<SimplexType>> chain(allocation);
    auto id_dist = -1;
    for (std::size_t i = 0; i < SimplexType::dimension(); ++i) {
        auto array_ = array;
        auto j = array_.begin() + id_dist + 1;
        auto id = std::find_if(j, array_.end(), [](int k) { return k != 0; });
        id_dist = std::distance(array_.begin(), id);
        *id = 0;
        typename SimplexType::discrete_vector_type vect_;
        ddc::detail::array(vect_) = array_;
        chain += boundary_t<SimplexType>(elem, vect_, (negative + i) % 2);
        j = id + 1;
    }
    return chain;
}

} // namespace detail

template <misc::Specialization<Kokkos::View> AllocationType, class SimplexType>
KOKKOS_FUNCTION Chain<boundary_t<SimplexType>> boundary(
        AllocationType allocation,
        SimplexType simplex)
{
    Chain<boundary_t<SimplexType>> chain(allocation);
    detail::generate_half_subchain<SimplexType>(
            Kokkos::
                    subview(allocation,
                            std::pair<std::size_t, std::size_t>(0, SimplexType::dimension())),
            simplex.discrete_element(),
            simplex.discrete_vector(),
            SimplexType::dimension() % 2);
    chain += SimplexType::dimension();
    detail::generate_half_subchain<SimplexType>(
            Kokkos::
                    subview(allocation,
                            std::pair<std::size_t, std::size_t>(
                                    SimplexType::dimension(),
                                    2 * SimplexType::dimension())),
            simplex.discrete_element() + simplex.discrete_vector(),
            -simplex.discrete_vector());
    chain += SimplexType::dimension();
    chain *= (SimplexType::dimension() % 2 ? 1 : -1) * (simplex.negative() ? -1 : 1);
    return chain;
}

template <misc::Specialization<Kokkos::View> AllocationType, class SimplexType>
KOKKOS_FUNCTION Chain<boundary_t<SimplexType>> boundary(
        AllocationType allocation,
        Chain<SimplexType> chain)
{
    Chain<boundary_t<SimplexType>> boundary_chain(allocation);
    for (auto i = chain.begin(); i < chain.end(); ++i) {
        std::size_t const distance = Kokkos::Experimental::distance(chain.begin(), i);
        boundary(
                Kokkos::
                        subview(allocation,
                                std::pair<std::size_t, std::size_t>(
                                        2 * SimplexType::dimension() * distance,
                                        2 * SimplexType::dimension() * (distance + 1))),
                *i);
        boundary_chain += 2 * SimplexType::dimension();
    }
    boundary_chain.optimize();
    return boundary_chain;
}

} // namespace exterior

} // namespace sil
