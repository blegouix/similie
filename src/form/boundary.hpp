// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <ddc/ddc.hpp>

#include "chain.hpp"
#include "specialization.hpp"

namespace sil {

namespace form {

namespace detail {

template <class SimplexType>
struct BoundaryType;

template <std::size_t K, class... Tag>
struct BoundaryType<Simplex<K, Tag...>>
{
    using type = Simplex<K - 1, Tag...>;
};

} // namespace detail

template <class SimplexType>
using boundary_t = typename detail::BoundaryType<SimplexType>::type;

namespace detail {


template <class SimplexType>
KOKKOS_FUNCTION constexpr Chain<boundary_t<SimplexType>> permutations_subset(
        typename SimplexType::elem_type elem,
        typename SimplexType::vect_type vect,
        bool negative = false)
{
    auto array = ddc::detail::array(vect);
    Chain<boundary_t<SimplexType>> chain;
    auto id_dist = -1;
    for (std::size_t i = 0; i < SimplexType::dimension(); ++i) {
        auto array_ = array;
        auto j = array_.begin() + id_dist + 1;
        auto id = std::find_if(j, array_.end(), [](int k) { return k != 0; });
        id_dist = std::distance(array_.begin(), id);
        *id = 0;
        typename SimplexType::vect_type vect_;
        ddc::detail::array(vect_) = array_;
        chain.push_back(boundary_t<SimplexType>(elem, vect_, (negative + i) % 2));
        j = id + 1;
    }
    return chain;
}

} // namespace detail

template <class SimplexType>
KOKKOS_FUNCTION Chain<boundary_t<SimplexType>> boundary(SimplexType simplex)
{
    return Chain<boundary_t<SimplexType>>(
                   detail::permutations_subset<SimplexType>(
                           simplex.discrete_element(),
                           simplex.discrete_vector(),
                           SimplexType::dimension() % 2)
                   + detail::permutations_subset<SimplexType>(
                           simplex.discrete_element() + simplex.discrete_vector(),
                           -simplex.discrete_vector()))
           * (SimplexType::dimension() % 2 ? 1 : -1) * (simplex.negative() ? -1 : 1);
}

template <class SimplexType>
KOKKOS_FUNCTION Chain<boundary_t<SimplexType>> boundary(Chain<SimplexType> chain)
{
    Chain<boundary_t<SimplexType>> boundary_chain;
    for (auto& simplex : chain) {
        Chain<boundary_t<SimplexType>> boundary_simplex = boundary(simplex);
        for (auto& simplex_ : boundary_simplex) {
            boundary_chain.push_back(simplex_);
        }
    }
    boundary_chain.optimize();
    return boundary_chain;
}

} // namespace form

} // namespace sil
