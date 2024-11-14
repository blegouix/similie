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

/*
 Compute all permutations on a subset of indices in idx_to_permute, keep the
 rest of the indices where they are.
 */
template <class SimplexType>
KOKKOS_FUNCTION constexpr Chain<boundary_t<SimplexType>> permutations_subset(
        typename SimplexType::elem_type elem,
        typename SimplexType::vect_type vect)
{
    auto array = ddc::detail::array(vect);
    Chain<boundary_t<SimplexType>> chain;
    auto id_dist = 0;
    for (std::size_t i = 0; i < SimplexType::dimension(); ++i) {
        auto array_ = array;
        auto j = array_.begin() + id_dist + 1;
        auto id = std::find_if(j, array_.end(), [](int k) { return k != 0; });
        id_dist = std::distance(array_.begin(), id);
        *id = 0;
        typename SimplexType::vect_type vect_;
        ddc::detail::array(vect_) = array_;
        chain.push_back(boundary_t<SimplexType>(elem, vect_));
        j = id + 1;
    }
    return chain;
}

} // namespace detail

template <class SimplexType>
KOKKOS_FUNCTION Chain<boundary_t<SimplexType>> boundary(SimplexType simplex)
{
    return Chain<boundary_t<SimplexType>>(
            detail::permutations_subset<
                    SimplexType>(simplex.discrete_element(), simplex.discrete_vector())
            - detail::permutations_subset<SimplexType>(
                    simplex.discrete_element() + simplex.discrete_vector(),
                    -simplex.discrete_vector()));
}

} // namespace form

} // namespace sil
