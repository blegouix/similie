// SPDX-License-Identifier: GPL-3.0

#include <ddc/ddc.hpp>

#include "mesher.hpp"

template <class X, std::size_t D>
Mesher<X, D>::Mesher(double x_start, double x_end, std::size_t nb_x_points)
{
    ddc::init_discrete_space<discrete_dimension_type>(
            ddc::Coordinate<X>(x_start),
            ddc::Coordinate<X>(x_end),
            nb_x_points);
    ddc::init_discrete_space<discrete_dimension_type>(
            greville_points_type::template get_sampling<discrete_dimension_type>());
    printf("Mesher created");
}

template <class X, std::size_t D>
ddc::DiscreteDomain<typename Mesher<X, D>::discrete_dimension_type> Mesher<X, D>::get_domain()
{
    return greville_points_type::template get_domain<discrete_dimension_type>();
}
