// SPDX-License-Identifier: GPL-3.0

#pragma once

#include <ddc/ddc.hpp>
#include <ddc/kernels/splines.hpp>

template <class X, std::size_t D>
class Mesher
{
public:
    static constexpr ddc::BoundCond BoundCond = ddc::BoundCond::GREVILLE;

    using bsplines_type = ddc::UniformBSplines<X, D>;

private:
    template <class T>
    using greville_points_type = ddc::GrevilleInterpolationPoints<T, BoundCond, BoundCond>;

public:
    using discrete_dimension_type =
            typename greville_points_type<bsplines_type>::interpolation_discrete_dimension_type;

    template <class DDimX, class BSplinesX>
    ddc::DiscreteDomain<DDimX> mesh(double x_start, double x_end, std::size_t nb_x_points)
    {
        static_assert(
                std::is_base_of_v<discrete_dimension_type, DDimX>,
                "DDimX has to derivate from discrete_dimension_type");
        static_assert(
                std::is_base_of_v<bsplines_type, BSplinesX>,
                "BSplinesX has to derivate from bsplines_type");

        ddc::init_discrete_space<
                BSplinesX>(ddc::Coordinate<X>(x_start), ddc::Coordinate<X>(x_end), nb_x_points);
        ddc::init_discrete_space<DDimX>(
                greville_points_type<BSplinesX>::template get_sampling<DDimX>());
        return greville_points_type<BSplinesX>::template get_domain<DDimX>();
    };
};
