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
    using greville_points_type
            = ddc::GrevilleInterpolationPoints<bsplines_type, BoundCond, BoundCond>;

public:
    using discrete_dimension_type =
            typename greville_points_type::interpolation_discrete_dimension_type;

    explicit Mesher(double x_start, double x_end, std::size_t nb_x_points);

    ddc::DiscreteDomain<discrete_dimension_type> get_domain();
};
