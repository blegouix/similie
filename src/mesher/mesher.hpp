// SPDX-License-Identifier: GPL-3.0

#pragma once

#include <ddc/ddc.hpp>
#include <ddc/kernels/splines.hpp>

template <class BSplines>
class Mesher
{
public:
    static constexpr ddc::BoundCond BoundCond = ddc::BoundCond::GREVILLE;

    using bsplines_type = BSplines;

    using greville_points = ddc::GrevilleInterpolationPoints<BSplines, BoundCond, BoundCond>;

    explicit Mesher();
};
