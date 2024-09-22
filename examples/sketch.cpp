// SPDX-License-Identifier: GPL-3.0

#include <ddc/ddc.hpp>
#include <ddc/kernels/splines.hpp>

#include "tensor.hpp"

static constexpr std::size_t s_degree_x = 3;

struct X
{
    static constexpr bool PERIODIC = false;
};

struct BSplinesX : ddc::UniformBSplines<X, s_degree_x>
{
};

static constexpr ddc::BoundCond BoundCond = ddc::BoundCond::GREVILLE;

using GrevillePoints = ddc::GrevilleInterpolationPoints<BSplinesX, BoundCond, BoundCond>;

struct DDimX : GrevillePoints::interpolation_discrete_dimension_type
{
};

int main(int argc, char** argv)
{
    Kokkos::ScopeGuard const kokkos_scope(argc, argv);
    ddc::ScopeGuard const ddc_scope(argc, argv);

    printf("start example\n");

    // Initialization of the global domain in X
    ddc::init_discrete_space<BSplinesX>(ddc::Coordinate<X>(0), ddc::Coordinate<X>(1), 1000);
    ddc::init_discrete_space<DDimX>(GrevillePoints::get_sampling<DDimX>());

    ddc::DiscreteDomain<DDimX> const x_domain = GrevillePoints::get_domain<DDimX>();
}
