// SPDX-License-Identifier: GPL-3.0

#include <ddc/ddc.hpp>
#include <ddc/kernels/splines.hpp>

#include "mesher.hpp"
#include "tensor.hpp"

static constexpr std::size_t s_degree_x = 3;

struct X
{
    static constexpr bool PERIODIC = false;
};

using MesherX = Mesher1D<X, s_degree_x>;

struct BSplinesX : MesherX::bsplines_type
{
};

struct DDimX : MesherX::discrete_dimension_type
{
};

int main(int argc, char** argv)
{
    Kokkos::ScopeGuard const kokkos_scope(argc, argv);
    ddc::ScopeGuard const ddc_scope(argc, argv);

    printf("start example\n");

    MesherX mesher;
    ddc::DiscreteDomain<DDimX> dom_x = mesher.template mesh<DDimX, BSplinesX>(0., 1., 1000);
}
