// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: MIT

#include <ddc/ddc.hpp>
#include <ddc/kernels/splines.hpp>

#include <similie/similie.hpp>

#include "scalar_field_hamiltonian.hpp"

static constexpr std::size_t s_degree = 3;

struct X
{
    static constexpr bool PERIODIC = false;
};

struct Y
{
    static constexpr bool PERIODIC = false;
};

using MesherXY = sil::mesher::Mesher<s_degree, X, Y>;

struct BSplinesX : MesherXY::template bsplines_type<X>
{
};

struct DDimX : MesherXY::template discrete_dimension_type<X>
{
};

struct BSplinesY : MesherXY::template bsplines_type<Y>
{
};

struct DDimY : MesherXY::template discrete_dimension_type<Y>
{
};

using Mu = sil::tensor::TensorNaturalIndex<X, Y>;

int main(int argc, char** argv)
{
    Kokkos::ScopeGuard const kokkos_scope(argc, argv);
    ddc::ScopeGuard const ddc_scope(argc, argv);

    MesherXY mesher;
    ddc::Coordinate<X, Y> lower_bounds(-5., -5.);
    ddc::Coordinate<X, Y> upper_bounds(5., 5.);
    ddc::DiscreteVector<DDimX, DDimY> nb_cells(256, 256);
    ddc::DiscreteDomain<DDimX, DDimY> mesh_xy = mesher.template mesh<
            ddc::detail::TypeSeq<DDimX, DDimY>,
            ddc::detail::TypeSeq<BSplinesX, BSplinesY>>(lower_bounds, upper_bounds, nb_cells);

    [[maybe_unused]] sil::tensor::TensorAccessor<Mu> hamiltonian_grad_accessor;
    ddc::DiscreteDomain<DDimX, DDimY, Mu>
            hamiltonian_grad_dom(mesh_xy, hamiltonian_grad_accessor.domain());
    ddc::Chunk hamiltonian_grad_alloc(hamiltonian_grad_dom, ddc::DeviceAllocator<double>());
    sil::tensor::Tensor hamiltonian_grad(hamiltonian_grad_alloc);

    ddc::parallel_for_each(
            Kokkos::DefaultExecutionSpace(),
            mesh_xy,
            KOKKOS_LAMBDA(ddc::DiscreteElement<DDimX, DDimY> elem) {
                const std::pair<float, float> grad = ScalarFieldHamiltonian::
                        grad(ddc::coordinate(ddc::DiscreteElement<DDimX>(elem)),
                             ddc::coordinate(ddc::DiscreteElement<DDimX>(elem)));

                hamiltonian_grad(elem, ddc::DiscreteElement<Mu>(0)) = std::get<0>(grad);
                hamiltonian_grad(elem, ddc::DiscreteElement<Mu>(1)) = std::get<1>(grad);

                printf("%f ", hamiltonian_grad(elem, ddc::DiscreteElement<Mu>(0)));
            });
    Kokkos::fence();

    return EXIT_SUCCESS;
}
