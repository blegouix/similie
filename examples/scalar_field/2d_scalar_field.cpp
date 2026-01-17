// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: MIT

#include <ddc/ddc.hpp>
#include <ddc/kernels/splines.hpp>
#include <ddc/pdi.hpp>

#include <similie/similie.hpp>

#include "scalar_field_hamiltonian.hpp"

// PDI config
constexpr char const* const PDI_CFG = R"PDI_CFG(
metadata:
  Nx : int
  Ny : int

data:
  position:
    type: array
    subtype: double
    size: [ '$Nx', '$Ny' , 2]
  potential:
    type: array
    subtype: double
    size: [ '$Nx', '$Ny', 2 ]

plugins:
  decl_hdf5:
    - file: 'scalar_field.h5'
      on_event: [export]
      collision_policy: replace_and_warn
      write: [Nx, Ny, position, potential]
  #trace: ~
)PDI_CFG";

// XDMF
int write_xdmf(int Nx, int Ny)
{
    constexpr char const* const xdmf = R"XDMF(<?xml version="1.0" ?>
<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
<Xdmf Version="2.0">
 <Domain>
   <Grid Name="mesh1" GridType="Uniform">
     <Topology TopologyType="2DSMesh" NumberOfElements="%i %i"/>
     <Geometry GeometryType="XY">
       <DataItem Dimensions="%i %i 2" NumberType="Float" Precision="8" Format="HDF">
        2d_vector_laplacian.h5:/position
       </DataItem>
     </Geometry>
     <Attribute Name="Potential" AttributeType="Vector" Center="Cell"> // Cell enforced because of Paraview bug
       <DataItem Dimensions="%i %i 2" NumberType="Float" Precision="8" Format="HDF">
        scalar_field.h5:/potential
       </DataItem>
     </Attribute>
   </Grid>
 </Domain>
</Xdmf>
)XDMF";

    FILE* file = fopen("scalar_field.xmf", "w");
    fprintf(file, xdmf, Nx, Ny, Nx, Ny, Nx, Ny);
    fclose(file);

    return 1;
}

static constexpr std::size_t s_degree = 3;

struct X
{
    static constexpr bool PERIODIC = false;
};

struct Y
{
    static constexpr bool PERIODIC = false;
};

// Declare a metric
using MetricIndex = sil::tensor::TensorSymmetricIndex<
        sil::tensor::Covariant<sil::tensor::MetricIndex1<X, Y>>,
        sil::tensor::Covariant<sil::tensor::MetricIndex2<X, Y>>>;

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

// Declare natural indices taking values in {X, Y}
struct Mu : sil::tensor::TensorNaturalIndex<X, Y>
{
};

struct Nu : sil::tensor::TensorNaturalIndex<X, Y>
{
};

// Declare indices
using MuUp = sil::tensor::Contravariant<Mu>;
using MuLow = sil::tensor::Covariant<Mu>;
using NuUp = sil::tensor::Contravariant<Nu>;
using NuLow = sil::tensor::Covariant<Nu>;

using DummyIndex = sil::tensor::Covariant<sil::tensor::ScalarIndex>;

int main(int argc, char** argv)
{
    // Initialize PDI, Kokkos and DDC
    PC_tree_t conf_pdi = PC_parse_string(PDI_CFG);
    PC_errhandler(PC_NULL_HANDLER);
    PDI_init(conf_pdi);

    Kokkos::ScopeGuard const kokkos_scope(argc, argv);
    ddc::ScopeGuard const ddc_scope(argc, argv);

    // Produce mesh
    MesherXY mesher;
    ddc::Coordinate<X, Y> lower_bounds(-5., -5.);
    ddc::Coordinate<X, Y> upper_bounds(5., 5.);
    ddc::DiscreteVector<DDimX, DDimY> nb_cells(256, 256);
    ddc::DiscreteDomain<DDimX, DDimY> mesh_xy = mesher.template mesh<
            ddc::detail::TypeSeq<DDimX, DDimY>,
            ddc::detail::TypeSeq<BSplinesX, BSplinesY>>(lower_bounds, upper_bounds, nb_cells);
    assert(static_cast<std::size_t>(mesh_xy.template extent<DDimX>())
           == static_cast<std::size_t>(mesh_xy.template extent<DDimY>()));
    ddc::expose_to_pdi("Nx", static_cast<int>(mesh_xy.template extent<DDimX>()));
    ddc::expose_to_pdi("Ny", static_cast<int>(mesh_xy.template extent<DDimY>()));

    // Allocate and instantiate a position field (used only to be exported).
    [[maybe_unused]] sil::tensor::TensorAccessor<NuUp> position_accessor;
    ddc::DiscreteDomain<DDimX, DDimY, NuUp> position_dom(mesh_xy, position_accessor.domain());
    ddc::Chunk position_alloc(position_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor position(position_alloc);
    ddc::parallel_for_each(
            Kokkos::DefaultHostExecutionSpace(),
            mesh_xy,
            KOKKOS_LAMBDA(ddc::DiscreteElement<DDimX, DDimY> elem) {
                position(elem, position.accessor().access_element<X>())
                        = static_cast<double>(ddc::coordinate(ddc::DiscreteElement<DDimX>(elem)));
                position(elem, position.accessor().access_element<Y>())
                        = static_cast<double>(ddc::coordinate(ddc::DiscreteElement<DDimY>(elem)));
            });

    // Allocate and instantiate a metric tensor field.
    [[maybe_unused]] sil::tensor::TensorAccessor<MetricIndex> metric_accessor;
    ddc::DiscreteDomain<DDimX, DDimY, MetricIndex> metric_dom(mesh_xy, metric_accessor.domain());
    ddc::Chunk metric_alloc(metric_dom, ddc::DeviceAllocator<double>());
    sil::tensor::Tensor metric(metric_alloc);
    ddc::parallel_for_each(
            Kokkos::DefaultExecutionSpace(),
            mesh_xy,
            KOKKOS_LAMBDA(ddc::DiscreteElement<DDimX, DDimY> elem) {
                metric(elem, metric.accessor().access_element<X, X>()) = 1.;
                metric(elem, metric.accessor().access_element<X, Y>()) = 0.;
                metric(elem, metric.accessor().access_element<Y, Y>()) = 1.;
            });

    // Invert metric
    [[maybe_unused]] sil::tensor::TensorAccessor<sil::tensor::upper_t<MetricIndex>>
            inv_metric_accessor;
    ddc::DiscreteDomain<DDimX, DDimY, sil::tensor::upper_t<MetricIndex>>
            inv_metric_dom(mesh_xy, inv_metric_accessor.domain());
    ddc::Chunk inv_metric_alloc(inv_metric_dom, ddc::DeviceAllocator<double>());
    sil::tensor::Tensor inv_metric(inv_metric_alloc);
    sil::tensor::fill_inverse_metric<
            MetricIndex>(Kokkos::DefaultExecutionSpace(), inv_metric, metric);

    // Potential
    [[maybe_unused]] sil::tensor::TensorAccessor<DummyIndex> potential_accessor;
    ddc::DiscreteDomain<DDimX, DDimY, DummyIndex>
            potential_dom(metric.non_indices_domain(), potential_accessor.domain());
    ddc::Chunk potential_alloc(potential_dom, ddc::DeviceAllocator<double>());
    sil::tensor::Tensor potential(potential_alloc);

    double const R = 2.;
    double const L = ddc::coordinate(ddc::DiscreteElement<DDimX>(potential.domain().back()))
                     - ddc::coordinate(ddc::DiscreteElement<DDimX>(potential.domain().front()));
    double const alpha = (static_cast<double>(nb_cells.template get<DDimX>())
                          * static_cast<double>(nb_cells.template get<DDimY>()))
                         / L / 2 / L / 2;
    ddc::parallel_for_each(
            potential.domain(),
            KOKKOS_LAMBDA(ddc::DiscreteElement<DDimX, DDimY, DummyIndex> elem) {
                double const r = Kokkos::sqrt(
                        static_cast<double>(
                                ddc::coordinate(ddc::DiscreteElement<DDimX>(elem))
                                * ddc::coordinate(ddc::DiscreteElement<DDimX>(elem)))
                        + static_cast<double>(
                                ddc::coordinate(ddc::DiscreteElement<DDimY>(elem))
                                * ddc::coordinate(ddc::DiscreteElement<DDimY>(elem))));
                if (r <= R) {
                    potential.mem(elem) = -alpha * r * r;
                } else {
                    potential.mem(elem) = alpha * R * R * (2 * Kokkos::log(R / r) - 1);
                }
            });
    auto potential_host
            = ddc::create_mirror_view_and_copy(Kokkos::DefaultHostExecutionSpace(), potential);

    // Moments
    [[maybe_unused]] sil::tensor::TensorAccessor<Mu> moments_accessor;
    ddc::DiscreteDomain<DDimX, DDimY, Mu> moments_dom(mesh_xy, moments_accessor.domain());
    ddc::Chunk moments_alloc(moments_dom, ddc::DeviceAllocator<double>());
    sil::tensor::Tensor moments(moments_alloc);

    sil::exterior::deriv<Mu, DummyIndex>(Kokkos::DefaultHostExecutionSpace(), moments, potential);
    Kokkos::fence();

    // Fill Hamiltonian grad
    [[maybe_unused]] sil::tensor::TensorAccessor<Mu> hamiltonian_grad_accessor;
    ddc::DiscreteDomain<DDimX, DDimY, Mu>
            hamiltonian_grad_dom(mesh_xy, hamiltonian_grad_accessor.domain());
    ddc::Chunk hamiltonian_grad_alloc(hamiltonian_grad_dom, ddc::DeviceAllocator<double>());
    sil::tensor::Tensor hamiltonian_grad(hamiltonian_grad_alloc);

    double const mass = 1.;

    ddc::parallel_for_each(
            Kokkos::DefaultExecutionSpace(),
            mesh_xy,
            KOKKOS_LAMBDA(ddc::DiscreteElement<DDimX, DDimY> elem) {
                const double phi = potential.mem(elem, ddc::DiscreteElement<DummyIndex>());
                const std::array<const double, 2>
                        pi {moments(elem, ddc::DiscreteElement<Mu>(0)),
                            moments(elem, ddc::DiscreteElement<Mu>(1))};

                const std::array<const double, 3> dH = ScalarFieldHamiltonian(mass).d(phi, pi);

                hamiltonian_grad(elem, ddc::DiscreteElement<Mu>(0)) = dH[1]; // dH/dpi_x
                hamiltonian_grad(elem, ddc::DiscreteElement<Mu>(1)) = dH[2]; // dH/dpi_y

                // printf("%f ", hamiltonian_grad(elem, ddc::DiscreteElement<Mu>(0)));
                // printf("%f ", phi); // dH/dphi
                // printf("%f ", std::get<0>(dH)); // dH/dphi
                printf("%f ", ScalarFieldHamiltonian(mass).value(phi, pi));
            });
    Kokkos::fence();

    return EXIT_SUCCESS;
}
