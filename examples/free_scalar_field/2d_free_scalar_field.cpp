// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: MIT

#include <ddc/ddc.hpp>
#include <ddc/kernels/splines.hpp>
#include <ddc/pdi.hpp>

#include <similie/similie.hpp>

#include "free_scalar_field_hamiltonian.hpp"

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
    - file: 'free_scalar_field.h5'
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
        free_scalar_field.h5:/potential
       </DataItem>
     </Attribute>
   </Grid>
 </Domain>
</Xdmf>
)XDMF";

    FILE* file = fopen("free_scalar_field.xmf", "w");
    fprintf(file, xdmf, Nx, Ny, Nx, Ny, Nx, Ny);
    fclose(file);

    return 1;
}

static constexpr std::size_t s_degree = 3;

struct T
{
};

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
struct Alpha : sil::tensor::TensorNaturalIndex<X, Y>
{
};

struct Beta : sil::tensor::TensorNaturalIndex<X, Y>
{
};

// Declare indices
using AlphaUp = sil::tensor::Contravariant<Alpha>;
using AlphaLow = sil::tensor::Covariant<Alpha>;
using BetaUp = sil::tensor::Contravariant<Beta>;
using BetaLow = sil::tensor::Covariant<Beta>;

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
    [[maybe_unused]] sil::tensor::TensorAccessor<AlphaUp> position_accessor;
    ddc::DiscreteDomain<DDimX, DDimY, AlphaUp> position_dom(mesh_xy, position_accessor.domain());
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

    // Potential gradient
    [[maybe_unused]] sil::tensor::TensorAccessor<AlphaLow> potential_grad_accessor;
    ddc::DiscreteDomain<DDimX, DDimY, AlphaLow>
            potential_grad_dom(mesh_xy, potential_grad_accessor.domain());
    ddc::Chunk potential_grad_alloc(potential_grad_dom, ddc::DeviceAllocator<double>());
    sil::tensor::Tensor potential_grad(potential_grad_alloc);

    sil::exterior::deriv<
            AlphaLow,
            DummyIndex>(Kokkos::DefaultHostExecutionSpace(), potential_grad, potential);
    Kokkos::fence();

    // Compute the spatial moments pi_\alpha by solving dphi/dx^\alpha = -dH/dpi_\alpha
    [[maybe_unused]] sil::tensor::TensorAccessor<AlphaLow> spatial_moments_accessor;
    ddc::DiscreteDomain<DDimX, DDimY, AlphaLow>
            spatial_moments_dom(mesh_xy, spatial_moments_accessor.domain());
    ddc::Chunk spatial_moments_alloc(spatial_moments_dom, ddc::DeviceAllocator<double>());
    sil::tensor::Tensor spatial_moments(spatial_moments_alloc);

    double const mass = 1.;

    ddc::parallel_for_each(
            Kokkos::DefaultExecutionSpace(),
            mesh_xy,
            KOKKOS_LAMBDA(ddc::DiscreteElement<DDimX, DDimY> elem) {
                // const double phi = potential.mem(elem, ddc::DiscreteElement<DummyIndex>());
                /*
                const std::array<const double, 2> spatial_pi {
                        spatial_moments(elem, ddc::DiscreteElement<Alpha>(0)),
                        spatial_moments(elem, ddc::DiscreteElement<Alpha>(1))};
			*/
                // dH/dpi_x
                spatial_moments(elem, ddc::DiscreteElement<AlphaLow>(0))
                        = FreeScalarFieldHamiltonian(mass).pi1(
                                potential_grad(elem, ddc::DiscreteElement<AlphaLow>(0)));
                spatial_moments(elem, ddc::DiscreteElement<AlphaLow>(1))
                        = FreeScalarFieldHamiltonian(mass).pi2(
                                potential_grad(elem, ddc::DiscreteElement<AlphaLow>(1)));
            });
    Kokkos::fence();

    // Compute the divergence dpi_\alpha/dx^\alpha of the spatial moments, which is the codifferential \delta pi of the spatial moments
    [[maybe_unused]] sil::tensor::TensorAccessor<DummyIndex> spatial_moments_div_accessor;
    ddc::DiscreteDomain<DDimX, DDimY, DummyIndex>
            spatial_moments_div_dom(mesh_xy, spatial_moments_div_accessor.domain());
    ddc::Chunk spatial_moments_div_alloc(spatial_moments_div_dom, ddc::DeviceAllocator<double>());
    sil::tensor::Tensor spatial_moments_div(spatial_moments_div_alloc);

    sil::exterior::codifferential<MetricIndex, AlphaLow, AlphaLow>(
            Kokkos::DefaultHostExecutionSpace(),
            spatial_moments_div,
            spatial_moments,
            inv_metric);
    Kokkos::fence();

    // Compute dpi_0/dx^0 by solving dpi_mu/dx^\mu = dH/d\phi and advect pi_0 by a time step dx^0. Then, compute dphi/dx^0 by solving dphi/dx^0 = -dH/dpi_0 and advect phi by a time step dx^0.
    // TODO use better temporal integration scheme like Runge-Kutta
    [[maybe_unused]] sil::tensor::TensorAccessor<DummyIndex> temporal_moment_accessor;
    ddc::DiscreteDomain<DDimX, DDimY, DummyIndex>
            temporal_moment_dom(mesh_xy, temporal_moment_accessor.domain());
    ddc::Chunk temporal_moment_alloc(temporal_moment_dom, ddc::DeviceAllocator<double>());
    sil::tensor::Tensor temporal_moment(temporal_moment_alloc);
    ddc::parallel_for_each(
            Kokkos::DefaultExecutionSpace(),
            spatial_moments_div.domain(),
            KOKKOS_LAMBDA(ddc::DiscreteElement<DDimX, DDimY, DummyIndex> elem) {
                temporal_moment(elem) = 0.;
            });
    Kokkos::fence();

    const float dt = 1e-3;

    ddc::parallel_for_each(
            Kokkos::DefaultExecutionSpace(),
            spatial_moments_div.domain(),
            KOKKOS_LAMBDA(ddc::DiscreteElement<DDimX, DDimY, DummyIndex> elem) {
                temporal_moment(elem) += (FreeScalarFieldHamiltonian(mass).dH_dphi(potential(elem))
                                          - spatial_moments_div(elem))
                                         * dt;
                potential(elem)
                        += -FreeScalarFieldHamiltonian(mass).dH_dpi0(temporal_moment(elem)) * dt;
            });
    Kokkos::fence();

    return EXIT_SUCCESS;
}
