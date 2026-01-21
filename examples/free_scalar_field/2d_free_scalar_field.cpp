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
    size: [ '$Nx', '$Ny' ]
  temporal_moment:
    type: array
    subtype: double
    size: [ '$Nx', '$Ny' ]
  spatial_moments:
    type: array
    subtype: double
    size: [ '$Nx-1', '$Ny-1', 2 ]

plugins:
  decl_hdf5:
    - file: '2d_free_scalar_field.h5'
      on_event: [export]
      collision_policy: replace_and_warn
      write: [Nx, Ny, position, potential, temporal_moment, spatial_moments]
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
        2d_free_scalar_field.h5:/position
       </DataItem>
     </Geometry>
     <Attribute Name="Potential" AttributeType="Scalar" Center="Node">
       <DataItem Dimensions="%i %i" NumberType="Float" Precision="8" Format="HDF">
        2d_free_scalar_field.h5:/potential
       </DataItem>
     </Attribute>
     <Attribute Name="Temporal moment" AttributeType="Scalar" Center="Node">
       <DataItem Dimensions="%i %i" NumberType="Float" Precision="8" Format="HDF">
        2d_free_scalar_field.h5:/temporal_moment
       </DataItem>
     </Attribute>
     <Attribute Name="Spatial moments" AttributeType="Vector" Center="Cell">
       <DataItem Dimensions="%i %i 2" NumberType="Float" Precision="8" Format="HDF">
        2d_free_scalar_field.h5:/spatial_moments
       </DataItem>
     </Attribute>
   </Grid>
 </Domain>
</Xdmf>
)XDMF";

    FILE* file = fopen("2d_free_scalar_field.xmf", "w");
    fprintf(file, xdmf, Nx, Ny, Nx, Ny, Nx, Ny, Nx, Ny, Nx - 1, Ny - 1);
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

    // ------------------------------------------
    // ----- ALLOCATIONS AND INSTANTIATIONS -----
    // ------------------------------------------

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
            Kokkos::DefaultExecutionSpace(),
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

    ddc::parallel_for_each(
            potential.domain(),
            KOKKOS_LAMBDA(ddc::DiscreteElement<DDimX, DDimY, DummyIndex> elem) {
                potential(elem) = elem
                                                  == ddc::DiscreteElement<DDimX, DDimY, DummyIndex>(
                                                          potential.extent<DDimX>() / 2,
                                                          potential.extent<DDimY>() / 2,
                                                          0)
                                          ? 1.
                                          : 0.;
            });
    auto potential_host
            = ddc::create_mirror_view_and_copy(Kokkos::DefaultHostExecutionSpace(), potential);

    // Potential gradient
    [[maybe_unused]] sil::tensor::TensorAccessor<AlphaLow> potential_grad_accessor;
    ddc::DiscreteDomain<DDimX, DDimY, AlphaLow>
            potential_grad_dom(mesh_xy, potential_grad_accessor.domain());
    ddc::Chunk potential_grad_alloc(potential_grad_dom, ddc::DeviceAllocator<double>());
    sil::tensor::Tensor potential_grad(potential_grad_alloc);

    // Spatial moments
    auto& spatial_moments
            = potential_grad; // We can perform the computations inplace so spatial_moments is just an alias of potential_grad

    auto spatial_moments_host = ddc::
            create_mirror_view_and_copy(Kokkos::DefaultHostExecutionSpace(), spatial_moments);

    // Spatial moments divergence
    [[maybe_unused]] sil::tensor::TensorAccessor<DummyIndex> spatial_moments_div_accessor;
    ddc::DiscreteDomain<DDimX, DDimY, DummyIndex>
            spatial_moments_div_dom(mesh_xy, spatial_moments_div_accessor.domain());
    ddc::Chunk spatial_moments_div_alloc(spatial_moments_div_dom, ddc::DeviceAllocator<double>());
    sil::tensor::Tensor spatial_moments_div(spatial_moments_div_alloc);

    // Temporal moment
    [[maybe_unused]] sil::tensor::TensorAccessor<DummyIndex> temporal_moment_accessor;
    ddc::DiscreteDomain<DDimX, DDimY, DummyIndex>
            temporal_moment_dom(mesh_xy, temporal_moment_accessor.domain());
    ddc::Chunk temporal_moment_alloc(temporal_moment_dom, ddc::DeviceAllocator<double>());
    sil::tensor::Tensor temporal_moment(temporal_moment_alloc);
    ddc::parallel_fill(Kokkos::DefaultExecutionSpace(), temporal_moment, 0.);

    auto temporal_moment_host = ddc::
            create_mirror_view_and_copy(Kokkos::DefaultHostExecutionSpace(), temporal_moment);

    // ------------------
    // ----- SOLVER -----
    // ------------------

    double const mass = 1.;
    int const nb_iter_between_exports = 10;
    int const nb_iter = 10000;
    double const dt = 1e-3;

    for (int i = 0; i < nb_iter; i++) {
        std::cout << "Start iteration " << i << std::endl;

        // Compute the potential gradient
        sil::exterior::deriv<
                AlphaLow,
                DummyIndex>(Kokkos::DefaultExecutionSpace(), potential_grad, potential);
        Kokkos::fence();

        // Compute the spatial moments pi_\alpha by solving dphi/dx^\alpha = -dH/dpi_\alpha
        ddc::parallel_for_each(
                Kokkos::DefaultExecutionSpace(),
                mesh_xy,
                KOKKOS_LAMBDA(ddc::DiscreteElement<DDimX, DDimY> elem) {
                    spatial_moments(elem, ddc::DiscreteElement<AlphaLow>(0))
                            = FreeScalarFieldHamiltonian(mass).pi1(
                                    potential_grad(elem, ddc::DiscreteElement<AlphaLow>(0)));
                    spatial_moments(elem, ddc::DiscreteElement<AlphaLow>(1))
                            = FreeScalarFieldHamiltonian(mass).pi2(
                                    potential_grad(elem, ddc::DiscreteElement<AlphaLow>(1)));
                });
        Kokkos::fence();

        ddc::parallel_deepcopy(
                Kokkos::DefaultExecutionSpace(),
                spatial_moments_host,
                spatial_moments);

        // Compute the divergence dpi_\alpha/dx^\alpha of the spatial moments, which is the codifferential \delta pi of the spatial moments
        sil::exterior::codifferential<MetricIndex, AlphaLow, AlphaLow>(
                Kokkos::DefaultExecutionSpace(),
                spatial_moments_div,
                spatial_moments,
                inv_metric);
        Kokkos::fence();

        // Compute dpi_0/dx^0 by solving dpi_mu/dx^\mu = dH/d\phi and advect pi_0 by a time step dx^0. Then, compute dphi/dx^0 by solving dphi/dx^0 = -dH/dpi_0 and advect phi by a time step dx^0.
        // TODO use better temporal integration scheme like Runge-Kutta
        ddc::parallel_for_each(
                Kokkos::DefaultExecutionSpace(),
                spatial_moments_div.domain(),
                KOKKOS_LAMBDA(ddc::DiscreteElement<DDimX, DDimY, DummyIndex> elem) {
                    double temporal_moment_ = temporal_moment(elem);

                    temporal_moment_ += (FreeScalarFieldHamiltonian(mass).dH_dphi(potential(elem))
                                         - spatial_moments_div(elem))
                                        * dt;
                    potential(elem)
                            -= FreeScalarFieldHamiltonian(mass).dH_dpi0(temporal_moment_) * dt;

                    temporal_moment(elem) = temporal_moment_;
                });
        Kokkos::fence();

        ddc::parallel_deepcopy(
                Kokkos::DefaultExecutionSpace(),
                temporal_moment_host,
                temporal_moment);
        ddc::parallel_deepcopy(Kokkos::DefaultExecutionSpace(), potential_host, potential);

        // Export HDF5 and XDMF
        std::cout << "Potential center = "
                  << potential(
                             ddc::DiscreteElement<DDimX, DDimY, DummyIndex>(
                                     potential.extent<DDimX>() / 2,
                                     potential.extent<DDimY>() / 2,
                                     0))
                  << std::endl;
        if (i % nb_iter_between_exports == 0) {
            ddc::PdiEvent("export")
                    .with("position", position)
                    .and_with("potential", potential_host)
                    .and_with("temporal_moment", temporal_moment_host)
                    .and_with("spatial_moments", spatial_moments_host);
            std::cout << "Computation result exported in 2d_free_scalar_field.h5." << std::endl;

            write_xdmf(
                    static_cast<int>(mesh_xy.template extent<DDimX>()),
                    static_cast<int>(mesh_xy.template extent<DDimY>()));
            std::cout << "XDMF model exported in 2d_free_scalar_field.xmf." << std::endl;
        }
    }

    return EXIT_SUCCESS;
}
