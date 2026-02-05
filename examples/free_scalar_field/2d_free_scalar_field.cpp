// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <array>
#include <cmath>

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
    size: [ '$Nx', '$Ny', 2 ]
  spatial_moments_div:
    type: array
    subtype: double
    size: [ '$Nx', '$Ny' ]
  hamiltonian:
    type: array
    subtype: double
    size: [ '$Nx', '$Ny' ]

plugins:
  decl_hdf5:
    - file: '2d_free_scalar_field.h5'
      on_event: [export]
      collision_policy: replace_and_warn
      write:
        [Nx, Ny, position, potential, temporal_moment, spatial_moments, spatial_moments_div, hamiltonian]
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
     <Attribute Name="Spatial moments divergency" AttributeType="Scalar" Center="Node">
       <DataItem Dimensions="%i %i" NumberType="Float" Precision="8" Format="HDF">
        2d_free_scalar_field.h5:/spatial_moments_div
       </DataItem>
     </Attribute>
     <Attribute Name="Hamiltonian" AttributeType="Scalar" Center="Node">
       <DataItem Dimensions="%i %i" NumberType="Float" Precision="8" Format="HDF">
        2d_free_scalar_field.h5:/hamiltonian
       </DataItem>
     </Attribute>
   </Grid>
 </Domain>
</Xdmf>
)XDMF";

    FILE* file = fopen("2d_free_scalar_field.xmf", "w");
    fprintf(file, xdmf, Nx, Ny, Nx, Ny, Nx, Ny, Nx, Ny, Nx, Ny, Nx, Ny, Nx, Ny);
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
    ddc::DiscreteVector<DDimX, DDimY> nb_cells(1000, 1000);
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


    float const x_0 = -2.;
    float const y_0 = 0.;
    float const x_1 = 2.;
    float const y_1 = -0.3;
    float sigma = .2;

    double const v = .01;
    double const k = 1.;
    double const mass = 1e-6;

    ddc::parallel_for_each(
            potential.domain(),
            KOKKOS_LAMBDA(ddc::DiscreteElement<DDimX, DDimY, DummyIndex> elem) {
                double const x = ddc::coordinate(ddc::DiscreteElement<DDimX>(elem));
                double const y = ddc::coordinate(ddc::DiscreteElement<DDimY>(elem));
                // Two Gaussian wave packets
                potential(elem) = std::sin(k * (x - x_0))
                                          * std::exp(
                                                  -((x - x_0) * (x - x_0) + (y - y_0) * (y - y_0))
                                                  / 2. / sigma / sigma)
                                  + std::sin(k * (x - x_1))
                                            * std::exp(
                                                    -((x - x_1) * (x - x_1) + (y - y_1) * (y - y_1))
                                                    / 2. / sigma / sigma);
            });
    auto potential_host
            = ddc::create_mirror_view_and_copy(Kokkos::DefaultHostExecutionSpace(), potential);

    // Half-step potential: mid-point integration requires additional allocation
    ddc::Chunk half_step_potential_alloc(potential_dom, ddc::DeviceAllocator<double>());
    sil::tensor::Tensor half_step_potential(half_step_potential_alloc);

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
    auto h_spatial_moments_div = ddc::
            create_mirror_view_and_copy(Kokkos::DefaultHostExecutionSpace(), spatial_moments_div);

    // Temporal moment
    [[maybe_unused]] sil::tensor::TensorAccessor<DummyIndex> temporal_moment_accessor;
    ddc::DiscreteDomain<DDimX, DDimY, DummyIndex>
            temporal_moment_dom(mesh_xy, temporal_moment_accessor.domain());
    ddc::Chunk temporal_moment_alloc(temporal_moment_dom, ddc::DeviceAllocator<double>());
    sil::tensor::Tensor temporal_moment(temporal_moment_alloc);

    ddc::parallel_for_each(
            potential.domain(),
            KOKKOS_LAMBDA(ddc::DiscreteElement<DDimX, DDimY, DummyIndex> elem) {
                double const x = ddc::coordinate(ddc::DiscreteElement<DDimX>(elem));
                double const y = ddc::coordinate(ddc::DiscreteElement<DDimY>(elem)) - y_0;
                // v*dphi/dx of the left wave packet only to get a pure kick along x toward the immobile right one
                temporal_moment(elem) = -v
                                        * (-k * std::cos(k * (x - x_0))
                                           + (x - x_0) / sigma / sigma * std::sin(k * (x - x_0)))
                                        * std::exp(
                                                -((x - x_0) * (x - x_0) + (y - y_0) * (y - y_0))
                                                / 2. / sigma / sigma);
                // temporal_moment(elem) = 0;
            });

    auto temporal_moment_host = ddc::
            create_mirror_view_and_copy(Kokkos::DefaultHostExecutionSpace(), temporal_moment);

    // Hamiltonian, only for export
    ddc::Chunk hamiltonian_alloc(mesh_xy, ddc::DeviceAllocator<double>());
    sil::tensor::Tensor hamiltonian(hamiltonian_alloc);
    auto h_hamiltonian
            = ddc::create_mirror_view_and_copy(Kokkos::DefaultHostExecutionSpace(), hamiltonian);

    // ------------------
    // ----- SOLVER -----
    // ------------------

    int const nb_iter_between_exports = 50;
    int const nb_iter = 10000;
    double const dx = (ddc::get<X>(upper_bounds) - ddc::get<X>(lower_bounds))
                      / ddc::get<DDimX>(nb_cells);
    double const dy = (ddc::get<Y>(upper_bounds) - ddc::get<Y>(lower_bounds))
                      / ddc::get<DDimY>(nb_cells);
    double const cfl = 0.5;
    double const dt = cfl * std::min(dx, dy) / std::sqrt(2.0);

    /*
     * DeDonder-Weyl equations are commonly written:
     * dpi^\mu/dx^\mu = -dH/dphi
     * dphi/dx^\mu = dH/dpi^\mu
     *
     * But we follow the convention with pi being stored as covariant. Thus:
     * eta^\mu\nu dpi_\nu/dx^\mu = -dH/dphi
     * dphi/dx^\mu = eta^\mu\nu dH/dpi_\nu
     *
     * Or explicitely with a (-, +, +) metric:
     * - dpi_0/dx^0 + dpi_\alpha/dx^\alpha = -dH/dphi
     * dphi/dx^0 = - dH/dpi_0
     * dphi/dx^\alpha = dH/dpi_\alpha
     *
     * We implement mid-point explicit temporal integration scheme.
     */
    for (int i = 0; i < nb_iter; i++) {
        if (i % nb_iter_between_exports == 0) {
            std::cout << "Start iteration " << i << std::endl;
        }

        // First half-advect phi by solving dphi/dx^0 = -dH/dpi_0
        ddc::parallel_for_each(
                Kokkos::DefaultExecutionSpace(),
                half_step_potential.domain(),
                KOKKOS_LAMBDA(ddc::DiscreteElement<DDimX, DDimY, DummyIndex> elem) {
                    half_step_potential(elem)
                            = potential(elem)
                              - FreeScalarFieldHamiltonian(mass).dH_dpi0(temporal_moment(elem)) * dt
                                        / 2.;
                });

        // Compute the potential gradient
        sil::exterior::deriv<
                AlphaLow,
                DummyIndex>(Kokkos::DefaultExecutionSpace(), potential_grad, half_step_potential);

        // Compute the spatial moments pi_\alpha by solving dphi/dx^\alpha = dH/dpi_\alpha
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
        if (i % nb_iter_between_exports == 0) {
            ddc::parallel_deepcopy(spatial_moments_host, spatial_moments);
        }

        // Compute the divergence dpi_\alpha/dx^\alpha of the spatial moments, which is the codifferential \delta pi of the spatial moments
        sil::exterior::codifferential<MetricIndex, AlphaLow, AlphaLow>(
                Kokkos::DefaultExecutionSpace(),
                spatial_moments_div,
                spatial_moments,
                inv_metric);

        // Compute dpi_0/dx^0 by solving - dpi_0/dx^0 + dpi_\alpha/dx^\alpha = -dH/dphi and advect pi_0 by a time step dx^0. Also Then, perform the second phi half-advection by solving dphi/dx^0 = -dH/dpi_0
        double const dS = (ddc::get<X>(upper_bounds) - ddc::get<X>(lower_bounds))
                          * (ddc::get<Y>(upper_bounds) - ddc::get<Y>(lower_bounds))
                          / ddc::get<DDimX>(nb_cells) / ddc::get<DDimY>(nb_cells); // FIXME unused
        ddc::parallel_for_each(
                Kokkos::DefaultExecutionSpace(),
                spatial_moments_div.domain(),
                KOKKOS_LAMBDA(ddc::DiscreteElement<DDimX, DDimY, DummyIndex> elem) {
                    const double half_step_potential_ = half_step_potential(elem);
                    const double temporal_moment_ = temporal_moment(elem);
                    const double spatial_moments_div_ = spatial_moments_div(elem);

                    // Advect temporal moment by half-step, this is what is needed to perform the whole-step potential advection
                    const double half_step_temporal_moment_
                            = temporal_moment_
                              + (FreeScalarFieldHamiltonian(mass).dH_dphi(half_step_potential_)
                                 + spatial_moments_div_)
                                        * dt / 2;

                    // Whole-step advection of field state
                    potential(elem)
                            += FreeScalarFieldHamiltonian(mass).dH_dpi0(half_step_temporal_moment_)
                               * dt;
                    temporal_moment(elem)
                            += (FreeScalarFieldHamiltonian(mass).dH_dphi(half_step_potential_)
                                + spatial_moments_div_)
                               * dt;
                });

        if (i % nb_iter_between_exports == 0) {
            ddc::parallel_deepcopy(temporal_moment_host, temporal_moment);
            ddc::parallel_deepcopy(potential_host, potential);
            ddc::parallel_deepcopy(h_spatial_moments_div, spatial_moments_div);

            ddc::parallel_for_each(
                    Kokkos::DefaultExecutionSpace(),
                    mesh_xy,
                    KOKKOS_LAMBDA(ddc::DiscreteElement<DDimX, DDimY> elem) {
                        std::array<double, 3> pi
                                = {temporal_moment(elem, ddc::DiscreteElement<DummyIndex>()),
                                   spatial_moments(elem, ddc::DiscreteElement<AlphaLow>(0)),
                                   spatial_moments(elem, ddc::DiscreteElement<AlphaLow>(1))};
                        hamiltonian(elem)
                                = FreeScalarFieldHamiltonian(mass)
                                          .H(potential(elem, ddc::DiscreteElement<DummyIndex>()),
                                             pi);
                    });
            ddc::parallel_deepcopy(h_hamiltonian, hamiltonian);

            // Export HDF5 and XDMF
            std::cout << "Potential center = "
                      << potential_host(
                                 ddc::DiscreteElement<DDimX, DDimY, DummyIndex>(
                                         potential.extent<DDimX>() / 2,
                                         potential.extent<DDimY>() / 2,
                                         0))
                      << std::endl;
            ddc::PdiEvent("export")
                    .with("position", position)
                    .with("potential", potential_host)
                    .with("temporal_moment", temporal_moment_host)
                    .with("spatial_moments", spatial_moments_host)
                    .with("spatial_moments_div", h_spatial_moments_div)
                    .with("hamiltonian", h_hamiltonian);
            std::cout << "Computation result exported in 2d_free_scalar_field.h5." << std::endl;

            write_xdmf(
                    static_cast<int>(mesh_xy.template extent<DDimX>()),
                    static_cast<int>(mesh_xy.template extent<DDimY>()));
            std::cout << "XDMF model exported in 2d_free_scalar_field.xmf." << std::endl;
        }
    }

    return EXIT_SUCCESS;
}
