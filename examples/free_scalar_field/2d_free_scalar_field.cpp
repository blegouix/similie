// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>

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
  Nt : int

data:
  write_position:
    type: int
  export_id:
    type: int
  time:
    type: double
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
      collision_policy: write_into
      datasets:
        position:
          type: array
          subtype: double
          size: [ '$Nx', '$Ny' , 2]
        potential:
          type: array
          subtype: double
          size: [ '$Nt', '$Nx', '$Ny' ]
        temporal_moment:
          type: array
          subtype: double
          size: [ '$Nt', '$Nx', '$Ny' ]
        spatial_moments:
          type: array
          subtype: double
          size: [ '$Nt', '$Nx', '$Ny', 2 ]
        spatial_moments_div:
          type: array
          subtype: double
          size: [ '$Nt', '$Nx', '$Ny' ]
        hamiltonian:
          type: array
          subtype: double
          size: [ '$Nt', '$Nx', '$Ny' ]
        time:
          type: array
          subtype: double
          size: [ '$Nt' ]
      write:
        position:
          when: "$write_position"
        potential:
          dataset_selection:
            start: [ '$export_id', 0, 0 ]
            size: [ 1, '$Nx', '$Ny' ]
        temporal_moment:
          dataset_selection:
            start: [ '$export_id', 0, 0 ]
            size: [ 1, '$Nx', '$Ny' ]
        spatial_moments:
          dataset_selection:
            start: [ '$export_id', 0, 0, 0 ]
            size: [ 1, '$Nx', '$Ny', 2 ]
        spatial_moments_div:
          dataset_selection:
            start: [ '$export_id', 0, 0 ]
            size: [ 1, '$Nx', '$Ny' ]
        hamiltonian:
          dataset_selection:
            start: [ '$export_id', 0, 0 ]
            size: [ 1, '$Nx', '$Ny' ]
        time:
          dataset_selection:
            start: [ '$export_id' ]
            size: [ 1 ]
  #trace: ~
)PDI_CFG";

// XDMF
int write_xdmf(int Nx, int Ny, int total_steps, int exported_ids, double export_dt)
{
    std::ofstream file("2d_free_scalar_field.xmf", std::ios::trunc);
    file << R"XMF(<?xml version="1.0" ?>
<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
<Xdmf Version="2.0">
 <Domain>
  <Grid Name="TimeSeries" GridType="Collection" CollectionType="Temporal">
)XMF";

    int const step_width = std::max(1, static_cast<int>(std::to_string(total_steps - 1).size()));
    for (int step = 0; step < exported_ids; step++) {
        double const time = step * export_dt;
        std::ostringstream step_name;
        step_name << "Step" << std::setw(step_width) << std::setfill('0') << step;
        file << "   <Grid Name=\"" << step_name.str() << "\" GridType=\"Uniform\">\n";
        file << "    <Time Value=\"" << time << "\"/>\n";
        file << "    <Topology TopologyType=\"2DSMesh\" NumberOfElements=\"" << Nx << " " << Ny
             << "\"/>\n";
        file << R"XMF(    <Geometry GeometryType="XY">
      <DataItem Dimensions=")XMF"
             << Nx << " " << Ny << R"XMF( 2" NumberType="Float" Precision="8" Format="HDF">
       2d_free_scalar_field.h5:/position
      </DataItem>
    </Geometry>
)XMF";

        auto write_scalar_attribute = [&](char const* name, char const* dataset) {
            file << R"XMF(    <Attribute Name=")XMF" << name
                 << R"XMF(" AttributeType="Scalar" Center="Node">
      <DataItem ItemType="HyperSlab" Dimensions=")XMF"
                 << Nx << " " << Ny << R"XMF(">
        <DataItem Dimensions="3 3" Format="XML">
         )XMF" << step
                 << R"XMF( 0 0
         1 1 1
         1 )XMF" << Nx
                 << " " << Ny << R"XMF(
        </DataItem>
        <DataItem Dimensions=")XMF"
                 << total_steps << " " << Nx << " " << Ny
                 << R"XMF(" NumberType="Float" Precision="8" Format="HDF">
         2d_free_scalar_field.h5:/)XMF"
                 << dataset << R"XMF(
        </DataItem>
      </DataItem>
    </Attribute>
)XMF";
        };

        write_scalar_attribute("Potential", "potential");
        write_scalar_attribute("Temporal moment", "temporal_moment");

        file << R"XMF(    <Attribute Name="Spatial moments" AttributeType="Vector" Center="Cell">
      <DataItem ItemType="HyperSlab" Dimensions=")XMF"
             << Nx << " " << Ny << R"XMF( 2">
        <DataItem Dimensions="3 4" Format="XML">
         )XMF"
             << step << R"XMF( 0 0 0
         1 1 1 1
         1 )XMF"
             << Nx << " " << Ny << R"XMF( 2
        </DataItem>
        <DataItem Dimensions=")XMF"
             << total_steps << " " << Nx << " " << Ny
             << R"XMF( 2" NumberType="Float" Precision="8" Format="HDF">
         2d_free_scalar_field.h5:/spatial_moments
        </DataItem>
      </DataItem>
    </Attribute>
)XMF";

        write_scalar_attribute("Spatial moments divergency", "spatial_moments_div");
        write_scalar_attribute("Hamiltonian", "hamiltonian");

        file << "   </Grid>\n";
    }
    file << R"XMF(  </Grid>
 </Domain>
</Xdmf>
)XMF";

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
using MetricIndex = sil::tensor::TensorIdentityIndex<
        sil::tensor::Covariant<sil::tensor::MetricIndex1<X, Y>>,
        sil::tensor::Covariant<sil::tensor::MetricIndex2<X, Y>>>;

using MesherXY = sil::mesher::Mesher<s_degree, X, Y>;

struct DDimX : ddc::UniformPointSampling<X>
{
};

struct DDimY : ddc::UniformPointSampling<Y>
{
};

struct DDimXHalf : ddc::UniformPointSampling<X>
{
};

struct DDimYHalf : ddc::UniformPointSampling<Y>
{
};

/*
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
*/

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
    ddc::Coordinate<X, Y> lower_bounds(-5., -5.);
    ddc::Coordinate<X, Y> upper_bounds(5., 5.);
    ddc::DiscreteVector<DDimX, DDimY> nb_cells(1000, 1000);
    /*
    MesherXY mesher;
    ddc::DiscreteDomain<DDimX, DDimY> mesh_xy = mesher.template mesh<
            ddc::detail::TypeSeq<DDimX, DDimY>,
            ddc::detail::TypeSeq<BSplinesX, BSplinesY>>(lower_bounds, upper_bounds, nb_cells);
     */
    auto const x_dom = ddc::init_discrete_space<DDimX>(DDimX::init<DDimX>(
            ddc::select<X>(lower_bounds),
            ddc::select<X>(upper_bounds),
            ddc::select<DDimX>(nb_cells)));
    auto const y_dom = ddc::init_discrete_space<DDimY>(DDimY::init<DDimY>(
            ddc::select<Y>(lower_bounds),
            ddc::select<Y>(upper_bounds),
            ddc::select<DDimY>(nb_cells)));
    auto const x_face_x_dom = ddc::init_discrete_space<DDimXHalf>(DDimXHalf::init<DDimXHalf>(
            ddc::Coordinate<X>(
                    ddc::select<X>(lower_bounds)
                    + (ddc::select<X>(upper_bounds) - ddc::select<X>(lower_bounds))
                              / (2 * (ddc::select<DDimX>(nb_cells).value() - 1))),
            ddc::Coordinate<X>(
                    ddc::select<X>(upper_bounds)
                    - (ddc::select<X>(upper_bounds) - ddc::select<X>(lower_bounds))
                              / (2 * (ddc::select<DDimX>(nb_cells).value() - 1))),
            ddc::DiscreteVector<DDimXHalf>(ddc::select<DDimX>(nb_cells).value() - 1)));
    auto const y_face_y_dom = ddc::init_discrete_space<DDimYHalf>(DDimYHalf::init<DDimYHalf>(
            ddc::Coordinate<Y>(
                    ddc::select<Y>(lower_bounds)
                    + (ddc::select<Y>(upper_bounds) - ddc::select<Y>(lower_bounds))
                              / (2 * (ddc::select<DDimY>(nb_cells).value() - 1))),
            ddc::Coordinate<Y>(
                    ddc::select<Y>(upper_bounds)
                    - (ddc::select<Y>(upper_bounds) - ddc::select<Y>(lower_bounds))
                              / (2 * (ddc::select<DDimY>(nb_cells).value() - 1))),
            ddc::DiscreteVector<DDimYHalf>(ddc::select<DDimY>(nb_cells).value() - 1)));
    ddc::DiscreteDomain<DDimX, DDimY> mesh_xy(x_dom, y_dom);
    ddc::DiscreteDomain<DDimXHalf, DDimY> x_face_dom(x_face_x_dom, y_dom);
    ddc::DiscreteDomain<DDimX, DDimYHalf> y_face_dom(x_dom, y_face_y_dom);

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
    /*
    ddc::parallel_for_each(
            Kokkos::DefaultExecutionSpace(),
            mesh_xy,
            KOKKOS_LAMBDA(ddc::DiscreteElement<DDimX, DDimY> elem) {
                metric(elem, metric.accessor().access_element<X, X>()) = 1.;
                metric(elem, metric.accessor().access_element<X, Y>()) = 0.;
                metric(elem, metric.accessor().access_element<Y, Y>()) = 1.;
            });
    */

    // Invert metric
    [[maybe_unused]] sil::tensor::TensorAccessor<sil::tensor::upper_t<MetricIndex>>
            inv_metric_accessor;
    ddc::DiscreteDomain<DDimX, DDimY, sil::tensor::upper_t<MetricIndex>>
            inv_metric_dom(mesh_xy, inv_metric_accessor.domain());
    ddc::Chunk inv_metric_alloc(inv_metric_dom, ddc::DeviceAllocator<double>());
    sil::tensor::Tensor inv_metric(inv_metric_alloc);
    /* // Metric inversion generates small noise which makes the simulation unsable so we prefer to rely on an identity tensor index
    sil::tensor::fill_inverse_metric<
            MetricIndex>(Kokkos::DefaultExecutionSpace(), inv_metric, metric);
    */

    // Potential
    [[maybe_unused]] sil::tensor::TensorAccessor<DummyIndex> potential_accessor;
    ddc::DiscreteDomain<DDimX, DDimY, DummyIndex>
            potential_dom(metric.non_indices_domain(), potential_accessor.domain());
    ddc::Chunk potential_alloc(potential_dom, ddc::DeviceAllocator<double>());
    sil::tensor::Tensor potential(potential_alloc);


    double const x_0 = -2.;
    double const y_0 = 0.;
    double const x_1 = 2.;
    double const y_1 = -0.3;
    double const sigma = .5;

    double const v = 10.;
    double const k = 1.;
    double const mass = 1.;

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
                                  + 30. * std::sin(k * (x - x_1))
                                            * std::exp(
                                                    -((x - x_1) * (x - x_1) + (y - y_1) * (y - y_1))
                                                    / 2. / sigma / sigma);
            });
    auto potential_host
            = ddc::create_mirror_view_and_copy(Kokkos::DefaultHostExecutionSpace(), potential);

    // Half-step potential: mid-point integration requires additional allocation
    ddc::Chunk half_step_potential_alloc(potential_dom, ddc::DeviceAllocator<double>());
    sil::tensor::Tensor half_step_potential(half_step_potential_alloc);

    // Staggered potential gradients
    [[maybe_unused]] sil::tensor::TensorAccessor<DummyIndex> scalar_accessor;
    ddc::DiscreteDomain<DDimXHalf, DDimY, DummyIndex>
            potential_grad_x_dom(x_face_dom, scalar_accessor.domain());
    ddc::Chunk potential_grad_x_alloc(potential_grad_x_dom, ddc::DeviceAllocator<double>());
    sil::tensor::Tensor potential_grad_x(potential_grad_x_alloc);

    ddc::DiscreteDomain<DDimX, DDimYHalf, DummyIndex>
            potential_grad_y_dom(y_face_dom, scalar_accessor.domain());
    ddc::Chunk potential_grad_y_alloc(potential_grad_y_dom, ddc::DeviceAllocator<double>());
    sil::tensor::Tensor potential_grad_y(potential_grad_y_alloc);

    // Staggered spatial moments
    ddc::DiscreteDomain<DDimXHalf, DDimY, DummyIndex>
            spatial_moment_x_dom(x_face_dom, scalar_accessor.domain());
    ddc::Chunk spatial_moment_x_alloc(spatial_moment_x_dom, ddc::DeviceAllocator<double>());
    sil::tensor::Tensor spatial_moment_x(spatial_moment_x_alloc);

    ddc::DiscreteDomain<DDimX, DDimYHalf, DummyIndex>
            spatial_moment_y_dom(y_face_dom, scalar_accessor.domain());
    ddc::Chunk spatial_moment_y_alloc(spatial_moment_y_dom, ddc::DeviceAllocator<double>());
    sil::tensor::Tensor spatial_moment_y(spatial_moment_y_alloc);

    auto spatial_moment_x_host = ddc::
            create_mirror_view_and_copy(Kokkos::DefaultHostExecutionSpace(), spatial_moment_x);
    auto spatial_moment_y_host = ddc::
            create_mirror_view_and_copy(Kokkos::DefaultHostExecutionSpace(), spatial_moment_y);

    // Reconstructed spatial moments on nodes, only for export and diagnostics
    [[maybe_unused]] sil::tensor::TensorAccessor<AlphaLow> spatial_moments_accessor;
    ddc::DiscreteDomain<DDimX, DDimY, AlphaLow>
            spatial_moments_dom(mesh_xy, spatial_moments_accessor.domain());
    ddc::Chunk spatial_moments_alloc(spatial_moments_dom, ddc::DeviceAllocator<double>());
    sil::tensor::Tensor spatial_moments(spatial_moments_alloc);
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
    double const dx
            = (ddc::get<X>(upper_bounds) - ddc::get<X>(lower_bounds)) / ddc::get<DDimX>(nb_cells);
    double const dy
            = (ddc::get<Y>(upper_bounds) - ddc::get<Y>(lower_bounds)) / ddc::get<DDimY>(nb_cells);
    double const dt = .5 * std::min(dx, dy) / std::sqrt(2.0);
    std::cout << "Time step = " << dt << std::endl;
    ddc::expose_to_pdi("Nt", (nb_iter - 1) / nb_iter_between_exports + 1);
    std::remove("2d_free_scalar_field.h5");

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

        // Compute the staggered potential gradients
        sil::exterior::staggered_deriv_component<X>(
                Kokkos::DefaultExecutionSpace(),
                potential_grad_x,
                half_step_potential);
        sil::exterior::staggered_deriv_component<Y>(
                Kokkos::DefaultExecutionSpace(),
                potential_grad_y,
                half_step_potential);

        // Compute the spatial moments pi_\alpha by solving dphi/dx^\alpha = dH/dpi_\alpha
        ddc::parallel_for_each(
                Kokkos::DefaultExecutionSpace(),
                x_face_dom,
                KOKKOS_LAMBDA(ddc::DiscreteElement<DDimXHalf, DDimY> elem) {
                    spatial_moment_x(elem, ddc::DiscreteElement<DummyIndex>(0))
                            = FreeScalarFieldHamiltonian(mass).pi1(
                                    potential_grad_x(elem, ddc::DiscreteElement<DummyIndex>(0)));
                });
        ddc::parallel_for_each(
                Kokkos::DefaultExecutionSpace(),
                y_face_dom,
                KOKKOS_LAMBDA(ddc::DiscreteElement<DDimX, DDimYHalf> elem) {
                    spatial_moment_y(elem, ddc::DiscreteElement<DummyIndex>(0))
                            = FreeScalarFieldHamiltonian(mass).pi2(
                                    potential_grad_y(elem, ddc::DiscreteElement<DummyIndex>(0)));
                });

        // Compute the divergence dpi_\alpha/dx^\alpha of the spatial moments, which is the codifferential \delta pi of the spatial moments
        ddc::parallel_for_each(
                Kokkos::DefaultExecutionSpace(),
                spatial_moments_div.domain(),
                KOKKOS_LAMBDA(ddc::DiscreteElement<DDimX, DDimY, DummyIndex> elem) {
                    spatial_moments_div(elem) = 0.;
                });
        sil::exterior::add_staggered_codifferential_component<X, X>(
                Kokkos::DefaultExecutionSpace(),
                spatial_moments_div,
                spatial_moment_x,
                inv_metric);
        sil::exterior::add_staggered_codifferential_component<Y, Y>(
                Kokkos::DefaultExecutionSpace(),
                spatial_moments_div,
                spatial_moment_y,
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
                              + (spatial_moments_div_
                                 - FreeScalarFieldHamiltonian(mass).dH_dphi(half_step_potential_))
                                        * dt / 2;

                    // Whole-step advection of field state
                    potential(elem)
                            -= FreeScalarFieldHamiltonian(mass).dH_dpi0(half_step_temporal_moment_)
                               * dt;
                    temporal_moment(elem)
                            += (spatial_moments_div_
                                - FreeScalarFieldHamiltonian(mass).dH_dphi(half_step_potential_))
                               * dt;
                });

        if (i % nb_iter_between_exports == 0) {
            ddc::parallel_deepcopy(temporal_moment_host, temporal_moment);
            ddc::parallel_deepcopy(potential_host, potential);
            ddc::parallel_deepcopy(h_spatial_moments_div, spatial_moments_div);
            ddc::parallel_deepcopy(spatial_moment_x_host, spatial_moment_x);
            ddc::parallel_deepcopy(spatial_moment_y_host, spatial_moment_y);

            {
                ddc::DiscreteElement<DDimX, DDimY> const mesh_front = mesh_xy.front();
                for (std::size_t ix = 0; ix < mesh_xy.template extent<DDimX>(); ++ix) {
                    for (std::size_t iy = 0; iy < mesh_xy.template extent<DDimY>(); ++iy) {
                        ddc::DiscreteElement<DDimX, DDimY> const elem
                                = mesh_front + ddc::DiscreteVector<DDimX, DDimY>(ix, iy);
                        spatial_moments_host(elem, ddc::DiscreteElement<AlphaLow>(0)) = 0.;
                        spatial_moments_host(elem, ddc::DiscreteElement<AlphaLow>(1)) = 0.;
                    }
                }
            }
            {
                ddc::DiscreteElement<DDimXHalf, DDimY> const x_face_front = x_face_dom.front();
                for (std::size_t ix = 0; ix < x_face_dom.template extent<DDimXHalf>(); ++ix) {
                    for (std::size_t iy = 0; iy < x_face_dom.template extent<DDimY>(); ++iy) {
                        ddc::DiscreteElement<DDimXHalf, DDimY> const elem
                                = x_face_front + ddc::DiscreteVector<DDimXHalf, DDimY>(ix, iy);
                        ddc::DiscreteElement<DDimX, DDimY> const left_node(
                                ddc::DiscreteElement<DDimX>(ddc::uid<DDimXHalf>(elem)),
                                ddc::DiscreteElement<DDimY>(elem));
                        double const value
                                = spatial_moment_x_host(elem, ddc::DiscreteElement<DummyIndex>(0));
                        spatial_moments_host(left_node, ddc::DiscreteElement<AlphaLow>(0))
                                += value / 2.;
                        spatial_moments_host(
                                left_node + ddc::DiscreteVector<DDimX, DDimY>(1, 0),
                                ddc::DiscreteElement<AlphaLow>(0))
                                += value / 2.;
                    }
                }
            }
            {
                ddc::DiscreteElement<DDimX, DDimYHalf> const y_face_front = y_face_dom.front();
                for (std::size_t ix = 0; ix < y_face_dom.template extent<DDimX>(); ++ix) {
                    for (std::size_t iy = 0; iy < y_face_dom.template extent<DDimYHalf>(); ++iy) {
                        ddc::DiscreteElement<DDimX, DDimYHalf> const elem
                                = y_face_front + ddc::DiscreteVector<DDimX, DDimYHalf>(ix, iy);
                        ddc::DiscreteElement<DDimX, DDimY> const lower_node(
                                ddc::DiscreteElement<DDimX>(elem),
                                ddc::DiscreteElement<DDimY>(ddc::uid<DDimYHalf>(elem)));
                        double const value
                                = spatial_moment_y_host(elem, ddc::DiscreteElement<DummyIndex>(0));
                        spatial_moments_host(lower_node, ddc::DiscreteElement<AlphaLow>(1))
                                += value / 2.;
                        spatial_moments_host(
                                lower_node + ddc::DiscreteVector<DDimX, DDimY>(0, 1),
                                ddc::DiscreteElement<AlphaLow>(1))
                                += value / 2.;
                    }
                }
            }
            ddc::parallel_deepcopy(spatial_moments, spatial_moments_host);

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
            double const time = i * dt;
            ddc::PdiEvent("export")
                    .with("export_id", i / nb_iter_between_exports)
                    .with("write_position", i / nb_iter_between_exports == 0 ? 1 : 0)
                    .with("time", time)
                    .with("position", position)
                    .with("potential", potential_host)
                    .with("temporal_moment", temporal_moment_host)
                    .with("spatial_moments", spatial_moments_host)
                    .with("spatial_moments_div", h_spatial_moments_div)
                    .with("hamiltonian", h_hamiltonian);
            std::cout << "Computation result exported in 2d_free_scalar_field.h5." << std::endl;

            write_xdmf(
                    static_cast<int>(mesh_xy.template extent<DDimX>()),
                    static_cast<int>(mesh_xy.template extent<DDimY>()),
                    (nb_iter - 1) / nb_iter_between_exports + 1,
                    i / nb_iter_between_exports + 1,
                    dt * nb_iter_between_exports);
            std::cout << "XDMF model exported in 2d_free_scalar_field.xmf." << std::endl;
        }
    }

    return EXIT_SUCCESS;
}
