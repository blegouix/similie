// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include <ddc/ddc.hpp>
#include <ddc/kernels/splines.hpp>
#include <ddc/pdi.hpp>

#include <similie/physics/scalar_field/scalar_field_with_power_coupling.hpp>
#include <similie/similie.hpp>

#if defined(SIMILIE_ASSERT_EXAMPLE_RESULTS_CORRECTNESS)
void print_expected_central_potential_values(std::vector<double> const& central_potential_values)
{
    std::cerr << "std::array<double, " << central_potential_values.size()
              << "> const central_potential_values\n        = {";
    for (std::size_t i = 0; i < central_potential_values.size(); ++i) {
        if (i != 0) {
            std::cerr << ",";
        }
        if (i % 3 == 0) {
            std::cerr << "\n           ";
        } else {
            std::cerr << " ";
        }
        std::cerr << std::setprecision(17) << central_potential_values[i];
    }
    std::cerr << "};\n";
}
#endif

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
  minus_spatial_moments_div:
    type: array
    subtype: double
    size: [ '$Nx', '$Ny' ]
  hamiltonian:
    type: array
    subtype: double
    size: [ '$Nx', '$Ny' ]

plugins:
  decl_hdf5:
    - file: '2d_scalar_field.h5'
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
        minus_spatial_moments_div:
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
        minus_spatial_moments_div:
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
    std::ofstream file("2d_scalar_field.xmf", std::ios::trunc);
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
       2d_scalar_field.h5:/position
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
         2d_scalar_field.h5:/)XMF"
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
         2d_scalar_field.h5:/spatial_moments
        </DataItem>
      </DataItem>
    </Attribute>
)XMF";

        write_scalar_attribute("Minus spatial moments divergence", "minus_spatial_moments_div");
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
    ddc::DiscreteVector<DDimX, DDimY> nb_cells(2000, 2000);
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
    ddc::DiscreteDomain<DDimX, DDimY> mesh_xy(x_dom, y_dom);

    assert(static_cast<std::size_t>(mesh_xy.template extent<DDimX>())
           == static_cast<std::size_t>(mesh_xy.template extent<DDimY>()));
    ddc::expose_to_pdi("Nx", static_cast<int>(mesh_xy.template extent<DDimX>()));
    ddc::expose_to_pdi("Ny", static_cast<int>(mesh_xy.template extent<DDimY>()));

    // Allocate and instantiate the position field.
    [[maybe_unused]] sil::tensor::TensorAccessor<AlphaUp> position_accessor;
    ddc::DiscreteDomain<DDimX, DDimY, AlphaUp> position_dom(mesh_xy, position_accessor.domain());
    ddc::Chunk position_alloc(position_dom, ddc::DeviceAllocator<double>());
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
    auto position_host
            = ddc::create_mirror_view_and_copy(Kokkos::DefaultHostExecutionSpace(), position);

    // Allocate and instantiate a metric tensor field.
    [[maybe_unused]] sil::tensor::TensorAccessor<MetricIndex> metric_accessor;
    ddc::DiscreteDomain<DDimX, DDimY, MetricIndex> metric_dom(mesh_xy, metric_accessor.domain());
    ddc::Chunk metric_alloc(metric_dom, ddc::DeviceAllocator<double>());
    sil::tensor::Tensor metric(metric_alloc);

    // Potential
    [[maybe_unused]] sil::tensor::TensorAccessor<DummyIndex> potential_accessor;
    ddc::DiscreteDomain<DDimX, DDimY, DummyIndex>
            potential_dom(metric.non_indices_domain(), potential_accessor.domain());
    ddc::Chunk potential_alloc(potential_dom, ddc::DeviceAllocator<double>());
    sil::tensor::Tensor potential(potential_alloc);


    double const x_0 = -3.5;
    double const y_0 = 0.;
    double const x_1 = 0.;
    double const y_1 = -.5;
    double const sigma = .5;

    double const v = 0.5;
    assert(v >= 0. && v < 1.);
    double const mass = 150.;
    double const coupling_constant = 1.e15;
    double const coupling_power
            = 16.; // High coupling power is required to make coupling between wavepackets predominant compare to self-interactions, such that nice manifest macroscopic momentum transfert can be exhibited.
    double const k = mass * v / std::sqrt(1. - v * v);
    double const omega = std::sqrt(k * k + mass * mass);

    double const dx
            = (ddc::get<X>(upper_bounds) - ddc::get<X>(lower_bounds)) / ddc::get<DDimX>(nb_cells);
    double const dy
            = (ddc::get<Y>(upper_bounds) - ddc::get<Y>(lower_bounds)) / ddc::get<DDimY>(nb_cells);
    std::cout << "Space sampling = " << std::max(dx, dy)
              << " (2*pi/k = " << 2 * std::numbers::pi / k << ")" << std::endl;
    double const dt_max
            = 2.0 / std::sqrt(mass * mass + 4.0 / (dx * dx) + 4.0 / (dy * dy)); // TOJUSTIFY
    double const dt = 0.2 * dt_max;
    std::cout << "Time step = " << dt << " (2*pi/omega = " << 2 * std::numbers::pi / omega
              << " while CFL condition estimates maximal dt = " << dt_max << ")" << std::endl;
    std::cout << "Space sampling = " << std::max(dx, dy)
              << " (2*pi/k = " << 2 * std::numbers::pi / k << ")" << std::endl;

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
                                  + std::sqrt(1.264) * // Magic number to equalize Hamiltonians
                                            std::exp(
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
    ddc::Chunk spatial_moments_alloc(potential_grad_dom, ddc::DeviceAllocator<double>());
    sil::tensor::Tensor spatial_moments(spatial_moments_alloc);

    auto spatial_moments_host = ddc::
            create_mirror_view_and_copy(Kokkos::DefaultHostExecutionSpace(), spatial_moments);

    // Minus spatial moments divergence
    [[maybe_unused]] sil::tensor::TensorAccessor<DummyIndex> minus_spatial_moments_div_accessor;
    ddc::DiscreteDomain<DDimX, DDimY, DummyIndex>
            minus_spatial_moments_div_dom(mesh_xy, minus_spatial_moments_div_accessor.domain());
    ddc::Chunk minus_spatial_moments_div_alloc(
            minus_spatial_moments_div_dom,
            ddc::DeviceAllocator<double>());
    sil::tensor::Tensor minus_spatial_moments_div(minus_spatial_moments_div_alloc);
    auto h_minus_spatial_moments_div = ddc::create_mirror_view_and_copy(
            Kokkos::DefaultHostExecutionSpace(),
            minus_spatial_moments_div);

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
                double const y = ddc::coordinate(ddc::DiscreteElement<DDimY>(elem));
                // Group velocity toward the right for left wave packet
                temporal_moment(elem) = (-omega * std::cos(k * (x - x_0))
                                         + v * (x - x_0) / sigma / sigma * std::sin(k * (x - x_0)))
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

    // Codifferential
    auto codifferential = sil::exterior::make_staged_codifferential<
            MetricIndex,
            AlphaLow,
            AlphaLow>(Kokkos::DefaultExecutionSpace(), spatial_moments, metric, position);

    // ------------------
    // ----- SOLVER -----
    // ------------------

    int const nb_iter_between_exports = 100;
    int const nb_iter = 20000;
    ddc::expose_to_pdi("Nt", (nb_iter - 1) / nb_iter_between_exports + 1);
    std::remove("2d_scalar_field.h5");

#if defined(SIMILIE_ASSERT_EXAMPLE_RESULTS_CORRECTNESS)
    std::vector<double> central_potential_values;
#endif

    /*
     * DeDonder-Weyl equations are commonly written:
     * dpi^\mu/dx^\mu = -dH/dphi
     * dphi/dx^\mu = dH/dpi^\mu
     *
     * But we follow the convention with pi being stored as covariant. Thus:
     * eta^\mu\nu dpi_\nu/dx^\mu = -dH/dphi
     * dphi/dx^\mu = eta^\mu\nu dH/dpi_\nu
     *
     * Or explicitly with a (-, +, +) metric:
     * - dpi_0/dx^0 + dpi_\alpha/dx^\alpha = -dH/dphi
     * dphi/dx^0 = - dH/dpi_0
     * dphi/dx^\alpha = dH/dpi_\alpha
     *
     * In the exterior calculus formalism, we have :
     *
     * d \phi = dH/dpi_\alpha
     * \delta \pi = -dpi_\alpha/dx^\alpha
     *
     * Thus:
     *
     * dpi_0/dx^0 = dH/dphi - \delta \pi
     * dphi/dx^0 = - dH/dpi_0
     * d \phi = dH/dpi_\alpha
     *
     * We implement mid-point explicit temporal integration scheme.
     */
    similie::physics::scalar_field::ScalarFieldWithPowerCouplingHamiltonian<T, X, Y> const
            hamiltonian_model(mass, coupling_constant, coupling_power);
    similie::physics::DeDonderWeylEquations equations(hamiltonian_model);
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
                              - equations.template potential_grad<T>(temporal_moment(elem)) * dt
                                        / 2.;
                });

        // Compute the potential gradient
        sil::exterior::deriv<
                AlphaLow,
                DummyIndex>(Kokkos::DefaultExecutionSpace(), potential_grad, half_step_potential);

        // For this scalar model, the spatial moments cochain is exactly dphi.
        ddc::parallel_deepcopy(spatial_moments, potential_grad);
        if (i % nb_iter_between_exports == 0) {
            ddc::parallel_deepcopy(spatial_moments_host, spatial_moments);
        }

        // Compute minus the divergence \delta \pi of the spatial moments
        codifferential.run(minus_spatial_moments_div, spatial_moments);

        // Compute dpi_0/dx^0 = dH/dphi - \delta \pi from the DeDonder-Weyl equation then perform the whole-step advection
        ddc::parallel_for_each(
                Kokkos::DefaultExecutionSpace(),
                minus_spatial_moments_div.domain(),
                KOKKOS_LAMBDA(ddc::DiscreteElement<DDimX, DDimY, DummyIndex> elem) {
                    const double half_step_potential_ = half_step_potential(elem);
                    const double temporal_moment_ = temporal_moment(elem);
                    const double minus_spatial_moments_div_ = minus_spatial_moments_div(elem);

                    // Advect temporal moment by half-step, this is what is needed to perform the whole-step potential advection
                    const double half_step_temporal_moment_
                            = temporal_moment_
                              - (equations.moments_div(half_step_potential_)
                                 + minus_spatial_moments_div_)
                                        * dt / 2;

                    // Whole-step advection of field state
                    potential(elem)
                            -= equations.template potential_grad<T>(half_step_temporal_moment_)
                               * dt;
                    temporal_moment(elem) -= (equations.moments_div(half_step_potential_)
                                              + minus_spatial_moments_div_)
                                             * dt;
                });

        if (i % nb_iter_between_exports == 0) {
            ddc::parallel_deepcopy(temporal_moment_host, temporal_moment);
            ddc::parallel_deepcopy(potential_host, potential);
            ddc::parallel_deepcopy(h_minus_spatial_moments_div, minus_spatial_moments_div);

            ddc::parallel_for_each(
                    Kokkos::DefaultExecutionSpace(),
                    mesh_xy,
                    KOKKOS_LAMBDA(ddc::DiscreteElement<DDimX, DDimY> elem) {
                        std::array<double, 3> pi
                                = {temporal_moment(elem, ddc::DiscreteElement<DummyIndex>()),
                                   spatial_moments(elem, ddc::DiscreteElement<AlphaLow>(0)),
                                   spatial_moments(elem, ddc::DiscreteElement<AlphaLow>(1))};
                        hamiltonian(elem) = hamiltonian_model.hamiltonian(
                                potential(elem, ddc::DiscreteElement<DummyIndex>()),
                                pi);
                    });
            ddc::parallel_deepcopy(h_hamiltonian, hamiltonian);

            // Export HDF5 and XDMF
            const float central_potential_value = potential_host(
                    ddc::DiscreteElement<DDimX, DDimY, DummyIndex>(
                            potential.extent<DDimX>() / 2,
                            potential.extent<DDimY>() / 2,
                            0));
            std::cout << "Potential center = " << central_potential_value << std::endl;
#if defined(SIMILIE_ASSERT_EXAMPLE_RESULTS_CORRECTNESS)
            central_potential_values.push_back(central_potential_value);
#endif
            double const time = i * dt;
            ddc::PdiEvent("export")
                    .with("export_id", i / nb_iter_between_exports)
                    .with("write_position", i / nb_iter_between_exports == 0 ? 1 : 0)
                    .with("time", time)
                    .with("position", position_host)
                    .with("potential", potential_host)
                    .with("temporal_moment", temporal_moment_host)
                    .with("spatial_moments", spatial_moments_host)
                    .with("minus_spatial_moments_div", h_minus_spatial_moments_div)
                    .with("hamiltonian", h_hamiltonian);
            std::cout << "Computation result exported in 2d_scalar_field.h5." << std::endl;

            write_xdmf(
                    static_cast<int>(mesh_xy.template extent<DDimX>()),
                    static_cast<int>(mesh_xy.template extent<DDimY>()),
                    (nb_iter - 1) / nb_iter_between_exports + 1,
                    i / nb_iter_between_exports + 1,
                    dt * nb_iter_between_exports);
            std::cout << "XDMF model exported in 2d_scalar_field.xmf." << std::endl;
        }
    }

#if defined(SIMILIE_ASSERT_EXAMPLE_RESULTS_CORRECTNESS)
    std::array<double, 200> const expected_central_potential_values
            = {0.67492215011843792,    -0.40246684614292494,   -0.1336233694933959,
               0.58230657047397394,    -0.64970537624010127,   0.29154778164661815,
               0.25775695099772306,    -0.63858350766781313,   0.60143463969023925,
               -0.17037762542836751,   -0.37261942519765845,   0.67218282026632059,
               -0.53199627540628391,   0.043072531242274004,   0.47454177537656839,
               -0.67882507392402747,   0.43883571919243275,    0.076612650121525142,
               -0.53663608864732826,   0.66670058694160717,    -0.38299888272021027,
               -0.16533289183455882,   0.64454059373167638,    -0.7177506214139252,
               0.29257630752372193,    0.36656494047855209,    -0.79819539313270815,
               0.67800990138251493,    -0.072031384386125344,  -0.59860452036843936,
               0.85783698338217107,    -0.5165155937224305,    -0.18837091841285805,
               0.76498367040911752,    -0.81120985184082117,   0.29587804384570909,
               0.42229761070510358,    -0.84843971275161389,   0.69070106802060938,
               -0.061671869371967239,  -0.60629679802102843,   0.85826431788129998,
               -0.52509056382839636,   -0.16127044681469613,   0.73478137098817897,
               -0.80817773697185402,   0.33519588102888231,    0.35902270079111726,
               -0.80877161523443697,   0.71141775471478019,    -0.1363007735010108,
               -0.52320518333607591,   0.83220347858833243,    -0.57980539441849865,
               -0.06000556953023721,   0.6484589351366028,     -0.81094641784209176,
               0.42363531085695183,    0.24472931760468059,    -0.73116304473876315,
               0.75275863426846312,    -0.25133340295018908,   -0.41058658973212692,
               0.76885276646703249,    -0.6674486242775165,    0.068514582474183974,
               0.55077772849349815,    -0.76048726214933926,   0.56661539565087171,
               0.12355706330427974,    -0.65668097152014782,   0.70842877484744704,
               -0.4613745789433733,    -0.33043237019918714,   0.71116460122115077,
               -0.62654381697600259,   0.3534869396161478,     0.56942414577582801,
               -0.67138672312659653,   0.56949527040849179,    -0.16925212554587457,
               -0.8207601222378319,    0.4733720852537221,     -0.67259486186583151,
               -0.42857816637053447,   0.63841417323810923,    -0.76621386865003804,
               -0.25780206162003738,   0.45913677200765063,    -0.9686248858598363,
               -0.46663153984546074,   0.56104033941142073,    -0.65144739914597927,
               -0.39170935164562992,   0.93460264841584384,    0.68580862685431698,
               -0.030143078224102789,  0.97233174040237724,    0.19743096964239121,
               -1.0459484652497473,    -0.34502815567070882,   -0.10962795782172065,
               -0.95145976033080204,   0.21459742781821037,    0.68884661455204599,
               -0.3219469015149779,    -1.0066130684158683,    0.16507172550976201,
               1.0791047736485755,     0.27642801316874427,    -0.81430534756208972,
               -0.73931441092818595,   0.33519032444307412,    0.86014194443280678,
               0.6042114007040118,     -0.23746713634091235,   -0.90464599741689267,
               -0.58452445204407066,   0.33566847649543152,    0.26706663914774392,
               0.59817366565884522,    0.62481298505516625,    -0.51050949873969698,
               -0.76068840632751022,   -0.18440519540625461,   0.034308195342312685,
               0.39958075204770382,    0.47548910248629545,    -0.077849004232377006,
               -0.091656793432834463,  -0.075282108976138198,  -0.4320277204932918,
               -0.30696941770549929,   -0.068035079166870738,  -0.047063502910938318,
               0.29364901901387841,    0.085861087741449432,   -0.27677565788032399,
               -0.0083110337345212327, -0.15963931289688107,   -0.085149967081486255,
               0.27836227505509131,    -0.1223369962718327,    0.0046080992418764304,
               0.26763980210823279,    -0.20034575201706434,   -0.11198486987538842,
               0.092689456605984585,   -0.13350567940625935,   0.076176097279343422,
               0.068791253064879807,   -0.14425623202323354,   0.082489279311730099,
               0.043315131156853572,   -0.10259704276927549,   0.057786916093854397,
               -0.047660952975694601,  -0.032597546311375666,  0.16824189054027072,
               -0.063916218113173037,  -0.10406442087709983,   0.075941220174916363,
               -0.02047758503484402,   -0.0016397489894813071, 0.023792719794979288,
               -0.059685339787543208,  0.078021640709808271,   0.039397522741253709,
               -0.11724716663455606,   0.013628800902111405,   0.02242127721256594,
               0.024480403657905568,   0.034990248417244095,   -0.10844138973067387,
               0.021077851947577018,   0.099156359662950555,   -0.094524445345024355,
               0.037856271486780585,   0.056835376487312711,   -0.10951102417194017,
               0.067064667324745947,   0.026100640834147527,   -0.11184883056078285,
               0.089072704165883865,   0.00070407889583772758, -0.081224997567999987,
               0.080005613297491354,   -0.024283805955702099,  -0.031928530513750275,
               0.063975836665454894,   -0.059804223150268288,  0.0066904386215389947,
               0.052689714084930221,   -0.072903229452145502,  0.041931512187486822,
               0.02236982724628522,    -0.076438790712124241,  0.06908471510881016,
               -0.011153586561775475,  -0.050126111050055905};
    double constexpr tolerance = 1.e-6;
    double max_error = 0.;
    for (std::size_t i = 0; i < central_potential_values.size(); i++) {
        double const error
                = std::abs(central_potential_values[i] - expected_central_potential_values[i]);
        max_error = std::max(max_error, error);
        if (error > tolerance) {
            std::cerr << "ERROR: central potential non-regression value " << i << " expected "
                      << expected_central_potential_values[i] << " +/- " << tolerance << ", got "
                      << central_potential_values[i] << std::endl;
            print_expected_central_potential_values(central_potential_values);
            return EXIT_FAILURE;
        }
    }
    std::cout << "Central potential non-regression check passed with max error "
              << std::setprecision(17) << max_error << " <= " << tolerance << std::endl;
#endif

    return EXIT_SUCCESS;
}
