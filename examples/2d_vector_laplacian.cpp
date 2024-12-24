// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: GPL-3.0-or-later

#include <ddc/ddc.hpp>
#include <ddc/kernels/splines.hpp>
#include <ddc/pdi.hpp>

#include <similie/similie.hpp>

// PDI config
constexpr char const* const PDI_CFG = R"PDI_CFG(
metadata:
  Nx : int
  Ny : int

data:
  X:
    type: array
    subtype: double
    size: [ '$Nx', '$Ny' ]
  Y:
    type: array
    subtype: double
    size: [ '$Nx', '$Ny' ]
  potential_x:
    type: array
    subtype: double
    size: [ '$Nx', '$Ny' ]
  potential_y:
    type: array
    subtype: double
    size: [ '$Nx', '$Ny' ]
  laplacian_x:
    type: array
    subtype: double
    size: [ '$Nx-1', '$Ny-1' ]
  laplacian_y:
    type: array
    subtype: double
    size: [ '$Nx-1', '$Ny-1' ]

plugins:
  decl_hdf5:
    - file: '2d_vector_laplacian.h5'
      on_event: [export]
      collision_policy: replace_and_warn
      write: [Nx, Ny, X, Y, potential_x, potential_y, laplacian_x, laplacian_y]
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
     <Geometry GeometryType="X_Y">
       <DataItem Dimensions="%i %i" NumberType="Float" Precision="8" Format="HDF">
        2d_vector_laplacian.h5:/X
       </DataItem>
       <DataItem Dimensions="%i %i" NumberType="Float" Precision="8" Format="HDF">
        2d_vector_laplacian.h5:/Y
       </DataItem>
     </Geometry>
     <Attribute Name="PotentialX" AttributeType="Scalar" Center="Node">
       <DataItem Dimensions="%i %i" NumberType="Float" Precision="8" Format="HDF">
        2d_vector_laplacian.h5:/potential_x
       </DataItem>
     </Attribute>
    <Attribute Name="PotentialY" AttributeType="Scalar" Center="Node">
       <DataItem Dimensions="%i %i" NumberType="Float" Precision="8" Format="HDF">
        2d_vector_laplacian.h5:/potential_y
       </DataItem>
     </Attribute>
     <Attribute Name="LaplacianX" AttributeType="Scalar" Center="Cell">
       <DataItem Dimensions="%i %i" NumberType="Float" Precision="8" Format="HDF">
        2d_vector_laplacian.h5:laplacian_x/
       </DataItem>
     </Attribute>
     <Attribute Name="LaplacianY" AttributeType="Scalar" Center="Cell">
       <DataItem Dimensions="%i %i" NumberType="Float" Precision="8" Format="HDF">
        2d_vector_laplacian.h5:laplacian_y/
       </DataItem>
     </Attribute>

   </Grid>
 </Domain>
</Xdmf>
)XDMF";

    FILE* file = fopen("2d_vector_laplacian.xmf", "w");
    fprintf(file, xdmf, Nx, Ny, Nx, Ny, Nx, Ny, Nx, Ny, Nx, Ny, Nx - 1, Ny - 1, Nx - 1, Ny - 1);
    fclose(file);

    return 1;
}

static constexpr std::size_t s_degree = 3;

// Labelize the dimensions of space
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
        sil::tensor::TensorCovariantNaturalIndex<sil::tensor::MetricIndex1<X, Y>>,
        sil::tensor::TensorCovariantNaturalIndex<sil::tensor::MetricIndex2<X, Y>>>;

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
using MuUp = sil::tensor::TensorContravariantNaturalIndex<Mu>;
using NuLow = sil::tensor::TensorCovariantNaturalIndex<Nu>;
using NuUp = sil::tensor::TensorContravariantNaturalIndex<Nu>;

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
    ddc::DiscreteVector<DDimX, DDimY> nb_cells(1000, 1000);
    ddc::DiscreteDomain<DDimX, DDimY> mesh_xy = mesher.template mesh<
            ddc::detail::TypeSeq<DDimX, DDimY>,
            ddc::detail::TypeSeq<BSplinesX, BSplinesY>>(lower_bounds, upper_bounds, nb_cells);
    assert(static_cast<std::size_t>(mesh_xy.template extent<DDimX>())
           == static_cast<std::size_t>(mesh_xy.template extent<DDimY>()));
    ddc::expose_to_pdi("Nx", static_cast<int>(mesh_xy.template extent<DDimX>()));
    ddc::expose_to_pdi("Ny", static_cast<int>(mesh_xy.template extent<DDimY>()));

    // Allocate and instantiate a position field (used only to be exported).
    [[maybe_unused]] sil::tensor::TensorAccessor<NuUp> position_accessor;
    ddc::DiscreteDomain<NuUp, DDimX, DDimY> position_dom(mesh_xy, position_accessor.mem_domain());
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
    ddc::DiscreteDomain<DDimX, DDimY, MetricIndex>
            metric_dom(mesh_xy, metric_accessor.mem_domain());
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
    [[maybe_unused]] sil::tensor::TensorAccessor<sil::tensor::upper<MetricIndex>>
            inv_metric_accessor;
    ddc::DiscreteDomain<DDimX, DDimY, sil::tensor::upper<MetricIndex>>
            inv_metric_dom(mesh_xy, inv_metric_accessor.mem_domain());
    ddc::Chunk inv_metric_alloc(inv_metric_dom, ddc::DeviceAllocator<double>());
    sil::tensor::Tensor inv_metric(inv_metric_alloc);
    sil::tensor::fill_inverse_metric<
            MetricIndex>(Kokkos::DefaultExecutionSpace(), inv_metric, metric);
    // TODO weird -1 for X, X on GPU
    Kokkos::deep_copy(
            inv_metric.allocation_kokkos_view(),
            metric.allocation_kokkos_view()); // FIXME: Temporary patch
    /*
    auto debug = inv_metric;
    auto debug_host = ddc::create_mirror_view_and_copy(Kokkos::DefaultHostExecutionSpace(), debug);
    std::cout << "DEBUG:" << std::endl;
    std::cout << sil::tensor::Tensor(debug_host[debug.accessor().mem_domain().front()])
              << std::endl;
    */

    // Potential
    [[maybe_unused]] sil::tensor::TensorAccessor<MuUp> potential_accessor;
    ddc::DiscreteDomain<MuUp, DDimX, DDimY>
            potential_dom(metric.non_indices_domain(), potential_accessor.mem_domain());
    ddc::Chunk potential_alloc(potential_dom, ddc::DeviceAllocator<double>());
    sil::tensor::Tensor potential(potential_alloc);

    double const R = 2.;
    double const L = ddc::coordinate(ddc::DiscreteElement<DDimX>(potential.domain().back()))
                     - ddc::coordinate(ddc::DiscreteElement<DDimX>(potential.domain().front()));
    double const alpha = (static_cast<double>(nb_cells.template get<DDimX>())
                          * static_cast<double>(nb_cells.template get<DDimY>()))
                         / L / 2 / L / 2;
    ddc::parallel_for_each(
            potential.non_indices_domain(),
            KOKKOS_LAMBDA(ddc::DiscreteElement<DDimX, DDimY> elem) {
                double const r = Kokkos::sqrt(
                        static_cast<double>(
                                ddc::coordinate(ddc::DiscreteElement<DDimX>(elem))
                                * ddc::coordinate(ddc::DiscreteElement<DDimX>(elem)))
                        + static_cast<double>(
                                ddc::coordinate(ddc::DiscreteElement<DDimY>(elem))
                                * ddc::coordinate(ddc::DiscreteElement<DDimY>(elem))));
                double const theta = Kokkos::tan(
                                             ddc::coordinate(ddc::DiscreteElement<DDimY>(elem))
                                             / ddc::coordinate(ddc::DiscreteElement<DDimX>(elem)))
                                     + (ddc::coordinate(ddc::DiscreteElement<DDimX>(elem)) < 0
                                                ? Kokkos::numbers::pi_v<double>
                                                : 0);
                if (r <= R) {
                    potential.mem(elem, potential_accessor.access_element<X>())
                            = alpha * ((R * R) - (r * r)) * Kokkos::cos(theta);
                    potential.mem(elem, potential_accessor.access_element<Y>())
                            = alpha * ((R * R) - (r * r)) * Kokkos::sin(theta);
                } else {
                    potential.mem(elem)
                            = alpha * 2 * R * R * Kokkos::log(R / r) * Kokkos::cos(theta);
                    potential.mem(elem)
                            = alpha * 2 * R * R * Kokkos::log(R / r) * Kokkos::sin(theta);
                }
            });
    auto potential_host
            = ddc::create_mirror_view_and_copy(Kokkos::DefaultHostExecutionSpace(), potential);

    // Laplacian
    [[maybe_unused]] sil::tensor::TensorAccessor<MuUp> laplacian_accessor;
    ddc::DiscreteDomain<MuUp, DDimX, DDimY> laplacian_dom(
            mesh_xy.remove_last(ddc::DiscreteVector<DDimX, DDimY> {1, 1}),
            laplacian_accessor.mem_domain());
    ddc::Chunk laplacian_alloc(laplacian_dom, ddc::DeviceAllocator<double>());
    sil::tensor::Tensor laplacian(laplacian_alloc);

    sil::exterior::laplacian<
            MetricIndex,
            NuLow,
            MuUp>(Kokkos::DefaultExecutionSpace(), laplacian, potential, inv_metric);
    Kokkos::fence();

    auto laplacian_host
            = ddc::create_mirror_view_and_copy(Kokkos::DefaultHostExecutionSpace(), laplacian);

    // Export HDF5 and XDMF
    ddc::PdiEvent("export")
            .with("X", position[position.accessor().access_element<X>()])
            .and_with("Y", position[position.accessor().access_element<Y>()])
            .and_with("potential_x", potential_host[potential_accessor.access_element<X>()])
            .and_with("potential_y", potential_host[potential_accessor.access_element<Y>()])
            .and_with("laplacian_x", laplacian_host[laplacian_accessor.access_element<X>()])
            .and_with("laplacian_y", laplacian_host[laplacian_accessor.access_element<Y>()]);
    std::cout << "Computation result exported in 2d_vecotr_laplacian.h5." << std::endl;

    write_xdmf(
            static_cast<int>(mesh_xy.template extent<DDimX>()),
            static_cast<int>(mesh_xy.template extent<DDimY>()));
    std::cout << "XDMF model exported in 2d_vector_laplacian.xmf." << std::endl;

    // Finalize PDI
    PC_tree_destroy(&conf_pdi);
    PDI_finalize();

    return EXIT_SUCCESS;
}
