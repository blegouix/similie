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
  potential:
    type: array
    subtype: double
    size: [ '$Nx', '$Ny' ]
  laplacian:
    type: array
    subtype: double
    size: [ '$Nx-1', '$Ny-1' ]

plugins:
  decl_hdf5:
    - file: '2d_laplacian.h5'
      on_event: [export]
      collision_policy: replace_and_warn
      write: [Nx, Ny, X, Y, potential, laplacian]
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
        2d_laplacian.h5:/X
       </DataItem>
       <DataItem Dimensions="%i %i" NumberType="Float" Precision="8" Format="HDF">
        2d_laplacian.h5:/Y
       </DataItem>
     </Geometry>
     <Attribute Name="Potential" AttributeType="Scalar" Center="Node">
       <DataItem Dimensions="%i %i" NumberType="Float" Precision="8" Format="HDF">
        2d_laplacian.h5:/potential
       </DataItem>
     </Attribute>
     <Attribute Name="Laplacian" AttributeType="Scalar" Center="Cell">
       <DataItem Dimensions="%i %i" NumberType="Float" Precision="8" Format="HDF">
        2d_laplacian.h5:laplacian/
       </DataItem>
     </Attribute>
   </Grid>
 </Domain>
</Xdmf>
)XDMF";

    FILE* file = fopen("2d_laplacian.xmf", "w");
    fprintf(file, xdmf, Nx, Ny, Nx, Ny, Nx, Ny, Nx, Ny, Nx - 1, Ny - 1);
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

struct Rho : sil::tensor::TensorNaturalIndex<X, Y>
{
};

// Declare indices
using MuLow = sil::tensor::TensorCovariantNaturalIndex<Mu>;
using MuUp = sil::tensor::TensorContravariantNaturalIndex<Mu>;
using NuLow = sil::tensor::TensorCovariantNaturalIndex<Nu>;
using NuUp = sil::tensor::TensorContravariantNaturalIndex<Nu>;
using RhoLow = sil::tensor::TensorCovariantNaturalIndex<Rho>;
using RhoUp = sil::tensor::TensorContravariantNaturalIndex<Rho>;

using HodgeStarDomain = sil::exterior::
        hodge_star_domain_t<ddc::detail::TypeSeq<MuUp>, ddc::detail::TypeSeq<NuLow>>;
using HodgeStarDomain2 = sil::exterior::
        hodge_star_domain_t<ddc::detail::TypeSeq<RhoUp, NuUp>, ddc::detail::TypeSeq<>>;

using DummyIndex = sil::tensor::TensorCovariantNaturalIndex<sil::tensor::TensorNaturalIndex<>>;

int main(int argc, char** argv)
{
    PC_tree_t conf_pdi = PC_parse_string(PDI_CFG);
    PC_errhandler(PC_NULL_HANDLER);
    PDI_init(conf_pdi);

    Kokkos::ScopeGuard const kokkos_scope(argc, argv);
    ddc::ScopeGuard const ddc_scope(argc, argv);

    MesherXY mesher;
    ddc::Coordinate<X, Y> lower_bounds(-5., -5.);
    ddc::Coordinate<X, Y> upper_bounds(5., 5.);
    ddc::DiscreteVector<DDimX, DDimY> nb_cells(1000, 1000);
    ddc::DiscreteDomain<DDimX, DDimY> mesh_xy = mesher.template mesh<
            ddc::detail::TypeSeq<DDimX, DDimY>,
            ddc::detail::TypeSeq<BSplinesX, BSplinesY>>(lower_bounds, upper_bounds, nb_cells);
    ddc::expose_to_pdi("Nx", static_cast<int>(mesh_xy.template extent<DDimX>()));
    ddc::expose_to_pdi("Ny", static_cast<int>(mesh_xy.template extent<DDimY>()));

    // Allocate and instantiate a position field (used only to be exported).
    [[maybe_unused]] sil::tensor::TensorAccessor<MuUp> position_accessor;
    ddc::DiscreteDomain<MuUp, DDimX, DDimY> position_dom(mesh_xy, position_accessor.mem_domain());
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
    [[maybe_unused]] sil::tensor::TensorAccessor<DummyIndex> potential_accessor;
    ddc::DiscreteDomain<DDimX, DDimY, DummyIndex>
            potential_dom(metric.non_indices_domain(), potential_accessor.mem_domain());
    ddc::Chunk potential_alloc(potential_dom, ddc::DeviceAllocator<double>());
    sil::tensor::Tensor potential(potential_alloc);

    ddc::parallel_for_each(
            potential.domain(),
            KOKKOS_LAMBDA(ddc::DiscreteElement<DDimX, DDimY, DummyIndex> elem) {
                double const R = 2.;
                double const r = Kokkos::sqrt(
                        static_cast<double>(
                                ddc::coordinate(ddc::DiscreteElement<DDimX>(elem))
                                * ddc::coordinate(ddc::DiscreteElement<DDimX>(elem)))
                        + static_cast<double>(
                                ddc::coordinate(ddc::DiscreteElement<DDimY>(elem))
                                * ddc::coordinate(ddc::DiscreteElement<DDimY>(elem))));
                if (r <= R) {
                    potential.mem(elem) = 6.25 * r * r;
                } else {
                    potential.mem(elem) = 6.25 * (Kokkos::log(r / R) + (R * R));
                }
            });
    auto potential_host
            = ddc::create_mirror_view_and_copy(Kokkos::DefaultHostExecutionSpace(), potential);

    // Gradient
    [[maybe_unused]] sil::tensor::TensorAccessor<MuLow> gradient_accessor;
    ddc::DiscreteDomain<DDimX, DDimY, MuLow> gradient_dom(mesh_xy, gradient_accessor.mem_domain());
    ddc::Chunk gradient_alloc(gradient_dom, ddc::DeviceAllocator<double>());
    sil::tensor::Tensor gradient(gradient_alloc);
    sil::exterior::deriv<MuLow, DummyIndex>(Kokkos::DefaultExecutionSpace(), gradient, potential);
    Kokkos::fence();

    // Hodge star
    [[maybe_unused]] sil::tensor::tensor_accessor_for_domain_t<HodgeStarDomain> hodge_star_accessor;
    ddc::cartesian_prod_t<ddc::DiscreteDomain<DDimX, DDimY>, HodgeStarDomain>
            hodge_star_dom(metric.non_indices_domain(), hodge_star_accessor.mem_domain());
    ddc::Chunk hodge_star_alloc(hodge_star_dom, ddc::DeviceAllocator<double>());
    sil::tensor::Tensor hodge_star(hodge_star_alloc);

    sil::exterior::fill_hodge_star<
            sil::tensor::upper<MetricIndex>,
            ddc::detail::TypeSeq<MuUp>,
            ddc::detail::TypeSeq<NuLow>>(Kokkos::DefaultExecutionSpace(), hodge_star, inv_metric);
    Kokkos::fence();

    // Dual gradient
    [[maybe_unused]] sil::tensor::TensorAccessor<NuLow> dual_gradient_accessor;
    ddc::DiscreteDomain<DDimX, DDimY, NuLow>
            dual_gradient_dom(mesh_xy, dual_gradient_accessor.mem_domain());
    ddc::Chunk dual_gradient_alloc(dual_gradient_dom, ddc::DeviceAllocator<double>());
    sil::tensor::Tensor dual_gradient(dual_gradient_alloc);

    ddc::parallel_for_each(
            Kokkos::DefaultExecutionSpace(),
            dual_gradient.non_indices_domain(),
            KOKKOS_LAMBDA(ddc::DiscreteElement<DDimX, DDimY> elem) {
                sil::tensor::tensor_prod(dual_gradient[elem], gradient[elem], hodge_star[elem]);
            });
    Kokkos::fence();

    // Dual Laplacian
    [[maybe_unused]] sil::tensor::TensorAccessor<
            sil::tensor::TensorAntisymmetricIndex<RhoLow, NuLow>> dual_laplacian_accessor;
    ddc::DiscreteDomain<DDimX, DDimY, sil::tensor::TensorAntisymmetricIndex<RhoLow, NuLow>>
            dual_laplacian_dom(
                    mesh_xy.remove_last(ddc::DiscreteVector<DDimX, DDimY> {1, 1}),
                    dual_laplacian_accessor.mem_domain());
    ddc::Chunk dual_laplacian_alloc(dual_laplacian_dom, ddc::DeviceAllocator<double>());
    sil::tensor::Tensor dual_laplacian(dual_laplacian_alloc);
    sil::exterior::
            deriv<RhoLow, NuLow>(Kokkos::DefaultExecutionSpace(), dual_laplacian, dual_gradient);
    Kokkos::fence();

    // Hodge star 2
    [[maybe_unused]] sil::tensor::tensor_accessor_for_domain_t<HodgeStarDomain2>
            hodge_star_accessor2;
    ddc::cartesian_prod_t<ddc::DiscreteDomain<DDimX, DDimY>, HodgeStarDomain2>
            hodge_star_dom2(metric.non_indices_domain(), hodge_star_accessor2.mem_domain());
    ddc::Chunk hodge_star_alloc2(hodge_star_dom2, ddc::DeviceAllocator<double>());
    sil::tensor::Tensor hodge_star2(hodge_star_alloc2);

    sil::exterior::fill_hodge_star<
            sil::tensor::upper<MetricIndex>,
            ddc::detail::TypeSeq<RhoUp, NuUp>,
            ddc::detail::TypeSeq<>>(Kokkos::DefaultExecutionSpace(), hodge_star2, inv_metric);
    Kokkos::fence();

    // Laplacian
    [[maybe_unused]] sil::tensor::TensorAccessor<DummyIndex> laplacian_accessor;
    ddc::DiscreteDomain<DDimX, DDimY, DummyIndex> laplacian_dom(
            mesh_xy.remove_last(ddc::DiscreteVector<DDimX, DDimY> {1, 1}),
            laplacian_accessor.mem_domain());
    ddc::Chunk laplacian_alloc(laplacian_dom, ddc::DeviceAllocator<double>());
    sil::tensor::Tensor laplacian(laplacian_alloc);

    ddc::parallel_for_each(
            Kokkos::DefaultExecutionSpace(),
            laplacian.non_indices_domain(),
            KOKKOS_LAMBDA(ddc::DiscreteElement<DDimX, DDimY> elem) {
                sil::tensor::tensor_prod(laplacian[elem], dual_laplacian[elem], hodge_star2[elem]);
            });
    Kokkos::fence();

    auto laplacian_host
            = ddc::create_mirror_view_and_copy(Kokkos::DefaultHostExecutionSpace(), laplacian);

    ddc::PdiEvent("export")
            .with("X", position[position.accessor().access_element<X>()])
            .and_with("Y", position[position.accessor().access_element<Y>()])
            .and_with("potential", potential_host)
            .and_with("laplacian", laplacian_host);
    std::cout << "Computation result exported in 2d_laplacian.h5" << std::endl;
    PC_tree_destroy(&conf_pdi);
    PDI_finalize();

    write_xdmf(
            static_cast<int>(mesh_xy.template extent<DDimX>()),
            static_cast<int>(mesh_xy.template extent<DDimY>()));

    return EXIT_SUCCESS;
}
