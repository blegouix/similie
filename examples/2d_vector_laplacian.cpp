// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: MIT

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
  position:
    type: array
    subtype: double
    size: [ '$Nx', '$Ny' , 2]
  potential:
    type: array
    subtype: double
    size: [ '$Nx', '$Ny', 2 ]
  laplacian:
    type: array
    subtype: double
    size: [ '$Nx-1', '$Ny-1', 2 ]

plugins:
  decl_hdf5:
    - file: '2d_vector_laplacian.h5'
      on_event: [export]
      collision_policy: replace_and_warn
      write: [Nx, Ny, position, potential, laplacian]
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
        2d_vector_laplacian.h5:/potential
       </DataItem>
     </Attribute>
     <Attribute Name="Laplacian" AttributeType="Vector" Center="Cell">
       <DataItem Dimensions="%i %i 2" NumberType="Float" Precision="8" Format="HDF">
        2d_vector_laplacian.h5:/laplacian
       </DataItem>
     </Attribute>
   </Grid>
 </Domain>
</Xdmf>
)XDMF";

    FILE* file = fopen("2d_vector_laplacian.xmf", "w");
    fprintf(file, xdmf, Nx, Ny, Nx, Ny, Nx, Ny, Nx - 1, Ny - 1);
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

    // Potential
    [[maybe_unused]] sil::tensor::TensorAccessor<MuLow> potential_accessor;
    ddc::DiscreteDomain<DDimX, DDimY, MuLow>
            potential_dom(metric.non_indices_domain(), potential_accessor.domain());
    ddc::Chunk potential_alloc(potential_dom, ddc::DeviceAllocator<double>());
    sil::tensor::Tensor potential(potential_alloc);

    double const R = 2.;
    ddc::parallel_for_each(
            Kokkos::DefaultExecutionSpace(),
            potential.non_indices_domain(),
            KOKKOS_LAMBDA(ddc::DiscreteElement<DDimX, DDimY> elem) {
                double const r = Kokkos::sqrt(
                        static_cast<double>(
                                ddc::coordinate(ddc::DiscreteElement<DDimX>(elem))
                                * ddc::coordinate(ddc::DiscreteElement<DDimX>(elem)))
                        + static_cast<double>(
                                ddc::coordinate(ddc::DiscreteElement<DDimY>(elem))
                                * ddc::coordinate(ddc::DiscreteElement<DDimY>(elem))));
                if (r <= R) {
                    potential.mem(elem, potential_accessor.access_element<X>())
                            = ddc::coordinate(ddc::DiscreteElement<DDimY>(elem))
                              * (r / 3. - R / 2.);
                    potential.mem(elem, potential_accessor.access_element<Y>())
                            = -ddc::coordinate(ddc::DiscreteElement<DDimX>(elem))
                              * (r / 3. - R / 2.);
                } else {
                    potential.mem(elem, potential_accessor.access_element<X>())
                            = -ddc::coordinate(ddc::DiscreteElement<DDimY>(elem)) * R * R * R
                              / (6. * r * r);
                    potential.mem(elem, potential_accessor.access_element<Y>())
                            = ddc::coordinate(ddc::DiscreteElement<DDimX>(elem)) * R * R * R
                              / (6. * r * r);
                }
            });
    auto potential_host
            = ddc::create_mirror_view_and_copy(Kokkos::DefaultHostExecutionSpace(), potential);

    using LaplacianDummyIndex2 = sil::tensor::Covariant<
            sil::exterior::detail::LaplacianDummy2<sil::tensor::uncharacterize_t<MuLow>>>;
    using DerivativeIndex = sil::exterior::coboundary_index_t<LaplacianDummyIndex2, MuLow>;
    using DerivativeMuUpSeq = sil::tensor::upper_t<
            ddc::to_type_seq_t<sil::tensor::natural_domain_t<DerivativeIndex>>>;
    using DerivativeNuLowSeq = typename sil::exterior::detail::CodifferentialDummyIndexSeq<
            LaplacianDummyIndex2::size() - DerivativeIndex::rank(),
            LaplacianDummyIndex2>::type;
    using DerivativeRhoLowSeq
            = ddc::type_seq_merge_t<ddc::detail::TypeSeq<LaplacianDummyIndex2>, DerivativeNuLowSeq>;
    using DerivativeRhoUpSeq = sil::tensor::upper_t<DerivativeRhoLowSeq>;
    using DerivativeSigmaLowSeq = ddc::type_seq_remove_t<
            sil::tensor::lower_t<DerivativeMuUpSeq>,
            ddc::detail::TypeSeq<LaplacianDummyIndex2>>;
    using DerivativeDualIndex = sil::misc::
            convert_type_seq_to_t<sil::tensor::TensorAntisymmetricIndex, DerivativeNuLowSeq>;
    using MuUpSeq = sil::tensor::upper_t<ddc::to_type_seq_t<sil::tensor::natural_domain_t<MuLow>>>;
    using NuLowSeq = typename sil::exterior::detail::
            CodifferentialDummyIndexSeq<MuLow::size() - MuLow::rank(), MuLow>::type;
    using RhoLowSeq = ddc::type_seq_merge_t<ddc::detail::TypeSeq<MuLow>, NuLowSeq>;
    using RhoUpSeq = sil::tensor::upper_t<RhoLowSeq>;
    using SigmaLowSeq
            = ddc::type_seq_remove_t<sil::tensor::lower_t<MuUpSeq>, ddc::detail::TypeSeq<MuLow>>;
    using DualIndex
            = sil::misc::convert_type_seq_to_t<sil::tensor::TensorAntisymmetricIndex, NuLowSeq>;
    using CodifferentialIndex = sil::exterior::codifferential_index_t<MuLow, MuLow>;

    // Laplacian
    [[maybe_unused]] sil::tensor::TensorAccessor<MuLow> laplacian_accessor;
    ddc::DiscreteDomain<DDimX, DDimY, MuLow> laplacian_dom(
            mesh_xy.remove_last(ddc::DiscreteVector<DDimX, DDimY> {1, 1}),
            laplacian_accessor.domain());
    ddc::Chunk laplacian_alloc(laplacian_dom, ddc::DeviceAllocator<double>());
    sil::tensor::Tensor laplacian(laplacian_alloc);

    ddc::Chunk coboundary_of_codifferential_alloc(laplacian_dom, ddc::DeviceAllocator<double>());
    sil::tensor::Tensor coboundary_of_codifferential_buffer(coboundary_of_codifferential_alloc);

    [[maybe_unused]] sil::tensor::TensorAccessor<DerivativeIndex> derivative_accessor;
    ddc::DiscreteDomain<DDimX, DDimY, DerivativeIndex>
            derivative_dom(mesh_xy, derivative_accessor.domain());
    ddc::Chunk derivative_alloc(derivative_dom, ddc::DeviceAllocator<double>());
    sil::tensor::Tensor derivative_tensor_buffer(derivative_alloc);

    [[maybe_unused]] sil::tensor::tensor_accessor_for_domain_t<
            sil::exterior::hodge_star_domain_t<DerivativeMuUpSeq, DerivativeNuLowSeq>>
            derivative_hodge_star_accessor;
    ddc::cartesian_prod_t<
            decltype(metric.non_indices_domain()),
            sil::exterior::hodge_star_domain_t<DerivativeMuUpSeq, DerivativeNuLowSeq>>
            derivative_hodge_star_dom(
                    metric.non_indices_domain(),
                    derivative_hodge_star_accessor.domain());
    ddc::Chunk
            derivative_hodge_star_alloc(derivative_hodge_star_dom, ddc::DeviceAllocator<double>());
    sil::tensor::Tensor derivative_hodge_star(derivative_hodge_star_alloc);

    [[maybe_unused]] sil::tensor::tensor_accessor_for_domain_t<
            sil::exterior::hodge_star_domain_t<DerivativeRhoUpSeq, DerivativeSigmaLowSeq>>
            dual_derivative_hodge_star_accessor;
    ddc::cartesian_prod_t<
            decltype(metric.non_indices_domain()),
            sil::exterior::hodge_star_domain_t<DerivativeRhoUpSeq, DerivativeSigmaLowSeq>>
            dual_derivative_hodge_star_dom(
                    metric.non_indices_domain(),
                    dual_derivative_hodge_star_accessor.domain());
    ddc::Chunk dual_derivative_hodge_star_alloc(
            dual_derivative_hodge_star_dom,
            ddc::DeviceAllocator<double>());
    sil::tensor::Tensor dual_derivative_hodge_star(dual_derivative_hodge_star_alloc);

    [[maybe_unused]] sil::tensor::TensorAccessor<DerivativeDualIndex>
            derivative_dual_tensor_accessor;
    ddc::cartesian_prod_t<decltype(mesh_xy), ddc::DiscreteDomain<DerivativeDualIndex>>
            derivative_dual_tensor_dom(mesh_xy, derivative_dual_tensor_accessor.domain());
    ddc::Chunk derivative_dual_tensor_alloc(
            derivative_dual_tensor_dom,
            ddc::DeviceAllocator<double>());
    sil::tensor::Tensor derivative_dual_tensor_buffer(derivative_dual_tensor_alloc);

    [[maybe_unused]] sil::tensor::TensorAccessor<CodifferentialIndex> codifferential_accessor;
    ddc::DiscreteDomain<DDimX, DDimY, CodifferentialIndex>
            codifferential_dom(mesh_xy, codifferential_accessor.domain());
    ddc::Chunk codifferential_alloc(codifferential_dom, ddc::DeviceAllocator<double>());
    sil::tensor::Tensor codifferential_tensor_buffer(codifferential_alloc);

    [[maybe_unused]] sil::tensor::tensor_accessor_for_domain_t<
            sil::exterior::hodge_star_domain_t<MuUpSeq, NuLowSeq>> hodge_star_accessor;
    ddc::cartesian_prod_t<
            decltype(metric.non_indices_domain()),
            sil::exterior::hodge_star_domain_t<MuUpSeq, NuLowSeq>>
            hodge_star_dom(metric.non_indices_domain(), hodge_star_accessor.domain());
    ddc::Chunk hodge_star_alloc(hodge_star_dom, ddc::DeviceAllocator<double>());
    sil::tensor::Tensor hodge_star(hodge_star_alloc);

    [[maybe_unused]] sil::tensor::tensor_accessor_for_domain_t<
            sil::exterior::hodge_star_domain_t<RhoUpSeq, SigmaLowSeq>> dual_hodge_star_accessor;
    ddc::cartesian_prod_t<
            decltype(metric.non_indices_domain()),
            sil::exterior::hodge_star_domain_t<RhoUpSeq, SigmaLowSeq>>
            dual_hodge_star_dom(metric.non_indices_domain(), dual_hodge_star_accessor.domain());
    ddc::Chunk dual_hodge_star_alloc(dual_hodge_star_dom, ddc::DeviceAllocator<double>());
    sil::tensor::Tensor dual_hodge_star(dual_hodge_star_alloc);

    [[maybe_unused]] sil::tensor::TensorAccessor<DualIndex> dual_tensor_accessor;
    ddc::cartesian_prod_t<decltype(mesh_xy), ddc::DiscreteDomain<DualIndex>>
            dual_tensor_dom(mesh_xy, dual_tensor_accessor.domain());
    ddc::Chunk dual_tensor_alloc(dual_tensor_dom, ddc::DeviceAllocator<double>());
    sil::tensor::Tensor dual_tensor_buffer(dual_tensor_alloc);

    sil::exterior::fill_discrete_hodge_star<DerivativeMuUpSeq, DerivativeNuLowSeq>(
            Kokkos::DefaultExecutionSpace(),
            derivative_hodge_star,
            metric,
            position);
    sil::exterior::fill_discrete_hodge_star<DerivativeRhoUpSeq, DerivativeSigmaLowSeq>(
            Kokkos::DefaultExecutionSpace(),
            dual_derivative_hodge_star,
            metric,
            position);
    sil::exterior::fill_discrete_hodge_star<
            MuUpSeq,
            NuLowSeq>(Kokkos::DefaultExecutionSpace(), hodge_star, metric, position);
    sil::exterior::fill_discrete_hodge_star<
            RhoUpSeq,
            SigmaLowSeq>(Kokkos::DefaultExecutionSpace(), dual_hodge_star, metric, position);

    sil::exterior::laplacian<MetricIndex, MuLow, MuLow>(
            Kokkos::DefaultExecutionSpace(),
            laplacian,
            potential,
            derivative_hodge_star,
            dual_derivative_hodge_star,
            derivative_dual_tensor_buffer,
            derivative_tensor_buffer,
            hodge_star,
            dual_hodge_star,
            dual_tensor_buffer,
            codifferential_tensor_buffer,
            coboundary_of_codifferential_buffer);
    Kokkos::fence();

    auto laplacian_host
            = ddc::create_mirror_view_and_copy(Kokkos::DefaultHostExecutionSpace(), laplacian);

    // Export HDF5 and XDMF
    ddc::PdiEvent("export")
            .with("position", position)
            .with("potential", potential_host)
            .with("laplacian", laplacian_host);
    std::cout << "Computation result exported in 2d_vector_laplacian.h5." << std::endl;

    write_xdmf(
            static_cast<int>(mesh_xy.template extent<DDimX>()),
            static_cast<int>(mesh_xy.template extent<DDimY>()));
    std::cout << "XDMF model exported in 2d_vector_laplacian.xmf." << std::endl;

    // Finalize PDI
    PC_tree_destroy(&conf_pdi);
    PDI_finalize();

    return EXIT_SUCCESS;
}
