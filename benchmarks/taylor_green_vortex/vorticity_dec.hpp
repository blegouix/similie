// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#pragma once

#include <cmath>

#include <similie/exterior/interior_product/staged_vertex_form_composition.hpp>
#include <similie/exterior/reduction_and_reconstruction/vertex_form_transfer.hpp>
#include <similie/tensor/periodic_extrapolation_rule.hpp>

#include "structured_grid.hpp"

namespace taylor_green_vortex {
using ThreeForm = sil::tensor::TensorAntisymmetricIndex<
        sil::tensor::Covariant<VolumeDirection>,
        sil::tensor::Covariant<CurlDirection>,
        Vector>;
using ThreeFormTensor = sil::tensor::Tensor<
        double,
        ddc::DiscreteDomain<GridX, GridY, GridZ, ThreeForm>,
        Kokkos::layout_right,
        Field::memory_space>;

/** Compatible vorticity complex. W=d U is a primal two-cochain.
 * B is the primal two-cochain representing the Hodge dual of vector potential A:
 * Delta_H B=W, d B=0 (Coulomb gauge), U=delta B. Thus delta U=0 and
 * d U=W up to the elliptic solve residual. Pointwise fields are reconstructed
 * only for nonlinear transport and output; they never replace these identities.
 * \important This operator and documentation is fully AI-generated.
 */
class VorticityDEC
{
    StructuredGrid& m_grid;
    using TwoCodifferential = sil::exterior::StagedOperator<
            sil::exterior::Codifferential<
                    MetricIndex,
                    sil::tensor::Covariant<CurlDirection>,
                    TwoForm,
                    sil::exterior::Cochain<TwoFormTensor>,
                    MetricTensor,
                    PositionTensor>,
            typename sil::exterior::Codifferential<
                    MetricIndex,
                    sil::tensor::Covariant<CurlDirection>,
                    TwoForm,
                    sil::exterior::Cochain<TwoFormTensor>,
                    MetricTensor,
                    PositionTensor>::staged_storage_type::
                    template rebind<Kokkos::DefaultExecutionSpace>>;
    using OneCodifferential = sil::exterior::StagedOperator<
            sil::exterior::Codifferential<
                    MetricIndex,
                    Vector,
                    Vector,
                    sil::exterior::Cochain<VectorTensor>,
                    MetricTensor,
                    PositionTensor>,
            typename sil::exterior::Codifferential<
                    MetricIndex,
                    Vector,
                    Vector,
                    sil::exterior::Cochain<VectorTensor>,
                    MetricTensor,
                    PositionTensor>::staged_storage_type::
                    template rebind<Kokkos::DefaultExecutionSpace>>;
    using InverseMetric = sil::tensor::relabelize_indices_of_t<
            MetricTensor,
            ddc::to_type_seq_t<typename MetricTensor::accessor_t::natural_domain_t>,
            sil::tensor::upper_t<
                    ddc::to_type_seq_t<typename MetricTensor::accessor_t::natural_domain_t>>>;
    using StagedInteriorProduct
            = sil::exterior::StagedOperator<sil::exterior::VertexFormBilinearComposition<
                    sil::exterior::
                            InteriorProduct<sil::tensor::Contravariant<CurlDirection>, TwoForm>,
                    Vector,
                    TwoForm,
                    Vector,
                    SpatialDomain,
                    Field::memory_space,
                    InverseMetric>>;
    std::unique_ptr<StagedInteriorProduct> m_interior_product;
    Field m_one, m_two, m_two_output, m_physical_one, m_physical_two, m_three, m_axial;
    std::unique_ptr<LaplacianOperator<sil::exterior::Cochain<TwoFormTensor>>> m_laplacian;
    std::unique_ptr<TwoCodifferential> m_codifferential;
    std::unique_ptr<OneCodifferential> m_one_codifferential;

    template <class Index>
    sil::tensor::Tensor<
            double,
            ddc::DiscreteDomain<GridX, GridY, GridZ, Index>,
            Kokkos::layout_right,
            Field::memory_space>
    tensor(Field const& storage) const
    {
        [[maybe_unused]] sil::tensor::TensorAccessor<Index> accessor;
        ddc::DiscreteDomain<GridX, GridY, GridZ, Index> const
                domain(m_grid.domain(), accessor.domain());
        return sil::tensor::Tensor<
                double,
                ddc::DiscreteDomain<GridX, GridY, GridZ, Index>,
                Kokkos::layout_right,
                Field::memory_space>(ddc::ChunkSpan<
                                     double,
                                     ddc::DiscreteDomain<GridX, GridY, GridZ, Index>,
                                     Kokkos::layout_right,
                                     Field::memory_space>(storage.data(), domain));
    }

public:
    // NVCC requires methods enclosing extended device lambdas to be public.
    void pack(Field const& source, Field const& target)
    {
        Kokkos::parallel_for(
                "pack_cochain",
                Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {m_grid.cells, 3}),
                KOKKOS_LAMBDA(int i, int component) {
                    target(0, 3 * i + component) = source(component, i);
                });
    }
    void unpack(Field const& source, Field const& target)
    {
        Kokkos::parallel_for(
                "unpack_cochain",
                Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {m_grid.cells, 3}),
                KOKKOS_LAMBDA(int i, int component) {
                    target(component, i) = source(0, 3 * i + component);
                });
    }

public:
    explicit VorticityDEC(StructuredGrid& grid)
        : m_grid(grid)
        , m_one("compatible_one", 1, 3 * grid.cells)
        , m_two("compatible_two", 1, 3 * grid.cells)
        , m_two_output("compatible_two_output", 1, 3 * grid.cells)
        , m_physical_one("one_form_coefficients", 1, 3 * grid.cells)
        , m_physical_two("two_form_coefficients", 1, 3 * grid.cells)
        , m_three("closure", 1, grid.cells)
        , m_axial("physical_axial_components", 1, 3 * grid.cells)
    {
        m_codifferential = std::make_unique<TwoCodifferential>(
                Kokkos::DefaultExecutionSpace(),
                sil::exterior::Cochain(tensor<TwoForm>(m_two)),
                grid.euclidean_metric(),
                grid.geometry(),
                sil::tensor::PeriodicExtrapolationRule {});
        m_one_codifferential = std::make_unique<OneCodifferential>(
                Kokkos::DefaultExecutionSpace(),
                sil::exterior::Cochain(tensor<Vector>(m_one)),
                grid.euclidean_metric(),
                grid.geometry(),
                sil::tensor::PeriodicExtrapolationRule {});
        InverseMetric const inverse_metric = sil::tensor::relabelize_indices_of<
                ddc::to_type_seq_t<typename MetricTensor::accessor_t::natural_domain_t>,
                sil::tensor::upper_t<
                        ddc::to_type_seq_t<typename MetricTensor::accessor_t::natural_domain_t>>>(
                grid.euclidean_metric());
        m_interior_product = std::make_unique<StagedInteriorProduct>(
                Kokkos::DefaultExecutionSpace(),
                sil::exterior::Cochain(tensor<Vector>(m_two_output)),
                sil::exterior::Cochain(tensor<Vector>(m_one)),
                sil::exterior::Cochain(tensor<TwoForm>(m_two)),
                grid.geometry(),
                inverse_metric);
        Kokkos::fence("vorticity_hodge_setup_complete");
    }
    void reduce_one_form(Field const& coefficients, Field const& cochain)
    {
        pack(coefficients, m_physical_one);
        sil::exterior::Form<VectorTensor> const physical(tensor<Vector>(m_physical_one));
        sil::exterior::Cochain<VectorTensor> const reduced(tensor<Vector>(m_one));
        m_grid.reduce(reduced, physical);
        unpack(m_one, cochain);
    }

    void reduce_two_form(Field const& values, Field const& cochain, bool project_vertices = false)
    {
        pack(values, m_axial);
        sil::exterior::Form<TwoFormTensor> const physical(tensor<TwoForm>(m_physical_two));
        sil::exterior::Cochain<TwoFormTensor> const reduced(tensor<TwoForm>(m_two));
        sil::tensor::Tensor const axial = tensor<sil::tensor::Covariant<VolumeDirection>>(m_axial);
        MetricTensor const metric = m_grid.euclidean_metric();
        ddc::parallel_for_each(
                "axial_to_two_form",
                Kokkos::DefaultExecutionSpace(),
                m_grid.domain(),
                KOKKOS_LAMBDA(SpatialDomain::discrete_element_type element) {
                    sil::exterior::LeviCivitaContinuousHodgeStar<
                            ddc::TypeSeq<sil::tensor::Contravariant<VolumeDirection>>,
                            ddc::TypeSeq<sil::tensor::Covariant<CurlDirection>, Vector>,
                            MetricTensor,
                            SpatialDomain::discrete_element_type>::
                    operator()(
                            physical[element],
                            sil::exterior::Form(axial[element]),
                            metric,
                            element);
                });
        if (project_vertices) {
            PositionTensor const geometry = m_grid.geometry();
            ddc::parallel_for_each(
                    "project_two_form",
                    Kokkos::DefaultExecutionSpace(),
                    m_grid.domain(),
                    KOKKOS_LAMBDA(SpatialDomain::discrete_element_type element) {
                        sil::exterior::VertexFormProjection<
                                ddc::TypeSeq<sil::tensor::Covariant<CurlDirection>, Vector>>::
                        operator()(
                                reduced[element],
                                physical,
                                geometry,
                                element,
                                sil::tensor::PeriodicExtrapolationRule {});
                    });
        } else
            m_grid.reduce(reduced, physical);
        unpack(m_two, cochain);
    }
    void reconstruct_one_form(Field const& cochain, Field const& output, bool at_vertices = false)
    {
        pack(cochain, m_one);
        sil::exterior::Cochain<VectorTensor> const reduced(tensor<Vector>(m_one));
        sil::exterior::Form<VectorTensor> const physical(tensor<Vector>(m_physical_one));
        if (at_vertices) {
            PositionTensor const geometry = m_grid.geometry();
            ddc::parallel_for_each(
                    "observe_one_form",
                    Kokkos::DefaultExecutionSpace(),
                    m_grid.domain(),
                    KOKKOS_LAMBDA(SpatialDomain::discrete_element_type element) {
                        sil::exterior::VertexFormReconstruction<ddc::TypeSeq<Vector>>::operator()(
                                physical[element],
                                reduced,
                                geometry,
                                element,
                                sil::tensor::PeriodicExtrapolationRule {});
                    });
        } else
            m_grid.reconstruct(physical, reduced);
        unpack(m_physical_one, output);
    }
    void reconstruct_two_form(Field const& cochain, Field const& output, bool at_vertices = false)
    {
        pack(cochain, m_two);
        sil::exterior::Cochain<TwoFormTensor> const reduced(tensor<TwoForm>(m_two));
        sil::exterior::Form<TwoFormTensor> const physical(tensor<TwoForm>(m_physical_two));
        if (at_vertices) {
            PositionTensor const geometry = m_grid.geometry();
            ddc::parallel_for_each(
                    "observe_two_form",
                    Kokkos::DefaultExecutionSpace(),
                    m_grid.domain(),
                    KOKKOS_LAMBDA(SpatialDomain::discrete_element_type element) {
                        sil::exterior::VertexFormReconstruction<
                                ddc::TypeSeq<sil::tensor::Covariant<CurlDirection>, Vector>>::
                        operator()(
                                physical[element],
                                reduced,
                                geometry,
                                element,
                                sil::tensor::PeriodicExtrapolationRule {});
                    });
        } else
            m_grid.reconstruct(physical, reduced);
        sil::tensor::Tensor const axial = tensor<sil::tensor::Covariant<VolumeDirection>>(m_axial);
        MetricTensor const metric = m_grid.euclidean_metric();
        ddc::parallel_for_each(
                "two_form_to_axial",
                Kokkos::DefaultExecutionSpace(),
                m_grid.domain(),
                KOKKOS_LAMBDA(SpatialDomain::discrete_element_type element) {
                    sil::exterior::LeviCivitaContinuousHodgeStar<
                            ddc::TypeSeq<
                                    sil::tensor::Contravariant<CurlDirection>,
                                    sil::tensor::Contravariant<Direction>>,
                            ddc::TypeSeq<sil::tensor::Covariant<VolumeDirection>>,
                            MetricTensor,
                            SpatialDomain::discrete_element_type>::
                    operator()(
                            sil::exterior::Form(axial[element]),
                            physical[element],
                            metric,
                            element);
                });
        unpack(m_axial, output);
    }
    void derivative_one_form(Field const& one_cochain, Field const& two_cochain)
    {
        pack(one_cochain, m_one);
        sil::exterior::deriv<sil::tensor::Covariant<CurlDirection>, Vector>(
                Kokkos::DefaultExecutionSpace(),
                sil::exterior::Cochain(tensor<TwoForm>(m_two)),
                sil::exterior::Cochain(tensor<Vector>(m_one)),
                sil::tensor::PeriodicExtrapolationRule {});
        unpack(m_two, two_cochain);
    }
    void interior_product(
            Field const& velocity_cochain,
            Field const& two_cochain,
            Field const& output,
            std::array<double, 3> const& harmonic)
    {
        pack(velocity_cochain, m_one);
        pack(two_cochain, m_two);
        Field const one = m_one;
        double const h = m_grid.h;
        Kokkos::parallel_for(
                "interior_product_harmonic_velocity",
                m_grid.cells,
                KOKKOS_LAMBDA(int i) {
                    for (int a = 0; a < 3; ++a)
                        one(0, 3 * i + a) += h * harmonic[a];
                });
        (*m_interior_product)(
                sil::exterior::Cochain(tensor<Vector>(m_two_output)),
                sil::exterior::Cochain(tensor<Vector>(m_one)),
                sil::exterior::Cochain(tensor<TwoForm>(m_two)));
        unpack(m_two_output, output);
    }
    void codifferential(Field const& two_cochain, Field const& one_cochain)
    {
        pack(two_cochain, m_two);
        (*m_codifferential)(
                sil::exterior::Cochain(tensor<Vector>(m_one)),
                sil::exterior::Cochain(tensor<TwoForm>(m_two)));
        unpack(m_one, one_cochain);
    }
    void positive_laplacian(Field const& input, Field const& output)
    {
        // Electromagnetic helpers use the cochain maps without this Laplacian.
        // Avoid allocating a second full stencil cache for those helpers.
        if (!m_laplacian)
            m_laplacian = make_two_form_laplacian(
                    m_grid.configuration,
                    sil::exterior::Cochain(tensor<TwoForm>(m_two_output)),
                    sil::exterior::Cochain(tensor<TwoForm>(m_two)),
                    m_grid.euclidean_metric(),
                    m_grid.geometry());
        pack(input, m_two);
        m_laplacian
                ->apply(sil::exterior::Cochain(tensor<TwoForm>(m_two_output)),
                        sil::exterior::Cochain(tensor<TwoForm>(m_two)));
        unpack(m_two_output, output);
    }
    void negative_codifferential_one_form(Field const& one_cochain, Field const& output)
    {
        pack(one_cochain, m_one);
        (*m_one_codifferential)(
                sil::exterior::Cochain(m_grid.scalar(output)),
                sil::exterior::Cochain(tensor<Vector>(m_one)));
        Kokkos::parallel_for(
                "physical_cochain_divergence",
                m_grid.cells,
                KOKKOS_LAMBDA(int i) { output(0, i) *= -1; });
    }


    double closure_rms(Field const& two_cochain)
    {
        pack(two_cochain, m_two);
        sil::exterior::deriv<sil::tensor::Covariant<VolumeDirection>, TwoForm>(
                Kokkos::DefaultExecutionSpace(),
                sil::exterior::Cochain(tensor<ThreeForm>(m_three)),
                sil::exterior::Cochain(tensor<TwoForm>(m_two)),
                sil::tensor::PeriodicExtrapolationRule {});
        Field const residual = m_three;
        double sum;
        Kokkos::parallel_reduce(
                "two_cochain_closure",
                m_grid.cells,
                KOKKOS_LAMBDA(int i, double& value) { value += residual(0, i) * residual(0, i); },
                sum);
        return std::sqrt(sum / m_grid.cells) / (m_grid.h * m_grid.h * m_grid.h);
    }
};
} // namespace taylor_green_vortex
