// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#pragma once

#include <cmath>

#include "structured_grid.hpp"

namespace taylor_green_vortex {
struct VolumeDirection : sil::tensor::TensorNaturalIndex<X, Y, Z>
{
};
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
    using TwoLaplacian = sil::exterior::StagedLaplacian<
            MetricIndex,
            sil::tensor::Covariant<CurlDirection>,
            TwoForm,
            TwoFormTensor,
            MetricTensor,
            PositionTensor,
            Kokkos::DefaultExecutionSpace,
            sil::exterior::CartesianPeriodicStencil>;
    using TwoCodifferential = sil::exterior::StagedCodifferential<
            MetricIndex,
            sil::tensor::Covariant<CurlDirection>,
            TwoForm,
            TwoFormTensor,
            MetricTensor,
            PositionTensor,
            Kokkos::DefaultExecutionSpace>;
    using OneCodifferential = sil::exterior::StagedCodifferential<
            MetricIndex,
            Vector,
            Vector,
            VectorTensor,
            MetricTensor,
            PositionTensor,
            Kokkos::DefaultExecutionSpace>;
    Field m_one, m_two, m_two_output, m_physical_one, m_physical_two, m_three, m_edges;
    std::unique_ptr<TwoLaplacian> m_laplacian;
    std::unique_ptr<TwoCodifferential> m_codifferential;
    std::unique_ptr<OneCodifferential> m_one_codifferential;

    template <class Index>
    sil::tensor::Tensor<
            double,
            ddc::DiscreteDomain<GridX, GridY, GridZ, Index>,
            Kokkos::layout_right,
            Field::memory_space>
    tensor(Field storage) const
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
    // Device lambda instantiations in exterior must be able to name this type.
    struct Periodic
    {
        static constexpr bool is_periodic = true;

        template <class Tensor, class Element>
        KOKKOS_FUNCTION auto operator()(Tensor const& field, Element element) const
        {
            SpatialDomain const domain(field.non_indices_domain());
            std::ptrdiff_t const n = domain.extents().template get<GridX>();
            SpatialDomain::discrete_element_type const
                    wrapped((static_cast<std::ptrdiff_t>(element.template uid<GridX>()) - 1 + n) % n
                                    + 1,
                            (static_cast<std::ptrdiff_t>(element.template uid<GridY>()) - 1 + n) % n
                                    + 1,
                            (static_cast<std::ptrdiff_t>(element.template uid<GridZ>()) - 1 + n) % n
                                    + 1);
            return sil::tensor::detail::pointwise_tensor<Tensor>(
                    [&](auto component) { return field.mem(wrapped, component); });
        }
    };

    // NVCC requires methods enclosing extended device lambdas to be public.
    void pack(Field source, Field target)
    {
        Kokkos::parallel_for(
                "pack_cochain",
                m_grid.cells,
                KOKKOS_LAMBDA(int i) {
                    for (int c = 0; c < 3; ++c)
                        target(0, 3 * i + c) = source(c, i);
                });
    }
    void unpack(Field source, Field target)
    {
        Kokkos::parallel_for(
                "unpack_cochain",
                m_grid.cells,
                KOKKOS_LAMBDA(int i) {
                    for (int c = 0; c < 3; ++c)
                        target(c, i) = source(0, 3 * i + c);
                });
    }

private:
    KOKKOS_FUNCTION static double axial(TwoFormTensor field, int i, int a, int n)
    {
        SpatialDomain::discrete_element_type const
                elem(i / (n * n) + 1, (i / n) % n + 1, i % n + 1);
        if (a == 0)
            return field(elem, field.accessor().template access_element<Y, Z>());
        if (a == 1)
            return -field(elem, field.accessor().template access_element<X, Z>());
        return field(elem, field.accessor().template access_element<X, Y>());
    }

public:
    explicit VorticityDEC(StructuredGrid& grid)
        : m_grid(grid)
        , m_one("compatible_one", 1, 3 * grid.cells)
        , m_two("compatible_two", 1, 3 * grid.cells)
        , m_two_output("compatible_two_output", 1, 3 * grid.cells)
        , m_physical_one("physical_edges", 1, 3 * grid.cells)
        , m_physical_two("physical_faces", 1, 3 * grid.cells)
        , m_three("closure", 1, grid.cells)
        , m_edges("transport_edge_values", 3, grid.cells)
    {
        m_laplacian = std::make_unique<TwoLaplacian>(
                Kokkos::DefaultExecutionSpace(),
                tensor<TwoForm>(m_two_output),
                tensor<TwoForm>(m_two),
                grid.euclidean_metric(),
                grid.geometry(),
                Periodic {},
                Periodic {});
        m_codifferential = std::make_unique<TwoCodifferential>(
                Kokkos::DefaultExecutionSpace(),
                tensor<TwoForm>(m_two),
                grid.euclidean_metric(),
                grid.geometry(),
                Periodic {});
        m_one_codifferential = std::make_unique<OneCodifferential>(
                Kokkos::DefaultExecutionSpace(),
                tensor<Vector>(m_one),
                grid.euclidean_metric(),
                grid.geometry(),
                Periodic {});
        Kokkos::fence("vorticity_hodge_setup_complete");
    }
    // Physical vector components supplied at their respective forward edges.
    void reduce_edges(Field edge_values, Field cochain)
    {
        pack(edge_values, m_physical_one);
        VectorTensor const physical = tensor<Vector>(m_physical_one),
                           reduced = tensor<Vector>(m_one);
        PositionTensor const geometry = m_grid.geometry();
        ddc::parallel_for_each(
                Kokkos::DefaultExecutionSpace(),
                m_grid.domain(),
                KOKKOS_LAMBDA(SpatialDomain::discrete_element_type elem) {
                    sil::exterior::Reduction<
                            ddc::TypeSeq<Vector>,
                            PositionTensor,
                            SpatialDomain::discrete_element_type>::
                    operator()(reduced[elem], physical[elem], geometry, elem);
                });
        unpack(m_one, cochain);
    }
    void reduce_nodes(Field values, Field cochain)
    {
        Field const edges = m_edges;
        int const n = m_grid.n;
        Kokkos::parallel_for(
                "transport_to_edges",
                m_grid.cells,
                KOKKOS_LAMBDA(int i) {
                    for (int a = 0; a < 3; ++a)
                        edges(a, i) = 0.5
                                      * (values(a, i)
                                         + values(a, StructuredGrid::neighbor(i, a, 1, n)));
                });
        reduce_edges(edges, cochain);
    }
    // Physical axial components sampled at forward face centres.
    void reduce_faces(Field values, Field cochain)
    {
        Field const storage = m_physical_two;
        Kokkos::parallel_for(
                "place_axial_faces",
                m_grid.cells,
                KOKKOS_LAMBDA(int i) {
                    storage(0, 3 * i) = values(2, i);
                    storage(0, 3 * i + 1) = -values(1, i);
                    storage(0, 3 * i + 2) = values(0, i);
                });
        TwoFormTensor const physical = tensor<TwoForm>(m_physical_two),
                            reduced = tensor<TwoForm>(m_two);
        PositionTensor const geometry = m_grid.geometry();
        ddc::parallel_for_each(
                Kokkos::DefaultExecutionSpace(),
                m_grid.domain(),
                KOKKOS_LAMBDA(SpatialDomain::discrete_element_type elem) {
                    sil::exterior::Reduction<
                            ddc::TypeSeq<sil::tensor::Covariant<CurlDirection>, Vector>,
                            PositionTensor,
                            SpatialDomain::discrete_element_type>::
                    operator()(reduced[elem], physical[elem], geometry, elem);
                });
        unpack(m_two, cochain);
    }
    void edge_values(Field cochain, Field output)
    {
        pack(cochain, m_one);
        VectorTensor const reduced = tensor<Vector>(m_one),
                           physical = tensor<Vector>(m_physical_one);
        m_grid.reconstruct(physical, reduced);
        unpack(m_physical_one, output);
    }
    void face_values(Field cochain, Field output)
    {
        pack(cochain, m_two);
        TwoFormTensor const reduced = tensor<TwoForm>(m_two),
                            physical = tensor<TwoForm>(m_physical_two);
        m_grid.reconstruct(physical, reduced);
        int const n = m_grid.n;
        Kokkos::parallel_for(
                "physical_axial_faces",
                m_grid.cells,
                KOKKOS_LAMBDA(int i) {
                    for (int a = 0; a < 3; ++a)
                        output(a, i) = axial(physical, i, a, n);
                });
    }
    void exterior_curl(Field one_cochain, Field two_cochain)
    {
        pack(one_cochain, m_one);
        sil::exterior::deriv<sil::tensor::Covariant<CurlDirection>, Vector>(
                Kokkos::DefaultExecutionSpace(),
                tensor<TwoForm>(m_two),
                tensor<Vector>(m_one),
                Periodic {});
        unpack(m_two, two_cochain);
    }
    void codifferential(Field two_cochain, Field one_cochain)
    {
        pack(two_cochain, m_two);
        (*m_codifferential)(tensor<Vector>(m_one), tensor<TwoForm>(m_two));
        unpack(m_one, one_cochain);
    }
    void positive_laplacian(Field input, Field output)
    {
        pack(input, m_two);
        (*m_laplacian)(tensor<TwoForm>(m_two_output), tensor<TwoForm>(m_two));
        unpack(m_two_output, output);
    }
    void divergence(Field one_cochain, Field output)
    {
        pack(one_cochain, m_one);
        (*m_one_codifferential)(m_grid.scalar(output), tensor<Vector>(m_one));
        Kokkos::parallel_for(
                "physical_cochain_divergence",
                m_grid.cells,
                KOKKOS_LAMBDA(int i) { output(0, i) *= -1; });
    }
    void velocity(Field cochain, Field output)
    {
        pack(cochain, m_one);
        VectorTensor const reduced = tensor<Vector>(m_one),
                           physical = tensor<Vector>(m_physical_one);
        m_grid.reconstruct(physical, reduced);
        Field const edge = m_physical_one;
        int const n = m_grid.n;
        Kokkos::parallel_for(
                "compatible_velocity_to_nodes",
                m_grid.cells,
                KOKKOS_LAMBDA(int i) {
                    for (int a = 0; a < 3; ++a)
                        output(a, i) = 0.5
                                       * (edge(0, 3 * i + a)
                                          + edge(0, 3 * StructuredGrid::neighbor(i, a, -1, n) + a));
                });
    }
    void axial_nodes(Field cochain, Field output)
    {
        pack(cochain, m_two);
        TwoFormTensor const reduced = tensor<TwoForm>(m_two),
                            physical = tensor<TwoForm>(m_physical_two);
        m_grid.reconstruct(physical, reduced);
        int const n = m_grid.n;
        Kokkos::parallel_for(
                "axial_faces_to_nodes",
                m_grid.cells,
                KOKKOS_LAMBDA(int i) {
                    for (int a = 0; a < 3; ++a) {
                        int const b = (a + 1) % 3, c = (a + 2) % 3;
                        int const ib = StructuredGrid::neighbor(i, b, -1, n),
                                  ic = StructuredGrid::neighbor(i, c, -1, n);
                        int const ibc = StructuredGrid::neighbor(ib, c, -1, n);
                        output(a, i) = 0.25
                                       * (axial(physical, i, a, n) + axial(physical, ib, a, n)
                                          + axial(physical, ic, a, n) + axial(physical, ibc, a, n));
                    }
                });
    }
    double closure_rms(Field two_cochain)
    {
        pack(two_cochain, m_two);
        sil::exterior::deriv<sil::tensor::Covariant<VolumeDirection>, TwoForm>(
                Kokkos::DefaultExecutionSpace(),
                tensor<ThreeForm>(m_three),
                tensor<TwoForm>(m_two),
                Periodic {});
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
