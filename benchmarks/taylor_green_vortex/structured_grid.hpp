// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#pragma once

#include <cstddef>
#include <memory>
#include <numbers>

#include <ddc/ddc.hpp>

#include <similie/exterior/exterior.hpp>
#include <similie/tensor/scalar_extrapolation_rules.hpp>
#include <similie/tensor/tensor_indices/identity_tensor.hpp>

#include "configuration.hpp"
#include "grid_types.hpp"
#include "laplacian_mode.hpp"

namespace taylor_green_vortex {
/** Benchmark storage and field placement, delegating all differential operators.
 * Geometry and Hodge stars are computed by exterior, never supplied as h powers.
 * Periodic cochain samples wrap; geometry is unwrapped, including a halo, so the
 * seam never becomes an edge of length 2*pi-h. Node/edge/face arithmetic averages
 * are field-placement operations, not alternative differential stencils.
 * \important This operator and documentation is fully AI-generated.
 */
class StructuredGrid
{
    using VectorCodifferential = sil::exterior::StagedOperator<
            sil::exterior::Codifferential<
                    MetricIndex,
                    Vector,
                    Vector,
                    VectorTensor,
                    MetricTensor,
                    PositionTensor>,
            typename sil::exterior::Codifferential<
                    MetricIndex,
                    Vector,
                    Vector,
                    VectorTensor,
                    MetricTensor,
                    PositionTensor>::staged_storage_type::
                    template rebind<Kokkos::DefaultExecutionSpace>>;
    template <template <class...> class Coefficients = sil::tensor::TensorFullIndex>
    using OneReconstruction = sil::exterior::StagedOperator<sil::exterior::Reconstruction<
            ddc::TypeSeq<Vector>,
            PositionTensor,
            typename VectorTensor::non_indices_domain_t::discrete_element_type,
            sil::exterior::CellComplex::Primal,
            Coefficients>>;
    template <template <class...> class Coefficients = sil::tensor::TensorFullIndex>
    using OneReduction = sil::exterior::StagedOperator<sil::exterior::Reduction<
            ddc::TypeSeq<Vector>,
            PositionTensor,
            typename VectorTensor::non_indices_domain_t::discrete_element_type,
            sil::exterior::CellComplex::Primal,
            Coefficients>>;
    template <template <class...> class Coefficients = sil::tensor::TensorFullIndex>
    using TwoReconstruction = sil::exterior::StagedOperator<sil::exterior::Reconstruction<
            ddc::TypeSeq<sil::tensor::Covariant<CurlDirection>, Vector>,
            PositionTensor,
            typename TwoFormTensor::non_indices_domain_t::discrete_element_type,
            sil::exterior::CellComplex::Primal,
            Coefficients>>;
    template <template <class...> class Coefficients = sil::tensor::TensorFullIndex>
    using TwoReduction = sil::exterior::StagedOperator<sil::exterior::Reduction<
            ddc::TypeSeq<sil::tensor::Covariant<CurlDirection>, Vector>,
            PositionTensor,
            typename TwoFormTensor::non_indices_domain_t::discrete_element_type,
            sil::exterior::CellComplex::Primal,
            Coefficients>>;
    SpatialDomain m_domain;
    SpatialDomain m_geometry_domain;
    Field m_position, m_metric, m_oneform, m_pointwise, m_twoform, m_face_pointwise, m_scalar;
    std::unique_ptr<LaplacianOperator<ScalarTensor>> m_laplacian;
    std::unique_ptr<VectorCodifferential> m_codifferential;
    std::unique_ptr<OneReconstruction<>> m_one_reconstruction;
    std::unique_ptr<OneReconstruction<sil::tensor::TensorDiagonalIndex>>
            m_one_diagonal_reconstruction;
    std::unique_ptr<OneReduction<>> m_one_reduction;
    std::unique_ptr<OneReduction<sil::tensor::TensorDiagonalIndex>> m_one_diagonal_reduction;
    std::unique_ptr<TwoReconstruction<>> m_two_reconstruction;
    std::unique_ptr<TwoReconstruction<sil::tensor::TensorDiagonalIndex>>
            m_two_diagonal_reconstruction;
    std::unique_ptr<TwoReduction<>> m_two_reduction;
    std::unique_ptr<TwoReduction<sil::tensor::TensorDiagonalIndex>> m_two_diagonal_reduction;

public:
    // Device lambda instantiations in exterior must be able to name this type.
    struct Periodic
    {
        template <class Tensor, class Element>
        KOKKOS_FUNCTION auto operator()(Tensor const& field, Element element) const
        {
            SpatialDomain const domain(field.non_indices_domain());
            std::ptrdiff_t const n = domain.extents().template get<GridX>();
            ddc::DiscreteElement<GridX, GridY, GridZ> const
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

private:
    PositionTensor position() const
    {
        [[maybe_unused]] sil::tensor::TensorAccessor<PositionIndex> accessor;
        ddc::DiscreteDomain<GridX, GridY, GridZ, PositionIndex> const
                domain(m_geometry_domain, accessor.domain());
        return PositionTensor(
                ddc::ChunkSpan<
                        double,
                        ddc::DiscreteDomain<GridX, GridY, GridZ, PositionIndex>,
                        Kokkos::layout_right,
                        Field::memory_space>(m_position.data(), domain));
    }
    MetricTensor metric() const
    {
        [[maybe_unused]] sil::tensor::TensorAccessor<MetricIndex> accessor;
        ddc::DiscreteDomain<GridX, GridY, GridZ, MetricIndex> const
                domain(m_domain, accessor.domain());
        return MetricTensor(
                ddc::ChunkSpan<
                        double,
                        ddc::DiscreteDomain<GridX, GridY, GridZ, MetricIndex>,
                        Kokkos::layout_right,
                        Field::memory_space>(m_metric.data(), domain));
    }
    VectorTensor vector(Field const& field) const
    {
        [[maybe_unused]] sil::tensor::TensorAccessor<Vector> accessor;
        ddc::DiscreteDomain<GridX, GridY, GridZ, Vector> const domain(m_domain, accessor.domain());
        return VectorTensor(
                ddc::ChunkSpan<
                        double,
                        ddc::DiscreteDomain<GridX, GridY, GridZ, Vector>,
                        Kokkos::layout_right,
                        Field::memory_space>(field.data(), domain));
    }
    TwoFormTensor twoform(Field const& field) const
    {
        [[maybe_unused]] sil::tensor::TensorAccessor<TwoForm> accessor;
        ddc::DiscreteDomain<GridX, GridY, GridZ, TwoForm> const domain(m_domain, accessor.domain());
        return TwoFormTensor(
                ddc::ChunkSpan<
                        double,
                        ddc::DiscreteDomain<GridX, GridY, GridZ, TwoForm>,
                        Kokkos::layout_right,
                        Field::memory_space>(field.data(), domain));
    }
    KOKKOS_FUNCTION static double face_curl(
            TwoFormTensor const& field,
            int index,
            int component,
            int nodes)
    {
        SpatialDomain::discrete_element_type const
                elem(index / (nodes * nodes) + 1, (index / nodes) % nodes + 1, index % nodes + 1);
        if (component == 0)
            return field(elem, field.accessor().template access_element<Y, Z>());
        if (component == 1)
            return -field(elem, field.accessor().template access_element<X, Z>());
        return field(elem, field.accessor().template access_element<X, Y>());
    }

public:
    // NVCC requires methods enclosing extended device lambdas to be public.
    void reduce_one_form()
    {
        VectorTensor const form = vector(m_pointwise), cochain = vector(m_oneform);
        reduce(cochain, form);
    }
    void negative_reduced_codifferential(Field const& output, int component)
    {
        reduce_one_form();
        (*m_codifferential)(scalar(output, component), vector(m_oneform));
        Kokkos::parallel_for(
                "codifferential_to_divergence",
                cells,
                KOKKOS_LAMBDA(int i) { output(component, i) *= -1; });
    }
    void reconstruct_gradient(Field const& input, int component)
    {
        VectorTensor const cochain = vector(m_oneform), pointwise = vector(m_pointwise);
        sil::exterior::deriv<Vector, Scalar>(
                Kokkos::DefaultExecutionSpace(),
                cochain,
                scalar(input, component),
                Periodic {});
        reconstruct(pointwise, cochain);
    }

    Configuration const configuration;
    int const n, cells;
    double const h;
    SpatialDomain domain() const
    {
        return m_domain;
    }
    PositionTensor geometry() const
    {
        return position();
    }
    MetricTensor euclidean_metric() const
    {
        return metric();
    }
    explicit StructuredGrid(int nodes, Configuration const& selected = {})
        : m_domain(
                  ddc::DiscreteElement<GridX, GridY, GridZ>(1, 1, 1),
                  ddc::DiscreteVector<GridX, GridY, GridZ>(nodes, nodes, nodes))
        , m_geometry_domain(
                  ddc::DiscreteElement<GridX, GridY, GridZ>(0, 0, 0),
                  ddc::DiscreteVector<GridX, GridY, GridZ>(nodes + 2, nodes + 2, nodes + 2))
        , m_position("position_with_halo", 1, 3 * (nodes + 2) * (nodes + 2) * (nodes + 2))
        , m_metric("metric", 1, nodes * nodes * nodes)
        , m_oneform("one_cochain", 1, 3 * nodes * nodes * nodes)
        , m_pointwise("reconstructed_one_form", 1, 3 * nodes * nodes * nodes)
        , m_twoform("two_cochain", 1, 3 * nodes * nodes * nodes)
        , m_face_pointwise("reconstructed_two_form", 1, 3 * nodes * nodes * nodes)
        , m_scalar("scalar_operator_workspace", 1, nodes * nodes * nodes)
        , configuration(selected)
        , n(nodes)
        , cells(nodes * nodes * nodes)
        , h(2 * std::numbers::pi / nodes)
    {
        initialize_geometry();
        PositionTensor const geometry = position();
        if (configuration.reconstruction_mode() == 2) {
            m_one_diagonal_reconstruction
                    = std::make_unique<OneReconstruction<sil::tensor::TensorDiagonalIndex>>(
                            Kokkos::DefaultExecutionSpace(),
                            vector(m_pointwise),
                            vector(m_oneform),
                            geometry);
            m_two_diagonal_reconstruction
                    = std::make_unique<TwoReconstruction<sil::tensor::TensorDiagonalIndex>>(
                            Kokkos::DefaultExecutionSpace(),
                            twoform(m_face_pointwise),
                            twoform(m_twoform),
                            geometry);
        } else if (configuration.staged_reconstruction) {
            m_one_reconstruction = std::make_unique<OneReconstruction<>>(
                    Kokkos::DefaultExecutionSpace(),
                    vector(m_pointwise),
                    vector(m_oneform),
                    geometry);
            m_two_reconstruction = std::make_unique<TwoReconstruction<>>(
                    Kokkos::DefaultExecutionSpace(),
                    twoform(m_face_pointwise),
                    twoform(m_twoform),
                    geometry);
        }
        if (configuration.reduction_mode() == 2) {
            m_one_diagonal_reduction
                    = std::make_unique<OneReduction<sil::tensor::TensorDiagonalIndex>>(
                            Kokkos::DefaultExecutionSpace(),
                            vector(m_oneform),
                            vector(m_pointwise),
                            geometry);
            m_two_diagonal_reduction
                    = std::make_unique<TwoReduction<sil::tensor::TensorDiagonalIndex>>(
                            Kokkos::DefaultExecutionSpace(),
                            twoform(m_twoform),
                            twoform(m_face_pointwise),
                            geometry);
        } else if (configuration.staged_reduction) {
            m_one_reduction = std::make_unique<OneReduction<>>(
                    Kokkos::DefaultExecutionSpace(),
                    vector(m_oneform),
                    vector(m_pointwise),
                    geometry);
            m_two_reduction = std::make_unique<TwoReduction<>>(
                    Kokkos::DefaultExecutionSpace(),
                    twoform(m_twoform),
                    twoform(m_face_pointwise),
                    geometry);
        }
        m_laplacian = make_scalar_laplacian(
                configuration,
                scalar(m_scalar),
                scalar(m_scalar),
                metric(),
                geometry);
        m_codifferential = std::make_unique<VectorCodifferential>(
                Kokkos::DefaultExecutionSpace(),
                vector(m_oneform),
                metric(),
                geometry,
                Periodic {});
    }
    // An extended device lambda must also be enclosed by an addressable function.
    void initialize_geometry()
    {
        PositionTensor const geometry = position();
        double const spacing = h;
        ddc::parallel_for_each(
                Kokkos::DefaultExecutionSpace(),
                m_geometry_domain,
                KOKKOS_LAMBDA(SpatialDomain::discrete_element_type elem) {
                    geometry.mem(elem, ddc::DiscreteElement<PositionIndex>(0))
                            = -std::numbers::pi
                              + (static_cast<double>(elem.uid<GridX>()) - 1) * spacing;
                    geometry.mem(elem, ddc::DiscreteElement<PositionIndex>(1))
                            = -std::numbers::pi
                              + (static_cast<double>(elem.uid<GridY>()) - 1) * spacing;
                    geometry.mem(elem, ddc::DiscreteElement<PositionIndex>(2))
                            = -std::numbers::pi
                              + (static_cast<double>(elem.uid<GridZ>()) - 1) * spacing;
                });
        Kokkos::deep_copy(m_metric, 1.0);
    }
    ScalarTensor scalar(Field const& field, int component = 0) const
    {
        [[maybe_unused]] sil::tensor::TensorAccessor<Scalar> accessor;
        ddc::DiscreteDomain<GridX, GridY, GridZ, Scalar> const domain(m_domain, accessor.domain());
        return ScalarTensor(
                ddc::ChunkSpan<
                        double,
                        ddc::DiscreteDomain<GridX, GridY, GridZ, Scalar>,
                        Kokkos::layout_right,
                        Field::memory_space>(field.data() + component * cells, domain));
    }
    void reconstruct(VectorTensor const& output, VectorTensor const& input) const
    {
        if (m_one_diagonal_reconstruction)
            (*m_one_diagonal_reconstruction)(output, input);
        else if (m_one_reconstruction)
            (*m_one_reconstruction)(output, input);
        else {
            PositionTensor const geometry = position();
            bool const diagonal = configuration.diagonal_reconstruction;
            ddc::parallel_for_each(
                    "tgv_direct_reconstruction",
                    Kokkos::DefaultExecutionSpace(),
                    m_domain,
                    KOKKOS_LAMBDA(SpatialDomain::discrete_element_type element) {
                        if (diagonal) {
                            sil::exterior::Reconstruction<
                                    ddc::TypeSeq<Vector>,
                                    PositionTensor,
                                    SpatialDomain::discrete_element_type,
                                    sil::exterior::CellComplex::Primal,
                                    sil::tensor::TensorDiagonalIndex>::
                            operator()(output[element], input[element], geometry, element);
                        } else {
                            sil::exterior::Reconstruction<
                                    ddc::TypeSeq<Vector>,
                                    PositionTensor,
                                    SpatialDomain::discrete_element_type>::
                            operator()(output[element], input[element], geometry, element);
                        }
                    });
        }
    }
    void reduce(VectorTensor const& output, VectorTensor const& input) const
    {
        if (m_one_diagonal_reduction)
            (*m_one_diagonal_reduction)(output, input);
        else if (m_one_reduction)
            (*m_one_reduction)(output, input);
        else {
            PositionTensor const geometry = position();
            bool const diagonal = configuration.diagonal_reduction;
            ddc::parallel_for_each(
                    "tgv_direct_reduction",
                    Kokkos::DefaultExecutionSpace(),
                    m_domain,
                    KOKKOS_LAMBDA(SpatialDomain::discrete_element_type element) {
                        if (diagonal) {
                            sil::exterior::Reduction<
                                    ddc::TypeSeq<Vector>,
                                    PositionTensor,
                                    SpatialDomain::discrete_element_type,
                                    sil::exterior::CellComplex::Primal,
                                    sil::tensor::TensorDiagonalIndex>::
                            operator()(output[element], input[element], geometry, element);
                        } else {
                            sil::exterior::Reduction<
                                    ddc::TypeSeq<Vector>,
                                    PositionTensor,
                                    SpatialDomain::discrete_element_type>::
                            operator()(output[element], input[element], geometry, element);
                        }
                    });
        }
    }
    void reconstruct(TwoFormTensor const& output, TwoFormTensor const& input) const
    {
        if (m_two_diagonal_reconstruction)
            (*m_two_diagonal_reconstruction)(output, input);
        else if (m_two_reconstruction)
            (*m_two_reconstruction)(output, input);
        else {
            PositionTensor const geometry = position();
            bool const diagonal = configuration.diagonal_reconstruction;
            ddc::parallel_for_each(
                    "tgv_direct_reconstruction",
                    Kokkos::DefaultExecutionSpace(),
                    m_domain,
                    KOKKOS_LAMBDA(SpatialDomain::discrete_element_type element) {
                        if (diagonal) {
                            sil::exterior::Reconstruction<
                                    ddc::TypeSeq<sil::tensor::Covariant<CurlDirection>, Vector>,
                                    PositionTensor,
                                    SpatialDomain::discrete_element_type,
                                    sil::exterior::CellComplex::Primal,
                                    sil::tensor::TensorDiagonalIndex>::
                            operator()(output[element], input[element], geometry, element);
                        } else {
                            sil::exterior::Reconstruction<
                                    ddc::TypeSeq<sil::tensor::Covariant<CurlDirection>, Vector>,
                                    PositionTensor,
                                    SpatialDomain::discrete_element_type>::
                            operator()(output[element], input[element], geometry, element);
                        }
                    });
        }
    }
    void reduce(TwoFormTensor const& output, TwoFormTensor const& input) const
    {
        if (m_two_diagonal_reduction)
            (*m_two_diagonal_reduction)(output, input);
        else if (m_two_reduction)
            (*m_two_reduction)(output, input);
        else {
            PositionTensor const geometry = position();
            bool const diagonal = configuration.diagonal_reduction;
            ddc::parallel_for_each(
                    "tgv_direct_reduction",
                    Kokkos::DefaultExecutionSpace(),
                    m_domain,
                    KOKKOS_LAMBDA(SpatialDomain::discrete_element_type element) {
                        if (diagonal) {
                            sil::exterior::Reduction<
                                    ddc::TypeSeq<sil::tensor::Covariant<CurlDirection>, Vector>,
                                    PositionTensor,
                                    SpatialDomain::discrete_element_type,
                                    sil::exterior::CellComplex::Primal,
                                    sil::tensor::TensorDiagonalIndex>::
                            operator()(output[element], input[element], geometry, element);
                        } else {
                            sil::exterior::Reduction<
                                    ddc::TypeSeq<sil::tensor::Covariant<CurlDirection>, Vector>,
                                    PositionTensor,
                                    SpatialDomain::discrete_element_type>::
                            operator()(output[element], input[element], geometry, element);
                        }
                    });
        }
    }
    KOKKOS_FUNCTION static int neighbor(int index, int axis, int offset, int nodes)
    {
        int const stride = axis == 0 ? nodes * nodes : (axis == 1 ? nodes : 1);
        int const coordinate = (index / stride) % nodes;
        return index + (((coordinate + offset + nodes) % nodes) - coordinate) * stride;
    }
    void edge_derivative_component(
            Field const& input,
            int component,
            Field const& output,
            int out_component,
            int axis)
    {
        reconstruct_gradient(input, component);
        Field const gradient = m_pointwise;
        Kokkos::parallel_for(
                "sample_native_edge_gradient",
                cells,
                KOKKOS_LAMBDA(int i) { output(out_component, i) = gradient(0, 3 * i + axis); });
    }
    void nodal_derivative_component(
            Field const& input,
            int component,
            Field const& output,
            int out_component,
            int axis)
    {
        reconstruct_gradient(input, component);
        Field const gradient = m_pointwise;
        int const nodes = n;
        Kokkos::parallel_for(
                "edge_gradient_to_node",
                cells,
                KOKKOS_LAMBDA(int i) {
                    int const previous = neighbor(i, axis, -1, nodes);
                    output(out_component, i)
                            = 0.5 * (gradient(0, 3 * i + axis) + gradient(0, 3 * previous + axis));
                });
    }
    void negative_face_codifferential(
            Field const& flux,
            int component,
            Field const& output,
            int out_component,
            int axis)
    {
        Field const pointwise = m_pointwise;
        Kokkos::parallel_for(
                "place_face_flux",
                cells,
                KOKKOS_LAMBDA(int i) {
                    for (int a = 0; a < 3; ++a)
                        pointwise(0, 3 * i + a) = a == axis ? flux(component, i) : 0;
                });
        negative_reduced_codifferential(output, out_component);
    }
    void negative_reduced_codifferential(Field const& velocity, Field const& output)
    {
        Field const placed = m_pointwise;
        int const nodes = n;
        Kokkos::parallel_for(
                "velocity_to_dual_faces",
                cells,
                KOKKOS_LAMBDA(int i) {
                    for (int a = 0; a < 3; ++a)
                        placed(0, 3 * i + a)
                                = 0.5 * (velocity(a, i) + velocity(a, neighbor(i, a, 1, nodes)));
                });
        negative_reduced_codifferential(output, 0);
    }

    void laplacian(Field const& input, int component, Field const& output, int out_component)
    {
        m_laplacian->apply(scalar(output, out_component), scalar(input, component));
        // exterior uses the positive Hodge Laplacian; viscosity uses its negative.
        Kokkos::parallel_for(
                "hodge_laplacian_to_diffusion",
                cells,
                KOKKOS_LAMBDA(int i) { output(out_component, i) *= -1; });
    }
    void reconstructed_derivative_one_form(Field const& velocity, Field const& output)
    {
        Field const placed = m_pointwise;
        int const nodes = n;
        Kokkos::parallel_for(
                "velocity_to_edges",
                cells,
                KOKKOS_LAMBDA(int i) {
                    for (int a = 0; a < 3; ++a)
                        placed(0, 3 * i + a)
                                = 0.5 * (velocity(a, i) + velocity(a, neighbor(i, a, 1, nodes)));
                });
        reduce_one_form();
        VectorTensor const cochain = vector(m_oneform);
        TwoFormTensor const rotation = twoform(m_twoform),
                            reconstructed = twoform(m_face_pointwise);
        sil::exterior::deriv<
                sil::tensor::Covariant<CurlDirection>,
                Vector>(Kokkos::DefaultExecutionSpace(), rotation, cochain, Periodic {});
        reconstruct(reconstructed, rotation);
        Kokkos::parallel_for(
                "vorticity_faces_to_nodes",
                cells,
                KOKKOS_LAMBDA(int i) {
                    for (int a = 0; a < 3; ++a) {
                        int const b = (a + 1) % 3, c = (a + 2) % 3;
                        int const ib = neighbor(i, b, -1, nodes), ic = neighbor(i, c, -1, nodes),
                                  ibc = neighbor(ib, c, -1, nodes);
                        output(a, i) = 0.25
                                       * (face_curl(reconstructed, i, a, nodes)
                                          + face_curl(reconstructed, ib, a, nodes)
                                          + face_curl(reconstructed, ic, a, nodes)
                                          + face_curl(reconstructed, ibc, a, nodes));
                    }
                });
    }
};
} // namespace taylor_green_vortex
