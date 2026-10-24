// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#pragma once

#include <cstddef>
#include <memory>
#include <numbers>

#include <ddc/ddc.hpp>

#include <similie/exterior/exterior.hpp>
#include <similie/tensor/periodic_extrapolation_rule.hpp>
#include <similie/tensor/scalar_extrapolation_rules.hpp>
#include <similie/tensor/tensor_indices/identity_tensor.hpp>

#include "configuration.hpp"
#include "grid_types.hpp"
#include "laplacian_mode.hpp"

namespace taylor_green_vortex {
/** Benchmark storage and field placement, delegating all differential operators.
 * Geometry and Hodge stars are computed by exterior, never supplied as h powers.
 * Periodic cochain samples wrap; geometry is unwrapped, including a halo, so the
 * seam never becomes an edge of length 2*pi-h. Cochains and cell-local physical
 * coefficients are connected only by the library reduction/reconstruction maps.
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
    Field m_position, m_metric, m_oneform, m_pointwise, m_twoform, m_face_pointwise, m_scalar,
            m_gradient_cochains, m_gradient_coefficients;
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
                sil::tensor::PeriodicExtrapolationRule {});
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
        , m_gradient_cochains("velocity_gradient_cochains", 1, 9 * nodes * nodes * nodes)
        , m_gradient_coefficients("velocity_gradient_coefficients", 1, 9 * nodes * nodes * nodes)
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
                sil::tensor::PeriodicExtrapolationRule {});
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
    template <class Operation, class Output, class Input>
    void reconstruct_batch(Operation const& operation, Output const& output, Input const& input)
            const
    {
        PositionTensor const geometry = position();
        using BatchDomain = decltype(ddc::remove_dims_of<Vector>(output.domain()));
        ddc::parallel_for_each(
                sil::misc::Specialization<Operation, sil::exterior::StagedOperator>
                        ? "similie_apply_staged_local_operator"
                        : "tgv_direct_reconstruction",
                Kokkos::DefaultExecutionSpace(),
                ddc::remove_dims_of<Vector>(output.domain()),
                KOKKOS_LAMBDA(typename BatchDomain::discrete_element_type element) {
                    operation(
                            output[element],
                            input[element],
                            geometry,
                            SpatialDomain::discrete_element_type(element));
                });
    }

    template <class Output, class Input>
    void reconstruct(Output const& output, Input const& input) const
    {
        if (m_one_diagonal_reconstruction)
            reconstruct_batch(*m_one_diagonal_reconstruction, output, input);
        else if (m_one_reconstruction)
            reconstruct_batch(*m_one_reconstruction, output, input);
        else if (configuration.diagonal_reconstruction)
            reconstruct_batch(
                    sil::exterior::Reconstruction<
                            ddc::TypeSeq<Vector>,
                            PositionTensor,
                            SpatialDomain::discrete_element_type,
                            sil::exterior::CellComplex::Primal,
                            sil::tensor::TensorDiagonalIndex> {},
                    output,
                    input);
        else
            reconstruct_batch(
                    sil::exterior::Reconstruction<
                            ddc::TypeSeq<Vector>,
                            PositionTensor,
                            SpatialDomain::discrete_element_type> {},
                    output,
                    input);
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

    void gradient(Field const& input, Field const& output, int component = 0)
    {
        reconstruct_gradient(input, component);
        Field const coefficients = m_pointwise;
        Kokkos::parallel_for(
                "unpack_gradient_coefficients",
                Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {cells, 3}),
                KOKKOS_LAMBDA(int i, int direction) {
                    output(direction, i) = coefficients(0, 3 * i + direction);
                });
    }

    // A velocity is a vector-valued zero-form. Its vector index is a spectator
    // of the exterior derivative, which acts on the complete field at once.
    void velocity_gradient(Field const& input, Field const& output, int first_component = 0)
    {
        ddc::DiscreteDomain<VelocityComponent, GridX, GridY, GridZ, Scalar> const input_domain(
                sil::tensor::TensorAccessor<VelocityComponent>::domain(),
                m_domain,
                sil::tensor::TensorAccessor<Scalar>::domain());
        ddc::DiscreteDomain<VelocityComponent, GridX, GridY, GridZ, Vector> const gradient_domain(
                sil::tensor::TensorAccessor<VelocityComponent>::domain(),
                m_domain,
                sil::tensor::TensorAccessor<Vector>::domain());
        sil::tensor::Tensor const values(
                ddc::ChunkSpan<
                        double,
                        ddc::DiscreteDomain<VelocityComponent, GridX, GridY, GridZ, Scalar>,
                        Kokkos::layout_right,
                        Field::memory_space>(input.data() + first_component * cells, input_domain));
        sil::tensor::Tensor const cochains(
                ddc::ChunkSpan<
                        double,
                        ddc::DiscreteDomain<VelocityComponent, GridX, GridY, GridZ, Vector>,
                        Kokkos::layout_right,
                        Field::memory_space>(m_gradient_cochains.data(), gradient_domain));
        sil::tensor::Tensor const coefficients(
                ddc::ChunkSpan<
                        double,
                        ddc::DiscreteDomain<VelocityComponent, GridX, GridY, GridZ, Vector>,
                        Kokkos::layout_right,
                        Field::memory_space>(m_gradient_coefficients.data(), gradient_domain));
        sil::exterior::deriv<Vector, Scalar>(
                Kokkos::DefaultExecutionSpace(),
                cochains,
                values,
                sil::tensor::PeriodicExtrapolationRule {});
        reconstruct(coefficients, cochains);
        Field const storage = m_gradient_coefficients;
        int const count = cells;
        Kokkos::parallel_for(
                "unpack_velocity_gradient",
                Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {3, cells, 3}),
                KOKKOS_LAMBDA(int component, int i, int direction) {
                    output(3 * component + direction, i)
                            = storage(0, 3 * (component * count + i) + direction);
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
        Kokkos::parallel_for(
                "pack_one_form_coefficients",
                Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {cells, 3}),
                KOKKOS_LAMBDA(int i, int component) {
                    placed(0, 3 * i + component) = velocity(component, i);
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
        Kokkos::parallel_for(
                "pack_velocity_coefficients",
                Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {cells, 3}),
                KOKKOS_LAMBDA(int i, int component) {
                    placed(0, 3 * i + component) = velocity(component, i);
                });
        reduce_one_form();
        VectorTensor const cochain = vector(m_oneform);
        TwoFormTensor const rotation = twoform(m_twoform),
                            reconstructed = twoform(m_face_pointwise);
        sil::exterior::deriv<sil::tensor::Covariant<CurlDirection>, Vector>(
                Kokkos::DefaultExecutionSpace(),
                rotation,
                cochain,
                sil::tensor::PeriodicExtrapolationRule {});
        reconstruct(reconstructed, rotation);
        ddc::DiscreteDomain<GridX, GridY, GridZ, sil::tensor::Covariant<VolumeDirection>> const
                axial_domain(
                        m_domain,
                        sil::tensor::TensorAccessor<
                                sil::tensor::Covariant<VolumeDirection>>::domain());
        sil::tensor::Tensor const axial(
                ddc::ChunkSpan<
                        double,
                        ddc::DiscreteDomain<
                                GridX,
                                GridY,
                                GridZ,
                                sil::tensor::Covariant<VolumeDirection>>,
                        Kokkos::layout_right,
                        Field::memory_space>(placed.data(), axial_domain));
        MetricTensor const metric = euclidean_metric();
        ddc::parallel_for_each(
                "reconstruct_axial_curl",
                Kokkos::DefaultExecutionSpace(),
                m_domain,
                KOKKOS_LAMBDA(SpatialDomain::discrete_element_type element) {
                    sil::exterior::LeviCivitaContinuousHodgeStar<
                            ddc::TypeSeq<
                                    sil::tensor::Contravariant<CurlDirection>,
                                    sil::tensor::Contravariant<Direction>>,
                            ddc::TypeSeq<sil::tensor::Covariant<VolumeDirection>>,
                            MetricTensor,
                            SpatialDomain::discrete_element_type>::
                    operator()(axial[element], reconstructed[element], metric, element);
                });
        Kokkos::parallel_for(
                "unpack_curl_coefficients",
                Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {cells, 3}),
                KOKKOS_LAMBDA(int i, int component) {
                    output(component, i) = placed(0, 3 * i + component);
                });
    }
};
} // namespace taylor_green_vortex
