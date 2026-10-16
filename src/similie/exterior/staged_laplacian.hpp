// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#pragma once

#include "laplacian.hpp"

namespace sil::exterior {
namespace detail {

template <class Memory>
struct LaplacianStorage
{
    Kokkos::View<std::size_t*, Memory> rows, columns;
    Kokkos::View<double*, Memory> values, offset;
    Kokkos::View<double**, Kokkos::LayoutRight, Memory> constant_response;
};

template <
        class Operator,
        class Index,
        class Cochain,
        class Exec,
        class Tensor,
        class Geometry,
        class Primal,
        class Dual>
LaplacianStorage<typename Tensor::memory_space> assemble_laplacian(
        Exec const& exec,
        Tensor output,
        Tensor input,
        Geometry geometry,
        Primal primal,
        Dual dual)
{
    using Memory = typename Tensor::memory_space;
    using Domain = typename Tensor::non_indices_domain_t;
    LaplacianStorage<Memory> storage;
    constexpr std::size_t capacity = [] {
        std::size_t count = Cochain::mem_size();
        for (std::size_t axis = 0; axis < Index::size(); ++axis)
            count *= 3;
        return count;
    }();
    std::size_t const row_count = output.non_indices_domain().size() * Cochain::mem_size();
    Kokkos::View<std::size_t**, Kokkos::LayoutRight, Memory>
            columns("laplacian_candidates", row_count, capacity);
    Kokkos::View<double**, Kokkos::LayoutRight, Memory>
            values("laplacian_weights", row_count, capacity);
    Kokkos::View<std::size_t*, Memory> counts("laplacian_counts", row_count);
    Kokkos::View<int, Memory> overflow("laplacian_stencil_overflow");
    storage.rows = decltype(storage.rows)("laplacian_rows", row_count + 1);
    storage.offset = decltype(storage.offset)("laplacian_offset", row_count);
    storage.constant_response = decltype(storage.constant_response)(
            "laplacian_constant_response",
            row_count,
            Cochain::mem_size());
    auto const offset = storage.offset;
    auto const constant_response = storage.constant_response;
    Domain const domain = input.non_indices_domain();
    Kokkos::parallel_for(
            "similie_assemble_laplacian_weights",
            Kokkos::RangePolicy<Exec>(exec, 0, row_count),
            KOKKOS_LAMBDA(std::size_t row) {
                auto const element = detail::laplacian_element(domain, row / Cochain::mem_size());
                ddc::DiscreteElement<Cochain> const target(row % Cochain::mem_size());
                std::size_t count = 0;
                auto baseline = Operator::operator()(
                        geometry,
                        input,
                        [&](auto sample, auto component) {
                            std::size_t const column = detail::laplacian_node_id(domain, sample)
                                                               * Cochain::mem_size()
                                                       + component.template uid<Cochain>();
                            std::size_t entry = 0;
                            for (; entry < count; ++entry)
                                if (columns(row, entry) == column)
                                    break;
                            if (entry == count) {
                                if (count < capacity)
                                    columns(row, count++) = column;
                                else
                                    Kokkos::atomic_exchange(&overflow(), 1);
                            }
                            return 0.0;
                        },
                        element,
                        primal,
                        dual);
                offset(row) = baseline.mem(target);
                for (std::size_t source = 0; source < Cochain::mem_size(); ++source) {
                    auto constant = Operator::operator()(
                            geometry,
                            input,
                            [&](auto, auto component) {
                                return component.template uid<Cochain>() == source ? 1.0 : 0.0;
                            },
                            element,
                            primal,
                            dual);
                    constant_response(row, source) = constant.mem(target) - offset(row);
                }
                std::size_t nonzero = 0;
                for (std::size_t entry = 0; entry < count; ++entry) {
                    std::size_t const column = columns(row, entry);
                    double const weight
                            = Operator::
                                      value(geometry, input, element, target, column, primal, dual)
                              - offset(row);
                    if (weight != 0.0) {
                        columns(row, nonzero) = column;
                        values(row, nonzero++) = weight;
                    }
                }
                counts(row) = nonzero;
            });
    int exceeded;
    Kokkos::deep_copy(exceeded, overflow);
    if (exceeded)
        throw std::invalid_argument("Laplacian boundary rule exceeds the local stencil capacity");
    auto const rows = storage.rows;
    Kokkos::parallel_scan(
            "similie_laplacian_row_offsets",
            Kokkos::RangePolicy<Exec>(exec, 0, row_count + 1),
            KOKKOS_LAMBDA(std::size_t row, std::size_t& total, bool final) {
                if (final)
                    rows(row) = total;
                if (row < row_count)
                    total += counts(row);
            });
    std::size_t nonzeros;
    Kokkos::deep_copy(nonzeros, Kokkos::subview(rows, row_count));
    storage.columns = decltype(storage.columns)("laplacian_columns", nonzeros);
    storage.values = decltype(storage.values)("laplacian_values", nonzeros);
    auto const stored_columns = storage.columns;
    auto const stored_values = storage.values;
    Kokkos::parallel_for(
            "similie_store_laplacian_weights",
            Kokkos::RangePolicy<Exec>(exec, 0, row_count),
            KOKKOS_LAMBDA(std::size_t row) {
                for (std::size_t entry = 0; entry < counts(row); ++entry) {
                    stored_columns(rows(row) + entry) = columns(row, entry);
                    stored_values(rows(row) + entry) = values(row, entry);
                }
            });
    exec.fence();
    return storage;
}

} // namespace detail

/** Cache the complete cochain Laplacian as sparse rows for repeated application.
 * \important This operator and documentation is fully AI-generated.
 * Construction uses Laplacian::value() and retains all nonzero coefficients,
 * including cross-component terms and boundary contributions. Application is a
 * single sparse stencil kernel. Geometry and linear or affine boundary rules
 * are fixed at construction; rebuild the object when either changes. Nonlinear
 * boundary rules are not supported. Input and output domains must remain equal
 * to the construction domains, and the allocations must not overlap.
 * Each row is evaluated using differences from the cochain at its center.
 * The response to each constant component is cached separately. This equivalent
 * form preserves constant null modes without cancellation of stencil weights.
 */
template <
        tensor::TensorIndex MetricIndex,
        tensor::TensorNatIndex Index,
        tensor::TensorIndex Cochain,
        misc::Specialization<tensor::Tensor> Tensor,
        misc::Specialization<tensor::Tensor> Metric,
        misc::Specialization<tensor::Tensor> Position,
        class Exec>
class StagedLaplacian
{
    using Memory = typename Tensor::memory_space;
    using Domain = typename Tensor::non_indices_domain_t;
    using Operator = Laplacian<MetricIndex, Index, Cochain, Tensor, Metric, Position, Exec>;
    Exec m_exec;
    Domain m_input_domain, m_output_domain;
    Kokkos::View<std::size_t*, Memory> m_rows, m_columns;
    Kokkos::View<double*, Memory> m_values, m_offset;
    Kokkos::View<double**, Kokkos::LayoutRight, Memory> m_constant_response;

public:
    template <
            class Primal = tensor::ClampExtrapolationRule,
            class Dual = tensor::ZeroExtrapolationRule>
    StagedLaplacian(
            Exec const& exec,
            Tensor output,
            Tensor input,
            Metric metric,
            Position position,
            Primal primal = {},
            Dual dual = {})
        : m_exec(exec)
        , m_input_domain(input.non_indices_domain())
        , m_output_domain(output.non_indices_domain())
    {
        if (m_input_domain != m_output_domain)
            throw std::invalid_argument(
                    "StagedLaplacian requires equal input/output spatial domains");
        auto storage = detail::assemble_laplacian<Operator, Index, Cochain>(
                exec,
                output,
                input,
                Operator::geometry(exec, metric, position),
                primal,
                dual);
        m_rows = std::move(storage.rows);
        m_columns = std::move(storage.columns);
        m_values = std::move(storage.values);
        m_offset = std::move(storage.offset);
        m_constant_response = std::move(storage.constant_response);
    }

    Tensor operator()(Tensor output, Tensor input) const
    {
        if (input.non_indices_domain() != m_input_domain
            || output.non_indices_domain() != m_output_domain)
            throw std::invalid_argument("StagedLaplacian domain changed");
        auto const rows = m_rows;
        auto const columns = m_columns;
        auto const values = m_values;
        auto const offset = m_offset;
        auto const constant_response = m_constant_response;
        Domain const source_domain = m_input_domain, target_domain = m_output_domain;
        Kokkos::parallel_for(
                "similie_apply_staged_laplacian",
                Kokkos::RangePolicy<Exec>(m_exec, 0, m_output_domain.size() * Cochain::mem_size()),
                KOKKOS_LAMBDA(std::size_t row) {
                    auto const element
                            = detail::laplacian_element(target_domain, row / Cochain::mem_size());
                    std::array<double, Cochain::mem_size()> reference;
                    double sum = offset(row);
                    for (std::size_t component = 0; component < Cochain::mem_size(); ++component) {
                        reference[component]
                                = input.mem(element, ddc::DiscreteElement<Cochain>(component));
                        sum += constant_response(row, component) * reference[component];
                    }
                    for (std::size_t entry = rows(row); entry < rows(row + 1); ++entry) {
                        std::size_t const column = columns(entry);
                        sum += values(entry)
                               * (input
                                          .mem(detail::laplacian_element(
                                                       source_domain,
                                                       column / Cochain::mem_size()),
                                               ddc::DiscreteElement<Cochain>(
                                                       column % Cochain::mem_size()))
                                  - reference[column % Cochain::mem_size()]);
                    }
                    output.mem(
                            detail::laplacian_element(target_domain, row / Cochain::mem_size()),
                            ddc::DiscreteElement<Cochain>(row % Cochain::mem_size()))
                            = sum;
                });
        return output;
    }
};

static_assert(StagedOperatorConcept<
              StagedLaplacian<
                      detail::MetricIndexConcept,
                      detail::DirectionConcept,
                      detail::ScalarConcept,
                      detail::FieldConcept<detail::ScalarConcept>,
                      detail::FieldConcept<detail::MetricIndexConcept>,
                      detail::FieldConcept<detail::PositionIndexConcept>,
                      Kokkos::DefaultHostExecutionSpace>,
              detail::FieldConcept<detail::ScalarConcept>>);

template <
        tensor::TensorIndex MetricIndex,
        tensor::TensorNatIndex Index,
        tensor::TensorIndex Cochain,
        class Exec,
        class Tensor,
        class Metric,
        class Position,
        class Primal = tensor::ClampExtrapolationRule,
        class Dual = tensor::ZeroExtrapolationRule>
StagedLaplacian<MetricIndex, Index, Cochain, Tensor, Metric, Position, Exec> make_staged_laplacian(
        Exec const& exec,
        Tensor output,
        Tensor input,
        Metric metric,
        Position position,
        Primal primal = {},
        Dual dual = {})
{
    return {exec, output, input, metric, position, primal, dual};
}

} // namespace sil::exterior
