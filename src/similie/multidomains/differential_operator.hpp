// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#pragma once

#include <stdexcept>
#include <tuple>

#include <similie/solvers/stencil_assembly.hpp>

#include "multidomain_operator.hpp"
#include "stencil_composition.hpp"

namespace sil::multidomains {
/** Spatial coordinates carried by differential stencil entries.
 * \important This documentation is fully AI-generated.
 * Relative coordinates encode offsets plus one: the center is at one.
 */
enum class StencilCoordinates { Absolute, Relative };

namespace detail {
using similie::solvers::detail::RowAssembly;
/** Compose a local differential stencil with the input domain's donor rules.
 * \important This operator and documentation is fully AI-generated.
 * StencilFactory calls the local operator's value() with its required arguments
 * and returns coefficients for one output component. The output and input may
 * have different component indices; global_index() supplies their row/column
 * numbers. A caller may add boundary constraints to the same RowAssembly.
 */
template <
        bool Matrix,
        StencilCoordinates Coordinates = StencilCoordinates::Absolute,
        std::size_t Capacity,
        class Output,
        class Input,
        class Operator,
        class Rules,
        class StencilFactory>
void assemble_differential_rows(
        Kokkos::DefaultExecutionSpace const& stream,
        RowAssembly<Capacity> system,
        Output const& output,
        Input const& input,
        Operator op,
        Rules const& rules,
        StencilFactory stencil_factory)
{
    std::tuple<Operator, Input, Rules, StencilFactory> const
            bound(op, input, rules, stencil_factory);
    ddc::parallel_for_each(
            "assemble_multidomain_differential_rows",
            stream,
            output.non_indices_domain(),
            KOKKOS_LAMBDA(typename Output::non_indices_domain_t::discrete_element_type elem) {
                auto const& local_op = std::get<0>(bound);
                auto const& local_input = std::get<1>(bound);
                auto const& local_rules = std::get<2>(bound);
                auto const& value = std::get<3>(bound);
                ddc::device_for_each(output.indices_domain(), [&](auto component) {
                    std::size_t const row = output.global_index(elem, component);
                    if (row >= system.constants.extent(0)) {
                        system.fail(4);
                        return;
                    }
                    auto const coefficients
                            = value(local_op, local_input, local_rules, elem, component);
                    ddc::device_for_each(coefficients.domain(), [&](auto entry) {
                        double const weight = coefficients.mem(entry);
                        if (weight != 0.0)
                            visit_rule_basis(
                                    local_rules,
                                    local_input,
                                    [&] {
                                        if constexpr (Coordinates == StencilCoordinates::Relative)
                                            return sil::tensor::detail::stencil_sample(elem, entry);
                                        else
                                            return typename Input::non_indices_domain_t::
                                                    discrete_element_type(entry);
                                    }(),
                                    typename Input::indices_domain_t::discrete_element_type(entry),
                                    weight,
                                    [&](auto const& donor,
                                        auto sample,
                                        auto source_component,
                                        double coefficient) {
                                        if constexpr (Matrix)
                                            system.add(
                                                    row,
                                                    donor.global_index(sample, source_component),
                                                    coefficient);
                                    },
                                    [&](double constant) { system.constants(row) += constant; });
                    });
                });
            });
}
} // namespace detail

/** Assemble any local differential operator with a stencil-producing value().
 * \important This operator and documentation is fully AI-generated.
 * The factory adapts the operator-specific value() arguments to
 * `(operator, input, rules, element, output_component)`. InputDomains supplies
 * donor and boundary rules; OutputDomains supplies globally numbered rows.
 * Coordinates selects absolute samples (Coboundary) or offsets plus one
 * (the scalar Codifferential stencil). Rows and columns need not have equal
 * counts. The returned rhs is the negative affine contribution of the rules.
 * InputRole selects the primal, flux or geometry sampling policies. Every
 * output row must have a unique owner. Empty rows retain a unit scaling.
 * The factory must be copyable and callable in device kernels. For prescribed
 * nodal constraints, use
 * the lower-level detail::assemble_differential_rows and prescribe rows before
 * finalizing, as the scalar Laplacian adapter does.
 */
template <
        std::size_t ColumnCapacity,
        StencilCoordinates Coordinates = StencilCoordinates::Absolute,
        FieldRole InputRole = FieldRole::Primal,
        class OutputDomains,
        class InputDomains,
        class Operator,
        class StencilFactory>
similie::solvers::AssembledLinearSystem assemble_multidomain_differential_operator(
        Kokkos::DefaultExecutionSpace const& exec,
        OutputDomains const& output_domains,
        InputDomains const& input_domains,
        MultidomainOperator<Operator> const& operators,
        StencilFactory stencil_factory,
        std::size_t rows,
        std::size_t columns,
        bool normalize = true)
{
    if (output_domains.domain_count() != input_domains.domain_count()
        || operators.domain_count() != output_domains.domain_count())
        throw std::invalid_argument("differential operator domain counts differ");
    if (columns == 0)
        throw std::invalid_argument("differential operator requires input columns");
    detail::RowAssembly<ColumnCapacity> const system(rows, true, columns);
    operators.for_each(
            exec,
            [&](std::size_t id, Kokkos::DefaultExecutionSpace const& stream, Operator const& op) {
                detail::assemble_differential_rows<true, Coordinates>(
                        stream,
                        system,
                        output_domains.field(id),
                        input_domains.template field<InputRole>(id),
                        op,
                        input_domains.template extrapolation_rules<InputRole>(id),
                        stencil_factory);
            });
    return system.finalize(exec, normalize, true);
}
} // namespace sil::multidomains
