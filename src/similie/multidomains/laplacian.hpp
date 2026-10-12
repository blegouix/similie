// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#pragma once

#include <tuple>

#include <similie/exterior/laplacian.hpp>
#include <similie/solvers/stencil_assembly.hpp>

#include "indexed_scalar_field.hpp"
#include "multidomain_operator.hpp"
#include "multidomains.hpp"
#include "stencil_composition.hpp"

namespace sil::multidomains {
/** Bind lazy *d(phi) cochains, so dual topology rules enter the donor's DEC flux.
 * \important This operator and documentation is fully AI-generated.
 * Every physical node retains its primal field, Cartesian geometry and physics.
 * Boundary nodes retain their policies. Direction is the covariant natural index
 * of the logical axes. Material coefficients multiply the constitutive flux.
 */
template <class Direction, class Domains, class Diffusivity>
auto make_laplacian_flux_domains(
        Domains const& domains,
        Diffusivity diffusivity,
        double jump_scale = 1.0)
{
    return domains.transform_data([&](std::size_t id, auto const& data) {
        double const coefficient = diffusivity(data.physics);
        if (!(coefficient > 0.0) || !std::isfinite(coefficient))
            throw std::invalid_argument("scalar diffusivity must be finite and positive");
        auto const field = domains.field(id);
        auto const flux = sil::exterior::make_dual_hodge_cochain<Direction>(
                sil::tensor::make_tensor_with_extrapolation_rules(
                        field,
                        domains.extrapolation_rules(id, jump_scale)),
                data.geometry,
                domains.template extrapolation_rules<sil::multidomains::FieldRole::Geometry>(id),
                domains.template boundary_rules<sil::multidomains::FieldRole::Flux>(id),
                coefficient);
        return sil::multidomains::domain_data(data.field, data.geometry, data.physics, flux);
    });
}

namespace detail {
using similie::solvers::detail::RowAssembly;
template <std::size_t Axis, bool Upper, std::size_t Capacity, class Field, class Primal>
void prescribe_face(
        Kokkos::DefaultExecutionSpace const& exec,
        RowAssembly<Capacity> system,
        Field field,
        Primal primal)
{
    if (sil::tensor::detail::is_dirichlet(primal.template boundary_rule<Axis, Upper>())) {
        auto nodes = field.non_indices_domain();
        auto front = nodes.front();
        auto extent = nodes.extents();
        ddc::detail::array(front)[Axis]
                = ddc::detail::array(Upper ? nodes.back() : nodes.front())[Axis];
        ddc::detail::array(extent)[Axis] = 1;
        // The final endpoint of a physical face can belong to a donor domain.
        [&]<std::size_t... Tangent>(std::index_sequence<Tangent...>) {
            (
                    [&] {
                        if constexpr (Tangent != Axis)
                            if (sil::tensor::detail::is_interface(
                                        primal.template boundary_rule<Tangent, true>())
                                && !sil::tensor::detail::shared_boundary(
                                        primal.template boundary_rule<Tangent, true>()))
                                ++ddc::detail::array(extent)[Tangent];
                    }(),
                    ...);
        }(std::make_index_sequence<
                ddc::type_seq_size_v<ddc::to_type_seq_t<typename Field::non_indices_domain_t>>> {});
        nodes = typename Field::non_indices_domain_t(front, extent);
        ddc::parallel_for_each(
                "prescribe_dec_face",
                exec,
                nodes,
                KOKKOS_LAMBDA(typename Field::non_indices_domain_t::discrete_element_type elem) {
                    std::array<std::size_t, Capacity> indices {};
                    auto weights = similie::solvers::detail::make_row_stencil<Capacity>();
                    std::size_t count = 0;
                    double constant = 0.0;
                    detail::visit_rule_basis(
                            primal,
                            field,
                            elem,
                            ddc::DiscreteElement<typename Field::component_type>(0),
                            1.0,
                            [&](auto const& donor, auto sample, auto component, double weight) {
                                if (!similie::solvers::detail::add_stencil_coefficient(
                                            weights,
                                            indices.data(),
                                            count,
                                            donor.global_index(sample, component),
                                            weight))
                                    system.fail(2);
                            },
                            [&](double value) { constant += value; });
                    std::size_t nonzero = 0;
                    std::size_t selected = 0;
                    for (std::size_t i = 0; i < count; ++i)
                        if (weights.mem(
                                    ddc::DiscreteElement<similie::solvers::detail::StencilColumn>(
                                            i))
                            != 0.0) {
                            ++nonzero;
                            selected = i;
                        }
                    auto exterior = elem;
                    if constexpr (Upper)
                        ++ddc::detail::array(exterior)[Axis];
                    else
                        --ddc::detail::array(exterior)[Axis];
                    auto const& face = primal.template boundary_rule<Axis, Upper>();
                    auto const& boundary = [&]() -> auto const& {
                        if constexpr (requires { face.boundary; })
                            return face.boundary;
                        else
                            return face;
                    }();
                    double const value = sil::tensor::detail::extrapolation_source_value(
                            boundary,
                            field,
                            exterior,
                            ddc::DiscreteElement<typename Field::component_type>(0));
                    if (nonzero != 1)
                        system.fail(8);
                    else
                        system.prescribe(
                                indices[selected],
                                (value - constant)
                                        / weights.mem(
                                                ddc::DiscreteElement<
                                                        similie::solvers::detail::StencilColumn>(
                                                        selected)));
                });
    }
}

template <std::size_t Capacity, class Field, class Primal, std::size_t... Axis>
void prescribe_faces(
        Kokkos::DefaultExecutionSpace const& exec,
        RowAssembly<Capacity> system,
        Field field,
        Primal primal,
        std::index_sequence<Axis...>)
{
    (prescribe_face<Axis, false, Capacity>(exec, system, field, primal), ...);
    (prescribe_face<Axis, true, Capacity>(exec, system, field, primal), ...);
}

template <class Direction, std::size_t Requested>
constexpr std::size_t column_capacity()
{
    if constexpr (Requested != 0)
        return Requested;
    std::size_t samples = 1;
    for (std::size_t axis = 0; axis < Direction::size(); ++axis)
        samples *= 5;
    return samples;
}

template <bool Matrix, std::size_t Capacity, class Field, class Operator, class Flux, class Rules>
void assemble_rows(
        Kokkos::DefaultExecutionSpace const& stream,
        RowAssembly<Capacity> system,
        Field field,
        Operator op,
        Flux flux,
        Rules rules)
{
    std::tuple<Operator, Flux, Rules> const bound(op, flux, rules);
    ddc::parallel_for_each(
            "assemble_dec_laplacian_rows",
            stream,
            field.non_indices_domain(),
            KOKKOS_LAMBDA(typename Field::non_indices_domain_t::discrete_element_type elem) {
                auto const& op = std::get<0>(bound);
                auto const& flux = std::get<1>(bound);
                auto const& rules = std::get<2>(bound);
                std::size_t const row = field.global_index(
                        elem,
                        ddc::DiscreteElement<typename Field::component_type>(0));
                if (row >= system.constants.extent(0)) {
                    system.fail(4);
                    return;
                }
                auto const coefficients = op.value(flux, flux, elem);
                ddc::device_for_each(coefficients.domain(), [&](auto entry) {
                    double const weight = coefficients.mem(entry);
                    if (weight != 0.0)
                        detail::visit_rule_basis(
                                rules,
                                flux,
                                sil::tensor::detail::stencil_sample(elem, entry),
                                typename Flux::indices_domain_t::discrete_element_type(entry),
                                weight,
                                [&](auto const& donor,
                                    auto sample,
                                    auto component,
                                    double coefficient) {
                                    if constexpr (Matrix)
                                        system
                                                .add(row,
                                                     donor.global_index(sample, component),
                                                     coefficient);
                                },
                                [&](double value) { system.constants(row) += value; });
                });
            });
}

template <class Direction, bool Matrix, std::size_t Capacity, class Domains, class Diffusivity>
void assemble_laplacian_rows(
        Kokkos::DefaultExecutionSpace const& exec,
        RowAssembly<Capacity> system,
        Domains const& domains,
        Diffusivity diffusivity,
        double jump_scale)
{
    auto const flux_domains
            = make_laplacian_flux_domains<Direction>(domains, diffusivity, jump_scale);
    MultidomainOperator const operation(
            std::vector<sil::exterior::Codifferential<Direction, Direction>>(
                    domains.domain_count()));
    operation.for_each(
            exec,
            [&](std::size_t id, Kokkos::DefaultExecutionSpace const& stream, auto const& op) {
                auto const field = domains.field(id);
                auto const flux = flux_domains.template field<FieldRole::Flux>(id);
                auto const rules = flux_domains.template extrapolation_rules<FieldRole::Flux>(id);
                assemble_rows<Matrix>(stream, system, field, op, flux, rules);
                prescribe_faces<Capacity>(
                        stream,
                        system,
                        field,
                        domains.extrapolation_rules(id, jump_scale),
                        std::make_index_sequence<Direction::size()> {});
            });
}
} // namespace detail

/** Assemble scalar DEC tensor stencils through topology-generated rules.
 * \important This operator and documentation is fully AI-generated.
 * Each domain owns disjoint nodal rows. Coefficients are composed through donor
 * mappings; prescribed values and interface jumps contribute separately to rhs.
 */
template <class Direction, std::size_t ColumnCapacity = 0, class Domains, class Diffusivity>
similie::solvers::AssembledLinearSystem assemble_multidomain_laplacian(
        Kokkos::DefaultExecutionSpace const& exec,
        Domains const& domains,
        std::size_t unknowns,
        Diffusivity diffusivity,
        double jump_scale = 1.0,
        bool normalize = true)
{
    detail::RowAssembly<detail::column_capacity<Direction, ColumnCapacity>()> const system(
            unknowns);
    detail::assemble_laplacian_rows<
            Direction,
            true>(exec, system, domains, diffusivity, jump_scale);
    return system.finalize(exec, normalize);
}

/** Evaluate a new boundary/interface load for an already assembled operator.
 * \important This operator and documentation is fully AI-generated.
 * Geometry, material coefficients, ownership and boundary kinds must match the
 * original system. Its row scaling is reused, including constrained nodal rows.
 */
template <class Direction, std::size_t ColumnCapacity = 0, class Domains, class Diffusivity>
Kokkos::View<double**> assemble_multidomain_laplacian_rhs(
        Kokkos::DefaultExecutionSpace const& exec,
        Domains const& domains,
        similie::solvers::AssembledLinearSystem const& matrix_system,
        Diffusivity diffusivity,
        double jump_scale = 1.0)
{
    if (matrix_system.row_scales.extent(0) != matrix_system.rhs.extent(0))
        throw std::invalid_argument("RHS assembly requires the original DEC row scaling");
    detail::RowAssembly<detail::column_capacity<Direction, ColumnCapacity>()> const
            system(matrix_system.rhs.extent(0), false);
    detail::assemble_laplacian_rows<
            Direction,
            false>(exec, system, domains, diffusivity, jump_scale);
    return system.finalize_rhs(exec, matrix_system.row_scales);
}
} // namespace sil::multidomains
