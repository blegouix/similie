// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#pragma once

#include <similie/exterior/laplacian.hpp>
#include <similie/solvers/affine_assembly.hpp>

#include "indexed_scalar_field.hpp"
#include "multidomain_operator.hpp"
#include "multidomains.hpp"

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
        auto const flux = sil::exterior::make_scalar_laplacian_flux<Direction>(
                domains.field(id),
                data.geometry,
                domains.extrapolation_rules(id, jump_scale),
                domains.template extrapolation_rules<sil::multidomains::FieldRole::Geometry>(id),
                domains.template boundary_rules<sil::multidomains::FieldRole::Flux>(id),
                coefficient);
        return sil::multidomains::domain_data(data.field, data.geometry, data.physics, flux);
    });
}

namespace detail {
using similie::solvers::detail::assemble_dec_rows;
using similie::solvers::detail::RowAssembly;
using similie::solvers::detail::sample_affine;
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
                    int error = 0;
                    auto const sample = sample_affine<Capacity>(
                            [&](auto sampler) {
                                return sil::tensor::detail::EvaluateRuleValue {sampler}(
                                        primal,
                                        field,
                                        elem,
                                        ddc::DiscreteElement<typename Field::component_type>(0));
                            },
                            error);
                    auto exterior = elem;
                    if constexpr (Upper)
                        ++ddc::detail::array(exterior)[Axis];
                    else
                        --ddc::detail::array(exterior)[Axis];
                    double const value = sil::tensor::detail::EvaluateRuleValue<
                            sil::exterior::detail::
                                    ZeroScalarSampler> {sil::exterior::detail::
                                                                ZeroScalarSampler {}}(
                            primal.template boundary_rule<Axis, Upper>(),
                            field,
                            exterior,
                            ddc::DiscreteElement<typename Field::component_type>(0));
                    if (error)
                        system.fail(error);
                    else
                        system.prescribe(sample, value);
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

} // namespace detail
/** Assemble scalar DEC stencils through topology-generated primal and dual rules.
 * \important This operator and documentation is fully AI-generated.
 * Each domain owns disjoint nodal rows and uses its partitioned execution space.
 * Dirichlet constraints include junction endpoints owned by connected domains.
 * Row stencils are packed into CSR on the device after all domains finish.
 */
template <class Direction, std::size_t ProbeCapacity = 0, class Domains, class Diffusivity>
similie::solvers::ScalarLinearSystem assemble_multidomain_laplacian(
        Kokkos::DefaultExecutionSpace const& exec,
        Domains const& domains,
        std::size_t unknowns,
        Diffusivity diffusivity,
        double jump_scale = 1.0,
        bool normalize = true)
{
    auto const flux_domains
            = make_laplacian_flux_domains<Direction>(domains, diffusivity, jump_scale);
    constexpr std::size_t capacity = [] {
        if constexpr (ProbeCapacity != 0)
            return ProbeCapacity;
        // Whole dual tensors visit every normal at each incidence sample.
        // Incidence and incident-cell offsets reach two nodes backward; cell
        // corners and full forward gradients reach two nodes forward. Bound
        // their union by this five-node box in every logical direction. Probes
        // whose assembled coefficient is zero are discarded by sample_affine.
        std::size_t samples = 1;
        for (std::size_t axis = 0; axis < Direction::size(); ++axis)
            samples *= 5;
        return samples;
    }();
    auto const make_operator = [&](std::size_t id) {
        auto const flux = flux_domains.template field<FieldRole::Flux>(id);
        auto const dual = flux_domains.template extrapolation_rules<FieldRole::Flux>(id);
        return sil::exterior::ScalarLaplacian<
                Direction,
                std::remove_cvref_t<decltype(flux)>,
                std::remove_cvref_t<decltype(dual)>> {flux, dual};
    };
    std::vector<std::remove_cvref_t<decltype(make_operator(0))>> operators;
    operators.reserve(domains.domain_count());
    for (std::size_t id = 0; id < domains.domain_count(); ++id)
        operators.push_back(make_operator(id));
    MultidomainOperator const operation(std::move(operators));
    detail::RowAssembly<capacity> const system(unknowns);
    operation.for_each(
            exec,
            [&](std::size_t id, Kokkos::DefaultExecutionSpace const& stream, auto const& op) {
                auto const field = domains.field(id);
                detail::assemble_dec_rows<capacity>(stream, system, field, op);
                detail::prescribe_faces<capacity>(
                        stream,
                        system,
                        field,
                        domains.extrapolation_rules(id, jump_scale),
                        std::make_index_sequence<Direction::size()> {});
            });
    return system.finalize(exec, normalize);
}
} // namespace sil::multidomains
