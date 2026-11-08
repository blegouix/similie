// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#pragma once

#include <tuple>

#include <similie/exterior/stencil.hpp>

namespace sil::exterior {

/** Cache a bilinear spatial map without freezing either operand.
 * Operator::value(geometry, first, second, element, arguments...) returns a
 * BilinearStencil. Input fields supply domains and types, not frozen values.
 * As for linear spatial staging, point indices use periodic addressing;
 * value() must fold other homogeneous boundary rules into the coefficients.
 * Output must not alias either operand. Rebuild after geometry or rules change.
 * \important This operator and documentation is fully AI-generated.
 */
template <
        class Operator,
        class Index,
        class First,
        class Second,
        class Target,
        std::size_t FirstRadius,
        std::size_t SecondRadius,
        class Domain,
        class Memory,
        class Exec = typename Memory::execution_space>
class BilinearSpatialOperatorStorage
{
    using Coefficients = BilinearStencil<Index, First, Second, Target, FirstRadius, SecondRadius>;
    Domain m_domain, m_first_domain, m_second_domain;
    Kokkos::View<
            double * [Target::mem_size() * Coefficients::first_size * Coefficients::second_size],
            Kokkos::LayoutLeft,
            Memory>
            m_weights;

public:
    using execution_space = Exec;
    template <class OtherExec>
    using rebind = BilinearSpatialOperatorStorage<
            Operator,
            Index,
            First,
            Second,
            Target,
            FirstRadius,
            SecondRadius,
            Domain,
            Memory,
            OtherExec>;

    template <class Geometry, class A, class B, class Arguments, std::size_t... I>
    KOKKOS_FUNCTION static Coefficients evaluate_value(
            Geometry const& geometry,
            A const& first,
            B const& second,
            typename Domain::discrete_element_type element,
            Arguments const& arguments,
            std::index_sequence<I...>)
    {
        return Operator::value(geometry, first, second, element, std::get<I>(arguments)...);
    }

    template <class FirstField, class SecondField, class Geometry, class... Arguments>
        requires(
                std::same_as<typename FirstField::indices_domain_t, ddc::DiscreteDomain<First>>
                && std::same_as<
                        typename SecondField::indices_domain_t,
                        ddc::DiscreteDomain<Second>>)
    static BilinearSpatialOperatorStorage build(
            Exec const& exec,
            FirstField const& first,
            SecondField const& second,
            Geometry const& geometry,
            Arguments const&... arguments)
    {
        BilinearSpatialOperatorStorage result;
        result.m_domain = first.non_indices_domain();
        result.m_first_domain = first.non_indices_domain();
        result.m_second_domain = second.non_indices_domain();
        // Equal spatial domains keep sampling and cached boundary rules unambiguous.
        if (result.m_first_domain != result.m_second_domain)
            throw std::invalid_argument("Bilinear staging requires equal input spatial domains");
        result.m_weights = decltype(m_weights)(
                Kokkos::view_alloc(Kokkos::WithoutInitializing, "staged_bilinear_stencils"),
                result.m_domain.size());
        auto const weights = result.m_weights;
        Domain const domain = result.m_domain;
        std::tuple<Arguments...> const cached_arguments(arguments...);
        Kokkos::View<int, Memory> incomplete("staged_bilinear_incomplete");
        Kokkos::parallel_for(
                "similie_cache_bilinear_operator_stencils",
                Kokkos::RangePolicy<Exec>(exec, 0, domain.size()),
                KOKKOS_LAMBDA(std::size_t node) {
                    Coefficients const stencil = evaluate_value(
                            geometry,
                            first,
                            second,
                            detail::stencil_element(domain, node),
                            cached_arguments,
                            std::index_sequence_for<Arguments...>());
                    if (!stencil.complete)
                        Kokkos::atomic_max(&incomplete(), 1);
                    for (std::size_t entry = 0; entry < stencil.weights.size(); ++entry)
                        weights(node, entry) = stencil.weights[entry];
                });
        int failed;
        Kokkos::deep_copy(exec, failed, incomplete);
        exec.fence();
        if (failed)
            throw std::invalid_argument("Bilinear boundary closure exceeds cached stencil support");
        return result;
    }

    [[nodiscard]] KOKKOS_FUNCTION Coefficients
    value(typename Domain::discrete_element_type element) const
    {
        Coefficients result;
        std::size_t const node = detail::stencil_node(m_domain, element);
        for (std::size_t entry = 0; entry < result.weights.size(); ++entry)
            result.weights[entry] = m_weights(node, entry);
        return result;
    }

    template <class Output, class FirstTensor, class SecondTensor>
        requires(
                FirstRadius == 0 && SecondRadius == 0
                && std::same_as<typename FirstTensor::indices_domain_t, ddc::DiscreteDomain<First>>
                && std::same_as<
                        typename SecondTensor::indices_domain_t,
                        ddc::DiscreteDomain<Second>>
                && std::same_as<typename Output::indices_domain_t, ddc::DiscreteDomain<Target>>)
    KOKKOS_FUNCTION void apply_pointwise(
            Output const& output,
            FirstTensor const& first,
            SecondTensor const& second,
            typename Domain::discrete_element_type element) const
    {
        std::size_t const node = detail::stencil_node(m_domain, element);
        for (std::size_t target = 0; target < Target::mem_size(); ++target) {
            double sum = 0.0;
            for (std::size_t a = 0; a < First::mem_size(); ++a)
                for (std::size_t b = 0; b < Second::mem_size(); ++b)
                    sum += m_weights(
                                   node,
                                   (target * First::mem_size() + a) * Second::mem_size() + b)
                           * first.mem(ddc::DiscreteElement<First>(a))
                           * second.mem(ddc::DiscreteElement<Second>(b));
            output.mem(ddc::DiscreteElement<Target>(target)) = sum;
        }
    }

    template <class Output, class FirstField, class SecondField>
        requires(
                std::same_as<typename FirstField::indices_domain_t, ddc::DiscreteDomain<First>>
                && std::same_as<typename SecondField::indices_domain_t, ddc::DiscreteDomain<Second>>
                && std::same_as<typename Output::indices_domain_t, ddc::DiscreteDomain<Target>>)
    Output apply(
            Exec const& exec,
            Output const& output,
            FirstField const& first,
            SecondField const& second) const
    {
        if (first.non_indices_domain() != m_first_domain
            || second.non_indices_domain() != m_second_domain)
            throw std::invalid_argument("Staged bilinear operator input domain changed");
        Domain const out_domain = output.non_indices_domain();
        if (!out_domain.empty()
            && (!misc::domain_contains(m_domain, out_domain.front())
                || !misc::domain_contains(m_domain, out_domain.back())))
            throw std::invalid_argument("Staged bilinear operator output exceeds cached domain");
        auto const weights = m_weights;
        Domain const domain = m_domain;
        Kokkos::parallel_for(
                "similie_apply_staged_bilinear_stencil",
                Kokkos::RangePolicy<Exec>(exec, 0, out_domain.size() * Target::mem_size()),
                KOKKOS_LAMBDA(std::size_t row) {
                    auto const element
                            = detail::stencil_element(out_domain, row / Target::mem_size());
                    std::size_t const node = detail::stencil_node(domain, element),
                                      target = row % Target::mem_size();
                    double sum = 0.0;
                    for (std::size_t a = 0; a < Coefficients::first_size; ++a)
                        for (std::size_t b = 0; b < Coefficients::second_size; ++b) {
                            double const coefficient = weights(
                                    node,
                                    (target * Coefficients::first_size + a)
                                                    * Coefficients::second_size
                                            + b);
                            if (coefficient != 0.0)
                                sum += coefficient
                                       * first
                                                 .mem(detail::stencil_source<Index, FirstRadius>(
                                                              domain,
                                                              element,
                                                              a / First::mem_size()),
                                                      ddc::DiscreteElement<First>(
                                                              a % First::mem_size()))
                                       * second
                                                 .mem(detail::stencil_source<Index, SecondRadius>(
                                                              domain,
                                                              element,
                                                              b / Second::mem_size()),
                                                      ddc::DiscreteElement<Second>(
                                                              b % Second::mem_size()));
                        }
                    output.mem(element, ddc::DiscreteElement<Target>(target)) = sum;
                });
        return output;
    }
};

// A local bilinear map has no spatial offsets for either operand.
template <
        class Operator,
        class Index,
        class First,
        class Second,
        class Target,
        class Domain,
        class Memory,
        class Exec = typename Memory::execution_space>
using BilinearLocalOperatorStorage = BilinearSpatialOperatorStorage<
        Operator,
        Index,
        First,
        Second,
        Target,
        0,
        0,
        Domain,
        Memory,
        Exec>;

} // namespace sil::exterior
