// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#pragma once

#include <array>
#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include "laplacian.hpp"

namespace sil::exterior {
namespace detail {

// Store topology once instead of copying full LocalChains through nested calls.
template <std::size_t Rank, class Domain, class Memory>
class LaplacianChain
{
    using Basis = decltype(tangent_basis<Rank, Domain, Memory>(
            typename Domain::discrete_element_type {}));
    Kokkos::View<Basis, Memory> m_basis;

public:
    using simplex_type = typename Basis::simplex_type;
    using memory_space = Memory;
    LaplacianChain()
    {
        Basis const basis
                = tangent_basis<Rank, Domain, Memory>(typename Domain::discrete_element_type {});
        m_basis = decltype(m_basis)(
                Kokkos::view_alloc(Kokkos::WithoutInitializing, "laplacian_topology"));
        Kokkos::deep_copy(m_basis, basis);
    }
    KOKKOS_FORCEINLINE_FUNCTION std::size_t size() const
    {
        return m_basis().size();
    }
    KOKKOS_FORCEINLINE_FUNCTION simplex_type const& operator[](std::size_t i) const
    {
        return m_basis()[i];
    }
    KOKKOS_FORCEINLINE_FUNCTION auto begin() const
    {
        return m_basis().begin();
    }
    KOKKOS_FORCEINLINE_FUNCTION auto end() const
    {
        return m_basis().end();
    }
};

template <std::size_t Rank, class Domain, class Memory>
struct LaplacianChains
{
    LaplacianChain<Rank, Domain, Memory> chain;
    std::conditional_t<
            (Rank < Domain::rank()),
            LaplacianChains<Rank + 1, Domain, Memory>,
            std::nullptr_t>
            tail;
    template <std::size_t Selected>
    KOKKOS_FORCEINLINE_FUNCTION auto get() const
    {
        if constexpr (Selected == Rank)
            return chain;
        else
            return tail.template get<Selected>();
    }
};

template <class Source, class Target, class Domain, class Memory>
struct LaplacianHodge
{
    using source_index = Source;
    Domain domain;
    Kokkos::View<double***, Kokkos::LayoutRight, Memory> coefficients;
    template <class Element>
    KOKKOS_FORCEINLINE_FUNCTION double operator()(
            Element element,
            std::size_t target,
            std::size_t source) const
    {
        std::size_t node = 0;
        for (std::size_t axis = 0; axis < Domain::rank(); ++axis)
            node = node * ddc::detail::array(domain.extents())[axis]
                   + ddc::detail::array(element)[axis] - ddc::detail::array(domain.front())[axis];
        return coefficients(node, target, source);
    }
};

// Fold tensor contraction conventions into canonical component matrices once.
template <class Source, class Target, class Exec, class Hodge>
LaplacianHodge<Source, Target, typename Hodge::non_indices_domain_t, typename Hodge::memory_space>
cache_laplacian_hodge(Exec const& exec, Hodge hodge)
{
    using Domain = typename Hodge::non_indices_domain_t;
    using Memory = typename Hodge::memory_space;
    Domain const domain = hodge.non_indices_domain();
    Kokkos::View<double***, Kokkos::LayoutRight, Memory> coefficients(
            "laplacian_hodge_components",
            domain.size(),
            Target::mem_size(),
            Source::mem_size());
    ddc::parallel_for_each(
            "similie_cache_laplacian_hodge",
            exec,
            domain,
            KOKKOS_LAMBDA(typename Domain::discrete_element_type element) {
                auto unit = make_stencil<Memory, Source>(ddc::DiscreteElement<>());
                auto output = make_stencil<Memory, Target>(ddc::DiscreteElement<>());
                std::size_t node = 0;
                for (std::size_t axis = 0; axis < Domain::rank(); ++axis)
                    node = node * ddc::detail::array(domain.extents())[axis]
                           + ddc::detail::array(element)[axis]
                           - ddc::detail::array(domain.front())[axis];
                for (std::size_t source = 0; source < Source::mem_size(); ++source) {
                    unit.mem(ddc::DiscreteElement<Source>(source)) = 1.0;
                    tensor::tensor_prod(
                            tensor::Tensor(output.span_view()),
                            tensor::Tensor(unit.span_view()),
                            hodge[element]);
                    for (std::size_t target = 0; target < Target::mem_size(); ++target)
                        coefficients(node, target, source)
                                = output.mem(ddc::DiscreteElement<Target>(target));
                    unit.mem(ddc::DiscreteElement<Source>(source)) = 0.0;
                }
            });
    return LaplacianHodge<Source, Target, Domain, Memory> {domain, coefficients};
}

template <class Index, class Cochain, class Exec, class Hodge, class Inverse>
auto cache_laplacian_hodge_pair(Exec const& exec, Hodge hodge, Inverse inverse)
{
    using Indices = codifferential_hodge_output_indices_t<Index::size() - Cochain::rank(), Index>;
    using Dual = misc::convert_type_seq_to_t<tensor::TensorAntisymmetricIndex, Indices>;
    using Intermediate = misc::convert_type_seq_to_t<
            tensor::TensorAntisymmetricIndex,
            ddc::type_seq_merge_t<ddc::TypeSeq<Index>, Indices>>;
    return std::
            pair {cache_laplacian_hodge<Cochain, Dual>(exec, hodge),
                  cache_laplacian_hodge<
                          Intermediate,
                          codifferential_index_t<Index, Cochain>>(exec, inverse)};
}

template <class A, class B, class C, class D, class Chains>
struct LaplacianGeometry
{
    A derivative_hodge;
    B inverse_derivative_hodge;
    C hodge;
    D inverse_hodge;
    Chains chains;
};

// Present a lazy cochain through the sampling interface used by boundary rules.
template <class Index, class Domain, class Memory, class Evaluate>
class LaplacianField
{
    Domain m_domain;
    Evaluate m_evaluate;

public:
    using memory_space = Memory;
    using non_indices_domain_t = Domain;
    using indices_domain_t = ddc::DiscreteDomain<Index>;

    KOKKOS_FORCEINLINE_FUNCTION LaplacianField(Domain domain, Evaluate evaluate)
        : m_domain(domain)
        , m_evaluate(evaluate)
    {
    }

    KOKKOS_FORCEINLINE_FUNCTION Domain non_indices_domain() const
    {
        return m_domain;
    }
    KOKKOS_FORCEINLINE_FUNCTION static tensor::TensorAccessor<Index> accessor()
    {
        return {};
    }

    template <class Element, class Component>
    KOKKOS_FORCEINLINE_FUNCTION double mem(Element element, Component component) const
    {
        return m_evaluate(element, component);
    }

    template <class Element>
    KOKKOS_FORCEINLINE_FUNCTION auto operator[](Element element) const
    {
        return tensor::detail::pointwise_tensor<indices_domain_t, Memory>(
                [&](auto component) { return mem(element, component); });
    }
};

template <class Index, class Field, class Rules, class Element, class Chains>
KOKKOS_FORCEINLINE_FUNCTION auto laplacian_derivative(
        Field field,
        Rules rules,
        Element element,
        Chains const& chains)
{
    using Source = ddc::type_seq_element_t<0, ddc::to_type_seq_t<typename Field::indices_domain_t>>;
    using Target = coboundary_index_t<Index, Source>;
    auto output = make_stencil<typename Field::memory_space, Target>(ddc::DiscreteElement<>());
    auto const chain = chains.template get<Source::rank() + 1>();
    auto const lower = chains.template get<Source::rank()>();
    auto const boundary = [&] {
        if constexpr (misc::Specialization<Rules, tensor::ExtrapolationRules>)
            return rules;
        else
            return tensor::make_extrapolation_rules(field, rules);
    }();
    apply_incidence<false, Index, Source>(
            tensor::Tensor(output.span_view()),
            [&](auto sample, auto component) { return boundary(field, sample).mem(component); },
            chain,
            lower,
            element);
    return output;
}

template <
        class Index,
        class Field,
        class Hodge,
        class Inverse,
        class Rules,
        class Element,
        class Chains>
KOKKOS_FORCEINLINE_FUNCTION auto laplacian_codifferential(
        Field field,
        Hodge hodge,
        Inverse inverse,
        Rules rules,
        Element element,
        Chains const& chains)
{
    using Source = ddc::type_seq_element_t<0, ddc::to_type_seq_t<typename Field::indices_domain_t>>;
    using Dual = misc::convert_type_seq_to_t<
            tensor::TensorAntisymmetricIndex,
            codifferential_hodge_output_indices_t<Index::size() - Source::rank(), Index>>;
    using Target = codifferential_index_t<Index, Source>;
    auto evaluate = [&](auto sample, auto component) {
        double value = 0.0;
        for (std::size_t source = 0; source < Source::mem_size(); ++source)
            value += hodge(sample, component.template uid<Dual>(), source)
                     * field.mem(sample, ddc::DiscreteElement<Source>(source));
        return value;
    };
    LaplacianField<
            Dual,
            typename Field::non_indices_domain_t,
            typename Field::memory_space,
            decltype(evaluate)> const dual(field.non_indices_domain(), evaluate);
    auto const boundary = [&] {
        if constexpr (misc::Specialization<Rules, tensor::ExtrapolationRules>)
            return rules;
        else
            return tensor::make_extrapolation_rules(dual, rules);
    }();
    auto output = make_stencil<typename Field::memory_space, Target>(ddc::DiscreteElement<>());
    auto const chain = chains.template get<Dual::rank() + 1>();
    auto const lower = chains.template get<Dual::rank()>();
    using Intermediate = typename Inverse::source_index;
    auto derivative
            = make_stencil<typename Field::memory_space, Intermediate>(ddc::DiscreteElement<>());
    apply_incidence<true, Index, Dual>(
            tensor::Tensor(derivative.span_view()),
            [&](auto sample, auto component) { return boundary(dual, sample).mem(component); },
            chain,
            lower,
            element);
    ddc::device_for_each(output.domain(), [&](auto target) {
        double value = 0.0;
        for (std::size_t source = 0; source < Intermediate::mem_size(); ++source)
            value += inverse(element, target.template uid<Target>(), source)
                     * derivative.mem(ddc::DiscreteElement<Intermediate>(source));
        if constexpr ((Index::size() * (Source::rank() + 1) + 1) % 2 == 1)
            value = -value;
        output.mem(target) = value;
    });
    return output;
}

template <class Domain, class Element>
KOKKOS_FORCEINLINE_FUNCTION std::size_t laplacian_node_id(Domain domain, Element element)
{
    std::size_t id = 0;
    for (std::size_t axis = 0; axis < Domain::rank(); ++axis)
        id = id * ddc::detail::array(domain.extents())[axis] + ddc::detail::array(element)[axis]
             - ddc::detail::array(domain.front())[axis];
    return id;
}

template <class Domain>
KOKKOS_FORCEINLINE_FUNCTION typename Domain::discrete_element_type laplacian_element(
        Domain domain,
        std::size_t id)
{
    typename Domain::discrete_element_type element = domain.front();
    for (std::size_t axis = Domain::rank(); axis-- > 0;) {
        ddc::detail::array(element)[axis] += id % ddc::detail::array(domain.extents())[axis];
        id /= ddc::detail::array(domain.extents())[axis];
    }
    return element;
}

} // namespace detail

/** Pointwise composition of the two branches of the Hodge Laplacian.
 * \important This operator and documentation is fully AI-generated.
 * The supplied staged operator provides geometry coefficients. value() evaluates
 * the response to a unit cochain. For affine boundary rules, subtract the
 * response to the zero cochain to obtain a matrix entry. Boundary rules act at
 * the same intermediate stages as in StagedLaplacian.
 */
template <
        class MetricIndex,
        class Index,
        class Cochain,
        class Tensor,
        class Metric,
        class Position,
        class Exec>
struct Laplacian<MetricIndex, Index, Cochain, Tensor, Metric, Position, Exec>
{
    static auto geometry(
            StagedLaplacian<MetricIndex, Index, Cochain, Tensor, Metric, Position, Exec> const&
                    staged)
    {
        using Domain = typename detail::
                NonSpectatorDimension<Index, typename Tensor::non_indices_domain_t>::type;
        detail::LaplacianChains<0, Domain, typename Tensor::memory_space> const chains {};
        auto derivative = [&] {
            if constexpr (Cochain::rank() < Index::size()) {
                using Added = tensor::Covariant<IndexForCodifferentialOfCoboundaryInLaplacian<
                        tensor::uncharacterize_t<Index>>>;
                return detail::
                        cache_laplacian_hodge_pair<Added, coboundary_index_t<Added, Cochain>>(
                                staged.m_exec_space,
                                *staged.m_derivative_hodge_star,
                                *staged.m_dual_derivative_hodge_star);
            } else
                return std::pair {0, 0};
        }();
        auto codifferential = [&] {
            if constexpr (Cochain::rank() > 0)
                return detail::cache_laplacian_hodge_pair<Index, Cochain>(
                        staged.m_exec_space,
                        *staged.m_hodge_star,
                        *staged.m_dual_hodge_star);
            else
                return std::pair {0, 0};
        }();
        return detail::LaplacianGeometry {
                derivative.first,
                derivative.second,
                codifferential.first,
                codifferential.second,
                chains};
    }

    template <class Geometry, class Evaluate, class Element, class Primal, class Dual>
    KOKKOS_FORCEINLINE_FUNCTION static auto operator()(
            Geometry const& staged,
            Tensor input,
            Evaluate evaluate,
            Element element,
            Primal primal,
            Dual dual)
    {
        using Memory = typename Tensor::memory_space;
        using Domain = typename Tensor::non_indices_domain_t;
        detail::LaplacianField<Cochain, Domain, Memory, Evaluate> const
                field(input.non_indices_domain(), evaluate);
        auto output = detail::make_stencil<Memory, Cochain>(ddc::DiscreteElement<>());
        if constexpr (Cochain::rank() < Index::size()) {
            using Added = tensor::Covariant<
                    IndexForCodifferentialOfCoboundaryInLaplacian<tensor::uncharacterize_t<Index>>>;
            using Derived = coboundary_index_t<Added, Cochain>;
            auto derivative = [&](auto sample, auto component) {
                return detail::laplacian_derivative<Added>(field, primal, sample, staged.chains)
                        .mem(component);
            };
            detail::LaplacianField<Derived, Domain, Memory, decltype(derivative)> const
                    derivatives(input.non_indices_domain(), derivative);
            auto result = detail::laplacian_codifferential<Added>(
                    derivatives,
                    staged.derivative_hodge,
                    staged.inverse_derivative_hodge,
                    dual,
                    element,
                    staged.chains);
            ddc::device_for_each(output.domain(), [&](auto component) {
                output.mem(component) += result.mem(component);
            });
        }
        if constexpr (Cochain::rank() > 0) {
            using Lower = codifferential_index_t<Index, Cochain>;
            auto codifferential = [&](auto sample, auto component) {
                return detail::laplacian_codifferential<Index>(
                               field,
                               staged.hodge,
                               staged.inverse_hodge,
                               dual,
                               sample,
                               staged.chains)
                        .mem(component);
            };
            detail::LaplacianField<Lower, Domain, Memory, decltype(codifferential)> const
                    codifferentials(input.non_indices_domain(), codifferential);
            auto result = detail::laplacian_derivative<
                    Index>(codifferentials, primal, element, staged.chains);
            ddc::device_for_each(output.domain(), [&](auto component) {
                output.mem(component) += result.mem(component);
            });
        }
        return output;
    }

    template <class Geometry, class Element, class Component, class Primal, class Dual>
    KOKKOS_FORCEINLINE_FUNCTION static double value(
            Geometry const& staged,
            Tensor input,
            Element element,
            Component target,
            std::size_t column,
            Primal primal,
            Dual dual)
    {
        auto result = operator()(
                staged,
                input,
                [&](auto sample, auto component) {
                    return detail::laplacian_node_id(input.non_indices_domain(), sample)
                                                           * Cochain::mem_size()
                                                   + component.template uid<Cochain>()
                                           == column
                                   ? 1.0
                                   : 0.0;
                },
                element,
                primal,
                dual);
        return result.mem(target);
    }
};

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
class AggressivelyStagedLaplacian
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
    AggressivelyStagedLaplacian(
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
                    "AggressivelyStagedLaplacian requires equal input/output spatial domains");
        StagedLaplacian<MetricIndex, Index, Cochain, Tensor, Metric, Position, Exec>
                staged(exec, output, input, metric, position);
        // All geometry kernels must finish before their coefficients are sampled.
        exec.fence();
        auto storage = detail::assemble_laplacian<
                Operator,
                Index,
                Cochain>(exec, output, input, Operator::geometry(staged), primal, dual);
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
            throw std::invalid_argument("AggressivelyStagedLaplacian domain changed");
        auto const rows = m_rows;
        auto const columns = m_columns;
        auto const values = m_values;
        auto const offset = m_offset;
        auto const constant_response = m_constant_response;
        Domain const source_domain = m_input_domain, target_domain = m_output_domain;
        Kokkos::parallel_for(
                "similie_apply_aggressively_staged_laplacian",
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
              AggressivelyStagedLaplacian<
                      detail::MetricIndexConcept,
                      detail::DirectionConcept,
                      detail::ScalarConcept,
                      detail::FieldConcept<detail::ScalarConcept>,
                      detail::FieldConcept<detail::MetricIndexConcept>,
                      detail::FieldConcept<detail::PositionIndexConcept>,
                      Kokkos::DefaultHostExecutionSpace>,
              detail::FieldConcept<detail::ScalarConcept>>);

} // namespace sil::exterior
