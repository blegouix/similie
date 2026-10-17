// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#pragma once

#include <array>
#include <cassert>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include <similie/exterior/coboundary/coboundary.hpp>
#include <similie/tensor/tensor_indices/diagonal_tensor.hpp>

#include "stencil_indices.hpp"

namespace sil::exterior {

// Full coefficient layout: output component, spatial point, input component.
// Offsets and constant responses support affine closures and constant null modes.
template <class Index, class Source, class Target, std::size_t Radius>
struct Stencil
{
    std::array<
            double,
            Target::mem_size() * Source::mem_size() * detail::StencilShape<Index, Radius>::size>
            weights {};
    std::array<double, Target::mem_size()> offset {};
    std::array<double, Target::mem_size() * Source::mem_size()> constant_response {};
    bool complete = true;
};

namespace detail {

// Static tensor coefficients, without spatial dimensions or heap allocation.
template <class Memory, class... Indices>
KOKKOS_FUNCTION auto coefficient_stencil(ddc::DiscreteDomain<Indices...> domain)
{
    std::array<double, (Indices::mem_size() * ... * std::size_t(1))> values {};
    tensor::Tensor<double, ddc::DiscreteDomain<Indices...>, Kokkos::layout_right, Memory> view(
            ddc::ChunkSpan<
                    double,
                    ddc::DiscreteDomain<Indices...>,
                    Kokkos::layout_right,
                    Memory>(values.data(), domain));
    return tensor::OwningTensor(view, std::move(values));
}

// A local operator uses the same ordering with one spatial point.
template <class Source, class Target>
struct LocalOperatorStencil
{
    std::array<double, Source::mem_size() * Target::mem_size()> weights {};
};

// Recover a natural tensor coefficient from canonical cochain components.
// Antisymmetric permutations and factorial conventions are applied only here.
template <
        class Source,
        class Target,
        bool NormalizeSource = true,
        class Stencil,
        std::size_t SourceRank,
        std::size_t TargetRank>
KOKKOS_FUNCTION double local_stencil_coefficient(
        Stencil const& stencil,
        std::array<std::size_t, SourceRank> source,
        std::array<std::size_t, TargetRank> target)
{
    int sign = 1;
    for (std::size_t i = 0; i < source.size(); ++i)
        for (std::size_t j = i + 1; j < source.size(); ++j) {
            if (source[i] == source[j])
                return 0.0;
            if (source[i] > source[j]) {
                Kokkos::kokkos_swap(source[i], source[j]);
                sign = -sign;
            }
        }
    for (std::size_t i = 0; i < target.size(); ++i)
        for (std::size_t j = i + 1; j < target.size(); ++j) {
            if (target[i] == target[j])
                return 0.0;
            if (target[i] > target[j]) {
                Kokkos::kokkos_swap(target[i], target[j]);
                sign = -sign;
            }
        }
    std::size_t factorial = 1;
    if constexpr (NormalizeSource)
        for (std::size_t i = 2; i <= source.size(); ++i)
            factorial *= i;
    return sign
           * stencil.weights
                     [stencil_component_index<Target>(target) * Source::mem_size()
                      + stencil_component_index<Source>(source)]
           / factorial;
}

// Stored topology is shared by pointwise compositions, not copied down their call tree.
template <std::size_t Rank, class Domain, class Memory>
class StencilChain
{
    using Basis = decltype(tangent_basis<Rank, Domain, Memory>(
            typename Domain::discrete_element_type {}));
    Kokkos::View<Basis, Memory> m_basis;

public:
    using simplex_type = typename Basis::simplex_type;
    using memory_space = Memory;
    StencilChain() : m_basis(Kokkos::view_alloc(Kokkos::WithoutInitializing, "stencil_topology"))
    {
        Basis const basis
                = tangent_basis<Rank, Domain, Memory>(typename Domain::discrete_element_type {});
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
struct StencilChains
{
    StencilChain<Rank, Domain, Memory> chain;
    std::conditional_t<
            (Rank < Domain::rank()),
            StencilChains<Rank + 1, Domain, Memory>,
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

template <class Domain, class Element>
KOKKOS_FORCEINLINE_FUNCTION std::size_t stencil_node(Domain domain, Element element)
{
    std::size_t node = 0;
    for (std::size_t axis = 0; axis < Domain::rank(); ++axis)
        node = node * ddc::detail::array(domain.extents())[axis] + ddc::detail::array(element)[axis]
               - ddc::detail::array(domain.front())[axis];
    return node;
}

template <class Domain>
KOKKOS_FORCEINLINE_FUNCTION typename Domain::discrete_element_type stencil_element(
        Domain domain,
        std::size_t node)
{
    typename Domain::discrete_element_type element = domain.front();
    for (std::size_t axis = Domain::rank(); axis-- > 0;) {
        ddc::detail::array(element)[axis] += node % ddc::detail::array(domain.extents())[axis];
        node /= ddc::detail::array(domain.extents())[axis];
    }
    return element;
}

template <class Index, class... Dimensions>
constexpr std::array<bool, sizeof...(Dimensions)> stencil_axes(ddc::DiscreteDomain<Dimensions...>)
{
    return {ddc::type_seq_contains_v<
            ddc::TypeSeq<typename Dimensions::continuous_dimension_type>,
            typename Index::type_seq_dimensions>...};
}

// Cyclic addressing supplies an implicit numbering at periodic boundaries.
// Nonperiodic closures simply give zero coefficients to unused wrapped samples.
template <class Index, std::size_t Radius, class Domain, class Element>
KOKKOS_FORCEINLINE_FUNCTION Element
stencil_source(Domain domain, Element element, std::size_t point)
{
    auto const offset = StencilShape<Index, Radius>::offset(point);
    auto const active = stencil_axes<Index>(Domain {});
    std::size_t physical_axis = 0;
    for (std::size_t axis = 0; axis < Domain::rank(); ++axis) {
        if (!active[axis])
            continue;
        std::ptrdiff_t const front = ddc::detail::array(domain.front())[axis];
        std::ptrdiff_t const count = ddc::detail::array(domain.extents())[axis];
        std::ptrdiff_t position = std::ptrdiff_t(ddc::detail::array(element)[axis]) - front
                                  + offset[physical_axis++];
        if (position < 0 || position >= count)
            position = (position % count + count) % count;
        ddc::detail::array(element)[axis] = front + position;
    }
    return element;
}

template <
        class Index,
        class Source,
        class Target,
        std::size_t Radius,
        class Domain,
        class Element,
        class Evaluate>
KOKKOS_FORCEINLINE_FUNCTION Stencil<Index, Source, Target, Radius> operator_stencil(
        Domain domain,
        Element element,
        Evaluate evaluate)
{
    Stencil<Index, Source, Target, Radius> stencil;
    auto baseline = evaluate([&](auto sample, auto) {
        bool found = false;
        for (std::size_t point = 0; point < StencilShape<Index, Radius>::size; ++point)
            found |= stencil_source<Index, Radius>(domain, element, point) == sample;
        stencil.complete &= found;
        return 0.0;
    });
    for (std::size_t target = 0; target < Target::mem_size(); ++target)
        stencil.offset[target] = baseline.mem(ddc::DiscreteElement<Target>(target));
    for (std::size_t source = 0; source < Source::mem_size(); ++source) {
        auto response = evaluate([&](auto, auto component) {
            return component.template uid<Source>() == source ? 1.0 : 0.0;
        });
        for (std::size_t target = 0; target < Target::mem_size(); ++target)
            stencil.constant_response[target * Source::mem_size() + source]
                    = response.mem(ddc::DiscreteElement<Target>(target)) - stencil.offset[target];
    }
    for (std::size_t point = 0; point < StencilShape<Index, Radius>::size; ++point) {
        Element const sample = stencil_source<Index, Radius>(domain, element, point);
        bool duplicate = false;
        for (std::size_t earlier = 0; earlier < point; ++earlier)
            duplicate |= stencil_source<Index, Radius>(domain, element, earlier) == sample;
        if (duplicate)
            continue;
        for (std::size_t source = 0; source < Source::mem_size(); ++source) {
            auto response = evaluate([&](auto sampled_element, auto component) {
                return sampled_element == sample && component.template uid<Source>() == source
                               ? 1.0
                               : 0.0;
            });
            for (std::size_t target = 0; target < Target::mem_size(); ++target)
                stencil.weights
                        [(target * StencilShape<Index, Radius>::size + point) * Source::mem_size()
                         + source]
                        = response.mem(ddc::DiscreteElement<Target>(target))
                          - stencil.offset[target];
        }
    }
    return stencil;
}

// A lazy field lets the fundamental operator evaluate unit inputs without an allocation.
template <class Index, class Domain, class Memory, class Evaluate>
class StencilField
{
    Domain m_domain;
    Evaluate m_evaluate;

public:
    using memory_space = Memory;
    using non_indices_domain_t = Domain;
    using indices_domain_t = ddc::DiscreteDomain<Index>;
    KOKKOS_FORCEINLINE_FUNCTION StencilField(Domain domain, Evaluate evaluate)
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

template <class Index, class Source, class Target, std::size_t Radius, class Domain, class Memory>
struct StencilStorage
{
    Domain domain;
    Kokkos::View<
            double * [Target::mem_size() * StencilShape<Index, Radius>::size * Source::mem_size()],
            Kokkos::LayoutLeft,
            Memory>
            weights;
    Kokkos::View<double * [Target::mem_size()], Kokkos::LayoutLeft, Memory> offset;
    Kokkos::View<double * [Target::mem_size()][Source::mem_size()], Kokkos::LayoutLeft, Memory>
            constant_response;
};

template <class Operator, class Position>
struct ReconstructionStencilValue
{
    Position position;
    template <class Element>
    KOKKOS_FUNCTION auto operator()(Element element) const
    {
        return Operator::value(position, element);
    }
};

template <class Operator, class Metric, class Position>
struct HodgeStencilValue
{
    Metric metric;
    Position position;
    template <class Element>
    KOKKOS_FUNCTION auto operator()(Element element) const
    {
        return Operator::value(metric, position, element);
    }
};

// Matrix indices enumerate independent form components, not physical directions.
template <std::size_t Component>
struct LocalComponent
{
};

template <class Form, bool Row, class Sequence = std::make_index_sequence<Form::mem_size()>>
struct LocalComponentIndex;

template <class Form, bool Row, std::size_t... Components>
struct LocalComponentIndex<Form, Row, std::index_sequence<Components...>>
    : tensor::TensorNaturalIndex<LocalComponent<Components>...>
{
};

template <class Source, class Target, template <class...> class Coefficients>
using LocalCoefficientIndex
        = Coefficients<LocalComponentIndex<Target, true>, LocalComponentIndex<Source, false>>;

template <
        class Source,
        class Target,
        class Domain,
        class Memory,
        template <class...> class Coefficients = tensor::TensorFullIndex>
struct LocalStencilStorage
{
    Domain domain;
    Kokkos::View<
            double * [LocalCoefficientIndex<Source, Target, Coefficients>::mem_size()],
            Kokkos::LayoutLeft,
            Memory>
            weights;
};

template <
        class Source,
        class Target,
        class Memory,
        template <class...> class Coefficients = tensor::TensorFullIndex,
        class Exec,
        class Domain,
        class Evaluate>
LocalStencilStorage<Source, Target, Domain, Memory, Coefficients> cache_local_stencils(
        Exec const& exec,
        Domain domain,
        Evaluate evaluate)
{
    using Index = LocalCoefficientIndex<Source, Target, Coefficients>;
    static_assert(
            misc::Specialization<Index, tensor::TensorFullIndex>
            || misc::Specialization<Index, tensor::TensorDiagonalIndex>);
    if constexpr (misc::Specialization<Index, tensor::TensorDiagonalIndex>)
        static_assert(Source::mem_size() == Target::mem_size());
    LocalStencilStorage<Source, Target, Domain, Memory, Coefficients> storage;
    storage.domain = domain;
    storage.weights = decltype(storage.weights)(
            Kokkos::view_alloc(Kokkos::WithoutInitializing, "staged_local_stencils"),
            domain.size());
    auto const weights = storage.weights;
    Kokkos::parallel_for(
            "similie_cache_local_stencils",
            Kokkos::RangePolicy<Exec>(exec, 0, domain.size()),
            KOKKOS_LAMBDA(std::size_t node) {
                auto const stencil = evaluate(stencil_element(domain, node));
#ifndef NDEBUG
                if constexpr (misc::Specialization<Index, tensor::TensorDiagonalIndex>) {
                    double scale = 0.0;
                    for (double coefficient : stencil.weights)
                        scale = Kokkos::max(scale, Kokkos::abs(coefficient));
                    for (std::size_t row = 0; row < Target::mem_size(); ++row)
                        for (std::size_t col = 0; col < Source::mem_size(); ++col)
                            assert((row == col
                                    || Kokkos::abs(stencil.weights[row * Source::mem_size() + col])
                                               <= 128.0 * std::numeric_limits<double>::epsilon()
                                                          * scale)
                                   && "Diagonal staged operator requires diagonal coefficients");
                }
#endif
                for (std::size_t entry = 0; entry < Index::mem_size(); ++entry) {
                    auto const ids = Index::mem_id_to_canonical_natural_ids(entry);
                    weights(node, entry) = stencil.weights[ids[0] * Source::mem_size() + ids[1]];
                }
            });
    return storage;
}

struct NoStencilRule
{
};

template <class Operator, class Geometry, class Input, class Element, class Rule1, class Rule2>
KOKKOS_FORCEINLINE_FUNCTION auto stencil_value(
        Geometry const& geometry,
        Input input,
        Element element,
        Rule1 rule1,
        Rule2 rule2)
{
    if constexpr (std::is_same_v<Rule2, NoStencilRule>)
        return Operator::value(geometry, input, element, rule1);
    else
        return Operator::value(geometry, input, element, rule1, rule2);
}

template <
        class Operator,
        class Index,
        class Source,
        class Target,
        std::size_t Radius,
        class Exec,
        class Tensor,
        class Geometry,
        class Rule1,
        class Rule2 = NoStencilRule>
StencilStorage<
        Index,
        Source,
        Target,
        Radius,
        typename Tensor::non_indices_domain_t,
        typename Tensor::memory_space>
cache_stencils(Exec const& exec, Tensor input, Geometry geometry, Rule1 rule1, Rule2 rule2 = {})
{
    using Domain = typename Tensor::non_indices_domain_t;
    using Memory = typename Tensor::memory_space;
    StencilStorage<Index, Source, Target, Radius, Domain, Memory> storage;
    storage.domain = input.non_indices_domain();
    storage.weights = decltype(storage.weights)(
            Kokkos::view_alloc(Kokkos::WithoutInitializing, "staged_stencils"),
            storage.domain.size());
    storage.offset = decltype(storage.offset)(
            Kokkos::view_alloc(Kokkos::WithoutInitializing, "staged_offsets"),
            storage.domain.size());
    storage.constant_response = decltype(storage.constant_response)(
            Kokkos::view_alloc(Kokkos::WithoutInitializing, "staged_constant_response"),
            storage.domain.size());
    Kokkos::View<int, Memory> incomplete("staged_stencil_incomplete");
    auto const weights = storage.weights;
    auto const offset = storage.offset;
    auto const constant_response = storage.constant_response;
    Domain const domain = storage.domain;
    Kokkos::parallel_for(
            "similie_cache_operator_stencils",
            Kokkos::RangePolicy<Exec>(exec, 0, domain.size()),
            KOKKOS_LAMBDA(std::size_t node) {
                auto const stencil = stencil_value<
                        Operator>(geometry, input, stencil_element(domain, node), rule1, rule2);
                if (!stencil.complete)
                    Kokkos::atomic_max(&incomplete(), 1);
                for (std::size_t target = 0; target < Target::mem_size(); ++target) {
                    offset(node, target) = stencil.offset[target];
                    for (std::size_t source = 0; source < Source::mem_size(); ++source)
                        constant_response(node, target, source)
                                = stencil.constant_response[target * Source::mem_size() + source];
                }
                for (std::size_t entry = 0; entry < stencil.weights.size(); ++entry)
                    weights(node, entry) = stencil.weights[entry];
            });
    int failed;
    Kokkos::deep_copy(failed, incomplete);
    exec.fence();
    if (failed)
        throw std::invalid_argument(
                "Boundary closure exceeds the operator's fixed stencil support");
    return storage;
}

template <
        class Index,
        class Source,
        class Target,
        std::size_t Radius,
        class Exec,
        class Output,
        class Input,
        class Domain,
        class Memory>
Output apply_stencils(
        Exec const& exec,
        Output output,
        Input input,
        StencilStorage<Index, Source, Target, Radius, Domain, Memory> storage)
{
    if (input.non_indices_domain() != storage.domain)
        throw std::invalid_argument("Staged operator input domain changed");
    auto const weights = storage.weights;
    auto const offset = storage.offset;
    auto const constant_response = storage.constant_response;
    Domain const domain = storage.domain;
    Domain const output_domain = output.non_indices_domain();
    if (!output_domain.empty()
        && (!misc::domain_contains(domain, output_domain.front())
            || !misc::domain_contains(domain, output_domain.back())))
        throw std::invalid_argument("Staged operator output domain exceeds its cached domain");
    Kokkos::parallel_for(
            "similie_apply_staged_stencil",
            Kokkos::RangePolicy<Exec>(exec, 0, output_domain.size() * Target::mem_size()),
            KOKKOS_LAMBDA(std::size_t row) {
                auto const element = stencil_element(output_domain, row / Target::mem_size());
                std::size_t const node = stencil_node(domain, element),
                                  target = row % Target::mem_size();
                std::array<double, Source::mem_size()> reference;
                double sum = offset(node, target);
                for (std::size_t source = 0; source < Source::mem_size(); ++source) {
                    reference[source] = input.mem(element, ddc::DiscreteElement<Source>(source));
                    sum += constant_response(node, target, source) * reference[source];
                }
                for (std::size_t entry
                     = target * StencilShape<Index, Radius>::size * Source::mem_size();
                     entry < (target + 1) * StencilShape<Index, Radius>::size * Source::mem_size();
                     ++entry) {
                    std::size_t const source = entry % Source::mem_size();
                    std::size_t const point
                            = (entry / Source::mem_size()) % StencilShape<Index, Radius>::size;
                    double const weight = weights(node, entry);
                    if (weight != 0.0)
                        sum += weight
                               * (input
                                          .mem(stencil_source<
                                                       Index,
                                                       Radius>(domain, element, point),
                                               ddc::DiscreteElement<Source>(source))
                                  - reference[source]);
                }
                output.mem(element, ddc::DiscreteElement<Target>(target)) = sum;
            });
    return output;
}

} // namespace detail
} // namespace sil::exterior
