// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#pragma once

#include <array>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include <similie/exterior/codifferential/codifferential.hpp>
#include <similie/exterior/stencil.hpp>

namespace sil::exterior {

template <class T>
struct IndexForCodifferentialOfCoboundaryInLaplacian : T
{
};


namespace detail {

template <std::size_t Rank, class Domain, class Memory>
using LaplacianChains = StencilChains<Rank, Domain, Memory>;

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
template <class Source, class Target, class Exec, class Hodge, class Metric, class Position>
LaplacianHodge<Source, Target, typename Metric::non_indices_domain_t, typename Metric::memory_space>
cache_laplacian_hodge(
        Exec const& exec,
        Hodge const& hodge,
        Metric const& metric,
        Position const& position)
{
    using Domain = typename Metric::non_indices_domain_t;
    using Memory = typename Metric::memory_space;
    Domain const domain = metric.non_indices_domain();
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
                std::size_t const node = [&] {
                    std::size_t result = 0;
                    for (std::size_t axis = 0; axis < Domain::rank(); ++axis)
                        result = result * ddc::detail::array(domain.extents())[axis]
                                 + ddc::detail::array(element)[axis]
                                 - ddc::detail::array(domain.front())[axis];
                    return result;
                }();
                for (std::size_t source = 0; source < Source::mem_size(); ++source) {
                    unit.mem(ddc::DiscreteElement<Source>(source)) = 1.0;
                    hodge(tensor::Tensor(output.span_view()),
                          tensor::Tensor(unit.span_view()),
                          metric,
                          position,
                          element);
                    for (std::size_t target = 0; target < Target::mem_size(); ++target)
                        coefficients(node, target, source)
                                = output.mem(ddc::DiscreteElement<Target>(target));
                    unit.mem(ddc::DiscreteElement<Source>(source)) = 0.0;
                }
            });
    // The temporary Hodge policy may own coefficients read by this kernel.
    exec.fence();
    return {domain, coefficients};
}

template <class Hodge, class Exec, class Metric, class Position>
Hodge make_laplacian_hodge(Exec const& exec, Metric const& metric, Position const& position)
{
    if constexpr (std::is_constructible_v<Hodge, Exec const&, Metric, Position>)
        return Hodge(exec, metric, position);
    else
        return Hodge {};
}

template <
        class Index,
        class Cochain,
        class Metric,
        class Position,
        template <CellComplex, class, class, class, class, class> class Hodge>
struct LaplacianHodgePair
{
    using SourceIndices = tensor::upper_t<ddc::to_type_seq_t<tensor::natural_domain_t<Cochain>>>;
    using TargetIndices
            = codifferential_hodge_output_indices_t<Index::size() - Cochain::rank(), Index>;
    using InverseSourceIndices = ddc::type_seq_merge_t<ddc::TypeSeq<Index>, TargetIndices>;
    using InverseTargetIndices
            = ddc::type_seq_remove_t<tensor::lower_t<SourceIndices>, ddc::TypeSeq<Index>>;
    using Dual = misc::convert_type_seq_to_t<tensor::TensorAntisymmetricIndex, TargetIndices>;
    using Intermediate
            = misc::convert_type_seq_to_t<tensor::TensorAntisymmetricIndex, InverseSourceIndices>;
    using Element = typename Metric::non_indices_domain_t::discrete_element_type;
    using Forward
            = Hodge<CellComplex::Primal, SourceIndices, TargetIndices, Metric, Position, Element>;
    using Inverse
            = Hodge<CellComplex::Primal,
                    tensor::upper_t<InverseSourceIndices>,
                    InverseTargetIndices,
                    Metric,
                    Position,
                    Element>;
    static_assert(DiscreteHodgeStarConcept<
                  Forward,
                  tensor::Tensor<
                          double,
                          ddc::DiscreteDomain<Dual>,
                          Kokkos::layout_right,
                          typename Metric::memory_space>,
                  tensor::Tensor<
                          double,
                          ddc::DiscreteDomain<Cochain>,
                          Kokkos::layout_right,
                          typename Metric::memory_space>,
                  Metric,
                  Position,
                  Element,
                  typename tensor::tensor_accessor_for_domain_t<
                          hodge_star_domain_t<SourceIndices, TargetIndices>>::natural_domain_t::
                          discrete_element_type>);
    static_assert(DiscreteHodgeStarConcept<
                  Inverse,
                  tensor::Tensor<
                          double,
                          ddc::DiscreteDomain<codifferential_index_t<Index, Cochain>>,
                          Kokkos::layout_right,
                          typename Metric::memory_space>,
                  tensor::Tensor<
                          double,
                          ddc::DiscreteDomain<Intermediate>,
                          Kokkos::layout_right,
                          typename Metric::memory_space>,
                  Metric,
                  Position,
                  Element,
                  typename tensor::tensor_accessor_for_domain_t<hodge_star_domain_t<
                          tensor::upper_t<InverseSourceIndices>,
                          InverseTargetIndices>>::natural_domain_t::discrete_element_type>);
    Forward forward;
    Inverse inverse;
    template <class Exec>
    LaplacianHodgePair(Exec const& exec, Metric const& metric, Position const& position)
        : forward(make_laplacian_hodge<Forward>(exec, metric, position))
        , inverse(make_laplacian_hodge<Inverse>(exec, metric, position))
    {
    }
    template <class Exec>
    auto canonical(Exec const& exec, Metric const& metric, Position const& position) const
    {
        return std::
                pair {cache_laplacian_hodge<Cochain, Dual>(exec, forward, metric, position),
                      cache_laplacian_hodge<
                              Intermediate,
                              codifferential_index_t<
                                      Index,
                                      Cochain>>(exec, inverse, metric, position)};
    }
};

template <class A, class B, class C, class D, class Chains>
struct LaplacianGeometry
{
    A derivative_hodge;
    B inverse_derivative_hodge;
    C hodge;
    D inverse_hodge;
    Chains chains;
};

template <class Index, class Domain, class Memory, class Evaluate>
using LaplacianField = StencilField<Index, Domain, Memory, Evaluate>;

template <class Index, class Field, class Rules, class Element, class Chains>
KOKKOS_FORCEINLINE_FUNCTION auto laplacian_derivative(
        Field const& field,
        Rules const& rules,
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
        Field const& field,
        Hodge const& hodge,
        Inverse const& inverse,
        Rules const& rules,
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
KOKKOS_FORCEINLINE_FUNCTION std::size_t laplacian_node_id(Domain const& domain, Element element)
{
    std::size_t id = 0;
    for (std::size_t axis = 0; axis < Domain::rank(); ++axis)
        id = id * ddc::detail::array(domain.extents())[axis] + ddc::detail::array(element)[axis]
             - ddc::detail::array(domain.front())[axis];
    return id;
}

template <class Domain>
KOKKOS_FORCEINLINE_FUNCTION typename Domain::discrete_element_type laplacian_element(
        Domain const& domain,
        std::size_t id)
{
    typename Domain::discrete_element_type element = domain.front();
    for (std::size_t axis = Domain::rank(); axis-- > 0;) {
        ddc::detail::array(element)[axis] += id % ddc::detail::array(domain.extents())[axis];
        id /= ddc::detail::array(domain.extents())[axis];
    }
    return element;
}

template <class Tensor>
struct LaplacianInputEvaluator
{
    Tensor input;
    template <class Output, class Element>
    KOKKOS_FUNCTION void operator()(Output const& output, Element element) const
    {
        ddc::device_for_each(output.domain(), [&](auto component) {
            output.mem(component) = input.mem(element, component);
        });
    }
};

template <class Index, class Cochain, class Tensor, class Boundary, class Chain, class Lower>
struct LaplacianDerivativeEvaluator
{
    Tensor input;
    Boundary boundary;
    Chain chain;
    Lower lower;
    template <class Output, class Element>
    KOKKOS_FUNCTION void operator()(Output const& output, Element element) const
    {
        Coboundary<Index, Cochain>::operator()(
                output,
                [&](auto sample, auto component) { return boundary(input, sample).mem(component); },
                chain,
                lower,
                element);
    }
};

template <
        class Index,
        class Cochain,
        class Tensor,
        class Metric,
        class Position,
        class Exec,
        template <CellComplex, class, class, class, class, class> class Hodge>
struct LaplacianCodifferential
{
    using Pair = LaplacianHodgePair<Index, Cochain, Metric, Position, Hodge>;
    using DualIndex = typename Pair::Dual;
    using Memory = typename Tensor::memory_space;
    using Domain = misc::
            cartesian_prod_t<typename Tensor::non_indices_domain_t, ddc::DiscreteDomain<DualIndex>>;
    Pair stars;
    ddc::Chunk<double, Domain, ddc::KokkosAllocator<double, Memory>> buffer;

    LaplacianCodifferential(
            Exec const& exec,
            Tensor const& input,
            Metric const& metric,
            Position const& position)
        : stars(exec, metric, position)
        , buffer(Domain(input.non_indices_domain(), tensor::TensorAccessor<DualIndex>::domain()),
                 ddc::KokkosAllocator<double, Memory>())
    {
    }

    template <class Output, class Evaluate, class Rules>
    void apply(
            Exec const& exec,
            Output output,
            Evaluate const& evaluate,
            Metric const& metric,
            Position const& position,
            Rules const& rules)
    {
        tensor::Tensor const dual(buffer.span_view());
        auto const& forward = stars.forward;
        auto const& inverse = stars.inverse;
        auto const boundary = [&] {
            if constexpr (misc::Specialization<Rules, tensor::ExtrapolationRules>)
                return rules;
            else
                return tensor::make_extrapolation_rules(dual, rules);
        }();
        using ChainDomain =
                typename NonSpectatorDimension<Index, typename Tensor::non_indices_domain_t>::type;
        auto const chain = tangent_basis<DualIndex::rank() + 1, ChainDomain>(exec);
        auto const lower = tangent_basis<DualIndex::rank(), ChainDomain>(exec);
        ddc::parallel_for_each(
                "similie_laplacian_first_hodge",
                exec,
                dual.non_indices_domain(),
                KOKKOS_LAMBDA(
                        typename Tensor::non_indices_domain_t::discrete_element_type element) {
                    auto point = make_stencil<Memory, Cochain>(ddc::DiscreteElement<>());
                    evaluate(tensor::Tensor(point.span_view()), element);
                    forward(dual[element],
                            tensor::Tensor(point.span_view()),
                            metric,
                            position,
                            element);
                });
        ddc::parallel_for_each(
                "similie_laplacian_second_hodge",
                exec,
                output.non_indices_domain(),
                KOKKOS_LAMBDA(
                        typename Tensor::non_indices_domain_t::discrete_element_type element) {
                    auto derivative = make_stencil<Memory, typename Pair::Intermediate>(
                            ddc::DiscreteElement<>());
                    TransposedCoboundary<Index, DualIndex>::operator()(
                            tensor::Tensor(derivative.span_view()),
                            [&](auto sample, auto component) {
                                return boundary(dual, sample).mem(component);
                            },
                            chain,
                            lower,
                            element);
                    inverse(output[element],
                            tensor::Tensor(derivative.span_view()),
                            metric,
                            position,
                            element);
                    if constexpr ((Index::size() * (Cochain::rank() + 1) + 1) % 2 == 1)
                        output[element] *= -1;
                });
    }
};

// Disabled endpoint terms do not instantiate nonexistent cochain degrees.
struct EmptyLaplacianTerm
{
    template <class... Args>
    EmptyLaplacianTerm(Args const&...)
    {
    }
};

template <
        bool Enabled,
        class Index,
        class Cochain,
        class Tensor,
        class Metric,
        class Position,
        class Exec,
        template <CellComplex, class, class, class, class, class> class Hodge>
struct LaplacianUpper : EmptyLaplacianTerm
{
    using EmptyLaplacianTerm::EmptyLaplacianTerm;
};

template <
        class Index,
        class Cochain,
        class Tensor,
        class Metric,
        class Position,
        class Exec,
        template <CellComplex, class, class, class, class, class> class Hodge>
struct LaplacianUpper<true, Index, Cochain, Tensor, Metric, Position, Exec, Hodge>
{
    using Added = tensor::Covariant<
            IndexForCodifferentialOfCoboundaryInLaplacian<tensor::uncharacterize_t<Index>>>;
    LaplacianCodifferential<
            Added,
            coboundary_index_t<Added, Cochain>,
            Tensor,
            Metric,
            Position,
            Exec,
            Hodge>
            operation;
    LaplacianUpper(
            Exec const& exec,
            Tensor const& input,
            Metric const& metric,
            Position const& position)
        : operation(exec, input, metric, position)
    {
    }
};

template <
        bool Enabled,
        class Index,
        class Cochain,
        class Tensor,
        class Metric,
        class Position,
        class Exec,
        template <CellComplex, class, class, class, class, class> class Hodge>
struct LaplacianLower : EmptyLaplacianTerm
{
    using EmptyLaplacianTerm::EmptyLaplacianTerm;
};

template <
        class Index,
        class Cochain,
        class Tensor,
        class Metric,
        class Position,
        class Exec,
        template <CellComplex, class, class, class, class, class> class Hodge>
struct LaplacianLower<true, Index, Cochain, Tensor, Metric, Position, Exec, Hodge>
{
    using Memory = typename Tensor::memory_space;
    using LowerDomain = misc::cartesian_prod_t<
            typename Tensor::non_indices_domain_t,
            ddc::DiscreteDomain<codifferential_index_t<Index, Cochain>>>;
    LaplacianCodifferential<Index, Cochain, Tensor, Metric, Position, Exec, Hodge> operation;
    ddc::Chunk<double, LowerDomain, ddc::KokkosAllocator<double, Memory>> lower;
    std::optional<ddc::Chunk<
            double,
            typename Tensor::discrete_domain_type,
            ddc::KokkosAllocator<double, Memory>>>
            contribution;
    LaplacianLower(
            Exec const& exec,
            Tensor const& output,
            Tensor const& input,
            Metric const& metric,
            Position const& position)
        : operation(exec, input, metric, position)
        , lower(LowerDomain(
                        input.non_indices_domain(),
                        tensor::TensorAccessor<codifferential_index_t<Index, Cochain>>::domain()),
                ddc::KokkosAllocator<double, Memory>())
    {
        if constexpr (Cochain::rank() < Index::size())
            contribution.emplace(output.domain(), ddc::KokkosAllocator<double, Memory>());
    }
};

} // namespace detail
} // namespace sil::exterior
