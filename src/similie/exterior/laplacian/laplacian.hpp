// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

#include <array>
#include <cstddef>
#include <limits>
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

// Stencil probes sum delta d and d delta after evaluating each contribution.
// Allow for roundoff in those compositions, relative to their pre-cancellation
// scale. A small coefficient without cancellation is therefore retained.
KOKKOS_FORCEINLINE_FUNCTION double laplacian_contribution_sum(double first, double second)
{
    double const sum = first + second;
    double const scale = Kokkos::max(Kokkos::abs(first), Kokkos::abs(second));
    return Kokkos::isfinite(sum)
                           && Kokkos::abs(sum)
                                      <= 128.0 * std::numeric_limits<double>::epsilon() * scale
                   ? 0.0
                   : sum;
}

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


namespace sil::exterior {

/** Hodge Laplacian assembled from independently selected Hodge-star policies.
 * \important This operator and documentation is fully AI-generated.
 * DiscreteHodgeStar evaluates geometry during application. Wrapping it in
 * StagedOperator freezes Hodge coefficients at construction. The same incidence operators,
 * intermediate cochains and boundary closures are used in both cases.
 * Policies have the DiscreteHodgeStar template signature and pointwise API.
 * Stateful policies are constructed with (execution space, metric, positions);
 * stateless policies may be default-constructed.
 */
template <
        tensor::TensorIndex MetricIndex,
        tensor::TensorNatIndex Index,
        tensor::TensorIndex Cochain,
        misc::Specialization<tensor::Tensor> Tensor,
        misc::Specialization<tensor::Tensor> Metric,
        misc::Specialization<tensor::Tensor> Position,
        class Exec,
        template <CellComplex, class, class, class, class, class> class Hodge = DiscreteHodgeStar>
class Laplacian
{
    Exec m_exec;
    Metric m_metric;
    Position m_position;
    detail::LaplacianUpper<
            (Cochain::rank() < Index::size()),
            Index,
            Cochain,
            Tensor,
            Metric,
            Position,
            Exec,
            Hodge>
            m_upper;
    detail::LaplacianLower<
            (Cochain::rank() > 0),
            Index,
            Cochain,
            Tensor,
            Metric,
            Position,
            Exec,
            Hodge>
            m_lower;

public:
    using staged_storage_type = detail::
            SpatialOperatorStorage<Laplacian, Index, Cochain, Cochain, 2, Tensor, Exec, true>;

    Laplacian(
            Exec const& exec,
            Tensor const& output,
            Tensor const& input,
            Metric const& metric,
            Position const& position)
        : m_exec(exec)
        , m_metric(metric)
        , m_position(position)
        , m_upper(exec, input, metric, position)
        , m_lower(exec, output, input, metric, position)
    {
    }

    template <
            class Primal = tensor::ClampExtrapolationRule,
            class Dual = tensor::ZeroExtrapolationRule>
    Tensor operator()(
            Tensor const& output,
            Tensor const& input,
            Primal const& primal = {},
            Dual const& dual = {})
    {
        if constexpr (Cochain::rank() < Index::size()) {
            using Added = tensor::Covariant<
                    IndexForCodifferentialOfCoboundaryInLaplacian<tensor::uncharacterize_t<Index>>>;
            using ChainDomain = typename detail::
                    NonSpectatorDimension<Index, typename Tensor::non_indices_domain_t>::type;
            auto const chain = tangent_basis<Cochain::rank() + 1, ChainDomain>(m_exec);
            auto const lower = tangent_basis<Cochain::rank(), ChainDomain>(m_exec);
            auto const boundary = [&] {
                if constexpr (misc::Specialization<Primal, tensor::ExtrapolationRules>)
                    return primal;
                else
                    return tensor::make_extrapolation_rules(input, primal);
            }();
            m_upper.operation
                    .apply(m_exec,
                           output,
                           detail::LaplacianDerivativeEvaluator<
                                   Added,
                                   Cochain,
                                   Tensor,
                                   decltype(boundary),
                                   decltype(chain),
                                   decltype(lower)> {input, boundary, chain, lower},
                           m_metric,
                           m_position,
                           dual);
        }
        if constexpr (Cochain::rank() > 0) {
            tensor::Tensor lower(m_lower.lower);
            m_lower.operation
                    .apply(m_exec,
                           lower,
                           detail::LaplacianInputEvaluator<Tensor> {input},
                           m_metric,
                           m_position,
                           dual);
            if constexpr (Cochain::rank() < Index::size()) {
                Tensor contribution(*m_lower.contribution);
                deriv<Index,
                      codifferential_index_t<Index, Cochain>>(m_exec, contribution, lower, primal);
                ddc::parallel_for_each(
                        "similie_add_laplacian_contributions",
                        m_exec,
                        output.domain(),
                        KOKKOS_LAMBDA(typename Tensor::discrete_element_type element) {
                            output.mem(element) += contribution.mem(element);
                        });
            } else
                deriv<Index, codifferential_index_t<Index, Cochain>>(m_exec, output, lower, primal);
        }
        return output;
    }

private:
    friend staged_storage_type;

    // Canonical component matrices are needed only while assembling a stencil.
    static auto geometry(Exec const& exec, Metric const& metric, Position const& position)
    {
        using Domain = typename detail::
                NonSpectatorDimension<Index, typename Tensor::non_indices_domain_t>::type;
        detail::LaplacianChains<0, Domain, typename Tensor::memory_space> const chains {};
        auto derivative = [&] {
            if constexpr (Cochain::rank() < Index::size()) {
                using Added = tensor::Covariant<IndexForCodifferentialOfCoboundaryInLaplacian<
                        tensor::uncharacterize_t<Index>>>;
                return detail::LaplacianHodgePair<
                               Added,
                               coboundary_index_t<Added, Cochain>,
                               Metric,
                               Position,
                               Hodge>(exec, metric, position)
                        .canonical(exec, metric, position);
            } else
                return std::pair {0, 0};
        }();
        auto codifferential = [&] {
            if constexpr (Cochain::rank() > 0)
                return detail::LaplacianHodgePair<
                               Index,
                               Cochain,
                               Metric,
                               Position,
                               Hodge>(exec, metric, position)
                        .canonical(exec, metric, position);
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

public:
    template <class Geometry, class Evaluate, class Element, class Primal, class Dual>
    KOKKOS_FORCEINLINE_FUNCTION static auto operator()(
            Geometry const& staged,
            Tensor const& input,
            Evaluate const& evaluate,
            Element element,
            Primal const& primal,
            Dual const& dual)
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
                output.mem(component) = detail::
                        laplacian_contribution_sum(output.mem(component), result.mem(component));
            });
        }
        return output;
    }

    /** Complete fixed stencil at one node, including component couplings.
     * \important This documentation is fully AI-generated.
     * Two incidence operations reach Manhattan distance two. The fixed support
     * also covers reflected linear boundary closures. All entries are retained.
     */
    template <class Geometry, class Element, class Primal, class Dual>
    KOKKOS_FORCEINLINE_FUNCTION static auto value(
            Geometry const& geometry,
            Tensor const& input,
            Element element,
            Primal const& primal,
            Dual const& dual)
    {
        return detail::operator_stencil<
                Index,
                Cochain,
                Cochain,
                2>(input.non_indices_domain(), element, [&](auto evaluate) {
            return operator()(geometry, input, evaluate, element, primal, dual);
        });
    }

    template <class Geometry, class Element, class Component, class Primal, class Dual>
    KOKKOS_FORCEINLINE_FUNCTION static double value(
            Geometry const& staged,
            Tensor const& input,
            Element element,
            Component target,
            std::size_t column,
            Primal const& primal,
            Dual const& dual)
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

template <
        tensor::TensorIndex MetricIndex,
        tensor::TensorNatIndex Index,
        tensor::TensorIndex Cochain,
        template <CellComplex, class, class, class, class, class> class Hodge = DiscreteHodgeStar,
        class Exec,
        class Tensor,
        class Metric,
        class Position>
Laplacian<MetricIndex, Index, Cochain, Tensor, Metric, Position, Exec, Hodge> make_laplacian(
        Exec const& exec,
        Tensor const& output,
        Tensor const& input,
        Metric const& metric,
        Position const& position)
{
    return {exec, output, input, metric, position};
}

namespace detail {

template <
        tensor::TensorIndex MetricIndex,
        tensor::TensorNatIndex LaplacianDummyIndex,
        tensor::TensorIndex CochainTag,
        misc::Specialization<tensor::Tensor> DualTensorBufferType,
        misc::Specialization<tensor::Tensor> TensorType,
        misc::Specialization<tensor::Tensor> HodgeStarType,
        misc::Specialization<tensor::Tensor> DualHodgeStarType,
        class ExecSpace,
        class PrimalExtrapolationRule = tensor::ClampExtrapolationRule,
        class DualExtrapolationRule = tensor::ZeroExtrapolationRule>
TensorType codifferential_of_coboundary(
        ExecSpace const& exec_space,
        TensorType out_tensor,
        TensorType const& tensor,
        HodgeStarType const& hodge_star,
        DualHodgeStarType const& dual_hodge_star,
        DualTensorBufferType const& dual_tensor_buffer,
        PrimalExtrapolationRule const& primal_extrapolation = {},
        DualExtrapolationRule const& dual_extrapolation = {})
{
    auto const primal_extrapolation_rules = [&]() {
        if constexpr (misc::Specialization<PrimalExtrapolationRule, tensor::ExtrapolationRules>)
            return primal_extrapolation;
        else
            return tensor::make_extrapolation_rules(tensor, primal_extrapolation);
    }();
    auto const dual_extrapolation_rules = [&]() {
        if constexpr (misc::Specialization<DualExtrapolationRule, tensor::ExtrapolationRules>)
            return dual_extrapolation;
        else
            return tensor::make_extrapolation_rules(dual_tensor_buffer, dual_extrapolation);
    }();
    using coboundary_output_index = coboundary_index_t<LaplacianDummyIndex, CochainTag>;
    using codifferential_hodge_output_indices = codifferential_hodge_output_indices_t<
            LaplacianDummyIndex::size() - coboundary_output_index::rank(),
            LaplacianDummyIndex>;
    using coboundary_dual_tensor_index = misc::convert_type_seq_to_t<
            tensor::TensorAntisymmetricIndex,
            codifferential_hodge_output_indices>;
    using dual_codifferential_hodge_input_indices = ddc::type_seq_merge_t<
            ddc::TypeSeq<LaplacianDummyIndex>,
            codifferential_hodge_output_indices>;
    using dual_codifferential_index = misc::convert_type_seq_to_t<
            tensor::TensorAntisymmetricIndex,
            dual_codifferential_hodge_input_indices>;
    auto const chain = tangent_basis<
            CochainTag::rank() + 1,
            typename detail::NonSpectatorDimension<
                    LaplacianDummyIndex,
                    typename TensorType::non_indices_domain_t>::type>(exec_space);
    auto const lower_chain = tangent_basis<
            CochainTag::rank(),
            typename detail::NonSpectatorDimension<
                    LaplacianDummyIndex,
                    typename TensorType::non_indices_domain_t>::type>(exec_space);
    auto dual_chain = tangent_basis<
            coboundary_dual_tensor_index::rank() + 1,
            typename detail::NonSpectatorDimension<
                    LaplacianDummyIndex,
                    typename TensorType::non_indices_domain_t>::type>(exec_space);
    auto dual_lower_chain = tangent_basis<
            coboundary_dual_tensor_index::rank(),
            typename detail::NonSpectatorDimension<
                    LaplacianDummyIndex,
                    typename TensorType::non_indices_domain_t>::type>(exec_space);

    SIMILIE_DEBUG_LOG("similie_deriv_and_apply_first_hodge_star_for_codifferential_of_coboundary");
    ddc::parallel_for_each(
            "similie_deriv_and_apply_first_hodge_star_for_codifferential_of_coboundary",
            exec_space,
            dual_tensor_buffer.non_indices_domain(),
            KOKKOS_LAMBDA(typename TensorType::non_indices_domain_t::discrete_element_type elem) {
                [[maybe_unused]] tensor::TensorAccessor<coboundary_output_index>
                        derivative_accessor;
                std::array<double, coboundary_output_index::mem_size()> derivative_alloc {};
                ddc::ChunkSpan<
                        double,
                        ddc::DiscreteDomain<coboundary_output_index>,
                        Kokkos::layout_right,
                        typename TensorType::memory_space>
                        derivative_span(derivative_alloc.data(), derivative_accessor.domain());
                sil::tensor::Tensor derivative_tensor(derivative_span);

                Coboundary<LaplacianDummyIndex, CochainTag>::operator()(
                        derivative_tensor,
                        [&](auto sampled_elem, auto cochain_elem) {
                            return primal_extrapolation_rules(tensor, sampled_elem)
                                    .mem(cochain_elem);
                        },
                        chain,
                        lower_chain,
                        elem);

                sil::tensor::
                        tensor_prod(dual_tensor_buffer[elem], derivative_tensor, hodge_star[elem]);
            });

    SIMILIE_DEBUG_LOG("similie_deriv_and_apply_second_hodge_star_for_codifferential_of_coboundary");
    ddc::parallel_for_each(
            "similie_deriv_and_apply_second_hodge_star_for_codifferential_of_coboundary",
            exec_space,
            out_tensor.non_indices_domain(),
            KOKKOS_LAMBDA(typename TensorType::non_indices_domain_t::discrete_element_type elem) {
                [[maybe_unused]] tensor::TensorAccessor<dual_codifferential_index>
                        dual_codifferential_accessor;
                std::array<double, dual_codifferential_index::mem_size()>
                        dual_codifferential_alloc {};
                ddc::ChunkSpan<
                        double,
                        ddc::DiscreteDomain<dual_codifferential_index>,
                        Kokkos::layout_right,
                        typename TensorType::memory_space>
                        dual_codifferential_span(
                                dual_codifferential_alloc.data(),
                                dual_codifferential_accessor.domain());
                sil::tensor::Tensor dual_codifferential(dual_codifferential_span);

                TransposedCoboundary<LaplacianDummyIndex, coboundary_dual_tensor_index>::operator()(
                        dual_codifferential,
                        [&](auto sampled_elem, auto dual_elem) {
                            return dual_extrapolation_rules(dual_tensor_buffer, sampled_elem)
                                    .mem(dual_elem);
                        },
                        dual_chain,
                        dual_lower_chain,
                        elem);

                sil::tensor::
                        tensor_prod(out_tensor[elem], dual_codifferential, dual_hodge_star[elem]);
                if constexpr (
                        (LaplacianDummyIndex::size() * (coboundary_output_index::rank() + 1) + 1)
                                % 2
                        == 1) {
                    out_tensor[elem] *= -1;
                }
            });

    return out_tensor;
}

template <class LaplacianDummyIndex, class CochainTag>
concept ZeroRankLaplacianCochain = CochainTag::rank() == 0;

template <class LaplacianDummyIndex, class CochainTag>
concept IntermediateRankLaplacianCochain
        = CochainTag::rank() > 0 && CochainTag::rank() < LaplacianDummyIndex::size();

template <class LaplacianDummyIndex, class CochainTag>
concept TopRankLaplacianCochain = CochainTag::rank() == LaplacianDummyIndex::size();

} // namespace detail

template <
        tensor::TensorIndex MetricIndex,
        tensor::TensorNatIndex LaplacianDummyIndex,
        tensor::TensorIndex CochainTag,
        misc::Specialization<tensor::Tensor> TensorType,
        misc::Specialization<tensor::Tensor> DerivativeHodgeStarType,
        misc::Specialization<tensor::Tensor> DualDerivativeHodgeStarType,
        misc::Specialization<tensor::Tensor> DerivativeDualTensorBufferType,
        class ExecSpace,
        class PrimalExtrapolationRule = tensor::ClampExtrapolationRule,
        class DualExtrapolationRule = tensor::ZeroExtrapolationRule>
    requires(detail::ZeroRankLaplacianCochain<LaplacianDummyIndex, CochainTag>)
TensorType laplacian(
        ExecSpace const& exec_space,
        TensorType const& laplacian_tensor,
        TensorType const& tensor,
        DerivativeHodgeStarType const& hodge_star,
        DualDerivativeHodgeStarType const& dual_hodge_star,
        DerivativeDualTensorBufferType const& dual_tensor_buffer,
        PrimalExtrapolationRule const& primal_extrapolation = {},
        DualExtrapolationRule const& dual_extrapolation = {})
{
    using codifferential_of_coboundary_index
            = tensor::Covariant<IndexForCodifferentialOfCoboundaryInLaplacian<
                    tensor::uncharacterize_t<LaplacianDummyIndex>>>;
    return detail::codifferential_of_coboundary<
            MetricIndex,
            codifferential_of_coboundary_index,
            CochainTag>(
            exec_space,
            laplacian_tensor,
            tensor,
            hodge_star,
            dual_hodge_star,
            dual_tensor_buffer,
            primal_extrapolation,
            dual_extrapolation);
}

template <
        tensor::TensorIndex MetricIndex,
        tensor::TensorNatIndex LaplacianDummyIndex,
        tensor::TensorIndex CochainTag,
        misc::Specialization<tensor::Tensor> TensorType,
        misc::Specialization<tensor::Tensor> DerivativeHodgeStarType,
        misc::Specialization<tensor::Tensor> DualDerivativeHodgeStarType,
        misc::Specialization<tensor::Tensor> DerivativeDualTensorBufferType,
        misc::Specialization<tensor::Tensor> HodgeStarType,
        misc::Specialization<tensor::Tensor> DualHodgeStarType,
        misc::Specialization<tensor::Tensor> DualTensorBufferType,
        misc::Specialization<tensor::Tensor> CodifferentialTensorBufferType,
        misc::Specialization<tensor::Tensor> CoboundaryOfCodifferentialBufferType,
        class ExecSpace,
        class PrimalExtrapolationRule = tensor::ClampExtrapolationRule,
        class DualExtrapolationRule = tensor::ZeroExtrapolationRule>
    requires(detail::IntermediateRankLaplacianCochain<LaplacianDummyIndex, CochainTag>)
TensorType laplacian(
        ExecSpace const& exec_space,
        TensorType const& laplacian_tensor,
        TensorType const& tensor,
        DerivativeHodgeStarType const& derivative_hodge_star,
        DualDerivativeHodgeStarType const& dual_derivative_hodge_star,
        DerivativeDualTensorBufferType const& derivative_dual_tensor_buffer,
        HodgeStarType const& hodge_star,
        DualHodgeStarType const& dual_hodge_star,
        DualTensorBufferType const& dual_tensor_buffer,
        CodifferentialTensorBufferType const& codifferential_tensor_buffer,
        CoboundaryOfCodifferentialBufferType const& coboundary_of_codifferential_buffer,
        PrimalExtrapolationRule const& primal_extrapolation = {},
        DualExtrapolationRule const& dual_extrapolation = {})
{
    using codifferential_of_coboundary_index
            = tensor::Covariant<IndexForCodifferentialOfCoboundaryInLaplacian<
                    tensor::uncharacterize_t<LaplacianDummyIndex>>>;
    auto exec_spaces = Kokkos::Experimental::partition_space(exec_space, 1, 1);

    detail::codifferential_of_coboundary<
            MetricIndex,
            codifferential_of_coboundary_index,
            CochainTag>(
            exec_spaces[0],
            laplacian_tensor,
            tensor,
            derivative_hodge_star,
            dual_derivative_hodge_star,
            derivative_dual_tensor_buffer,
            primal_extrapolation,
            dual_extrapolation);

    sil::exterior::codifferential<MetricIndex, LaplacianDummyIndex, CochainTag>(
            exec_spaces[1],
            codifferential_tensor_buffer,
            tensor,
            hodge_star,
            dual_hodge_star,
            dual_tensor_buffer,
            dual_extrapolation);
    sil::exterior::
            deriv<LaplacianDummyIndex, codifferential_index_t<LaplacianDummyIndex, CochainTag>>(
                    exec_spaces[1],
                    coboundary_of_codifferential_buffer,
                    codifferential_tensor_buffer,
                    primal_extrapolation);

    exec_spaces[0].fence();
    exec_spaces[1].fence();

    SIMILIE_DEBUG_LOG("similie_add_coboundary_of_codifferential_contribution_to_laplacian");
    ddc::parallel_for_each(
            "similie_add_coboundary_of_codifferential_contribution_to_laplacian",
            exec_space,
            laplacian_tensor.domain(),
            KOKKOS_LAMBDA(typename TensorType::discrete_element_type elem) {
                laplacian_tensor.mem(elem) += coboundary_of_codifferential_buffer.mem(elem);
            });

    return laplacian_tensor;
}

template <
        tensor::TensorIndex MetricIndex,
        tensor::TensorNatIndex LaplacianDummyIndex,
        tensor::TensorIndex CochainTag,
        misc::Specialization<tensor::Tensor> TensorType,
        misc::Specialization<tensor::Tensor> HodgeStarType,
        misc::Specialization<tensor::Tensor> DualHodgeStarType,
        misc::Specialization<tensor::Tensor> DualTensorBufferType,
        misc::Specialization<tensor::Tensor> CodifferentialTensorBufferType,
        class ExecSpace,
        class PrimalExtrapolationRule = tensor::ClampExtrapolationRule,
        class DualExtrapolationRule = tensor::ZeroExtrapolationRule>
    requires(detail::TopRankLaplacianCochain<LaplacianDummyIndex, CochainTag>)
TensorType laplacian(
        ExecSpace const& exec_space,
        TensorType const& laplacian_tensor,
        TensorType const& tensor,
        HodgeStarType const& hodge_star,
        DualHodgeStarType const& dual_hodge_star,
        DualTensorBufferType const& dual_tensor_buffer,
        CodifferentialTensorBufferType const& codifferential_tensor_buffer,
        PrimalExtrapolationRule const& primal_extrapolation = {},
        DualExtrapolationRule const& dual_extrapolation = {})
{
    sil::exterior::codifferential<MetricIndex, LaplacianDummyIndex, CochainTag>(
            exec_space,
            codifferential_tensor_buffer,
            tensor,
            hodge_star,
            dual_hodge_star,
            dual_tensor_buffer,
            dual_extrapolation);
    return sil::exterior::
            deriv<LaplacianDummyIndex, codifferential_index_t<LaplacianDummyIndex, CochainTag>>(
                    exec_space,
                    laplacian_tensor,
                    codifferential_tensor_buffer,
                    primal_extrapolation);
}


} // namespace sil::exterior
