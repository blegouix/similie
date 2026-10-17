// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

#include <similie/exterior/laplacian/laplacian_detail.hpp>

namespace sil::exterior {

template <
        tensor::TensorIndex MetricIndex,
        tensor::TensorNatIndex Index,
        tensor::TensorIndex Cochain,
        misc::Specialization<tensor::Tensor> Tensor,
        misc::Specialization<tensor::Tensor> Metric,
        misc::Specialization<tensor::Tensor> Position,
        class Exec>
class StagedLaplacian;

/** Hodge Laplacian assembled from independently selected Hodge-star policies.
 * \important This operator and documentation is fully AI-generated.
 * DiscreteHodgeStar evaluates geometry during application. StagedHodgeStar
 * freezes Hodge coefficients at construction. The same incidence operators,
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
    Laplacian(Exec const& exec, Tensor output, Tensor input, Metric metric, Position position)
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
    Tensor operator()(Tensor output, Tensor input, Primal primal = {}, Dual dual = {})
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
    friend class StagedLaplacian<MetricIndex, Index, Cochain, Tensor, Metric, Position, Exec>;

    // Canonical component matrices are needed only while assembling a stencil.
    static auto geometry(Exec const& exec, Metric metric, Position position)
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

    /** Complete fixed stencil at one node, including component couplings.
     * \important This documentation is fully AI-generated.
     * Two incidence operations reach Manhattan distance two. The fixed support
     * also covers reflected linear boundary closures. All entries are retained.
     */
    template <class Geometry, class Element, class Primal, class Dual>
    KOKKOS_FORCEINLINE_FUNCTION static auto value(
            Geometry const& geometry,
            Tensor input,
            Element element,
            Primal primal,
            Dual dual)
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
        Tensor output,
        Tensor input,
        Metric metric,
        Position position)
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
        TensorType tensor,
        HodgeStarType hodge_star,
        DualHodgeStarType dual_hodge_star,
        DualTensorBufferType dual_tensor_buffer,
        PrimalExtrapolationRule primal_extrapolation = {},
        DualExtrapolationRule dual_extrapolation = {})
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
    auto chain = tangent_basis<
            CochainTag::rank() + 1,
            typename detail::NonSpectatorDimension<
                    LaplacianDummyIndex,
                    typename TensorType::non_indices_domain_t>::type>(exec_space);
    auto lower_chain = tangent_basis<
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
        TensorType laplacian_tensor,
        TensorType tensor,
        DerivativeHodgeStarType hodge_star,
        DualDerivativeHodgeStarType dual_hodge_star,
        DerivativeDualTensorBufferType dual_tensor_buffer,
        PrimalExtrapolationRule primal_extrapolation = {},
        DualExtrapolationRule dual_extrapolation = {})
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
        TensorType laplacian_tensor,
        TensorType tensor,
        DerivativeHodgeStarType derivative_hodge_star,
        DualDerivativeHodgeStarType dual_derivative_hodge_star,
        DerivativeDualTensorBufferType derivative_dual_tensor_buffer,
        HodgeStarType hodge_star,
        DualHodgeStarType dual_hodge_star,
        DualTensorBufferType dual_tensor_buffer,
        CodifferentialTensorBufferType codifferential_tensor_buffer,
        CoboundaryOfCodifferentialBufferType coboundary_of_codifferential_buffer,
        PrimalExtrapolationRule primal_extrapolation = {},
        DualExtrapolationRule dual_extrapolation = {})
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
        TensorType laplacian_tensor,
        TensorType tensor,
        HodgeStarType hodge_star,
        DualHodgeStarType dual_hodge_star,
        DualTensorBufferType dual_tensor_buffer,
        CodifferentialTensorBufferType codifferential_tensor_buffer,
        PrimalExtrapolationRule primal_extrapolation = {},
        DualExtrapolationRule dual_extrapolation = {})
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

#include <similie/exterior/laplacian/staged_laplacian.hpp>
