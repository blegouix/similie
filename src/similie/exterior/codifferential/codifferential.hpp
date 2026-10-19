// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

#include <array>
#include <optional>
#include <utility>

#include <ddc/ddc.hpp>

#include <similie/exterior/coboundary/coboundary.hpp>
#include <similie/exterior/cochain.hpp>
#include <similie/exterior/concepts.hpp>
#include <similie/exterior/continuous_hodge_star/continuous_hodge_star.hpp>
#include <similie/exterior/cosimplex.hpp>
#include <similie/exterior/stencil.hpp>
#include <similie/misc/domain_contains.hpp>
#include <similie/misc/macros.hpp>
#include <similie/misc/specialization.hpp>
#include <similie/misc/type_seq_ext.hpp>
#include <similie/tensor/tensor_impl.hpp>
#include <similie/tensor/tensor_indices/character.hpp>

#include <Kokkos_StdAlgorithms.hpp>


namespace sil {

namespace exterior {

namespace detail {

template <class T>
struct CodifferentialType;

template <
        std::size_t K,
        class... Tag,
        class ElementType,
        class LayoutStridedPolicy1,
        class LayoutStridedPolicy2,
        class ExecSpace>
struct CodifferentialType<
        Cochain<Chain<Simplex<K, Tag...>, LayoutStridedPolicy1, ExecSpace>,
                ElementType,
                LayoutStridedPolicy2>>
{
    using type = Cosimplex<Simplex<K - 1, Tag...>, ElementType>;
};

} // namespace detail

template <misc::Specialization<Cochain> CochainType>
using codifferential_t = typename detail::CodifferentialType<CochainType>::type;

namespace detail {

template <class TagToRemoveFromCochain, class CochainTag>
struct CodifferentialIndex;

template <tensor::TensorNatIndex TagToRemoveFromCochain, tensor::TensorNatIndex CochainTag>
    requires(CochainTag::rank() == 1 && std::is_same_v<TagToRemoveFromCochain, CochainTag>)
struct CodifferentialIndex<TagToRemoveFromCochain, CochainTag>
{
    using type = tensor::Covariant<tensor::ScalarIndex>;
};

template <tensor::TensorNatIndex TagToRemoveFromCochain, tensor::TensorNatIndex Tag>
struct CodifferentialIndex<
        TagToRemoveFromCochain,
        tensor::TensorAntisymmetricIndex<TagToRemoveFromCochain, Tag>>
{
    using type = Tag;
};

template <tensor::TensorNatIndex TagToRemoveFromCochain, tensor::TensorNatIndex... Tag>
    requires(sizeof...(Tag) > 1)
struct CodifferentialIndex<
        TagToRemoveFromCochain,
        tensor::TensorAntisymmetricIndex<TagToRemoveFromCochain, Tag...>>
{
    using type = tensor::TensorAntisymmetricIndex<Tag...>;
};

} // namespace detail

template <class TagToRemoveFromCochain, class CochainTag>
using codifferential_index_t =
        typename detail::CodifferentialIndex<TagToRemoveFromCochain, CochainTag>::type;

namespace detail {

template <
        tensor::TensorNatIndex TagToRemoveFromCochain,
        tensor::TensorIndex CochainTag,
        misc::Specialization<tensor::Tensor> TensorType>
struct CodifferentialTensorType;

template <
        tensor::TensorNatIndex TagToRemoveFromCochain,
        tensor::TensorIndex CochainIndex,
        class ElementType,
        class... DDim,
        class SupportType,
        class MemorySpace>
    requires(ddc::type_seq_contains_v<ddc::TypeSeq<CochainIndex>, ddc::TypeSeq<DDim...>>)
struct CodifferentialTensorType<
        TagToRemoveFromCochain,
        CochainIndex,
        tensor::Tensor<ElementType, ddc::DiscreteDomain<DDim...>, SupportType, MemorySpace>>
{
    using type = tensor::Tensor<
            ElementType,
            ddc::replace_dim_of_t<
                    ddc::DiscreteDomain<DDim...>,
                    CochainIndex,
                    codifferential_index_t<TagToRemoveFromCochain, CochainIndex>>,
            SupportType,
            MemorySpace>;
};

} // namespace detail

template <
        tensor::TensorNatIndex TagToRemoveFromCochain,
        tensor::TensorIndex CochainTag,
        misc::Specialization<tensor::Tensor> TensorType>
using codifferential_tensor_t = typename detail::
        CodifferentialTensorType<TagToRemoveFromCochain, CochainTag, TensorType>::type;


template <class... Args>
struct Codifferential;

/** Pointwise codifferential with independently supplied Hodge operators.
 * \important This operator and documentation is fully AI-generated.
 * The evaluator samples the dual cochain *u, including its boundary policies.
 * The inverse Hodge maps its transposed coboundary back to a primal (k-1)-form.
 * Both geometric realizations use delta = (-1)^(n(k+1)+1) *^-1 d^T *;
 * incidence and the sign are independent of the chosen Hodge discretization.
 * Callbacks permit stored dual tensors and lazy constitutive cochains alike.
 */
template <tensor::TensorNatIndex TagToRemoveFromCochain, tensor::TensorIndex CochainTag>
struct Codifferential<TagToRemoveFromCochain, CochainTag>
{
    template <
            class Output,
            class DualEvaluator,
            class InverseHodge,
            class Chain,
            class LowerChain,
            class Element>
    KOKKOS_FUNCTION static void operator()(
            Output output,
            DualEvaluator const& evaluate_dual,
            InverseHodge const& inverse_hodge,
            Chain const& chain,
            LowerChain const& lower_chain,
            Element elem)
    {
        using DualIndices = codifferential_hodge_output_indices_t<
                TagToRemoveFromCochain::size() - CochainTag::rank(),
                TagToRemoveFromCochain>;
        using DualIndex
                = misc::convert_type_seq_to_t<tensor::TensorAntisymmetricIndex, DualIndices>;
        using TargetIndex = misc::convert_type_seq_to_t<
                tensor::TensorAntisymmetricIndex,
                ddc::type_seq_merge_t<ddc::TypeSeq<TagToRemoveFromCochain>, DualIndices>>;
        auto dual = detail::make_stencil<typename Output::memory_space, TargetIndex>(
                ddc::DiscreteElement<>());
        TransposedCoboundary<TagToRemoveFromCochain, DualIndex>::
        operator()(tensor::Tensor(dual.span_view()), evaluate_dual, chain, lower_chain, elem);
        inverse_hodge(output, tensor::Tensor(dual.span_view()), elem);
        if constexpr ((TagToRemoveFromCochain::size() * (CochainTag::rank() + 1) + 1) % 2 == 1)
            output *= -1;
    }

    /** Complete cochain stencil, with geometry and closure supplied separately.
     * \important This documentation is fully AI-generated.
     */
    template <class Geometry, class Field, class Element, class Rules>
        requires requires(Geometry const& geometry) {
            geometry.forward;
            geometry.inverse;
        }
    KOKKOS_FORCEINLINE_FUNCTION static auto value(
            Geometry const& geometry,
            Field const& input,
            Element element,
            Rules const& rules)
    {
        using Memory = typename Field::memory_space;
        using Domain = typename Field::non_indices_domain_t;
        using DualIndex = typename Geometry::DualIndex;
        return detail::operator_stencil<
                TagToRemoveFromCochain,
                CochainTag,
                codifferential_index_t<TagToRemoveFromCochain, CochainTag>,
                1>(input.non_indices_domain(), element, [&](auto evaluate) {
            auto evaluate_dual = [&](auto sample, auto component) {
                auto primal = detail::make_stencil<Memory, CochainTag>(ddc::DiscreteElement<>());
                auto dual = detail::make_stencil<Memory, DualIndex>(ddc::DiscreteElement<>());
                ddc::device_for_each(primal.domain(), [&](auto source) {
                    primal.mem(source) = evaluate(sample, source);
                });
                geometry.forward(
                        tensor::Tensor(dual.span_view()),
                        tensor::Tensor(primal.span_view()),
                        geometry.metric,
                        geometry.position,
                        sample);
                return dual.mem(component);
            };
            detail::StencilField<DualIndex, Domain, Memory, decltype(evaluate_dual)> const
                    dual(input.non_indices_domain(), evaluate_dual);
            auto const boundary = [&] {
                if constexpr (misc::Specialization<Rules, tensor::ExtrapolationRules>)
                    return rules;
                else
                    return tensor::make_extrapolation_rules(dual, rules);
            }();
            auto output = detail::make_stencil<
                    Memory,
                    codifferential_index_t<TagToRemoveFromCochain, CochainTag>>(
                    ddc::DiscreteElement<>());
            operator()(
                    tensor::Tensor(output.span_view()),
                    [&](auto sample, auto component) {
                        return boundary(dual, sample).mem(component);
                    },
                    [&](auto result, auto derivative, auto sample) {
                        geometry.inverse(
                                result,
                                derivative,
                                geometry.metric,
                                geometry.position,
                                sample);
                    },
                    geometry.chain,
                    geometry.lower,
                    element);
            return output;
        });
    }

    /** Scalar codifferential coefficients on the incident dual faces.
     * Spatial stencil coordinates are offsets plus one, with the center at one.
     * \important This operator and documentation is fully AI-generated.
     */
    template <class DualField, class InverseHodge, class Element>
        requires(CochainTag::rank() == 1)
    KOKKOS_FUNCTION static auto value(
            DualField const&,
            InverseHodge const& inverse_hodge,
            Element elem)
    {
        using DualIndex = misc::convert_type_seq_to_t<
                tensor::TensorAntisymmetricIndex,
                codifferential_hodge_output_indices_t<
                        TagToRemoveFromCochain::size() - 1,
                        TagToRemoveFromCochain>>;
        using TopIndex = misc::convert_type_seq_to_t<
                tensor::TensorAntisymmetricIndex,
                ddc::type_seq_merge_t<
                        ddc::TypeSeq<TagToRemoveFromCochain>,
                        codifferential_hodge_output_indices_t<
                                TagToRemoveFromCochain::size() - 1,
                                TagToRemoveFromCochain>>>;
        using ChainDomain = typename detail::NonSpectatorDimension<
                TagToRemoveFromCochain,
                typename DualField::non_indices_domain_t>::type;
        auto const chain = tangent_basis<
                TagToRemoveFromCochain::size(),
                ChainDomain,
                typename DualField::memory_space>(typename ChainDomain::discrete_element_type {});
        auto const lower = tangent_basis<
                TagToRemoveFromCochain::size() - 1,
                ChainDomain,
                typename DualField::memory_space>(typename ChainDomain::discrete_element_type {});
        // Build incidence at a positive local origin, independently of physical
        // coordinates, so a domain's first node never wraps the stencil front.
        Element origin = elem;
        for (std::size_t axis = 0; axis < TagToRemoveFromCochain::size(); ++axis) {
            ddc::detail::array(origin)[axis] = 2;
        }
        auto const incidence = TransposedCoboundary<TagToRemoveFromCochain, DualIndex>::
                value(chain,
                      lower,
                      origin,
                      tensor::TensorAccessor<TopIndex>::access_element(
                              tensor::TensorAccessor<TopIndex>::canonical_natural_element(
                                      ddc::DiscreteElement<TopIndex>(0))));
        auto unit = detail::make_stencil<typename DualField::memory_space, TopIndex>(
                ddc::DiscreteElement<>());
        unit.mem(ddc::DiscreteElement<TopIndex>(0)) = 1.0;
        double const scale
                = -inverse_hodge.inverse_zero_form(tensor::Tensor(unit.span_view()), elem);
        auto result = detail::make_stencil<typename DualField::memory_space, DualIndex>(Element {});
        for (std::size_t i = 0; i < result.storage().size(); ++i)
            result.storage()[i] = scale * incidence.storage()[i];
        return result;
    }

    template <class DualField, class DualRules, class InverseHodge, class Element>
        requires requires { typename DualField::non_indices_domain_t; }
    KOKKOS_FUNCTION static auto value(
            DualField const& dual_field,
            DualRules const& dual_rules,
            InverseHodge const& inverse_hodge,
            Element elem)
    {
        auto const rules = [&] {
            if constexpr (misc::Specialization<DualRules, tensor::ExtrapolationRules>)
                return dual_rules;
            else
                return tensor::make_extrapolation_rules(dual_field, dual_rules);
        }();
        using ChainDomain = typename detail::NonSpectatorDimension<
                TagToRemoveFromCochain,
                typename DualField::non_indices_domain_t>::type;
        auto const chain = tangent_basis<
                TagToRemoveFromCochain::size() - CochainTag::rank() + 1,
                ChainDomain,
                typename DualField::memory_space>(typename ChainDomain::discrete_element_type {});
        auto const lower = tangent_basis<
                TagToRemoveFromCochain::size() - CochainTag::rank(),
                ChainDomain,
                typename DualField::memory_space>(typename ChainDomain::discrete_element_type {});
        auto output = detail::make_stencil<
                typename DualField::memory_space,
                codifferential_index_t<TagToRemoveFromCochain, CochainTag>>(
                ddc::DiscreteElement<>());
        operator()(
                tensor::Tensor(output.span_view()),
                [&](auto sample, auto component) {
                    return rules(dual_field, sample).mem(component);
                },
                inverse_hodge,
                chain,
                lower,
                elem);
        return output;
    }
};

namespace detail {
template <class Index, class Cochain, class Tensor, class Metric, class Position>
struct CodifferentialGeometry;
} // namespace detail

template <
        tensor::TensorIndex MetricIndex,
        tensor::TensorNatIndex TagToRemoveFromCochain,
        tensor::TensorIndex CochainTag,
        misc::Specialization<tensor::Tensor> TensorType,
        misc::Specialization<tensor::Tensor> MetricType,
        misc::Specialization<tensor::Tensor> PositionType>
struct Codifferential<
        MetricIndex,
        TagToRemoveFromCochain,
        CochainTag,
        TensorType,
        MetricType,
        PositionType>
{
    using staged_storage_type = detail::SpatialOperatorStorage<
            Codifferential,
            TagToRemoveFromCochain,
            CochainTag,
            codifferential_index_t<TagToRemoveFromCochain, CochainTag>,
            1,
            TensorType>;

    template <class Exec>
    static auto geometry(Exec const& exec, MetricType const& metric, PositionType const& position)
    {
        return detail::CodifferentialGeometry<
                TagToRemoveFromCochain,
                CochainTag,
                TensorType,
                MetricType,
                PositionType>(exec, metric, position);
    }

    template <class Geometry, class Element, class Rules>
    KOKKOS_FUNCTION static auto value(
            Geometry const& geometry,
            TensorType const& input,
            Element element,
            Rules const& rules)
    {
        return Codifferential<TagToRemoveFromCochain, CochainTag>::
                value(geometry, input, element, rules);
    }

    template <class ChainType, class LowerChainType, class NaturalElem>
    KOKKOS_FUNCTION static auto value(
            TensorType const& tensor,
            MetricType const& metric,
            PositionType const& position,
            ChainType const& chain,
            LowerChainType const& lower_chain,
            typename TensorType::non_indices_domain_t::discrete_element_type elem,
            NaturalElem natural_elem)
    {
        auto stencil = detail::make_stencil<typename TensorType::memory_space, CochainTag>(
                detail::backward_stencil_front(elem, tensor.non_indices_domain()));
        ddc::device_for_each(stencil.domain(), [&](auto stencil_elem) {
            auto basis_stencil
                    = detail::make_stencil<typename TensorType::memory_space, CochainTag>(
                            stencil.non_indices_domain().front());
            basis_stencil.mem(stencil_elem) = 1.0;

            [[maybe_unused]] tensor::TensorAccessor<
                    codifferential_index_t<TagToRemoveFromCochain, CochainTag>>
                    codifferential_accessor;
            std::array<
                    double,
                    codifferential_index_t<TagToRemoveFromCochain, CochainTag>::mem_size()>
                    codifferential_alloc {};
            ddc::ChunkSpan<
                    double,
                    ddc::DiscreteDomain<codifferential_index_t<TagToRemoveFromCochain, CochainTag>>,
                    Kokkos::layout_right,
                    typename TensorType::memory_space>
                    codifferential_span(
                            codifferential_alloc.data(),
                            codifferential_accessor.domain());
            sil::tensor::Tensor codifferential_tensor(codifferential_span);

            operator()(
                    codifferential_tensor,
                    basis_stencil,
                    metric,
                    position,
                    chain,
                    lower_chain,
                    elem);
            stencil.mem(stencil_elem) = codifferential_tensor(natural_elem);
        });
        return stencil;
    }

    KOKKOS_FUNCTION static void operator()(
            auto codifferential_tensor,
            TensorType const& tensor,
            MetricType const& metric,
            PositionType const& position,
            auto const& chain,
            auto const& lower_chain,
            typename TensorType::non_indices_domain_t::discrete_element_type elem)
    {
        using source_hodge_input_indices
                = tensor::upper_t<ddc::to_type_seq_t<tensor::natural_domain_t<CochainTag>>>;
        using source_hodge_output_indices = codifferential_hodge_output_indices_t<
                TagToRemoveFromCochain::size() - CochainTag::rank(),
                TagToRemoveFromCochain>;
        using target_hodge_input_indices = ddc::
                type_seq_merge_t<ddc::TypeSeq<TagToRemoveFromCochain>, source_hodge_output_indices>;
        using dual_tensor_index = misc::convert_type_seq_to_t<
                tensor::TensorAntisymmetricIndex,
                source_hodge_output_indices>;
        auto dual_evaluator = [&](auto sampled_elem, auto dual_elem) {
            if (!misc::domain_contains(tensor.non_indices_domain(), sampled_elem)
                || !misc::domain_contains(metric.non_indices_domain(), sampled_elem)
                || !misc::domain_contains(position.non_indices_domain(), sampled_elem)) {
                return 0.0;
            }
            [[maybe_unused]] tensor::TensorAccessor<dual_tensor_index> dual_tensor_accessor;
            std::array<double, dual_tensor_index::mem_size()> dual_tensor_alloc {};
            ddc::ChunkSpan<
                    double,
                    ddc::DiscreteDomain<dual_tensor_index>,
                    Kokkos::layout_right,
                    typename TensorType::memory_space>
                    dual_tensor_span(dual_tensor_alloc.data(), dual_tensor_accessor.domain());
            sil::tensor::Tensor dual_tensor(dual_tensor_span);

            DiscreteHodgeStar<
                    CellComplex::CircumcentricDual,
                    source_hodge_input_indices,
                    source_hodge_output_indices,
                    MetricType,
                    PositionType,
                    typename TensorType::non_indices_domain_t::discrete_element_type> {}
                    .operator()(dual_tensor, tensor[sampled_elem], metric, position, sampled_elem);
            return dual_tensor.mem(dual_elem);
        };

        Codifferential<TagToRemoveFromCochain, CochainTag>::operator()(
                codifferential_tensor,
                dual_evaluator,
                [&](auto output, auto dual, auto sample) {
                    DiscreteHodgeStar<
                            CellComplex::CircumcentricDual,
                            tensor::upper_t<target_hodge_input_indices>,
                            ddc::type_seq_remove_t<
                                    tensor::lower_t<source_hodge_input_indices>,
                                    ddc::TypeSeq<TagToRemoveFromCochain>>,
                            MetricType,
                            PositionType,
                            typename TensorType::non_indices_domain_t::discrete_element_type> {}
                            .operator()(output, dual, metric, position, sample);
                },
                chain,
                lower_chain,
                elem);
    }
};

namespace detail {

template <class Index, class Cochain, class Tensor, class Metric, class Position>
struct CodifferentialGeometry
{
    using SourceIndices = tensor::upper_t<ddc::to_type_seq_t<tensor::natural_domain_t<Cochain>>>;
    using DualIndices
            = codifferential_hodge_output_indices_t<Index::size() - Cochain::rank(), Index>;
    using InverseSourceIndices = ddc::type_seq_merge_t<ddc::TypeSeq<Index>, DualIndices>;
    using InverseTargetIndices
            = ddc::type_seq_remove_t<tensor::lower_t<SourceIndices>, ddc::TypeSeq<Index>>;
    using DualIndex = misc::convert_type_seq_to_t<tensor::TensorAntisymmetricIndex, DualIndices>;
    using Element = typename Tensor::non_indices_domain_t::discrete_element_type;
    using ChainDomain =
            typename NonSpectatorDimension<Index, typename Tensor::non_indices_domain_t>::type;
    Metric metric;
    Position position;
    StagedOperator<DiscreteHodgeStar<
            CellComplex::Primal,
            SourceIndices,
            DualIndices,
            Metric,
            Position,
            Element>>
            forward;
    StagedOperator<DiscreteHodgeStar<
            CellComplex::Primal,
            tensor::upper_t<InverseSourceIndices>,
            InverseTargetIndices,
            Metric,
            Position,
            Element>>
            inverse;
    StencilChain<DualIndex::rank() + 1, ChainDomain, typename Tensor::memory_space> chain;
    StencilChain<DualIndex::rank(), ChainDomain, typename Tensor::memory_space> lower;
    template <class Exec>
    CodifferentialGeometry(Exec const& exec, Metric const& metric_, Position const& position_)
        : metric(metric_)
        , position(position_)
        , forward(exec, metric, position)
        , inverse(exec, metric, position)
    {
    }
};

} // namespace detail

/** Cache Codifferential::value() at every input node.
 * \important This operator and documentation is fully AI-generated.
 * A fixed stencil stores all component couplings and affine boundary terms.
 * Its Manhattan radius is one (seven spatial points in three dimensions).
 * Closures that sample outside this support are rejected during construction.
 * Geometry and the dual closure are fixed at construction. Application only
 * contracts the stored stencil with the current cochain values.
 */
template <
        tensor::TensorIndex MetricIndex,
        tensor::TensorNatIndex TagToRemoveFromCochain,
        tensor::TensorIndex CochainTag,
        class ExecSpace,
        misc::Specialization<tensor::Tensor> TensorType,
        misc::Specialization<tensor::Tensor> MetricType,
        misc::Specialization<tensor::Tensor> PositionType,
        class Rules = tensor::ZeroExtrapolationRule>
StagedOperator<
        Codifferential<
                MetricIndex,
                TagToRemoveFromCochain,
                CochainTag,
                TensorType,
                MetricType,
                PositionType>,
        typename Codifferential<
                MetricIndex,
                TagToRemoveFromCochain,
                CochainTag,
                TensorType,
                MetricType,
                PositionType>::staged_storage_type::template rebind<ExecSpace>>
make_staged_codifferential(
        ExecSpace const& exec_space,
        TensorType const& tensor,
        MetricType const& metric,
        PositionType const& position,
        Rules const& rules = {})
{
    return StagedOperator<
            Codifferential<
                    MetricIndex,
                    TagToRemoveFromCochain,
                    CochainTag,
                    TensorType,
                    MetricType,
                    PositionType>,
            typename Codifferential<
                    MetricIndex,
                    TagToRemoveFromCochain,
                    CochainTag,
                    TensorType,
                    MetricType,
                    PositionType>::staged_storage_type::
                    template rebind<ExecSpace>>(exec_space, tensor, metric, position, rules);
}

template <
        tensor::TensorIndex MetricIndex,
        tensor::TensorNatIndex TagToRemoveFromCochain,
        tensor::TensorIndex CochainTag,
        misc::Specialization<tensor::Tensor> DualTensorType,
        misc::Specialization<tensor::Tensor> TensorType,
        misc::Specialization<tensor::Tensor> HodgeStarType,
        misc::Specialization<tensor::Tensor> DualHodgeStarType,
        class ExecSpace,
        class DualExtrapolationRule = tensor::ZeroExtrapolationRule>
    requires(misc::NotSpecialization<DualExtrapolationRule, tensor::Tensor>)
codifferential_tensor_t<TagToRemoveFromCochain, CochainTag, TensorType> codifferential(
        ExecSpace const& exec_space,
        codifferential_tensor_t<TagToRemoveFromCochain, CochainTag, TensorType> const&
                codifferential_tensor,
        TensorType const& tensor,
        HodgeStarType const& hodge_star,
        DualHodgeStarType const& dual_hodge_star,
        DualTensorType const& dual_tensor_buffer,
        DualExtrapolationRule const& dual_extrapolation = {})
{
    auto const dual_extrapolation_rules = [&]() {
        if constexpr (misc::Specialization<DualExtrapolationRule, tensor::ExtrapolationRules>)
            return dual_extrapolation;
        else
            return tensor::make_extrapolation_rules(dual_tensor_buffer, dual_extrapolation);
    }();
    static_assert(tensor::is_covariant_v<TagToRemoveFromCochain>);
    using source_hodge_output_indices = codifferential_hodge_output_indices_t<
            TagToRemoveFromCochain::size() - CochainTag::rank(),
            TagToRemoveFromCochain>;
    using dual_tensor_index = misc::
            convert_type_seq_to_t<tensor::TensorAntisymmetricIndex, source_hodge_output_indices>;
    using non_spectator_dimensions = typename detail::NonSpectatorDimension<
            TagToRemoveFromCochain,
            typename TensorType::non_indices_domain_t>::type;
    auto const chain
            = tangent_basis<dual_tensor_index::rank() + 1, non_spectator_dimensions>(exec_space);
    auto lower_chain
            = tangent_basis<dual_tensor_index::rank(), non_spectator_dimensions>(exec_space);

    SIMILIE_DEBUG_LOG("similie_apply_first_hodge_star_for_codifferential");
    ddc::parallel_for_each(
            "similie_apply_first_hodge_star_for_codifferential",
            exec_space,
            dual_tensor_buffer.non_indices_domain(),
            KOKKOS_LAMBDA(typename TensorType::non_indices_domain_t::discrete_element_type elem) {
                sil::tensor::tensor_prod(dual_tensor_buffer[elem], tensor[elem], hodge_star[elem]);
            });

    SIMILIE_DEBUG_LOG("similie_apply_second_hodge_star_for_codifferential");
    ddc::parallel_for_each(
            "similie_apply_second_hodge_star_for_codifferential",
            exec_space,
            codifferential_tensor.non_indices_domain(),
            KOKKOS_LAMBDA(typename TensorType::non_indices_domain_t::discrete_element_type elem) {
                Codifferential<TagToRemoveFromCochain, CochainTag>::operator()(
                        codifferential_tensor[elem],
                        [&](auto sampled_elem, auto dual_elem) {
                            return dual_extrapolation_rules(dual_tensor_buffer, sampled_elem)
                                    .mem(dual_elem);
                        },
                        [&](auto output, auto dual, auto sample) {
                            sil::tensor::tensor_prod(output, dual, dual_hodge_star[sample]);
                        },
                        chain,
                        lower_chain,
                        elem);
            });

    return codifferential_tensor;
}

template <
        tensor::TensorIndex MetricIndex,
        tensor::TensorNatIndex TagToRemoveFromCochain,
        tensor::TensorIndex CochainTag,
        misc::Specialization<tensor::Tensor> TensorType,
        misc::Specialization<tensor::Tensor> MetricType,
        misc::Specialization<tensor::Tensor> PositionType,
        class ExecSpace,
        class DualExtrapolationRule = tensor::ZeroExtrapolationRule>
    requires(misc::NotSpecialization<DualExtrapolationRule, tensor::Tensor>)
codifferential_tensor_t<TagToRemoveFromCochain, CochainTag, TensorType> codifferential(
        ExecSpace const& exec_space,
        codifferential_tensor_t<TagToRemoveFromCochain, CochainTag, TensorType> const&
                codifferential_tensor,
        TensorType const& tensor,
        MetricType const& metric,
        PositionType const& position,
        DualExtrapolationRule const& dual_extrapolation = {})
{
    static_assert(tensor::is_covariant_v<TagToRemoveFromCochain>);
    return make_staged_codifferential<MetricIndex, TagToRemoveFromCochain, CochainTag>(
            exec_space,
            tensor,
            metric,
            position,
            dual_extrapolation)(codifferential_tensor, tensor);
}

} // namespace exterior

} // namespace sil
