// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

#include <array>
#include <stdexcept>

#include <ddc/ddc.hpp>

#include <similie/misc/factorial.hpp>
#include <similie/misc/macros.hpp>
#include <similie/misc/small_matrix.hpp>
#include <similie/misc/specialization.hpp>
#include <similie/misc/type_seq_conversion.hpp>
#include <similie/tensor/antisymmetric_tensor.hpp>
#include <similie/tensor/character.hpp>
#include <similie/tensor/full_tensor.hpp>
#include <similie/tensor/metric.hpp>

#include "coboundary.hpp"
#include "reduction_and_reconstruction.hpp"
#include "volume.hpp"

namespace sil {

namespace exterior {

template <misc::Specialization<ddc::TypeSeq> Indices1, misc::Specialization<ddc::TypeSeq> Indices2>
using hodge_star_domain_t
        = ddc::detail::convert_type_seq_to_discrete_domain_t<ddc::type_seq_merge_t<
                ddc::TypeSeq<misc::convert_type_seq_to_t<tensor::TensorFullIndex, Indices1>>,
                std::conditional_t<
                        (ddc::type_seq_size_v<Indices2> == 0),
                        ddc::TypeSeq<>,
                        ddc::TypeSeq<misc::convert_type_seq_to_t<
                                tensor::TensorAntisymmetricIndex,
                                Indices2>>>>>;

template <
        misc::Specialization<ddc::TypeSeq> Indices1,
        misc::Specialization<ddc::TypeSeq> Indices2,
        class MetricType,
        class BatchElem>
struct ContinuousHodgeStar;

template <
        CellComplex Complex,
        misc::Specialization<ddc::TypeSeq> Indices1,
        misc::Specialization<ddc::TypeSeq> Indices2,
        class MetricType,
        class PositionType,
        class BatchElem>
struct DiscreteHodgeStar;

namespace detail {

template <std::size_t N, std::size_t M>
KOKKOS_FUNCTION bool has_unique_ids(std::array<std::size_t, M> const& ids)
{
    if constexpr (M > 0) {
        for (std::size_t i = 0; i < M; ++i) {
            if (ids[i] >= N) {
                return false;
            }
            for (std::size_t j = i + 1; j < M; ++j) {
                if (ids[i] == ids[j]) {
                    return false;
                }
            }
        }
    }
    return true;
}

template <std::size_t N, std::size_t M1, std::size_t M2>
KOKKOS_FUNCTION int permutation_sign(
        std::array<std::size_t, M1> const& source_ids,
        std::array<std::size_t, M2> const& target_ids)
{
    std::array<std::size_t, N> permutation {};
    for (std::size_t i = 0; i < M1; ++i) {
        permutation[i] = source_ids[i];
    }
    if constexpr (M2 > 0) {
        for (std::size_t i = 0; i < M2; ++i) {
            permutation[M1 + i] = target_ids[i];
        }
    }

    bool odd = false;
    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = i + 1; j < N; ++j) {
            odd = (permutation[i] > permutation[j]) != odd;
        }
    }
    return odd ? -1 : 1;
}

template <std::size_t N, std::size_t M>
KOKKOS_FUNCTION std::array<std::size_t, N - M> complement_ids(std::array<std::size_t, M> const& ids)
{
    std::array<std::size_t, N - M> complement {};
    std::size_t complement_id = 0;
    for (std::size_t i = 0; i < N; ++i) {
        bool found = false;
        for (std::size_t id : ids) {
            found = found || (id == i);
        }
        if (!found) {
            complement[complement_id++] = i;
        }
    }
    return complement;
}

template <misc::Specialization<ddc::TypeSeq> Indices>
struct HodgeStarTargetSize
{
    static constexpr std::size_t value
            = misc::convert_type_seq_to_t<tensor::TensorAntisymmetricIndex, Indices>::access_size();
};

template <>
struct HodgeStarTargetSize<ddc::TypeSeq<>>
{
    static constexpr std::size_t value = 1;
};

template <
        CellComplex Complex,
        misc::Specialization<ddc::TypeSeq> Indices1,
        misc::Specialization<ddc::TypeSeq> Indices2,
        misc::Specialization<tensor::Tensor> HodgeStarType,
        misc::Specialization<tensor::Tensor> MetricType,
        misc::Specialization<tensor::Tensor> PositionType,
        class BatchElem>
struct FillDiscreteHodgeStarMem
{
    HodgeStarType m_hodge_star;
    MetricType m_metric;
    PositionType m_position;
    BatchElem m_elem;

    template <class MemElem>
    KOKKOS_FUNCTION void operator()(MemElem mem_elem) const
    {
        m_hodge_star.mem(typename HodgeStarType::discrete_element_type(m_elem, mem_elem))
                = DiscreteHodgeStar<
                        Complex,
                        Indices1,
                        Indices2,
                        MetricType,
                        PositionType,
                        BatchElem>::
                        value(m_metric,
                              m_position,
                              m_elem,
                              m_hodge_star.accessor().canonical_natural_element(mem_elem));
    }
};

template <
        misc::Specialization<ddc::TypeSeq> Indices1,
        misc::Specialization<ddc::TypeSeq> Indices2,
        misc::Specialization<tensor::Tensor> HodgeStarType,
        misc::Specialization<tensor::Tensor> MetricType,
        class BatchElem>
struct FillContinuousHodgeStarMem
{
    HodgeStarType m_hodge_star;
    MetricType m_metric;
    BatchElem m_elem;

    template <class MemElem>
    KOKKOS_FUNCTION void operator()(MemElem mem_elem) const
    {
        m_hodge_star.mem(typename HodgeStarType::discrete_element_type(m_elem, mem_elem))
                = ContinuousHodgeStar<Indices1, Indices2, MetricType, BatchElem>::
                        value(m_metric,
                              m_elem,
                              m_hodge_star.accessor().canonical_natural_element(mem_elem));
    }
};

} // namespace detail

template <
        CellComplex Complex,
        misc::Specialization<ddc::TypeSeq> Indices1,
        misc::Specialization<ddc::TypeSeq> Indices2,
        class MetricType,
        class PositionType,
        class BatchElem>
struct DiscreteHodgeStar
{
    /** Pair primal one-cochains using the full cell metric and orientation.
     * \important This operator and documentation is fully AI-generated.
     */
    KOKKOS_FUNCTION static auto one_form_pairing(
            MetricType metric,
            PositionType position,
            BatchElem elem)
    {
        using PositionIndex = ddc::
                type_seq_element_t<0, ddc::to_type_seq_t<typename PositionType::indices_domain_t>>;
        using MetricIndex = ddc::
                type_seq_element_t<0, ddc::to_type_seq_t<typename MetricType::indices_domain_t>>;
        constexpr std::size_t N = PositionIndex::size();
        std::array<double, N * N> jacobian {}, gram {}, inverse {}, workspace {};
        for (std::size_t axis = 0; axis < N; ++axis) {
            std::array<double, N> const edge
                    = detail::edge_vector<PositionIndex>(position, elem, axis);
            for (std::size_t physical = 0; physical < N; ++physical)
                jacobian[physical * N + axis] = edge[physical];
        }
        for (std::size_t i = 0; i < N; ++i)
            for (std::size_t j = 0; j < N; ++j)
                for (std::size_t a = 0; a < N; ++a)
                    for (std::size_t b = 0; b < N; ++b)
                        gram[i * N + j]
                                += jacobian[a * N + i] * jacobian[b * N + j]
                                   * metric.get(metric.access_element(
                                           elem,
                                           ddc::DiscreteElement<
                                                   tensor::metric_index_1<MetricIndex>,
                                                   tensor::metric_index_2<MetricIndex>>(a, b)));
        auto gram_view = misc::math::
                matrix_view<double, typename MetricType::memory_space>(gram.data(), N, N);
        auto inverse_view = misc::math::
                matrix_view<double, typename MetricType::memory_space>(inverse.data(), N, N);
        if (!misc::math::
                    invert(inverse_view,
                           gram_view,
                           misc::math::vector_view<
                                   double,
                                   typename MetricType::memory_space>(workspace.data(), N * N)))
            return inverse;
        double const orientation = detail::reduction_determinant<N>(jacobian);
        double const volume = (orientation < 0.0 ? -1.0 : 1.0)
                              * Kokkos::sqrt(Kokkos::abs(detail::reduction_determinant<N>(gram)));
        for (double& coefficient : inverse)
            coefficient *= volume;
        return inverse;
    }

    template <
            misc::Specialization<tensor::Tensor> HodgeTensorType,
            misc::Specialization<tensor::Tensor> FormTensorType>
    KOKKOS_FUNCTION static void operator()(
            HodgeTensorType hodge_tensor,
            FormTensorType form_tensor,
            MetricType metric,
            PositionType position,
            BatchElem elem)
    {
        using InputIndex = ddc::type_seq_element_t<
                0,
                ddc::to_type_seq_t<typename FormTensorType::indices_domain_t>>;
        using OutputIndex = ddc::type_seq_element_t<
                0,
                ddc::to_type_seq_t<typename HodgeTensorType::indices_domain_t>>;
        using InputIndexSeq
                = ddc::to_type_seq_t<typename FormTensorType::accessor_t::natural_domain_t>;
        using OutputIndexSeq
                = ddc::to_type_seq_t<typename HodgeTensorType::accessor_t::natural_domain_t>;

        [[maybe_unused]] tensor::TensorAccessor<InputIndex> reconstructed_accessor;
        std::array<double, InputIndex::access_size()> reconstructed_alloc {};
        ddc::ChunkSpan<
                double,
                ddc::DiscreteDomain<InputIndex>,
                Kokkos::layout_right,
                typename FormTensorType::memory_space>
                reconstructed_span(reconstructed_alloc.data(), reconstructed_accessor.domain());
        sil::tensor::Tensor reconstructed_form(reconstructed_span);

        [[maybe_unused]] tensor::TensorAccessor<OutputIndex> continuous_output_accessor;
        std::array<double, OutputIndex::access_size()> continuous_output_alloc {};
        ddc::ChunkSpan<
                double,
                ddc::DiscreteDomain<OutputIndex>,
                Kokkos::layout_right,
                typename HodgeTensorType::memory_space>
                continuous_output_span(
                        continuous_output_alloc.data(),
                        continuous_output_accessor.domain());
        sil::tensor::Tensor continuous_output(continuous_output_span);

        Reconstruction<InputIndexSeq, PositionType, BatchElem, CellComplex::Primal>::
        operator()(reconstructed_form, form_tensor, position, elem);
        ContinuousHodgeStar<Indices1, Indices2, MetricType, BatchElem>::
        operator()(continuous_output, reconstructed_form, metric, elem);
        Reduction<OutputIndexSeq, PositionType, BatchElem, Complex>::
        operator()(hodge_tensor, continuous_output, metric, position, elem);
    }

    KOKKOS_FUNCTION static double value(
            MetricType metric,
            PositionType position,
            BatchElem elem,
            auto natural_elem)
    {
        using SourceIndex = tensor::lower_t<reduction_index_t<Indices1>>;
        using TargetIndex = reduction_index_t<Indices2>;
        using SourceNaturalElem =
                typename tensor::natural_domain_t<SourceIndex>::discrete_element_type;
        using TargetNaturalElem =
                typename tensor::natural_domain_t<TargetIndex>::discrete_element_type;
        using SourceFullElem = misc::convert_type_seq_to_t<ddc::DiscreteElement, Indices1>;
        using TargetFullElem = misc::convert_type_seq_to_t<ddc::DiscreteElement, Indices2>;

        [[maybe_unused]] tensor::TensorAccessor<SourceIndex> source_accessor;
        std::array<double, SourceIndex::access_size()> source_alloc {};
        ddc::ChunkSpan<
                double,
                ddc::DiscreteDomain<SourceIndex>,
                Kokkos::layout_right,
                typename MetricType::memory_space>
                source_span(source_alloc.data(), source_accessor.domain());
        sil::tensor::Tensor source_tensor(source_span);
        ddc::device_for_each(source_tensor.domain(), [&](auto source_mem_elem) {
            source_tensor.mem(source_mem_elem) = 0.;
        });

        [[maybe_unused]] tensor::TensorAccessor<TargetIndex> target_accessor;
        std::array<double, TargetIndex::access_size()> target_alloc {};
        ddc::ChunkSpan<
                double,
                ddc::DiscreteDomain<TargetIndex>,
                Kokkos::layout_right,
                typename MetricType::memory_space>
                target_span(target_alloc.data(), target_accessor.domain());
        sil::tensor::Tensor target_tensor(target_span);
        ddc::device_for_each(target_tensor.domain(), [&](auto target_mem_elem) {
            target_tensor.mem(target_mem_elem) = 0.;
        });

        std::array source_ids = ddc::detail::array(SourceFullElem(natural_elem));
        if constexpr (ddc::type_seq_size_v<Indices1> > 1) {
            if (!detail::has_unique_ids<SourceIndex::size()>(source_ids)) {
                return 0.;
            }
        }
        bool odd = false;
        for (auto i = source_ids.begin(); i < source_ids.end(); ++i) {
            for (auto j = i + 1; j < source_ids.end(); ++j) {
                odd = (*i > *j) != odd;
            }
        }
        std::array canonical_source_ids(source_ids);
        if constexpr (source_ids.size() > 1) {
            for (auto i = canonical_source_ids.begin(); i < canonical_source_ids.end(); ++i) {
                for (auto j = i + 1; j < canonical_source_ids.end(); ++j) {
                    if (*j < *i) {
                        Kokkos::kokkos_swap(*i, *j);
                    }
                }
            }
        }

        SourceNaturalElem source_natural_elem;
        ddc::detail::array(source_natural_elem) = canonical_source_ids;
        source_tensor(source_tensor.accessor().access_element(source_natural_elem)) = 1.;

        operator()(target_tensor, source_tensor, metric, position, elem);

        double const source_factor
                = (odd ? -1. : 1.) / misc::factorial(ddc::type_seq_size_v<Indices1>);
        if constexpr (ddc::type_seq_size_v<Indices2> == 0) {
            return source_factor
                   * target_tensor(target_tensor.accessor().access_element(TargetNaturalElem()));
        } else {
            TargetNaturalElem target_natural_elem;
            ddc::detail::array(target_natural_elem)
                    = ddc::detail::array(TargetFullElem(natural_elem));
            return source_factor
                   * target_tensor.get(
                           target_tensor.accessor().access_element(target_natural_elem));
        }
    }
};

template <
        misc::Specialization<ddc::TypeSeq> Indices1,
        misc::Specialization<ddc::TypeSeq> Indices2,
        class MetricType,
        class BatchElem>
struct ContinuousHodgeStar
{
    template <
            misc::Specialization<tensor::Tensor> HodgeTensorType,
            misc::Specialization<tensor::Tensor> FormTensorType>
    KOKKOS_FUNCTION static void operator()(
            HodgeTensorType hodge_tensor,
            FormTensorType form_tensor,
            MetricType metric,
            BatchElem elem)
    {
        [[maybe_unused]] sil::tensor::tensor_accessor_for_domain_t<
                hodge_star_domain_t<Indices1, Indices2>> hodge_star_accessor;
        static constexpr std::size_t HODGE_STAR_SIZE
                = misc::convert_type_seq_to_t<tensor::TensorFullIndex, Indices1>::access_size()
                  * detail::HodgeStarTargetSize<Indices2>::value;
        std::array<double, HODGE_STAR_SIZE> hodge_star_alloc {};
        ddc::ChunkSpan<
                double,
                hodge_star_domain_t<Indices1, Indices2>,
                Kokkos::layout_right,
                typename HodgeTensorType::memory_space>
                hodge_star_span(hodge_star_alloc.data(), hodge_star_accessor.domain());
        sil::tensor::Tensor hodge_star(hodge_star_span);
        ddc::device_for_each(hodge_star.domain(), [&](auto it) {
            hodge_star.mem(it) = value(metric, elem, hodge_star.canonical_natural_element(it));
        });
        sil::tensor::tensor_prod(hodge_tensor, hodge_star, form_tensor);
    }

    KOKKOS_FUNCTION static double value(MetricType metric, BatchElem elem, auto natural_elem)
    {
        constexpr std::size_t N = ddc::type_seq_size_v<Indices1> + ddc::type_seq_size_v<Indices2>;
        constexpr std::size_t K = ddc::type_seq_size_v<Indices1>;
        using SourceElem = misc::convert_type_seq_to_t<ddc::DiscreteElement, Indices1>;
        using TargetElem = misc::convert_type_seq_to_t<ddc::DiscreteElement, Indices2>;
        std::array const source_ids = ddc::detail::array(SourceElem(natural_elem));
        std::array const target_ids = ddc::detail::array(TargetElem(natural_elem));

        if (!detail::has_unique_ids<N>(source_ids) || !detail::has_unique_ids<N>(target_ids)) {
            return 0.;
        }

        std::array<std::size_t, K> const complement = detail::complement_ids<N>(target_ids);
        using MetricIndex = ddc::
                type_seq_element_t<0, ddc::to_type_seq_t<typename MetricType::indices_domain_t>>;
        using MetricIndex1 = tensor::metric_index_1<MetricIndex>;
        using MetricIndex2 = tensor::metric_index_2<MetricIndex>;
        std::array<double, N * N> metric_alloc {};
        std::array<double, N * N> determinant_metric_alloc {};
        std::array<double, N * N> inverse_metric_alloc {};
        std::array<double, N * N> workspace_alloc {};
        std::array<double, K * K> submatrix_alloc {};
        for (std::size_t i = 0; i < N; ++i) {
            for (std::size_t j = 0; j < N; ++j) {
                metric_alloc[i * N + j] = metric.get(metric.access_element(
                        elem,
                        ddc::DiscreteElement<MetricIndex1, MetricIndex2>(i, j)));
            }
        }
        determinant_metric_alloc = metric_alloc;

        auto determinant_metric_view = misc::math::matrix_view<
                double,
                typename MetricType::memory_space>(determinant_metric_alloc.data(), N, N);
        auto metric_view = misc::math::
                matrix_view<double, typename MetricType::memory_space>(metric_alloc.data(), N, N);
        double const determinant = misc::math::determinant(determinant_metric_view);
        auto inverse_metric_view = misc::math::matrix_view<
                double,
                typename MetricType::memory_space>(inverse_metric_alloc.data(), N, N);
        auto workspace = misc::math::vector_view<
                double,
                typename MetricType::memory_space>(workspace_alloc.data(), N * N);
        if (!misc::math::invert(inverse_metric_view, metric_view, workspace)) {
            return 0.;
        }

        return Kokkos::sqrt(Kokkos::abs(determinant))
               * static_cast<double>(detail::permutation_sign<N>(complement, target_ids))
               * misc::math::submatrix_determinant<
                       K>(inverse_metric_view, source_ids, complement, submatrix_alloc)
               / misc::factorial(K);
    }
};

template <
        misc::Specialization<ddc::TypeSeq> Indices1,
        misc::Specialization<ddc::TypeSeq> Indices2,
        CellComplex Complex = CellComplex::Primal,
        misc::Specialization<tensor::Tensor> HodgeStarType,
        misc::Specialization<tensor::Tensor> MetricType,
        misc::Specialization<tensor::Tensor> PositionType,
        class ExecSpace>
HodgeStarType fill_discrete_hodge_star(
        ExecSpace const& exec_space,
        HodgeStarType hodge_star,
        MetricType metric,
        PositionType position)
{
    static_assert(
            ddc::type_seq_size_v<ddc::to_type_seq_t<typename PositionType::indices_domain_t>> == 1);

    SIMILIE_DEBUG_LOG("similie_compute_hodge_star");
    ddc::parallel_for_each(
            "similie_compute_hodge_star",
            exec_space,
            hodge_star.non_indices_domain(),
            KOKKOS_LAMBDA(
                    typename HodgeStarType::non_indices_domain_t::discrete_element_type elem) {
                ddc::device_for_each(
                        hodge_star.accessor().domain(),
                        detail::FillDiscreteHodgeStarMem<
                                Complex,
                                Indices1,
                                Indices2,
                                HodgeStarType,
                                MetricType,
                                PositionType,
                                typename HodgeStarType::non_indices_domain_t::
                                        discrete_element_type> {
                                hodge_star,
                                metric,
                                position,
                                elem});
            });
    return hodge_star;
}

template <
        misc::Specialization<ddc::TypeSeq> Indices1,
        misc::Specialization<ddc::TypeSeq> Indices2,
        misc::Specialization<tensor::Tensor> HodgeStarType,
        misc::Specialization<tensor::Tensor> MetricType,
        class ExecSpace>
HodgeStarType fill_continuous_hodge_star(
        ExecSpace const& exec_space,
        HodgeStarType hodge_star,
        MetricType metric)
{
    SIMILIE_DEBUG_LOG("similie_compute_continuous_hodge_star");
    ddc::parallel_for_each(
            "similie_compute_continuous_hodge_star",
            exec_space,
            hodge_star.non_indices_domain(),
            KOKKOS_LAMBDA(
                    typename HodgeStarType::non_indices_domain_t::discrete_element_type elem) {
                ddc::device_for_each(
                        hodge_star.accessor().domain(),
                        detail::FillContinuousHodgeStarMem<
                                Indices1,
                                Indices2,
                                HodgeStarType,
                                MetricType,
                                typename HodgeStarType::non_indices_domain_t::
                                        discrete_element_type> {hodge_star, metric, elem});
            });
    return hodge_star;
}

} // namespace exterior

} // namespace sil

namespace sil::exterior {
namespace detail {

template <std::size_t I, class T>
struct CodifferentialHodgeOutputIndex : tensor::uncharacterize_t<T>
{
};

template <class Ids, class T>
struct CodifferentialHodgeOutputIndexSeq_;

template <std::size_t... Id, class T>
struct CodifferentialHodgeOutputIndexSeq_<std::index_sequence<Id...>, T>
{
    using type = ddc::TypeSeq<tensor::Covariant<CodifferentialHodgeOutputIndex<Id, T>>...>;
};

template <std::size_t EndId, class T>
struct CodifferentialHodgeOutputIndexSeq;

template <std::size_t EndId, class T>
    requires(EndId == 0)
struct CodifferentialHodgeOutputIndexSeq<EndId, T>
{
    using type = ddc::TypeSeq<>;
};

template <std::size_t EndId, class T>
    requires(EndId > 0)
struct CodifferentialHodgeOutputIndexSeq<EndId, T>
{
    using type =
            typename CodifferentialHodgeOutputIndexSeq_<std::make_index_sequence<EndId>, T>::type;
};

} // namespace detail

template <std::size_t EndId, class T>
using codifferential_hodge_output_indices_t =
        typename detail::CodifferentialHodgeOutputIndexSeq<EndId, T>::type;

namespace detail {
template <std::size_t Axis, class Position, class Rules, class Element>
KOKKOS_FUNCTION void geometry_front(
        Position const& position,
        Rules const& rules,
        Element elem,
        Element& front)
{
    if (!tensor::detail::is_interface(rules.template boundary_rule<Axis, true>())) {
        if (ddc::detail::array(elem)[Axis]
                    == ddc::detail::array(position.non_indices_domain().back())[Axis]
            && ddc::detail::array(elem)[Axis]
                       > ddc::detail::array(position.non_indices_domain().front())[Axis])
            --ddc::detail::array(front)[Axis];
    }
}

template <class Position, class Rules, class Element, std::size_t... Axis>
KOKKOS_FUNCTION auto local_geometry(
        Position const& position,
        Rules const& rules,
        Element elem,
        std::index_sequence<Axis...>)
{
    using Index
            = ddc::type_seq_element_t<0, ddc::to_type_seq_t<typename Position::indices_domain_t>>;
    Element front = elem;
    (geometry_front<Axis>(position, rules, elem, front), ...);
    auto geometry = make_stencil<typename Position::memory_space, Index>(front);
    ddc::device_for_each(geometry.domain(), [&](auto sample) {
        geometry.mem(sample)
                = rules(position, Element(sample)).mem(ddc::DiscreteElement<Index>(sample));
    });
    return geometry;
}

template <class MemorySpace, class... Coordinates>
KOKKOS_FUNCTION auto euclidean_metric(ddc::TypeSeq<Coordinates...>)
{
    using Index = tensor::TensorIdentityIndex<
            tensor::Covariant<tensor::MetricIndex1<Coordinates...>>,
            tensor::Covariant<tensor::MetricIndex2<Coordinates...>>>;
    tensor::TensorAccessor<Index> accessor;
    // Identity tensors have no stored entries.
    return tensor::Tensor(
            ddc::ChunkSpan<
                    double,
                    ddc::DiscreteDomain<Index>,
                    Kokkos::layout_right,
                    MemorySpace>(nullptr, accessor.domain()));
}


} // namespace detail
/** Lazy dual cochain obtained by applying the Hodge star to a primal one-cochain.
 * \important This operator and documentation is fully AI-generated.
 * The input cochain can be stored or lazily evaluated. Geometry samples
 * cross interfaces through their rules. The local primal Hodge pairing reproduces
 * constant forms using the full cell Jacobian. Cross-axis terms use averages
 * of parallel primal edges. A symmetric projected diagonal pairing controls
 * unresolved edge modes while vanishing on constant physical forms, including
 * on warped cells, recovering axial DEC weights on orthogonal grids.
 * Incident physical cells supply the dual-face measure, including half faces
 * at walls.
 * At physical upper faces the outgoing
 * dual sample comes from the physical flux policy, including the zero wall law.
 */
template <
        class Direction,
        class Cochain,
        class Position,
        class GeometryRules,
        class BoundaryFluxRules>
struct DualHodgeCochain
{
    using source_indices = tensor::upper_t<ddc::to_type_seq_t<tensor::natural_domain_t<Direction>>>;
    using dual_indices = codifferential_hodge_output_indices_t<Direction::size() - 1, Direction>;
    using component_type
            = misc::convert_type_seq_to_t<tensor::TensorAntisymmetricIndex, dual_indices>;
    using non_indices_domain_t = typename Cochain::non_indices_domain_t;
    using indices_domain_t = ddc::DiscreteDomain<component_type>;
    using memory_space = typename Position::memory_space;
    using discrete_domain_type
            = misc::cartesian_prod_t<non_indices_domain_t, ddc::DiscreteDomain<component_type>>;
    Cochain cochain;
    Position position;
    GeometryRules geometry;
    BoundaryFluxRules boundary_flux;
    double diffusivity;

    KOKKOS_FUNCTION non_indices_domain_t non_indices_domain() const
    {
        return cochain.non_indices_domain();
    }

    template <std::size_t Axis, class Element, class Component>
    KOKKOS_FUNCTION bool upper_flux(
            Element elem,
            Component component,
            std::size_t normal,
            double& result) const
    {
        if (!tensor::detail::is_interface(geometry.template boundary_rule<Axis, true>())) {
            if (normal == Axis
                && ddc::detail::array(elem)[Axis]
                           == ddc::detail::array(non_indices_domain().back())[Axis]) {
                ++ddc::detail::array(elem)[Axis];
                auto const& rule = boundary_flux.template boundary_rule<Axis, true>();
                auto const coefficients = rule.value(non_indices_domain(), elem);
                result = tensor::detail::contract_extrapolation_basis(
                        coefficients,
                        [](auto) { return 0.0; },
                        [&](std::size_t) {
                            return tensor::detail::
                                    extrapolation_source_value(rule, *this, elem, component);
                        });
                return true;
            }
        }
        return false;
    }

    template <std::size_t Axis, class Element>
    KOKKOS_FUNCTION bool physical_cell(Element cell) const
    {
        if (!tensor::detail::is_interface(geometry.template boundary_rule<Axis, false>()))
            if (ddc::detail::array(cell)[Axis]
                < ddc::detail::array(non_indices_domain().front())[Axis])
                return false;
        if (!tensor::detail::is_interface(geometry.template boundary_rule<Axis, true>()))
            if (ddc::detail::array(cell)[Axis]
                >= ddc::detail::array(non_indices_domain().back())[Axis])
                return false;
        return true;
    }

    template <class Element>
    KOKKOS_FUNCTION auto cell_geometry(Element cell) const
    {
        constexpr std::size_t N = Direction::size();
        constexpr std::size_t edge_count = std::size_t(1) << (N - 1);
        using Index = ddc::
                type_seq_element_t<0, ddc::to_type_seq_t<typename Position::indices_domain_t>>;
        auto positions = detail::make_stencil<typename Position::memory_space, Index>(cell);
        std::array<double, N * N> jacobian {};
        for (std::size_t axis = 0; axis < N; ++axis) {
            for (std::size_t edge_id = 0; edge_id < edge_count; ++edge_id) {
                Element first = cell;
                std::size_t tangent = 0;
                for (std::size_t other = 0; other < N; ++other)
                    if (other != axis)
                        ddc::detail::array(first)[other] += (edge_id >> tangent++) & 1;
                Element second = first;
                ++ddc::detail::array(second)[axis];
                for (std::size_t physical = 0; physical < N; ++physical)
                    jacobian[physical * N + axis]
                            += (geometry(position, second)
                                        .mem(ddc::DiscreteElement<Index>(physical))
                                - geometry(position, first)
                                          .mem(ddc::DiscreteElement<Index>(physical)))
                               / edge_count;
            }
        }
        ddc::device_for_each(positions.domain(), [&](auto sample) {
            std::size_t const physical = ddc::DiscreteElement<Index>(sample).uid();
            double value = 0.0;
            for (std::size_t axis = 0; axis < N; ++axis)
                value += jacobian[physical * N + axis]
                         * (ddc::detail::array(Element(sample))[axis]
                            - ddc::detail::array(cell)[axis]);
            positions.mem(sample) = value;
        });
        return positions;
    }

    template <class Element, std::size_t... Axis>
    KOKKOS_FUNCTION double dual_volume(Element elem, std::index_sequence<Axis...>) const
    {
        constexpr std::size_t N = Direction::size();
        auto metric = detail::euclidean_metric<typename Position::memory_space>(
                typename tensor::uncharacterize_t<Direction>::type_seq_dimensions {});
        std::array<std::size_t, N> const ids {Axis...};
        double volume = 0.0;
        for (std::size_t vertex = 0; vertex < (std::size_t(1) << N); ++vertex) {
            Element cell = elem;
            for (std::size_t axis = 0; axis < N; ++axis)
                ddc::detail::array(cell)[axis] -= (vertex >> axis) & 1;
            if (!(physical_cell<Axis>(cell) && ...))
                continue;
            auto positions = cell_geometry(cell);
            tensor::Tensor<
                    double,
                    typename Position::discrete_domain_type,
                    Kokkos::layout_right,
                    typename Position::memory_space>
                    local_position(positions);
            volume += SimplexVolume<
                              CellComplex::Primal,
                              N,
                              decltype(metric),
                              decltype(local_position),
                              Element>::template operator()<N>(metric, local_position, cell, ids)
                      / (std::size_t(1) << N);
        }
        return volume;
    }

    template <class Component>
    KOKKOS_FUNCTION std::size_t normal_direction(Component component) const
    {
        tensor::TensorAccessor<component_type> accessor;
        auto const ids
                = detail::flat_natural_elem_ids(accessor.canonical_natural_element(component));
        for (std::size_t normal = 0; normal < Direction::size(); ++normal) {
            bool found = false;
            for (std::size_t id : ids)
                found = found || id == normal;
            if (!found)
                return normal;
        }
        return Direction::size();
    }

    template <class Element, class Component, std::size_t... Axis>
    KOKKOS_FUNCTION bool prescribed_flux(
            Element elem,
            Component component,
            double& prescribed,
            std::index_sequence<Axis...>) const
    {
        return (upper_flux<Axis>(elem, component, normal_direction(component), prescribed) || ...);
    }

    template <class Element, class Component>
    KOKKOS_FUNCTION bool prescribed_flux(Element elem, Component component, double& prescribed)
            const
    {
        return prescribed_flux(
                elem,
                component,
                prescribed,
                std::make_index_sequence<Direction::size()> {});
    }

    /** Constitutive coefficients on the neighbouring primal tensor nodes.
     * Spatial stencil coordinates are offsets plus one, with the center at one.
     * \important This operator and documentation is fully AI-generated.
     */
    template <class Element, class Component, std::size_t... Axis>
    KOKKOS_FUNCTION auto stencil(Element elem, Component component, std::index_sequence<Axis...>)
            const
    {
        using Index = ddc::
                type_seq_element_t<0, ddc::to_type_seq_t<typename Cochain::indices_domain_t>>;
        constexpr std::size_t N = Direction::size();
        constexpr std::size_t edge_count = std::size_t(1) << (N - 1);
        auto result = detail::make_stencil<typename Position::memory_space, Index, 3>(Element {});
        double prescribed = 0.0;
        if (prescribed_flux(elem, component, prescribed))
            return result;
        std::size_t const normal = normal_direction(component);
        tensor::TensorAccessor<component_type> accessor;
        auto const ids
                = detail::flat_natural_elem_ids(accessor.canonical_natural_element(component));
        auto metric = detail::euclidean_metric<typename Position::memory_space>(
                typename tensor::uncharacterize_t<Direction>::type_seq_dimensions {});
        for (std::size_t incident = 0; incident < edge_count; ++incident) {
            Element cell = elem;
            std::size_t bit = 0;
            for (std::size_t axis = 0; axis < N; ++axis)
                if (axis != normal)
                    ddc::detail::array(cell)[axis] -= (incident >> bit++) & 1;
            if (!(physical_cell<Axis>(cell) && ...))
                continue;
            auto positions = cell_geometry(cell);
            tensor::Tensor<
                    double,
                    typename Position::discrete_domain_type,
                    Kokkos::layout_right,
                    typename Position::memory_space>
                    local_position(positions);
            std::array<double, N * N> const pairing = DiscreteHodgeStar<
                    CellComplex::Primal,
                    source_indices,
                    dual_indices,
                    decltype(metric),
                    decltype(local_position),
                    Element>::one_form_pairing(metric, local_position, cell);
            using PositionIndex = ddc::
                    type_seq_element_t<0, ddc::to_type_seq_t<typename Position::indices_domain_t>>;
            std::array<double, N * edge_count * N> edges {};
            std::array<double, N * edge_count> projector_row {};
            std::array<double, N> weighted_projection {};
            std::array<double, N * N> gram {}, inverse {}, workspace {};
            std::size_t selected = 0;
            for (std::size_t axis = 0; axis < N; ++axis) {
                for (std::size_t edge_id = 0; edge_id < edge_count; ++edge_id) {
                    Element first = cell;
                    std::size_t tangent = 0;
                    for (std::size_t other = 0; other < N; ++other)
                        if (other != axis)
                            ddc::detail::array(first)[other] += (edge_id >> tangent++) & 1;
                    Element second = first;
                    ++ddc::detail::array(second)[axis];
                    std::size_t const edge = axis * edge_count + edge_id;
                    if (axis == normal && first == elem)
                        selected = edge;
                    for (std::size_t physical = 0; physical < N; ++physical)
                        edges[edge * N + physical]
                                = geometry(position, second)
                                          .mem(ddc::DiscreteElement<PositionIndex>(physical))
                                  - geometry(position, first)
                                            .mem(ddc::DiscreteElement<PositionIndex>(physical));
                    for (std::size_t i = 0; i < N; ++i) {
                        for (std::size_t j = 0; j < N; ++j)
                            gram[i * N + j] += edges[edge * N + i] * edges[edge * N + j];
                    }
                }
            }
            auto inverse_view = misc::math::
                    matrix_view<double, typename Position::memory_space>(inverse.data(), N, N);
            if (!misc::math::
                        invert(inverse_view,
                               misc::math::matrix_view<
                                       double,
                                       typename Position::memory_space>(gram.data(), N, N),
                               misc::math::vector_view<
                                       double,
                                       typename Position::memory_space>(workspace.data(), N * N)))
                continue;

            // One row of P W P, with P = I - E (E^T E)^-1 E^T.
            // It annihilates constant Cartesian forms, as in the pointwise Hodge.
            for (std::size_t edge = 0; edge < N * edge_count; ++edge) {
                projector_row[edge] = edge == selected ? 1.0 : 0.0;
                for (std::size_t i = 0; i < N; ++i)
                    for (std::size_t j = 0; j < N; ++j)
                        projector_row[edge] -= edges[selected * N + i] * inverse[i * N + j]
                                               * edges[edge * N + j];
                projector_row[edge] *= pairing[(edge / edge_count) * (N + 1)] / edge_count;
                for (std::size_t i = 0; i < N; ++i)
                    weighted_projection[i] += edges[edge * N + i] * projector_row[edge];
            }
            for (std::size_t axis = 0; axis < N; ++axis)
                for (std::size_t edge_id = 0; edge_id < edge_count; ++edge_id) {
                    std::size_t const edge = axis * edge_count + edge_id;
                    double weight = pairing[normal * N + axis] / (edge_count * edge_count)
                                    + projector_row[edge];
                    for (std::size_t i = 0; i < N; ++i)
                        for (std::size_t j = 0; j < N; ++j)
                            weight -= weighted_projection[i] * inverse[i * N + j]
                                      * edges[edge * N + j];
                    weight *= diffusivity
                              * detail::permutation_sign<
                                      N>(std::array<std::size_t, 1> {normal}, ids);
                    Element first = cell;
                    std::size_t tangent = 0;
                    for (std::size_t other = 0; other < N; ++other)
                        if (other != axis)
                            ddc::detail::array(first)[other] += (edge_id >> tangent++) & 1;
                    Element local = first;
                    for (std::size_t coordinate = 0; coordinate < N; ++coordinate)
                        ddc::detail::array(local)[coordinate]
                                = ddc::detail::array(first)[coordinate]
                                  - ddc::detail::array(elem)[coordinate] + 1;
                    if constexpr (Index::rank() == 0) {
                        // The scalar coboundary's tensor stencil is the oriented
                        // endpoint difference; retain that incidence independently of H.
                        result.mem(local, ddc::DiscreteElement<Index>(0)) -= weight;
                        ++ddc::detail::array(local)[axis];
                        result.mem(local, ddc::DiscreteElement<Index>(0)) += weight;
                    } else
                        result.mem(local, ddc::DiscreteElement<Index>(axis)) += weight;
                }
        }
        return result;
    }

    template <class Element, class Component>
    KOKKOS_FUNCTION auto value(Element elem, Component component) const
    {
        return stencil(elem, component, std::make_index_sequence<Direction::size()> {});
    }

    /** Inverse zero-form Hodge with the physical, truncated dual volume. */
    template <class TopTensor, class Element>
    KOKKOS_FUNCTION double inverse_zero_form(TopTensor top, Element elem) const
    {
        using TopIndices = ddc::type_seq_merge_t<ddc::TypeSeq<Direction>, dual_indices>;
        using Scalar = tensor::Covariant<tensor::ScalarIndex>;
        using Memory = typename Position::memory_space;
        using ScalarTensor
                = tensor::Tensor<double, ddc::DiscreteDomain<Scalar>, Kokkos::layout_right, Memory>;
        auto scalar = detail::make_stencil<Memory, Scalar>(ddc::DiscreteElement<>());
        auto positions = detail::local_geometry(
                position,
                geometry,
                elem,
                std::make_index_sequence<Direction::size()> {});
        tensor::Tensor<
                double,
                typename Position::discrete_domain_type,
                Kokkos::layout_right,
                Memory>
                local_position(positions);
        auto metric = detail::euclidean_metric<Memory>(
                typename tensor::uncharacterize_t<Direction>::type_seq_dimensions {});
        DiscreteHodgeStar<
                CellComplex::Primal,
                tensor::upper_t<TopIndices>,
                ddc::TypeSeq<>,
                decltype(metric),
                decltype(local_position),
                Element>::operator()(ScalarTensor(scalar), top, metric, local_position, elem);
        double const local_volume = SimplexVolume<
                CellComplex::Primal,
                Direction::size(),
                decltype(metric),
                decltype(local_position),
                Element>::template
        operator()<Direction::size()>(
                metric,
                local_position,
                elem,
                []<std::size_t... Axis>(std::index_sequence<Axis...>) {
                    return std::array<std::size_t, Direction::size()> {Axis...};
                }(std::make_index_sequence<Direction::size()> {}));
        // Each incident cell contributes one vertex share to the dual volume;
        // physical walls truncate it, while interface rules supply donor cells.
        return scalar.mem(ddc::DiscreteElement<Scalar>(0)) * local_volume
               / dual_volume(elem, std::make_index_sequence<Direction::size()> {});
    }

    /** Apply the inverse zero-form Hodge to a pointwise top cochain.
     * \important This operator and documentation is fully AI-generated.
     */
    template <class Output, class TopTensor, class Element>
    KOKKOS_FUNCTION void operator()(Output output, TopTensor top, Element elem) const
    {
        output.mem(ddc::DiscreteElement<tensor::Covariant<tensor::ScalarIndex>>(0))
                = inverse_zero_form(top, elem);
    }

    template <class Element, class Component>
    KOKKOS_FUNCTION double mem(Element elem, Component component) const
    {
        double prescribed = 0.0;
        if (prescribed_flux(elem, component, prescribed))
            return prescribed;
        auto const coefficients = value(elem, component);
        double result = 0.0;
        ddc::device_for_each(coefficients.domain(), [&](auto entry) {
            double const weight = coefficients.mem(entry);
            if (weight != 0.0)
                result += weight
                          * cochain
                                    .mem(tensor::detail::stencil_sample(elem, entry),
                                         typename Cochain::indices_domain_t::discrete_element_type(
                                                 entry));
        });
        return result;
    }
};

/** Normalize single policies locally, as for the staged differential operators.
 * \important This operator and documentation is fully AI-generated.
 */
template <class Direction, class Cochain, class Position, class Geometry, class Flux>
auto make_dual_hodge_cochain(
        Cochain cochain,
        Position const& position,
        Geometry geometry_rule,
        Flux flux_rule,
        double diffusivity = 1.0)
{
#if defined(KOKKOS_ENABLE_CUDA)
    if constexpr (Kokkos::SpaceAccessibility<Kokkos::Cuda, typename Position::memory_space>::
                          accessible) {
        // Debug builds cannot always infer the full sampling/Hodge call depth.
        std::size_t stack_size = 0;
        if (cudaDeviceGetLimit(&stack_size, cudaLimitStackSize) != cudaSuccess
            || (stack_size < 32768 && cudaDeviceSetLimit(cudaLimitStackSize, 32768) != cudaSuccess))
            throw std::runtime_error("cannot reserve the CUDA stack for scalar DEC sampling");
    }
#endif
    auto const geometry = [&] {
        if constexpr (misc::Specialization<Geometry, tensor::ExtrapolationRules>)
            return geometry_rule;
        else
            return tensor::make_extrapolation_rules(position, geometry_rule);
    }();
    auto const flux = [&] {
        if constexpr (misc::Specialization<Flux, tensor::ExtrapolationRules>)
            return flux_rule;
        else
            return tensor::make_extrapolation_rules(cochain, flux_rule);
    }();
    return DualHodgeCochain<
            Direction,
            Cochain,
            Position,
            std::remove_cvref_t<decltype(geometry)>,
            std::remove_cvref_t<decltype(flux)>> {cochain, position, geometry, flux, diffusivity};
}


} // namespace sil::exterior
