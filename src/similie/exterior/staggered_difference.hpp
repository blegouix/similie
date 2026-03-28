// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>

#include <ddc/ddc.hpp>

#include <similie/misc/specialization.hpp>
#include <similie/tensor/character.hpp>
#include <similie/tensor/tensor_impl.hpp>

namespace sil {

namespace exterior {

namespace detail {

template <class CDim, class TypeSeq>
struct DiscreteDimensionForTypeSeq;

template <class CDim, class DDim, class... Tail>
struct DiscreteDimensionForTypeSeq<CDim, ddc::detail::TypeSeq<DDim, Tail...>>
{
    using type = std::conditional_t<
            std::is_same_v<typename DDim::continuous_dimension_type, CDim>,
            DDim,
            typename DiscreteDimensionForTypeSeq<CDim, ddc::detail::TypeSeq<Tail...>>::type>;
};

template <class CDim>
struct DiscreteDimensionForTypeSeq<CDim, ddc::detail::TypeSeq<>>
{
    using type = void;
};

template <class CDim, class T>
using discrete_dimension_for_t
        = typename DiscreteDimensionForTypeSeq<CDim, ddc::to_type_seq_t<T>>::type;

template <class TargetDDim, class SourceElem>
KOKKOS_FUNCTION ddc::DiscreteElement<TargetDDim> remap_dimension(SourceElem const& source_elem)
{
    using source_d_dim_t
            = discrete_dimension_for_t<typename TargetDDim::continuous_dimension_type, SourceElem>;
    static_assert(!std::is_void_v<source_d_dim_t>);
    return ddc::DiscreteElement<TargetDDim>(ddc::uid<source_d_dim_t>(source_elem));
}

template <class TargetElem, class SourceElem>
struct RemapElement;

template <class... TargetDDims, class SourceElem>
struct RemapElement<ddc::DiscreteElement<TargetDDims...>, SourceElem>
{
    KOKKOS_FUNCTION static ddc::DiscreteElement<TargetDDims...> apply(SourceElem const& source_elem)
    {
        return ddc::DiscreteElement<TargetDDims...>(
                remap_dimension<TargetDDims>(source_elem)...);
    }
};

template <class TargetElem, class SourceElem>
KOKKOS_FUNCTION TargetElem remap_element(SourceElem const& source_elem)
{
    return RemapElement<TargetElem, SourceElem>::apply(source_elem);
}

template <class CDim, class Index>
KOKKOS_FUNCTION ddc::DiscreteElement<Index> component_element()
{
    using natural_index_t = tensor::uncharacterize_t<Index>;
    return ddc::DiscreteElement<Index>(
            ddc::type_seq_rank_v<CDim, typename natural_index_t::type_seq_dimensions>);
}

} // namespace detail

template <
        class CDim,
        misc::Specialization<tensor::Tensor> OutTensorType,
        misc::Specialization<tensor::Tensor> InTensorType,
        class ExecSpace>
OutTensorType staggered_deriv_component(
        ExecSpace const& exec_space,
        OutTensorType out,
        InTensorType in)
{
    using out_index_t = ddc::type_seq_element_t<
            0,
            ddc::to_type_seq_t<typename OutTensorType::indices_domain_t>>;
    using in_index_t = ddc::type_seq_element_t<
            0,
            ddc::to_type_seq_t<typename InTensorType::indices_domain_t>>;
    using in_d_dim_t = detail::discrete_dimension_for_t<CDim, typename InTensorType::non_indices_domain_t>;
    using in_elem_t = typename InTensorType::non_indices_domain_t::discrete_element_type;
    using out_elem_t = typename OutTensorType::non_indices_domain_t::discrete_element_type;
    static_assert(out_index_t::rank() == 0);
    static_assert(in_index_t::rank() == 0);
    static_assert(!std::is_void_v<in_d_dim_t>);

    ddc::parallel_for_each(
            "similie_staggered_deriv_component",
            exec_space,
            out.non_indices_domain(),
            KOKKOS_LAMBDA(out_elem_t out_elem) {
                in_elem_t const in_left = detail::remap_element<in_elem_t>(out_elem);
                in_elem_t const in_right = in_left + ddc::DiscreteVector<in_d_dim_t>(1);
                double const dx = static_cast<double>(
                        ddc::coordinate(ddc::DiscreteElement<in_d_dim_t>(in_right))
                        - ddc::coordinate(ddc::DiscreteElement<in_d_dim_t>(in_left)));
                out.mem(out_elem, ddc::DiscreteElement<out_index_t>(0))
                        = (in.get(in_right, ddc::DiscreteElement<in_index_t>(0))
                           - in.get(in_left, ddc::DiscreteElement<in_index_t>(0)))
                          / dx;
            });
    return out;
}

template <
        class AlphaCDim,
        class BetaCDim,
        misc::Specialization<tensor::Tensor> OutTensorType,
        misc::Specialization<tensor::Tensor> FluxTensorType,
        misc::Specialization<tensor::Tensor> MetricType,
        class ExecSpace>
OutTensorType add_staggered_codifferential_component(
        ExecSpace const& exec_space,
        OutTensorType out,
        FluxTensorType flux,
        MetricType inv_metric)
{
    using out_index_t = ddc::type_seq_element_t<
            0,
            ddc::to_type_seq_t<typename OutTensorType::indices_domain_t>>;
    using out_d_dim_t
            = detail::discrete_dimension_for_t<AlphaCDim, typename OutTensorType::non_indices_domain_t>;
    using flux_d_dim_t = detail::
            discrete_dimension_for_t<AlphaCDim, typename FluxTensorType::non_indices_domain_t>;
    using out_elem_t = typename OutTensorType::non_indices_domain_t::discrete_element_type;
    using flux_elem_t = typename FluxTensorType::non_indices_domain_t::discrete_element_type;
    using flux_index_t = ddc::type_seq_element_t<
            0,
            ddc::to_type_seq_t<typename FluxTensorType::indices_domain_t>>;
    static_assert(out_index_t::rank() == 0);
    static_assert(!std::is_void_v<out_d_dim_t>);
    static_assert(!std::is_void_v<flux_d_dim_t>);

    ddc::parallel_for_each(
            "similie_add_staggered_codifferential_component",
            exec_space,
            out.non_indices_domain(),
            KOKKOS_LAMBDA(out_elem_t out_elem) {
                ddc::DiscreteDomain<out_d_dim_t> dim_dom(out.non_indices_domain());
                ddc::DiscreteElement<out_d_dim_t> dim_elem(out_elem);
                if (dim_elem.uid() == dim_dom.front().uid() || dim_elem.uid() == dim_dom.back().uid()) {
                    return;
                }

                flux_elem_t const right_face = detail::remap_element<flux_elem_t>(out_elem);
                flux_elem_t const left_face = right_face - ddc::DiscreteVector<flux_d_dim_t>(1);
                double const dx = static_cast<double>(
                        ddc::coordinate(ddc::DiscreteElement<flux_d_dim_t>(right_face))
                        - ddc::coordinate(ddc::DiscreteElement<flux_d_dim_t>(left_face)));
                double const deriv = (flux.get(right_face, ddc::DiscreteElement<flux_index_t>(0))
                                      - flux.get(left_face, ddc::DiscreteElement<flux_index_t>(0)))
                                     / dx;
                out.mem(out_elem, ddc::DiscreteElement<out_index_t>(0))
                        += inv_metric.get(
                                   out_elem,
                                   inv_metric.accessor().template access_element<AlphaCDim, BetaCDim>())
                           * deriv;
            });
    return out;
}

} // namespace exterior

} // namespace sil
