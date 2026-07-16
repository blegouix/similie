// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

#include <ddc/ddc.hpp>

#include <similie/misc/macros.hpp>
#include <similie/misc/small_matrix.hpp>
#include <similie/misc/unsecure_parallel_deepcopy.hpp>

#include "character.hpp"
#include "diagonal_tensor.hpp"
#include "identity_tensor.hpp"
#include "lorentzian_sign_tensor.hpp"
#include "prime.hpp"
#include "relabelization.hpp"
#include "symmetric_tensor.hpp"
#include "tensor_prod.hpp"

namespace sil {

namespace tensor {

// Abstract indices used to characterize a generic metric

template <class... CDim>
struct MetricIndex1 : TensorNaturalIndex<CDim...>
{
};

template <class MetricIndex>
using metric_index_1
        = ddc::type_seq_element_t<0, ddc::to_type_seq_t<typename MetricIndex::subindices_domain_t>>;

template <class... CDim>
struct MetricIndex2 : TensorNaturalIndex<CDim...>
{
};

template <class MetricIndex>
using metric_index_2
        = ddc::type_seq_element_t<1, ddc::to_type_seq_t<typename MetricIndex::subindices_domain_t>>;

namespace detail {

template <class TypeSeqCDim>
struct ConvertTypeSeqToMetricIndex1;

template <class... CDim>
struct ConvertTypeSeqToMetricIndex1<ddc::detail::TypeSeq<CDim...>>
{
    using type = MetricIndex1<CDim...>;
};

template <class TypeSeqCDim>
struct ConvertTypeSeqToMetricIndex2;

template <class... CDim>
struct ConvertTypeSeqToMetricIndex2<ddc::detail::TypeSeq<CDim...>>
{
    using type = MetricIndex2<CDim...>;
};

} // namespace detail

// Relabelize metric
template <
        misc::Specialization<ddc::DiscreteDomain> Dom,
        TensorNatIndex Index1,
        TensorNatIndex Index2>
using relabelize_metric_in_domain_t = relabelize_indices_in_t<
        Dom,
        ddc::detail::TypeSeq<
                typename detail::ConvertTypeSeqToMetricIndex1<
                        typename Index1::type_seq_dimensions>::type,
                typename detail::ConvertTypeSeqToMetricIndex2<
                        typename Index2::type_seq_dimensions>::type>,
        ddc::detail::TypeSeq<uncharacterize_t<Index1>, uncharacterize_t<Index2>>>;

template <TensorNatIndex Index1, TensorNatIndex Index2, class Dom>
constexpr relabelize_metric_in_domain_t<Dom, Index1, Index2> relabelize_metric_in_domain(
        Dom metric_dom)
{
    return relabelize_indices_in<
            ddc::detail::TypeSeq<
                    typename detail::ConvertTypeSeqToMetricIndex1<
                            typename Index1::type_seq_dimensions>::type,
                    typename detail::ConvertTypeSeqToMetricIndex2<
                            typename Index2::type_seq_dimensions>::type>,
            ddc::detail::TypeSeq<uncharacterize_t<Index1>, uncharacterize_t<Index2>>>(metric_dom);
}

template <misc::Specialization<Tensor> TensorType, TensorNatIndex Index1, TensorNatIndex Index2>
using relabelize_metric_t = relabelize_indices_of_t<
        TensorType,
        ddc::detail::TypeSeq<
                typename detail::ConvertTypeSeqToMetricIndex1<
                        typename Index1::type_seq_dimensions>::type,
                typename detail::ConvertTypeSeqToMetricIndex2<
                        typename Index2::type_seq_dimensions>::type>,
        ddc::detail::TypeSeq<uncharacterize_t<Index1>, uncharacterize_t<Index2>>>;

template <TensorNatIndex Index1, TensorNatIndex Index2, misc::Specialization<Tensor> TensorType>
constexpr relabelize_metric_t<TensorType, Index1, Index2> relabelize_metric(TensorType tensor)
{
    return relabelize_indices_of<
            ddc::detail::TypeSeq<
                    typename detail::ConvertTypeSeqToMetricIndex1<
                            typename Index1::type_seq_dimensions>::type,
                    typename detail::ConvertTypeSeqToMetricIndex2<
                            typename Index2::type_seq_dimensions>::type>,
            ddc::detail::TypeSeq<uncharacterize_t<Index1>, uncharacterize_t<Index2>>>(tensor);
}

// Compute domain for a tensor product of metrics (ie. g_mu_muprime*g_nu_nuprime*...)
namespace detail {

template <class MetricIndex, class Indices1, class Indices2>
struct MetricProdDomainType;

template <class MetricIndex, class... Index1, class... Index2>
struct MetricProdDomainType<
        MetricIndex,
        ddc::detail::TypeSeq<Index1...>,
        ddc::detail::TypeSeq<Index2...>>
{
    static_assert(sizeof...(Index1) == sizeof...(Index2));
    using type = ddc::cartesian_prod_t<relabelize_metric_in_domain_t<
            ddc::DiscreteDomain<MetricIndex>,
            Index1,
            ddc::type_seq_element_t<
                    ddc::type_seq_rank_v<Index1, ddc::detail::TypeSeq<Index1...>>,
                    ddc::detail::TypeSeq<Index2...>>>...>;
};

} // namespace detail

template <
        TensorIndex MetricIndex,
        misc::Specialization<ddc::detail::TypeSeq> Indices1,
        misc::Specialization<ddc::detail::TypeSeq> Indices2>
using metric_prod_domain_t =
        typename detail::MetricProdDomainType<MetricIndex, Indices1, Indices2>::type;

namespace detail {

// Compute tensor product of metrics (ie. g_mu_muprime*g_nu_nuprime*...)
template <
        class NonMetricDom,
        class MetricIndex,
        class Indices1,
        class Indices2,
        class LayoutStridedPolicy,
        class MemorySpace>
struct MetricProdType;

template <
        class NonMetricDom,
        class MetricIndex,
        class... Index1,
        class... Index2,
        class LayoutStridedPolicy,
        class MemorySpace>
struct MetricProdType<
        NonMetricDom,
        MetricIndex,
        ddc::detail::TypeSeq<Index1...>,
        ddc::detail::TypeSeq<Index2...>,
        LayoutStridedPolicy,
        MemorySpace>
{
    using type = tensor::Tensor<
            double,
            ddc::cartesian_prod_t<
                    NonMetricDom,
                    metric_prod_domain_t<
                            MetricIndex,
                            ddc::detail::TypeSeq<Index1...>,
                            ddc::detail::TypeSeq<Index2...>>>,
            LayoutStridedPolicy,
            MemorySpace>;
};

template <class MetricType, class BatchElem, class Indices1, class Indices2>
struct MetricProdValue;

template <class MetricType, class BatchElem>
struct MetricProdValue<MetricType, BatchElem, ddc::detail::TypeSeq<>, ddc::detail::TypeSeq<>>
{
    KOKKOS_FUNCTION static double run(
            [[maybe_unused]] MetricType metric,
            [[maybe_unused]] BatchElem elem,
            [[maybe_unused]] auto natural_elem)
    {
        return 1.;
    }
};

template <
        class MetricType,
        class BatchElem,
        class HeadIndex1,
        class... TailIndex1,
        class HeadIndex2,
        class... TailIndex2>
struct MetricProdValue<
        MetricType,
        BatchElem,
        ddc::detail::TypeSeq<HeadIndex1, TailIndex1...>,
        ddc::detail::TypeSeq<HeadIndex2, TailIndex2...>>
{
    KOKKOS_FUNCTION static double run(MetricType metric, BatchElem elem, auto natural_elem)
    {
        auto const relabeled_metric = relabelize_metric<HeadIndex1, HeadIndex2>(metric);
        using metric_natural_elem_type = typename decltype(relabeled_metric)::accessor_t::
                natural_domain_t::discrete_element_type;
        metric_natural_elem_type const metric_natural_elem(natural_elem);
        return relabeled_metric.get(relabeled_metric.access_element(elem, metric_natural_elem))
               * MetricProdValue<
                       MetricType,
                       BatchElem,
                       ddc::detail::TypeSeq<TailIndex1...>,
                       ddc::detail::TypeSeq<TailIndex2...>>::run(metric, elem, natural_elem);
    }
};

} // namespace detail

template <
        misc::Specialization<ddc::DiscreteDomain> NonMetricDom,
        TensorIndex MetricIndex,
        misc::Specialization<ddc::detail::TypeSeq> Indices1,
        misc::Specialization<ddc::detail::TypeSeq> Indices2,
        class LayoutStridedPolicy,
        class MemorySpace>
using metric_prod_t = typename detail::MetricProdType<
        NonMetricDom,
        MetricIndex,
        Indices1,
        Indices2,
        LayoutStridedPolicy,
        MemorySpace>::type;

template <
        TensorIndex MetricIndex,
        misc::Specialization<ddc::detail::TypeSeq> Indices1,
        misc::Specialization<ddc::detail::TypeSeq> Indices2,
        misc::Specialization<Tensor> MetricType,
        class BatchElem>
struct MetricProd
{
    template <misc::Specialization<Tensor> MetricProdType_>
    KOKKOS_FUNCTION static void run(MetricProdType_ metric_prod, MetricType metric, BatchElem elem)
    {
        ddc::device_for_each(metric_prod.domain(), [&](auto mem_elem) {
            metric_prod.mem(mem_elem)
                    = value(metric, elem, metric_prod.canonical_natural_element(mem_elem));
        });
    }

    KOKKOS_FUNCTION static double value(MetricType metric, BatchElem elem, auto natural_elem)
    {
        return detail::MetricProdValue<MetricType, BatchElem, Indices1, Indices2>::
                run(metric, elem, natural_elem);
    }
};

template <
        TensorIndex MetricIndex,
        misc::Specialization<ddc::detail::TypeSeq> Indices1,
        misc::Specialization<ddc::detail::TypeSeq> Indices2,
        misc::Specialization<Tensor> MetricType,
        class ExecSpace>
metric_prod_t<
        typename MetricType::non_indices_domain_t,
        MetricIndex,
        Indices1,
        Indices2,
        typename MetricType::layout_type,
        typename MetricType::memory_space>
fill_metric_prod(
        ExecSpace const& exec_space,
        metric_prod_t<
                typename MetricType::non_indices_domain_t,
                MetricIndex,
                Indices1,
                Indices2,
                typename MetricType::layout_type,
                typename MetricType::memory_space> metric_prod,
        MetricType metric)
{
    SIMILIE_DEBUG_LOG("similie_compute_metric_prod");
    ddc::parallel_for_each(
            "similie_compute_metric_prod",
            exec_space,
            metric_prod.non_indices_domain(),
            KOKKOS_LAMBDA(
                    typename decltype(metric_prod)::non_indices_domain_t::discrete_element_type
                            elem) {
                MetricProd<MetricIndex, Indices1, Indices2, MetricType, decltype(elem)>::
                        run(metric_prod[elem], metric, elem);
            });
    return metric_prod;
}

namespace detail {

template <class Dom>
using non_primes = ddc::type_seq_remove_t<ddc::to_type_seq_t<Dom>, primes<ddc::to_type_seq_t<Dom>>>;

} // namespace detail

// Apply metrics inplace (like g_mu_muprime*T^muprime^nu)
template <
        misc::Specialization<Tensor> MetricType,
        misc::Specialization<Tensor> TensorType,
        class ExecSpace>
relabelize_indices_of_t<
        TensorType,
        swap_character_t<detail::non_primes<typename MetricType::accessor_t::natural_domain_t>>,
        detail::non_primes<typename MetricType::accessor_t::natural_domain_t>>
inplace_apply_metric(ExecSpace const& exec_space, TensorType tensor, MetricType metric_prod)
{
    ddc::Chunk result_alloc(
            relabelize_indices_in<
                    swap_character_t<
                            detail::non_primes<typename MetricType::accessor_t::natural_domain_t>>,
                    detail::non_primes<typename MetricType::accessor_t::natural_domain_t>>(
                    tensor.domain()),
            ddc::KokkosAllocator<double, typename ExecSpace::memory_space>());
    relabelize_indices_of_t<
            TensorType,
            swap_character_t<detail::non_primes<typename MetricType::accessor_t::natural_domain_t>>,
            detail::non_primes<typename MetricType::accessor_t::natural_domain_t>>
            result(result_alloc);
    SIMILIE_DEBUG_LOG("similie_inplace_apply_metric");
    ddc::parallel_for_each(
            "similie_inplace_apply_metric",
            exec_space,
            tensor.non_indices_domain(),
            KOKKOS_LAMBDA(typename TensorType::non_indices_domain_t::discrete_element_type elem) {
                tensor_prod(
                        result[elem],
                        metric_prod[elem],
                        relabelize_indices_of<
                                swap_character_t<detail::non_primes<
                                        typename MetricType::accessor_t::natural_domain_t>>,
                                primes<swap_character_t<detail::non_primes<
                                        typename MetricType::accessor_t::natural_domain_t>>>>(
                                tensor)[elem]);
            });
    misc::detail::unsecure_parallel_deepcopy(exec_space, tensor, result);

    return relabelize_indices_of<
            swap_character_t<detail::non_primes<typename MetricType::accessor_t::natural_domain_t>>,
            detail::non_primes<typename MetricType::accessor_t::natural_domain_t>>(tensor);
}

template <
        TensorIndex MetricIndex,
        TensorNatIndex... Index1,
        misc::Specialization<Tensor> MetricType,
        misc::Specialization<Tensor> TensorType,
        class ExecSpace>
relabelize_indices_of_t<
        TensorType,
        swap_character_t<ddc::detail::TypeSeq<Index1...>>,
        ddc::detail::TypeSeq<Index1...>>
inplace_apply_metric(ExecSpace const& exec_space, TensorType tensor, MetricType metric)
{
    tensor::tensor_accessor_for_domain_t<metric_prod_domain_t<
            MetricIndex,
            ddc::detail::TypeSeq<Index1...>,
            primes<ddc::detail::TypeSeq<Index1...>>>>
            metric_prod_accessor;
    ddc::Chunk metric_prod_alloc(
            ddc::cartesian_prod_t<
                    typename TensorType::non_indices_domain_t,
                    metric_prod_domain_t<
                            MetricIndex,
                            ddc::detail::TypeSeq<Index1...>,
                            primes<ddc::detail::TypeSeq<Index1...>>>>(
                    tensor.non_indices_domain(),
                    metric_prod_accessor.domain()),
            ddc::KokkosAllocator<double, typename ExecSpace::memory_space>());
    tensor::Tensor metric_prod(metric_prod_alloc);

    fill_metric_prod<
            MetricIndex,
            ddc::detail::TypeSeq<Index1...>,
            primes<ddc::detail::TypeSeq<Index1...>>>(exec_space, metric_prod, metric);

    return inplace_apply_metric(exec_space, tensor, metric_prod);
}

template <misc::Specialization<Tensor> MetricType>
using invert_metric_t = relabelize_indices_of_t<
        MetricType,
        ddc::to_type_seq_t<typename MetricType::accessor_t::natural_domain_t>,
        swap_character_t<ddc::to_type_seq_t<typename MetricType::accessor_t::natural_domain_t>>>;

template <TensorIndex MetricIndex, misc::Specialization<Tensor> MetricType, class BatchElem>
struct InverseMetric
{
    using output_tensor_type = invert_metric_t<MetricType>;

    template <misc::Specialization<Tensor> OutputTensorType>
    KOKKOS_FUNCTION static void run(OutputTensorType inv_metric, MetricType metric, BatchElem elem)
    {
        if constexpr (
                misc::Specialization<MetricIndex, TensorIdentityIndex>
                || misc::Specialization<MetricIndex, TensorLorentzianSignIndex>) {
            ddc::device_for_each(inv_metric.domain(), [&](auto mem_elem) {
                inv_metric.mem(mem_elem)
                        = value(metric, elem, inv_metric.canonical_natural_element(mem_elem));
            });
        } else if constexpr (misc::Specialization<MetricIndex, TensorDiagonalIndex>) {
            ddc::device_for_each(inv_metric.domain(), [&](auto mem_elem) {
                inv_metric.mem(mem_elem)
                        = value(metric, elem, inv_metric.canonical_natural_element(mem_elem));
            });
        } else if constexpr (misc::Specialization<MetricIndex, TensorSymmetricIndex>) {
            constexpr std::size_t N = metric_index_1<MetricIndex>::size();
            using memory_space = typename MetricType::memory_space;
            using metric_index_1_t = metric_index_1<MetricIndex>;
            using metric_index_2_t = metric_index_2<MetricIndex>;

            std::array<double, N * N> matrix_alloc {};
            std::array<double, N * N> inverse_alloc {};
            std::array<double, N * N> workspace_alloc {};
            auto matrix_view
                    = misc::math::matrix_view<double, memory_space>(matrix_alloc.data(), N, N);
            auto inverse_view
                    = misc::math::matrix_view<double, memory_space>(inverse_alloc.data(), N, N);
            auto workspace
                    = misc::math::vector_view<double, memory_space>(workspace_alloc.data(), N * N);

            for (std::size_t i = 0; i < N; ++i) {
                for (std::size_t j = 0; j < N; ++j) {
                    matrix_view(i, j) = metric.get(metric.access_element(
                            elem,
                            ddc::DiscreteElement<metric_index_1_t, metric_index_2_t>(i, j)));
                }
            }

            bool const success = misc::math::invert(inverse_view, matrix_view, workspace);
            assert(success && "Kokkos-kernels failed at inverting metric tensor");

            ddc::device_for_each(inv_metric.domain(), [&](auto mem_elem) {
                auto const natural_elem = inv_metric.canonical_natural_element(mem_elem);
                auto const ids = ddc::detail::array(natural_elem);
                inv_metric.mem(mem_elem) = inverse_view(ids[0], ids[1]);
            });
        }
    }

    KOKKOS_FUNCTION static double value(MetricType metric, BatchElem elem, auto natural_elem)
    {
        if constexpr (misc::Specialization<MetricIndex, TensorIdentityIndex>) {
            auto const ids = ddc::detail::array(natural_elem);
            return ids[0] == ids[1] ? 1. : 0.;
        } else if constexpr (misc::Specialization<MetricIndex, TensorLorentzianSignIndex>) {
            return metric.get(metric.access_element(elem, natural_elem));
        } else if constexpr (misc::Specialization<MetricIndex, TensorDiagonalIndex>) {
            return 1.
                   / metric.get(
                           relabelize_indices_of<
                                   swap_character_t<ddc::to_type_seq_t<
                                           typename MetricType::accessor_t::natural_domain_t>>,
                                   ddc::to_type_seq_t<
                                           typename MetricType::accessor_t::natural_domain_t>>(
                                   metric)
                                   .access_element(elem, natural_elem));
        } else if constexpr (misc::Specialization<MetricIndex, TensorSymmetricIndex>) {
            [[maybe_unused]] typename output_tensor_type::accessor_t accessor;
            std::array<double, output_tensor_type::accessor_t::domain().size()> alloc {};
            ddc::ChunkSpan<
                    double,
                    typename output_tensor_type::indices_domain_t,
                    Kokkos::layout_right,
                    typename MetricType::memory_space>
                    span(alloc.data(), accessor.domain());
            tensor::Tensor local_inverse(span);
            run(local_inverse, metric, elem);
            return local_inverse.get(local_inverse.access_element(natural_elem));
        } else {
            return 0.;
        }
    }
};

// Compute invert metric (g_mu_nu for gmunu or gmunu for g_mu_nu)
template <TensorIndex MetricIndex, misc::Specialization<Tensor> MetricType, class ExecSpace>
invert_metric_t<MetricType> fill_inverse_metric(
        ExecSpace const& exec_space,
        invert_metric_t<MetricType> inv_metric,
        MetricType metric)
{
    SIMILIE_DEBUG_LOG("similie_invert_metric");
    ddc::parallel_for_each(
            "similie_invert_metric",
            exec_space,
            inv_metric.non_indices_domain(),
            KOKKOS_LAMBDA(
                    typename invert_metric_t<
                            MetricType>::non_indices_domain_t::discrete_element_type elem) {
                InverseMetric<MetricIndex, MetricType, decltype(elem)>::
                        run(inv_metric[elem], metric, elem);
            });

    return inv_metric;
}

} // namespace tensor

} // namespace sil
