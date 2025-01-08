// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <ddc/ddc.hpp>

#include <similie/misc/unsecure_parallel_deepcopy.hpp>

#include <KokkosBatched_InverseLU_Decl.hpp>
#include <KokkosBatched_LU_Decl.hpp>

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

namespace detail {

template <class Indices1, class Indices2>
struct FillMetricProd;

template <>
struct FillMetricProd<ddc::detail::TypeSeq<>, ddc::detail::TypeSeq<>>
{
    template <class MetricProdType, class MetricType, class MetricProdType_, class ExecSpace>
    static MetricProdType run(
            ExecSpace const& exec_space,
            MetricProdType metric_prod,
            [[maybe_unused]] MetricType metric,
            MetricProdType_ metric_prod_)
    {
        misc::detail::unsecure_parallel_deepcopy(exec_space, metric_prod, metric_prod_);
        return metric_prod;
    }
};

template <class HeadIndex1, class... TailIndex1, class HeadIndex2, class... TailIndex2>
struct FillMetricProd<
        ddc::detail::TypeSeq<HeadIndex1, TailIndex1...>,
        ddc::detail::TypeSeq<HeadIndex2, TailIndex2...>>
{
    template <class MetricProdType, class MetricType, class MetricProdType_, class ExecSpace>
    static MetricProdType run(
            ExecSpace const& exec_space,
            MetricProdType metric_prod,
            MetricType metric,
            MetricProdType_ metric_prod_)
    {
        ddc::cartesian_prod_t<
                typename MetricProdType_::discrete_domain_type,
                relabelize_metric_in_domain_t<
                        typename MetricType::indices_domain_t,
                        HeadIndex1,
                        HeadIndex2>>
                new_metric_prod_dom_(
                        metric_prod_.domain(),
                        relabelize_metric_in_domain<HeadIndex1, HeadIndex2>(
                                metric.indices_domain()));
        ddc::Chunk new_metric_prod_alloc_(
                new_metric_prod_dom_,
                ddc::KokkosAllocator<double, typename ExecSpace::memory_space>());
        tensor::Tensor new_metric_prod_(new_metric_prod_alloc_);

        if (new_metric_prod_dom_.size() != 0) {
            ddc::parallel_for_each(
                    exec_space,
                    new_metric_prod_.non_indices_domain(),
                    KOKKOS_LAMBDA(typename decltype(new_metric_prod_)::non_indices_domain_t::
                                          discrete_element_type elem) {
                        tensor_prod(
                                new_metric_prod_[elem],
                                metric_prod_[elem],
                                relabelize_metric<HeadIndex1, HeadIndex2>(metric)[elem]);
                    });
        }


        return FillMetricProd<
                ddc::detail::TypeSeq<TailIndex1...>,
                ddc::detail::TypeSeq<TailIndex2...>>::
                run(exec_space, metric_prod, metric, new_metric_prod_);
    }
};
} // namespace detail

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
    typename MetricType::non_indices_domain_t dom_(metric.domain());
    ddc::Chunk metric_prod_alloc_(
            dom_,
            ddc::KokkosAllocator<double, typename ExecSpace::memory_space>());
    tensor::Tensor metric_prod_(metric_prod_alloc_);
    ddc::parallel_fill(exec_space, metric_prod_, 1.);

    return detail::FillMetricProd<Indices1, Indices2>::
            run(exec_space, metric_prod, metric, metric_prod_);
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
    ddc::parallel_for_each(
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

// Compute invert metric (g_mu_nu for gmunu or gmunu for g_mu_nu)
template <TensorIndex MetricIndex, misc::Specialization<Tensor> MetricType, class ExecSpace>
invert_metric_t<MetricType> fill_inverse_metric(
        ExecSpace const& exec_space,
        invert_metric_t<MetricType> inv_metric,
        MetricType metric)
{
    if constexpr (
            misc::Specialization<MetricIndex, TensorIdentityIndex>
            || misc::Specialization<MetricIndex, TensorLorentzianSignIndex>) {
    } else if (misc::Specialization<MetricIndex, TensorDiagonalIndex>) {
        ddc::parallel_for_each(
                exec_space,
                inv_metric.domain(),
                KOKKOS_LAMBDA(invert_metric_t<MetricType>::discrete_element_type elem) {
                    inv_metric.mem(elem)
                            = 1.
                              / metric.mem(relabelize_indices_in<
                                           swap_character_t<ddc::to_type_seq_t<
                                                   typename MetricType::accessor_t::
                                                           natural_domain_t>>,
                                           ddc::to_type_seq_t<typename MetricType::accessor_t::
                                                                      natural_domain_t>>(elem));
                });
    } else if (misc::Specialization<MetricIndex, TensorSymmetricIndex>) {
        // Allocate a buffer mirroring the metric as a full matrix
        ddc::Chunk buffer_alloc(
                ddc::cartesian_prod_t<
                        typename MetricType::non_indices_domain_t,
                        ddc::DiscreteDomain<
                                tensor::metric_index_1<MetricIndex>,
                                tensor::metric_index_2<MetricIndex>>>(
                        metric.non_indices_domain(),
                        ddc::DiscreteDomain<
                                tensor::metric_index_1<MetricIndex>,
                                tensor::metric_index_2<MetricIndex>>(metric.natural_domain())),
                ddc::KokkosAllocator<double, typename ExecSpace::memory_space>());
        ddc::ChunkSpan buffer(buffer_alloc);
        // Allocate a buffer for KokkosBatched::SerialInverseLU internal needs
        ddc::Chunk buffer_alloc2(
                ddc::cartesian_prod_t<
                        typename MetricType::non_indices_domain_t,
                        ddc::DiscreteDomain<tensor::metric_index_1<MetricIndex>>>(
                        metric.non_indices_domain(),
                        ddc::DiscreteDomain<tensor::metric_index_1<MetricIndex>>(
                                ddc::DiscreteElement<tensor::metric_index_1<MetricIndex>>(0),
                                ddc::DiscreteVector<tensor::metric_index_1<MetricIndex>>(
                                        metric.natural_domain()
                                                .template extent<
                                                        tensor::metric_index_1<MetricIndex>>()
                                        * metric.natural_domain()
                                                  .template extent<
                                                          tensor::metric_index_1<MetricIndex>>()))),
                ddc::KokkosAllocator<double, typename ExecSpace::memory_space>());
        ddc::ChunkSpan buffer2(buffer_alloc2);

        // process
        ddc::parallel_for_each(
                exec_space,
                inv_metric.non_indices_domain(),
                KOKKOS_LAMBDA(typename invert_metric_t<
                              MetricType>::non_indices_domain_t::discrete_element_type elem) {
                    ddc::annotated_for_each(
                            ddc::DiscreteDomain<
                                    tensor::metric_index_1<MetricIndex>,
                                    tensor::metric_index_2<MetricIndex>>(metric.natural_domain()),
                            [&](ddc::DiscreteElement<
                                    tensor::metric_index_1<MetricIndex>,
                                    tensor::metric_index_2<MetricIndex>> index) {
                                buffer(elem, index) = metric(metric.access_element(elem, index));
                            });

                    int err = KokkosBatched::SerialLU<KokkosBatched::Algo::SolveLU::Unblocked>::invoke(
                            buffer[elem]
                                    .allocation_kokkos_view()); // Seems to require diagonal-dominant
                    err += KokkosBatched::SerialInverseLU<KokkosBatched::Algo::SolveLU::Unblocked>::
                            invoke(buffer[elem].allocation_kokkos_view(),
                                   buffer2[elem].allocation_kokkos_view());
                    assert(err == 0 && "Kokkos-kernels failed at inverting metric tensor");
                    /*
                    Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> const
                            sol("metric_inversion_sol", n, n);
                    for (std::size_t i=0; i<n; ++i) {
                        Kokkos::View<double*, Kokkos::LayoutRight, Kokkos::HostSpace> const
                            rhs("metric_inversion_rhs", n);
                        for (std::size_t j=0; j<n; ++j) {
                            rhs(j) = 0.;
                        }
                        rhs(i) = 1.;
                        Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> const
                            tmp("metric_inversion_tmp", n, n+4);
                        const int err = KokkosBatched::SerialGesv<KokkosBatched::Gesv::NoPivoting>::invoke(metric_view, Kokkos::subview(sol, i, Kokkos::ALL), rhs, tmp);
                        assert(err==0 && "Kokkos-kernels failed at inverting metric tensor");
                    }
                    */

                    ddc::annotated_for_each(
                            tensor::swap_character_t<ddc::DiscreteDomain<MetricIndex>>(
                                    inv_metric.domain()),
                            [&](tensor::swap_character_t<ddc::DiscreteElement<MetricIndex>>
                                        mem_index) {
                                inv_metric.mem(elem, mem_index) = buffer(
                                        elem,
                                        tensor::relabelize_indices_in<
                                                tensor::swap_character_t<ddc::detail::TypeSeq<
                                                        tensor::metric_index_1<MetricIndex>,
                                                        tensor::metric_index_2<MetricIndex>>>,
                                                ddc::detail::TypeSeq<
                                                        tensor::metric_index_1<MetricIndex>,
                                                        tensor::metric_index_2<MetricIndex>>>(
                                                inv_metric.accessor().canonical_natural_element(
                                                        mem_index)));
                            });
                });
    }

    return inv_metric;
}

} // namespace tensor

} // namespace sil
