// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <ddc/ddc.hpp>

#include "character.hpp"
#include "prime.hpp"

namespace sil {

namespace tensor {

// Abstract indices used to characterize a generic metric

template <class... CDim>
struct MetricIndex1 : TensorNaturalIndex<CDim...>
{
};

template <class... CDim>
struct MetricIndex2 : TensorNaturalIndex<CDim...>
{
};

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
using relabelize_metric_in_domain_t = relabelize_indices_in_domain_t<
        Dom,
        ddc::detail::TypeSeq<
                typename detail::ConvertTypeSeqToMetricIndex1<
                        typename Index1::type_seq_dimensions>::type,
                typename detail::ConvertTypeSeqToMetricIndex2<
                        typename Index2::type_seq_dimensions>::type>,
        ddc::detail::TypeSeq<Index1, Index2>>;

template <TensorNatIndex Index1, TensorNatIndex Index2, class Dom>
relabelize_metric_in_domain_t<Dom, Index1, Index2> relabelize_metric_in_domain(Dom metric_dom)
{
    return relabelize_indices_in_domain<
            ddc::detail::TypeSeq<
                    typename detail::ConvertTypeSeqToMetricIndex1<
                            typename Index1::type_seq_dimensions>::type,
                    typename detail::ConvertTypeSeqToMetricIndex2<
                            typename Index2::type_seq_dimensions>::type>,
            ddc::detail::TypeSeq<Index1, Index2>>(metric_dom);
}

template <misc::Specialization<Tensor> TensorType, TensorNatIndex Index1, TensorNatIndex Index2>
using relabelize_metric_t = relabelize_indices_of_t<
        TensorType,
        ddc::detail::TypeSeq<
                typename detail::ConvertTypeSeqToMetricIndex1<
                        typename Index1::type_seq_dimensions>::type,
                typename detail::ConvertTypeSeqToMetricIndex2<
                        typename Index2::type_seq_dimensions>::type>,
        ddc::detail::TypeSeq<Index1, Index2>>;

template <TensorNatIndex Index1, TensorNatIndex Index2, misc::Specialization<Tensor> TensorType>
relabelize_metric_t<TensorType, Index1, Index2> relabelize_metric(TensorType tensor)
{
    return relabelize_indices_of<
            ddc::detail::TypeSeq<
                    typename detail::ConvertTypeSeqToMetricIndex1<
                            typename Index1::type_seq_dimensions>::type,
                    typename detail::ConvertTypeSeqToMetricIndex2<
                            typename Index2::type_seq_dimensions>::type>,
            ddc::detail::TypeSeq<Index1, Index2>>(tensor);
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
template <class NonMetricDom, class MetricIndex, class Indices1, class Indices2>
struct MetricProdType;

template <class NonMetricDom, class MetricIndex, class... Index1, class... Index2>
struct MetricProdType<
        NonMetricDom,
        MetricIndex,
        ddc::detail::TypeSeq<Index1...>,
        ddc::detail::TypeSeq<Index2...>>
{
    using type = tensor::Tensor<
            double,
            ddc::cartesian_prod_t<
                    NonMetricDom,
                    metric_prod_domain_t<
                            MetricIndex,
                            ddc::detail::TypeSeq<Index1...>,
                            ddc::detail::TypeSeq<Index2...>>>,
            Kokkos::layout_right,
            Kokkos::DefaultHostExecutionSpace::memory_space>;
};

} // namespace detail

template <
        misc::Specialization<ddc::DiscreteDomain> NonMetricDom,
        TensorIndex MetricIndex,
        misc::Specialization<ddc::detail::TypeSeq> Indices1,
        misc::Specialization<ddc::detail::TypeSeq> Indices2>
using metric_prod_t =
        typename detail::MetricProdType<NonMetricDom, MetricIndex, Indices1, Indices2>::type;

namespace detail {

template <class Indices1, class Indices2>
struct FillMetricProd;

template <>
struct FillMetricProd<ddc::detail::TypeSeq<>, ddc::detail::TypeSeq<>>
{
    template <class MetricProdType, class MetricType, class MetricProdType_>
    static MetricProdType run(
            MetricProdType metric_prod,
            MetricType metric,
            MetricProdType_ metric_prod_)
    {
        Kokkos::deep_copy(
                metric_prod.allocation_kokkos_view(),
                metric_prod_
                        .allocation_kokkos_view()); // We rely on Kokkos::deep_copy in place of ddc::parallel_deepcopy to avoid type verification of the type dimensions
        return metric_prod;
    }
};

template <class HeadIndex1, class... TailIndex1, class HeadIndex2, class... TailIndex2>
struct FillMetricProd<
        ddc::detail::TypeSeq<HeadIndex1, TailIndex1...>,
        ddc::detail::TypeSeq<HeadIndex2, TailIndex2...>>
{
    template <class MetricProdType, class MetricType, class MetricProdType_>
    static MetricProdType run(
            MetricProdType metric_prod,
            MetricType metric,
            MetricProdType_ metric_prod_)
    {
        ddc::cartesian_prod_t<
                typename MetricProdType_::discrete_domain_type,
                relabelize_metric_in_domain_t<
                        typename MetricType::discrete_domain_type,
                        HeadIndex1,
                        HeadIndex2>>
                new_metric_prod_dom_(
                        metric_prod_.domain(),
                        relabelize_metric_in_domain<HeadIndex1, HeadIndex2>(metric.domain()));
        ddc::Chunk new_metric_prod_alloc_(new_metric_prod_dom_, ddc::HostAllocator<double>());
        tensor::Tensor<
                double,
                ddc::cartesian_prod_t<
                        typename MetricProdType_::discrete_domain_type,
                        relabelize_metric_in_domain_t<
                                typename MetricType::discrete_domain_type,
                                HeadIndex1,
                                HeadIndex2>>,
                Kokkos::layout_right,
                Kokkos::DefaultHostExecutionSpace::memory_space>
                new_metric_prod_(new_metric_prod_alloc_);

        // TODO tensorial prod ? atm supports only Identity or Lorentzian metrics (new_metric_prod is empty)

        return FillMetricProd<
                ddc::detail::TypeSeq<TailIndex1...>,
                ddc::detail::TypeSeq<TailIndex2...>>::run(metric_prod, metric, new_metric_prod_);
    }
};
} // namespace detail

template <
        TensorIndex MetricIndex,
        misc::Specialization<ddc::detail::TypeSeq> Indices1,
        misc::Specialization<ddc::detail::TypeSeq> Indices2,
        misc::Specialization<Tensor> MetricType>
metric_prod_t<typename MetricType::non_indices_domain_t, MetricIndex, Indices1, Indices2>
fill_metric_prod(
        metric_prod_t<typename MetricType::non_indices_domain_t, MetricIndex, Indices1, Indices2>
                metric_prod,
        MetricType metric)
{
    ddc::DiscreteDomain<> dom_;
    ddc::Chunk metric_prod_alloc_(dom_, ddc::HostAllocator<double>());
    tensor::Tensor<
            double,
            ddc::DiscreteDomain<>,
            Kokkos::layout_right,
            Kokkos::DefaultHostExecutionSpace::memory_space>
            metric_prod_(metric_prod_alloc_);

    return detail::FillMetricProd<Indices1, Indices2>::run(metric_prod, metric, metric_prod_);
}

// Apply metrics inplace (like g_mu_muprime*T^muprime^nu)
template <
        TensorIndex MetricIndex,
        misc::Specialization<ddc::detail::TypeSeq> Indices1,
        misc::Specialization<ddc::detail::TypeSeq> Indices2,
        misc::Specialization<Tensor> MetricType,
        misc::Specialization<Tensor> TensorType>
relabelize_indices_of_t<TensorType, Indices2, Indices1> inplace_apply_metric(
        TensorType tensor,
        MetricType metric)
{
    tensor::tensor_accessor_for_domain_t<metric_prod_domain_t<MetricIndex, Indices1, Indices2>>
            metric_prod_accessor;
    ddc::Chunk metric_prod_alloc(
            ddc::cartesian_prod_t<
                    typename TensorType::non_indices_domain_t,
                    metric_prod_domain_t<MetricIndex, Indices1, Indices2>>(
                    tensor.non_indices_domain(),
                    metric_prod_accessor.mem_domain()),
            ddc::HostAllocator<double>());
    tensor::Tensor<
            double,
            ddc::cartesian_prod_t<
                    typename TensorType::non_indices_domain_t,
                    metric_prod_domain_t<MetricIndex, Indices1, Indices2>>,
            Kokkos::layout_right,
            Kokkos::DefaultHostExecutionSpace::memory_space>
            metric_prod(metric_prod_alloc);

    fill_metric_prod<MetricIndex, Indices1, Indices2>(metric_prod, metric);

    ddc::Chunk result_alloc(
            relabelize_indices_in_domain<Indices2, Indices1>(tensor.domain()),
            ddc::HostAllocator<double>());
    relabelize_indices_of_t<TensorType, Indices2, Indices1> result(result_alloc);
    ddc::for_each(tensor.non_indices_domain(), [&](auto elem) {
        std::cout << result[elem];
/*
        tensor_prod(
                result[elem],
                relabelize_indices_of<Indices2, primes<Indices1>>(metric_prod)[elem],
                relabelize_indices_of<Indices2, primes<Indices2>>(tensor)[elem]);
*/
    });
    Kokkos::deep_copy(
            tensor.allocation_kokkos_view(),
            result.allocation_kokkos_view()); // We rely on Kokkos::deep_copy in place of ddc::parallel_deepcopy to avoid type verification of the type dimensions

    return relabelize_indices_of<Indices2, Indices1>(tensor);
}

template <
        TensorIndex MetricIndex,
        TensorIndex Index1,
        TensorIndex Index2,
        misc::Specialization<Tensor> MetricType,
        misc::Specialization<Tensor> TensorType>
relabelize_index_of_t<TensorType, Index2, Index1> inplace_apply_metric(
        TensorType tensor,
        MetricType metric)
{
    return inplace_apply_metric<
            MetricIndex,
            ddc::detail::TypeSeq<Index1>,
            ddc::detail::TypeSeq<Index2>>(tensor, metric);
}

} // namespace tensor

} // namespace sil
