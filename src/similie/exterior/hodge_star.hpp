// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: MIT

#pragma once

#include <ddc/ddc.hpp>

#include <similie/misc/factorial.hpp>
#include <similie/misc/macros.hpp>
#include <similie/misc/specialization.hpp>
#include <similie/misc/type_seq_conversion.hpp>
#include <similie/tensor/antisymmetric_tensor.hpp>
#include <similie/tensor/character.hpp>
#include <similie/tensor/determinant.hpp>
#include <similie/tensor/full_tensor.hpp>
#include <similie/tensor/levi_civita_tensor.hpp>
#include <similie/tensor/metric.hpp>
#include <similie/tensor/prime.hpp>

namespace sil {

namespace exterior {

namespace detail {

template <misc::Specialization<ddc::detail::TypeSeq> Indices2, class Component>
KOKKOS_FUNCTION bool is_stored_hodge_component(Component const& component)
{
    if constexpr (ddc::type_seq_size_v<Indices2> == 0) {
        return true;
    } else {
        using dual_index_type
                = misc::convert_type_seq_to_t<tensor::TensorAntisymmetricIndex, Indices2>;
        if constexpr (dual_index_type::rank() > 1) {
            return component.template uid<dual_index_type>() != 0;
        } else {
            return true;
        }
    }
}

template <class DDim, class Elem, class Domain>
KOKKOS_FUNCTION double primal_mesh_measure(Elem const& elem, Domain const&)
{
    return static_cast<double>(ddc::distance_at_right(ddc::DiscreteElement<DDim>(elem)));
}

template <class DDim, class Elem, class Domain>
KOKKOS_FUNCTION double centered_dual_mesh_measure(Elem const& elem, Domain const& domain)
{
    ddc::DiscreteElement<DDim> const ddim_elem(elem);
    double const left = static_cast<double>(ddc::distance_at_left(ddim_elem));
    double const right = static_cast<double>(ddc::distance_at_right(ddim_elem));
    bool const is_front = ddim_elem == ddc::DiscreteElement<DDim>(domain.front());
    bool const is_back = ddim_elem == ddc::DiscreteElement<DDim>(domain.back());

    if (is_front && is_back) {
        return 0.5 * (left + right);
    }
    if (is_front) {
        return 0.5 * right;
    }
    if (is_back) {
        return 0.5 * left;
    }
    return 0.5 * (left + right);
}

template <bool DualMeasure, class DDim, class Elem, class Domain>
KOKKOS_FUNCTION double mesh_measure(Elem const& elem, Domain const& domain)
{
    if constexpr (DualMeasure) {
        return centered_dual_mesh_measure<DDim>(elem, domain);
    } else {
        return primal_mesh_measure<DDim>(elem, domain);
    }
}

template <bool DualMeasure, class NaturalIndex, class DDimSeq, class Elem, class Domain>
struct measure_from_natural_id;

template <bool DualMeasure, class NaturalIndex, class Elem, class Domain, class... DDim>
struct measure_from_natural_id<
        DualMeasure,
        NaturalIndex,
        ddc::detail::TypeSeq<DDim...>,
        Elem,
        Domain>
{
    KOKKOS_FUNCTION static double run(
            Elem const& elem,
            Domain const& domain,
            std::size_t const natural_id)
    {
        if constexpr (NaturalIndex::rank() == 0) {
            return 1.;
        } else {
            double measure = 1.;
            bool matched = false;
            (
                    [&] {
                        if constexpr (ddc::type_seq_contains_v<
                                              ddc::detail::TypeSeq<
                                                      typename DDim::continuous_dimension_type>,
                                              typename NaturalIndex::type_seq_dimensions>) {
                            if (natural_id
                                == NaturalIndex::template mem_id<
                                        typename DDim::continuous_dimension_type>()) {
                                measure = mesh_measure<DualMeasure, DDim>(elem, domain);
                                matched = true;
                            }
                        }
                    }(),
                    ...);
            return matched ? measure : 1.;
        }
    }
};

template <bool DualMeasure, class Index, class DDimSeq, class Elem, class Domain, class Component>
struct measure_from_index_component;

template <
        bool DualMeasure,
        class... SubIndex,
        class DDimSeq,
        class Elem,
        class Domain,
        class Component>
struct measure_from_index_component<
        DualMeasure,
        tensor::TensorFullIndex<SubIndex...>,
        DDimSeq,
        Elem,
        Domain,
        Component>
{
    KOKKOS_FUNCTION static double run(
            Elem const& elem,
            Domain const& domain,
            Component const& component)
    {
        using index_type = tensor::TensorFullIndex<SubIndex...>;
        constexpr std::size_t n_subindices = sizeof...(SubIndex);
        std::array<std::size_t, n_subindices> const natural_ids
                = index_type::mem_id_to_canonical_natural_ids(
                        index_type::access_id_to_mem_id(component.template uid<index_type>()));

        return (measure_from_natural_id<DualMeasure, SubIndex, DDimSeq, Elem, Domain>::
                        run(elem,
                            domain,
                            natural_ids[ddc::type_seq_rank_v<
                                    SubIndex,
                                    ddc::detail::TypeSeq<SubIndex...>>])
                * ...);
    }
};

template <bool DualMeasure, class DDimSeq, class Elem, class Domain, class Component>
struct measure_from_index_component<
        DualMeasure,
        tensor::TensorFullIndex<>,
        DDimSeq,
        Elem,
        Domain,
        Component>
{
    KOKKOS_FUNCTION static double run(Elem const&, Domain const&, Component const&)
    {
        return 1.;
    }
};

template <
        bool DualMeasure,
        class... SubIndex,
        class DDimSeq,
        class Elem,
        class Domain,
        class Component>
struct measure_from_index_component<
        DualMeasure,
        tensor::TensorAntisymmetricIndex<SubIndex...>,
        DDimSeq,
        Elem,
        Domain,
        Component>
{
    KOKKOS_FUNCTION static double run(
            Elem const& elem,
            Domain const& domain,
            Component const& component)
    {
        using index_type = tensor::TensorAntisymmetricIndex<SubIndex...>;
        if constexpr (index_type::rank() > 1) {
            if (component.template uid<index_type>() == 0) {
                return 1.;
            }
        }

        constexpr std::size_t n_subindices = sizeof...(SubIndex);
        std::array<std::size_t, n_subindices> const natural_ids
                = index_type::mem_id_to_canonical_natural_ids(
                        index_type::access_id_to_mem_id(component.template uid<index_type>()));

        return (measure_from_natural_id<DualMeasure, SubIndex, DDimSeq, Elem, Domain>::
                        run(elem,
                            domain,
                            natural_ids[ddc::type_seq_rank_v<
                                    SubIndex,
                                    ddc::detail::TypeSeq<SubIndex...>>])
                * ...);
    }
};

template <
        misc::Specialization<ddc::detail::TypeSeq> Indices1,
        misc::Specialization<ddc::detail::TypeSeq> Indices2,
        class NonIndicesDomain,
        class Elem,
        class Component>
KOKKOS_FUNCTION double discrete_hodge_measure_ratio(
        NonIndicesDomain const& domain,
        Elem const& elem,
        Component const& component)
{
    using ddim_seq = ddc::to_type_seq_t<NonIndicesDomain>;
    using primal_index_type = misc::convert_type_seq_to_t<tensor::TensorFullIndex, Indices1>;
    double const primal_measure = measure_from_index_component<
            false,
            primal_index_type,
            ddim_seq,
            Elem,
            NonIndicesDomain,
            Component>::run(elem, domain, component);

    if constexpr (ddc::type_seq_size_v<Indices2> == 0) {
        return 1. / primal_measure;
    } else {
        using dual_index_type
                = misc::convert_type_seq_to_t<tensor::TensorAntisymmetricIndex, Indices2>;
        double const dual_measure = measure_from_index_component<
                true,
                dual_index_type,
                ddim_seq,
                Elem,
                NonIndicesDomain,
                Component>::run(elem, domain, component);
        return dual_measure / primal_measure;
    }
}

} // namespace detail

template <
        misc::Specialization<ddc::detail::TypeSeq> Indices1,
        misc::Specialization<ddc::detail::TypeSeq> Indices2>
using hodge_star_domain_t
        = ddc::detail::convert_type_seq_to_discrete_domain_t<ddc::type_seq_merge_t<
                ddc::detail::TypeSeq<misc::convert_type_seq_to_t<
                        tensor::TensorFullIndex,
                        Indices1>>, // TODO clarify Antisymmetric
                std::conditional_t<
                        (ddc::type_seq_size_v<Indices2> == 0),
                        ddc::detail::TypeSeq<>,
                        ddc::detail::TypeSeq<misc::convert_type_seq_to_t<
                                tensor::TensorAntisymmetricIndex,
                                Indices2>>>>>;

template <
        misc::Specialization<ddc::detail::TypeSeq> Indices1,
        misc::Specialization<ddc::detail::TypeSeq> Indices2,
        misc::Specialization<tensor::Tensor> HodgeStarType,
        class MetricDeterminantType,
        misc::Specialization<tensor::Tensor> MetricProdType,
        class ExecSpace>
    requires(
            misc::Specialization<MetricDeterminantType, ddc::ChunkSpan>
            || misc::Specialization<MetricDeterminantType, tensor::Tensor>)
HodgeStarType fill_hodge_star(
        ExecSpace const& exec_space,
        HodgeStarType hodge_star,
        MetricDeterminantType metric_determinant,
        MetricProdType metric_prod)
{
    sil::tensor::TensorAccessor<misc::convert_type_seq_to_t<
            tensor::TensorLeviCivitaIndex,
            ddc::type_seq_merge_t<tensor::primes<tensor::lower_t<Indices1>>, Indices2>>>
            levi_civita_accessor;
    ddc::DiscreteDomain<misc::convert_type_seq_to_t<
            tensor::TensorLeviCivitaIndex,
            ddc::type_seq_merge_t<tensor::primes<tensor::lower_t<Indices1>>, Indices2>>>
            levi_civita_dom(levi_civita_accessor.domain());
    ddc::Chunk levi_civita_alloc(
            levi_civita_dom,
            ddc::KokkosAllocator<
                    double,
                    typename ExecSpace::memory_space>()); // TODO consider avoid allocation
    sil::tensor::Tensor levi_civita(levi_civita_alloc);

    SIMILIE_DEBUG_LOG("similie_compute_hodge_star");
    ddc::parallel_for_each(
            "similie_compute_hodge_star",
            exec_space,
            hodge_star.non_indices_domain(),
            KOKKOS_LAMBDA(
                    typename HodgeStarType::non_indices_domain_t::discrete_element_type elem) {
                tensor_prod(hodge_star[elem], metric_prod[elem], levi_civita);
                hodge_star[elem] *= Kokkos::sqrt(Kokkos::abs(metric_determinant(elem)))
                                    / misc::factorial(ddc::type_seq_size_v<Indices1>);
                ddc::device_for_each(hodge_star[elem].domain(), [&](auto index_elem) {
                    if (detail::is_stored_hodge_component<Indices2>(index_elem)) {
                        hodge_star[elem](index_elem) *= detail::discrete_hodge_measure_ratio<
                                Indices1,
                                Indices2>(hodge_star.non_indices_domain(), elem, index_elem);
                    }
                });
            });
    return hodge_star;
}

template <
        tensor::TensorIndex MetricIndex,
        misc::Specialization<ddc::detail::TypeSeq> Indices1,
        misc::Specialization<ddc::detail::TypeSeq> Indices2,
        misc::Specialization<tensor::Tensor> HodgeStarType,
        misc::Specialization<tensor::Tensor> MetricType,
        class ExecSpace>
HodgeStarType fill_hodge_star(
        ExecSpace const& exec_space,
        HodgeStarType hodge_star,
        MetricType metric)
{
    static_assert(tensor::are_contravariant_v<
                  ddc::to_type_seq_t<typename MetricIndex::subindices_domain_t>>);
    static_assert(tensor::are_contravariant_v<
                  ddc::to_type_seq_t<typename MetricType::accessor_t::natural_domain_t>>);
    static_assert(tensor::are_contravariant_v<Indices1>);
    static_assert(tensor::are_covariant_v<Indices2>);

    // Allocate metric_det to receive metric field determinant values
    ddc::Chunk metric_det_alloc(
            metric.non_indices_domain(),
            ddc::KokkosAllocator<double, typename ExecSpace::memory_space>());
    ddc::ChunkSpan metric_det(metric_det_alloc);
    // Allocate a buffer mirroring the metric as a full matrix, it will be overwritten by tensor::determinant() which involves a LU decomposition
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
    // Compute determinants
    SIMILIE_DEBUG_LOG("similie_compute_metric_determinant");
    ddc::parallel_for_each(
            "similie_compute_metric_determinant",
            exec_space,
            metric.non_indices_domain(),
            KOKKOS_LAMBDA(typename MetricType::non_indices_domain_t::discrete_element_type elem) {
                ddc::device_for_each(
                        ddc::DiscreteDomain<
                                tensor::metric_index_1<MetricIndex>,
                                tensor::metric_index_2<MetricIndex>>(metric.natural_domain()),
                        [&](ddc::DiscreteElement<
                                tensor::metric_index_1<MetricIndex>,
                                tensor::metric_index_2<MetricIndex>> index) {
                            buffer(elem, index) = metric.get(metric.access_element(
                                    elem,
                                    index)); // TODO: triggers a "nvlink warning : Stack size for entry function cannot be statically determined"
                        });
                metric_det(elem) = 1. / tensor::determinant(buffer[elem].allocation_kokkos_view());
            });

    // Allocate & compute the product of metrics
    tensor::tensor_accessor_for_domain_t<
            tensor::metric_prod_domain_t<MetricIndex, Indices1, tensor::primes<Indices1>>>
            metric_prod_accessor;
    ddc::Chunk metric_prod_alloc(
            ddc::cartesian_prod_t<
                    typename MetricType::non_indices_domain_t,
                    tensor::metric_prod_domain_t<MetricIndex, Indices1, tensor::primes<Indices1>>>(
                    metric.non_indices_domain(),
                    metric_prod_accessor.domain()),
            ddc::KokkosAllocator<double, typename ExecSpace::memory_space>());
    tensor::Tensor metric_prod(metric_prod_alloc);

    fill_metric_prod<
            MetricIndex,
            Indices1,
            tensor::primes<Indices1>>(exec_space, metric_prod, metric);

    // Compute Hodge star
    return fill_hodge_star<Indices1, Indices2>(exec_space, hodge_star, metric_det, metric_prod);
}

} // namespace exterior

} // namespace sil
