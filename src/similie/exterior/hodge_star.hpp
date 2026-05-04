// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: MIT

#pragma once

#include <array>

#include <ddc/ddc.hpp>

#include <similie/misc/factorial.hpp>
#include <similie/misc/macros.hpp>
#include <similie/misc/specialization.hpp>
#include <similie/misc/type_seq_conversion.hpp>
#include <similie/tensor/antisymmetric_tensor.hpp>
#include <similie/tensor/character.hpp>
#include <similie/tensor/full_tensor.hpp>

#include "volume.hpp"

namespace sil {

namespace exterior {

template <
        misc::Specialization<ddc::detail::TypeSeq> Indices1,
        misc::Specialization<ddc::detail::TypeSeq> Indices2>
using hodge_star_domain_t
        = ddc::detail::convert_type_seq_to_discrete_domain_t<ddc::type_seq_merge_t<
                ddc::detail::TypeSeq<
                        misc::convert_type_seq_to_t<tensor::TensorFullIndex, Indices1>>,
                std::conditional_t<
                        (ddc::type_seq_size_v<Indices2> == 0),
                        ddc::detail::TypeSeq<>,
                        ddc::detail::TypeSeq<misc::convert_type_seq_to_t<
                                tensor::TensorAntisymmetricIndex,
                                Indices2>>>>>;

namespace detail {

template <class Indices1, class Indices2>
struct AmbientDimension;

template <class HeadIndex, class... TailIndex, class Indices2>
struct AmbientDimension<ddc::detail::TypeSeq<HeadIndex, TailIndex...>, Indices2>
{
    static constexpr std::size_t value = HeadIndex::size();
};

template <class HeadIndex, class... TailIndex>
struct AmbientDimension<ddc::detail::TypeSeq<>, ddc::detail::TypeSeq<HeadIndex, TailIndex...>>
{
    static constexpr std::size_t value = HeadIndex::size();
};

template <class TypeSeq>
struct ExtractIds;

template <class... Indices>
struct ExtractIds<ddc::detail::TypeSeq<Indices...>>
{
    template <class NaturalElem>
    KOKKOS_FUNCTION static std::array<std::size_t, sizeof...(Indices)> run(NaturalElem natural_elem)
    {
        return std::array<std::size_t, sizeof...(Indices)> {
                static_cast<std::size_t>(natural_elem.template uid<Indices>())...};
    }
};

template <std::size_t N, std::size_t M>
KOKKOS_FUNCTION bool has_unique_ids(std::array<std::size_t, M> const& ids)
{
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
    return true;
}

template <std::size_t N, std::size_t M1, std::size_t M2>
KOKKOS_FUNCTION bool is_complete_permutation(
        std::array<std::size_t, M1> const& source_ids,
        std::array<std::size_t, M2> const& target_ids)
{
    std::array<int, N> counts {};
    for (std::size_t id : source_ids) {
        if (id >= N) {
            return false;
        }
        counts[id]++;
    }
    for (std::size_t id : target_ids) {
        if (id >= N) {
            return false;
        }
        counts[id]++;
    }
    for (int count : counts) {
        if (count != 1) {
            return false;
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
    for (std::size_t i = 0; i < M2; ++i) {
        permutation[M1 + i] = target_ids[i];
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
KOKKOS_FUNCTION std::array<bool, N> active_dimensions(std::array<std::size_t, M> const& ids)
{
    std::array<bool, N> active_dims {};
    for (std::size_t id : ids) {
        active_dims[id] = true;
    }
    return active_dims;
}

template <
        DualStrategy Strategy,
        class Indices1,
        class Indices2,
        class MetricType,
        class PositionType,
        class BatchElem>
KOKKOS_FUNCTION double hodge_star_entry(
        MetricType metric,
        PositionType position,
        BatchElem elem,
        auto natural_elem)
{
    constexpr std::size_t N = AmbientDimension<Indices1, Indices2>::value;
    std::array const source_ids = ExtractIds<Indices1>::run(natural_elem);
    std::array const target_ids = ExtractIds<Indices2>::run(natural_elem);

    if (!has_unique_ids<N>(source_ids) || !has_unique_ids<N>(target_ids)
        || !is_complete_permutation<N>(source_ids, target_ids)) {
        return 0.;
    }

    std::array<bool, N> const active_dims = active_dimensions<N>(source_ids);
    double const primal_volume = simplex_volume<N>(metric, position, elem, active_dims);
    if (primal_volume == 0.) {
        return 0.;
    }

    return static_cast<double>(permutation_sign<N>(source_ids, target_ids))
           * dual_simplex_volume<Strategy, N>(metric, position, elem, active_dims)
           / (primal_volume * misc::factorial(ddc::type_seq_size_v<Indices1>));
}

} // namespace detail

template <
        misc::Specialization<ddc::detail::TypeSeq> Indices1,
        misc::Specialization<ddc::detail::TypeSeq> Indices2,
        DualStrategy Strategy = DualStrategy::Circumcentric,
        misc::Specialization<tensor::Tensor> HodgeStarType,
        misc::Specialization<tensor::Tensor> MetricType,
        misc::Specialization<tensor::Tensor> PositionType,
        class ExecSpace>
HodgeStarType fill_hodge_star(
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
            hodge_star.domain(),
            KOKKOS_LAMBDA(
                    typename HodgeStarType::discrete_domain_type::discrete_element_type elem) {
                hodge_star.mem(elem) = detail::hodge_star_entry<Strategy, Indices1, Indices2>(
                        metric,
                        position,
                        typename HodgeStarType::non_indices_domain_t::discrete_element_type(elem),
                        hodge_star.canonical_natural_element(elem));
            });
    return hodge_star;
}

} // namespace exterior

} // namespace sil
