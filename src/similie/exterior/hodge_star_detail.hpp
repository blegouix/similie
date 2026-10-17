// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

#include <array>
#include <stdexcept>

#include <ddc/ddc.hpp>

#include <similie/exterior/coboundary/coboundary.hpp>
#include <similie/exterior/concepts.hpp>
#include <similie/exterior/reduction_and_reconstruction/reduction_and_reconstruction.hpp>
#include <similie/exterior/stencil.hpp>
#include <similie/exterior/volume.hpp>
#include <similie/misc/factorial.hpp>
#include <similie/misc/macros.hpp>
#include <similie/misc/small_matrix.hpp>
#include <similie/misc/specialization.hpp>
#include <similie/misc/type_seq_conversion.hpp>
#include <similie/tensor/metric.hpp>
#include <similie/tensor/tensor_indices/antisymmetric_tensor.hpp>
#include <similie/tensor/tensor_indices/character.hpp>
#include <similie/tensor/tensor_indices/full_tensor.hpp>

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
            = misc::convert_type_seq_to_t<tensor::TensorAntisymmetricIndex, Indices>::mem_size();
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


} // namespace exterior
} // namespace sil
