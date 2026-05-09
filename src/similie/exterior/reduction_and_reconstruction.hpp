// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: MIT

#pragma once

#include <array>

#include <ddc/ddc.hpp>

#include <similie/misc/clamp_to_domain.hpp>
#include <similie/misc/factorial.hpp>
#include <similie/misc/specialization.hpp>
#include <similie/misc/type_seq_conversion.hpp>
#include <similie/tensor/antisymmetric_tensor.hpp>
#include <similie/tensor/character.hpp>
#include <similie/tensor/full_tensor.hpp>
#include <similie/tensor/prime.hpp>
#include <similie/tensor/tensor_impl.hpp>

#include "volume.hpp"


namespace sil {

namespace exterior {

namespace detail {

template <misc::Specialization<ddc::detail::TypeSeq> Indices>
struct ReductionTargetIndex;

template <>
struct ReductionTargetIndex<ddc::detail::TypeSeq<>>
{
    using type = tensor::Covariant<tensor::ScalarIndex>;
};

template <tensor::TensorNatIndex Index>
struct ReductionTargetIndex<ddc::detail::TypeSeq<Index>>
{
    using type = Index;
};

template <tensor::TensorNatIndex... Index>
    requires(sizeof...(Index) > 1)
struct ReductionTargetIndex<ddc::detail::TypeSeq<Index...>>
{
    using type = tensor::TensorAntisymmetricIndex<Index...>;
};

template <std::size_t N>
KOKKOS_FUNCTION bool has_unique_reduction_ids(std::array<std::size_t, N> const& ids)
{
    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = i + 1; j < N; ++j) {
            if (ids[i] == ids[j]) {
                return false;
            }
        }
    }
    return true;
}

template <std::size_t N>
KOKKOS_FUNCTION bool have_same_reduction_ids(
        std::array<std::size_t, N> const& lhs,
        std::array<std::size_t, N> const& rhs)
{
    for (std::size_t i = 0; i < N; ++i) {
        if (lhs[i] != rhs[i]) {
            return false;
        }
    }
    return true;
}

template <std::size_t K>
KOKKOS_FUNCTION double reduction_determinant(std::array<double, K * K> const& matrix)
{
    if constexpr (K == 0) {
        return 1.;
    } else if constexpr (K == 1) {
        return matrix[0];
    } else {
        double det = 0.;
        for (std::size_t col = 0; col < K; ++col) {
            std::array<double, (K - 1) * (K - 1)> minor {};
            for (std::size_t row = 1; row < K; ++row) {
                std::size_t minor_col = 0;
                for (std::size_t source_col = 0; source_col < K; ++source_col) {
                    if (source_col == col) {
                        continue;
                    }
                    minor[(row - 1) * (K - 1) + minor_col] = matrix[row * K + source_col];
                    ++minor_col;
                }
            }
            double const sign = (col % 2 == 0) ? 1. : -1.;
            det += sign * matrix[col] * reduction_determinant<K - 1>(minor);
        }
        return det;
    }
}

template <class ReductionNaturalElemType, std::size_t N1, std::size_t N2>
KOKKOS_FUNCTION ReductionNaturalElemType merge_reduction_natural_elems(
        std::array<std::size_t, N1> const& first_ids,
        std::array<std::size_t, N2> const& second_ids)
{
    ReductionNaturalElemType reduction_natural_elem;
    auto reduction_ids = ddc::detail::array(reduction_natural_elem);
    for (std::size_t i = 0; i < N1; ++i) {
        reduction_ids[i] = first_ids[i];
    }
    for (std::size_t i = 0; i < N2; ++i) {
        reduction_ids[N1 + i] = second_ids[i];
    }
    ddc::detail::array(reduction_natural_elem) = reduction_ids;
    return reduction_natural_elem;
}

} // namespace detail

template <misc::Specialization<ddc::detail::TypeSeq> Indices>
using reduction_index_t = typename detail::ReductionTargetIndex<Indices>::type;

template <misc::Specialization<ddc::detail::TypeSeq> Indices>
using reduction_domain_t = ddc::detail::convert_type_seq_to_discrete_domain_t<ddc::type_seq_merge_t<
        ddc::detail::TypeSeq<misc::convert_type_seq_to_t<
                tensor::TensorFullIndex,
                tensor::primes<tensor::upper_t<Indices>>>>,
        ddc::detail::TypeSeq<reduction_index_t<Indices>>>>;

template <misc::Specialization<ddc::detail::TypeSeq> Indices>
using reconstruction_domain_t
        = ddc::detail::convert_type_seq_to_discrete_domain_t<ddc::type_seq_merge_t<
                ddc::detail::TypeSeq<reduction_index_t<Indices>>,
                ddc::detail::TypeSeq<reduction_index_t<tensor::primes<Indices>>>>>;

template <
        misc::Specialization<ddc::detail::TypeSeq> Indices,
        misc::Specialization<tensor::Tensor> PositionType,
        class BatchElem>
struct Reduction
{
    using source_index_type = misc::convert_type_seq_to_t<
            tensor::TensorFullIndex,
            tensor::primes<tensor::upper_t<Indices>>>;
    using target_index_type = reduction_index_t<Indices>;
    using source_natural_elem_type =
            typename tensor::natural_domain_t<source_index_type>::discrete_element_type;
    using target_natural_elem_type =
            typename tensor::natural_domain_t<target_index_type>::discrete_element_type;
    using position_index_type = ddc::
            type_seq_element_t<0, ddc::to_type_seq_t<typename PositionType::indices_domain_t>>;

    template <
            misc::Specialization<tensor::Tensor> ReductionTensorType,
            misc::Specialization<tensor::Tensor> FormTensorType>
    KOKKOS_FUNCTION static void run(
            ReductionTensorType reduced_tensor,
            FormTensorType form_tensor,
            PositionType position,
            BatchElem elem)
    {
        using reduction_accessor_t
                = sil::tensor::tensor_accessor_for_domain_t<reduction_domain_t<Indices>>;
        using reduction_natural_elem_type =
                typename reduction_accessor_t::natural_domain_t::discrete_element_type;
        using form_natural_elem_type =
                typename FormTensorType::accessor_t::natural_domain_t::discrete_element_type;
        if constexpr (target_index_type::rank() == 0) {
            reduced_tensor.mem(ddc::DiscreteElement<target_index_type>(0))
                    = form_tensor.get(form_tensor.access_element(form_natural_elem_type()));
        } else {
            [[maybe_unused]] sil::tensor::TensorAccessor<source_index_type> source_accessor;
            ddc::device_for_each(reduced_tensor.domain(), [&](auto target_mem_elem) {
                auto const target_natural_elem
                        = reduced_tensor.canonical_natural_element(target_mem_elem);
                double reduced_value = 0.;
                ddc::device_for_each(
                        source_accessor.natural_domain(),
                        [&](auto source_natural_elem) {
                            reduction_natural_elem_type reduction_natural_elem;
                            auto const source_ids = ddc::detail::array(source_natural_elem);
                            auto const target_ids = ddc::detail::array(target_natural_elem);
                            reduction_natural_elem = detail::merge_reduction_natural_elems<
                                    reduction_natural_elem_type>(source_ids, target_ids);

                            form_natural_elem_type form_natural_elem;
                            ddc::detail::array(form_natural_elem)
                                    = ddc::detail::array(source_natural_elem);

                            reduced_value += value(position, elem, reduction_natural_elem)
                                             * form_tensor.get(
                                                     form_tensor.access_element(form_natural_elem));
                        });
                reduced_tensor.mem(target_mem_elem) = reduced_value;
            });
        }
    }

    KOKKOS_FUNCTION static double value(PositionType position, BatchElem elem, auto natural_elem)
    {
        constexpr std::size_t K = target_index_type::rank();
        if constexpr (K == 0) {
            return 1.;
        } else {
            std::array const source_ids
                    = ddc::detail::array(source_natural_elem_type(natural_elem));
            std::array const target_ids
                    = ddc::detail::array(target_natural_elem_type(natural_elem));
            if (!detail::has_unique_reduction_ids<K>(source_ids)
                || !detail::has_unique_reduction_ids<K>(target_ids)) {
                return 0.;
            }

            std::array<double, K * K> jacobian_matrix {};
            for (std::size_t row = 0; row < K; ++row) {
                for (std::size_t col = 0; col < K; ++col) {
                    auto shifted_elem = elem;
                    ddc::detail::array(shifted_elem)[target_ids[col]]++;
                    shifted_elem
                            = misc::clamp_to_domain(position.non_indices_domain(), shifted_elem);
                    auto const position_component = position.accessor().canonical_natural_element(
                            ddc::DiscreteElement<position_index_type>(source_ids[row]));
                    jacobian_matrix[row * K + col] = position.mem(shifted_elem, position_component)
                                                     - position.mem(elem, position_component);
                }
            }
            return detail::reduction_determinant<K>(jacobian_matrix) / misc::factorial(K);
        }
    }
};

template <
        misc::Specialization<ddc::detail::TypeSeq> Indices,
        misc::Specialization<tensor::Tensor> ReductionTensorType,
        misc::Specialization<tensor::Tensor> PositionType,
        class ExecSpace>
ReductionTensorType fill_reduction_operator(
        ExecSpace const& exec_space,
        ReductionTensorType reduction_tensor,
        PositionType position)
{
    SIMILIE_DEBUG_LOG("similie_compute_reduction_operator");
    ddc::parallel_for_each(
            "similie_compute_reduction_operator",
            exec_space,
            reduction_tensor.non_indices_domain(),
            KOKKOS_LAMBDA(
                    typename ReductionTensorType::non_indices_domain_t::discrete_element_type
                            elem) {
                ddc::device_for_each(reduction_tensor.accessor().domain(), [&](auto mem_elem) {
                    reduction_tensor.mem(
                            typename ReductionTensorType::discrete_element_type(elem, mem_elem))
                            = Reduction<
                                    Indices,
                                    PositionType,
                                    typename ReductionTensorType::non_indices_domain_t::
                                            discrete_element_type>::
                                    value(position,
                                          elem,
                                          reduction_tensor.accessor().canonical_natural_element(
                                                  mem_elem));
                });
            });
    return reduction_tensor;
}

template <
        misc::Specialization<ddc::detail::TypeSeq> Indices,
        misc::Specialization<tensor::Tensor> PositionType,
        class BatchElem>
struct Reconstruction
{
    using source_index_type = reduction_index_t<Indices>;
    using target_index_type = reduction_index_t<tensor::primes<Indices>>;
    using source_natural_elem_type =
            typename tensor::natural_domain_t<source_index_type>::discrete_element_type;
    using target_natural_elem_type =
            typename tensor::natural_domain_t<target_index_type>::discrete_element_type;

    template <
            misc::Specialization<tensor::Tensor> ReconstructedTensorType,
            misc::Specialization<tensor::Tensor> CochainTensorType>
    KOKKOS_FUNCTION static void run(
            ReconstructedTensorType reconstructed_tensor,
            CochainTensorType cochain_tensor,
            PositionType position,
            BatchElem elem)
    {
        using reconstruction_accessor_t
                = sil::tensor::tensor_accessor_for_domain_t<reconstruction_domain_t<Indices>>;
        using reconstruction_natural_elem_type =
                typename reconstruction_accessor_t::natural_domain_t::discrete_element_type;
        using cochain_natural_elem_type =
                typename CochainTensorType::accessor_t::natural_domain_t::discrete_element_type;
        if constexpr (source_index_type::rank() == 0) {
            reconstructed_tensor.mem(ddc::DiscreteElement<source_index_type>(0))
                    = cochain_tensor.get(
                            cochain_tensor.access_element(cochain_natural_elem_type()));
        } else {
            [[maybe_unused]] sil::tensor::TensorAccessor<source_index_type> source_accessor;
            ddc::device_for_each(reconstructed_tensor.domain(), [&](auto target_mem_elem) {
                auto const target_natural_elem
                        = reconstructed_tensor.canonical_natural_element(target_mem_elem);
                double reconstructed_value = 0.;
                ddc::device_for_each(
                        source_accessor.natural_domain(),
                        [&](auto source_natural_elem) {
                            reconstruction_natural_elem_type reconstruction_natural_elem;
                            reconstruction_natural_elem = detail::merge_reduction_natural_elems<
                                    reconstruction_natural_elem_type>(
                                    ddc::detail::array(source_natural_elem),
                                    ddc::detail::array(target_natural_elem));

                            cochain_natural_elem_type cochain_natural_elem;
                            ddc::detail::array(cochain_natural_elem)
                                    = ddc::detail::array(source_natural_elem);

                            reconstructed_value
                                    += value(position, elem, reconstruction_natural_elem)
                                       * cochain_tensor.get(
                                               cochain_tensor.access_element(cochain_natural_elem));
                        });
                reconstructed_tensor.mem(target_mem_elem) = reconstructed_value;
            });
        }
    }

    KOKKOS_FUNCTION static double value(PositionType position, BatchElem elem, auto natural_elem)
    {
        constexpr std::size_t K = source_index_type::rank();
        if constexpr (K == 0) {
            return 1.;
        } else {
            std::array const source_ids
                    = ddc::detail::array(source_natural_elem_type(natural_elem));
            std::array const target_ids
                    = ddc::detail::array(target_natural_elem_type(natural_elem));
            using reduction_accessor_t
                    = sil::tensor::tensor_accessor_for_domain_t<reduction_domain_t<Indices>>;
            using reduction_natural_elem_type =
                    typename reduction_accessor_t::natural_domain_t::discrete_element_type;
            reduction_natural_elem_type reduction_natural_elem
                    = detail::merge_reduction_natural_elems<
                            reduction_natural_elem_type>(source_ids, target_ids);
            double const reduction_value = Reduction<Indices, PositionType, BatchElem>::
                    value(position, elem, reduction_natural_elem);

            if (!detail::have_same_reduction_ids<K>(source_ids, target_ids)) {
                assert(Kokkos::abs(reduction_value) < 1e-14
                       && "Reconstruction assumes a diagonal local reduction operator.");
                return 0.;
            }

            if (Kokkos::abs(reduction_value) < 1e-14) {
                return 0.;
            }
            return 1. / (misc::factorial(K) * reduction_value);
        }
    }
};

template <
        misc::Specialization<ddc::detail::TypeSeq> Indices,
        misc::Specialization<tensor::Tensor> ReconstructionTensorType,
        misc::Specialization<tensor::Tensor> PositionType,
        class ExecSpace>
ReconstructionTensorType fill_reconstruction_operator(
        ExecSpace const& exec_space,
        ReconstructionTensorType reconstruction_tensor,
        PositionType position)
{
    SIMILIE_DEBUG_LOG("similie_compute_reconstruction_operator");
    ddc::parallel_for_each(
            "similie_compute_reconstruction_operator",
            exec_space,
            reconstruction_tensor.non_indices_domain(),
            KOKKOS_LAMBDA(
                    typename ReconstructionTensorType::non_indices_domain_t::discrete_element_type
                            elem) {
                ddc::device_for_each(reconstruction_tensor.accessor().domain(), [&](auto mem_elem) {
                    reconstruction_tensor.mem(
                            typename ReconstructionTensorType::
                                    discrete_element_type(elem, mem_elem))
                            = Reconstruction<
                                    Indices,
                                    PositionType,
                                    typename ReconstructionTensorType::non_indices_domain_t::
                                            discrete_element_type>::
                                    value(position,
                                          elem,
                                          reconstruction_tensor.accessor()
                                                  .canonical_natural_element(mem_elem));
                });
            });
    return reconstruction_tensor;
}

} // namespace exterior

} // namespace sil
