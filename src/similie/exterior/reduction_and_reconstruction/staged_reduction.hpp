// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#pragma once

#include <similie/exterior/reduction_and_reconstruction/reduction_and_reconstruction.hpp>

namespace sil::exterior {
/** Cache reduction in the canonical form-component basis.
 * \important This operator and documentation is fully AI-generated.
 * Coefficients selects TensorFullIndex or TensorDiagonalIndex. The diagonal
 * choice is a geometric contract, checked by assertions only in Debug builds.
 * Rebuild when positions change, or when the metric changes for dual reduction.
 */
template <
        misc::Specialization<ddc::TypeSeq> Indices,
        misc::Specialization<tensor::Tensor> ReducedTensorType,
        misc::Specialization<tensor::Tensor> FormTensorType,
        misc::Specialization<tensor::Tensor> PositionType,
        class ExecSpace,
        CellComplex Complex = CellComplex::Primal,
        template <class...> class Coefficients = tensor::TensorFullIndex>
using StagedReduction = StagedOperator<
        Reduction<
                Indices,
                PositionType,
                typename ReducedTensorType::non_indices_domain_t::discrete_element_type,
                Complex>,
        typename Reduction<
                Indices,
                PositionType,
                typename ReducedTensorType::non_indices_domain_t::discrete_element_type,
                Complex>::staged_storage_type::template rebind<ExecSpace, Coefficients>>;

static_assert(StagedOperatorConcept<
              StagedReduction<
                      ddc::TypeSeq<detail::DirectionConcept>,
                      detail::FieldConcept<detail::DirectionConcept>,
                      detail::FieldConcept<detail::DirectionConcept>,
                      detail::FieldConcept<detail::PositionIndexConcept>,
                      Kokkos::DefaultHostExecutionSpace>,
              detail::FieldConcept<detail::DirectionConcept>>);


static_assert(LocalOperatorConcept<
              StagedReduction<
                      ddc::TypeSeq<detail::DirectionConcept>,
                      detail::FieldConcept<detail::DirectionConcept>,
                      detail::FieldConcept<detail::DirectionConcept>,
                      detail::FieldConcept<detail::PositionIndexConcept>,
                      Kokkos::DefaultHostExecutionSpace>,
              detail::FieldConcept<detail::DirectionConcept>,
              detail::FieldConcept<detail::DirectionConcept>,
              detail::FieldConcept<detail::PositionIndexConcept>,
              ddc::DiscreteElement<detail::GridConcept>>);
static_assert(LocalOperatorConcept<
              StagedReduction<
                      ddc::TypeSeq<detail::DirectionConcept>,
                      detail::FieldConcept<detail::DirectionConcept>,
                      detail::FieldConcept<detail::DirectionConcept>,
                      detail::FieldConcept<detail::PositionIndexConcept>,
                      Kokkos::DefaultHostExecutionSpace>,
              detail::FieldConcept<detail::DirectionConcept>,
              detail::FieldConcept<detail::DirectionConcept>,
              detail::FieldConcept<detail::MetricIndexConcept>,
              detail::FieldConcept<detail::PositionIndexConcept>,
              ddc::DiscreteElement<detail::GridConcept>>);

template <
        class Indices,
        CellComplex Complex = CellComplex::Primal,
        template <class...> class Coefficients = tensor::TensorFullIndex,
        class Exec,
        class Output,
        class Input,
        class Position>
StagedReduction<Indices, Output, Input, Position, Exec, Complex, Coefficients>
make_staged_reduction(Exec const& exec, Output output, Input input, Position position)
{
    return StagedReduction<
            Indices,
            Output,
            Input,
            Position,
            Exec,
            Complex,
            Coefficients>(exec, output, input, position);
}

} // namespace sil::exterior
