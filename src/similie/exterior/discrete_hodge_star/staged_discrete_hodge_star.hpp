// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#pragma once

#include <similie/exterior/continuous_hodge_star/continuous_hodge_star.hpp>

namespace sil::exterior {

/** Discrete Hodge star with geometry coefficients fixed at construction.
 * \important This operator and documentation is fully AI-generated.
 * The pointwise interface matches DiscreteHodgeStar. The metric and position
 * arguments of that interface are ignored: rebuild when geometry changes.
 * Whole-field application uses the execution space supplied at construction.
 */
template <
        CellComplex Complex,
        misc::Specialization<ddc::TypeSeq> Indices1,
        misc::Specialization<ddc::TypeSeq> Indices2,
        class MetricType,
        class PositionType,
        class BatchElem>
using StagedHodgeStar = StagedOperator<
        DiscreteHodgeStar<Complex, Indices1, Indices2, MetricType, PositionType, BatchElem>>;

static_assert(DiscreteHodgeStarConcept<
              StagedHodgeStar<
                      CellComplex::Primal,
                      ddc::TypeSeq<detail::DirectionConcept>,
                      ddc::TypeSeq<>,
                      detail::FieldConcept<detail::MetricIndexConcept>,
                      detail::FieldConcept<detail::PositionIndexConcept>,
                      ddc::DiscreteElement<detail::GridConcept>>,
              detail::FieldConcept<detail::ScalarConcept>,
              detail::FieldConcept<detail::DirectionConcept>,
              detail::FieldConcept<detail::MetricIndexConcept>,
              detail::FieldConcept<detail::PositionIndexConcept>,
              ddc::DiscreteElement<detail::GridConcept>,
              ddc::DiscreteElement<detail::DirectionConcept>>);

static_assert(StagedOperatorConcept<
              StagedHodgeStar<
                      CellComplex::Primal,
                      ddc::TypeSeq<detail::DirectionConcept>,
                      ddc::TypeSeq<>,
                      detail::FieldConcept<detail::MetricIndexConcept>,
                      detail::FieldConcept<detail::PositionIndexConcept>,
                      ddc::DiscreteElement<detail::GridConcept>>,
              detail::FieldConcept<detail::ScalarConcept>,
              detail::FieldConcept<detail::DirectionConcept>>);

} // namespace sil::exterior
