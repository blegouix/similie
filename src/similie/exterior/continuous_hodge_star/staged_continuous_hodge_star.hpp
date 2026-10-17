// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#pragma once

#include <similie/exterior/continuous_hodge_star/continuous_hodge_star.hpp>

namespace sil::exterior {
/** Cache the continuous metric-dependent Hodge map in canonical components.
 * \important This operator and documentation is fully AI-generated.
 * Rebuild when the metric changes. This does not cache geometric reduction or
 * reconstruction; StagedHodgeStar caches their complete discrete composition.
 */
template <
        misc::Specialization<ddc::TypeSeq> Indices1,
        misc::Specialization<ddc::TypeSeq> Indices2,
        class MetricType,
        class BatchElem>
using StagedContinuousHodgeStar
        = StagedOperator<ContinuousHodgeStar<Indices1, Indices2, MetricType, BatchElem>>;

static_assert(ContinuousHodgeStarConcept<
              StagedContinuousHodgeStar<
                      ddc::TypeSeq<detail::DirectionConcept>,
                      ddc::TypeSeq<>,
                      detail::FieldConcept<detail::MetricIndexConcept>,
                      ddc::DiscreteElement<detail::GridConcept>>,
              detail::FieldConcept<detail::ScalarConcept>,
              detail::FieldConcept<detail::DirectionConcept>,
              detail::FieldConcept<detail::MetricIndexConcept>,
              ddc::DiscreteElement<detail::GridConcept>,
              ddc::DiscreteElement<detail::DirectionConcept>>);


static_assert(StagedOperatorConcept<
              StagedContinuousHodgeStar<
                      ddc::TypeSeq<detail::DirectionConcept>,
                      ddc::TypeSeq<>,
                      detail::FieldConcept<detail::MetricIndexConcept>,
                      ddc::DiscreteElement<detail::GridConcept>>,
              detail::FieldConcept<detail::ScalarConcept>,
              detail::FieldConcept<detail::DirectionConcept>>);

} // namespace sil::exterior
