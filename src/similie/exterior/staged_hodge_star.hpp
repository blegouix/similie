// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#pragma once

#include "hodge_star.hpp"

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
class StagedHodgeStar
{
    using Memory = typename MetricType::memory_space;
    using Domain = misc::cartesian_prod_t<
            typename MetricType::non_indices_domain_t,
            hodge_star_domain_t<Indices1, Indices2>>;
    using Coefficients = tensor::Tensor<double, Domain, Kokkos::layout_right, Memory>;
    // Keep the execution instance on the host; device copies contain only a
    // View handle, including when the instance is Kokkos::Serial.
    Kokkos::View<typename Memory::execution_space, Kokkos::HostSpace> m_exec;
    // Keep ownership in a View, whose device copy is a lightweight handle.
    Kokkos::View<double*, Memory> m_storage;
    Coefficients m_coefficients;

public:
    template <class Exec>
    StagedHodgeStar(Exec const& exec, MetricType metric, PositionType position)
        : m_exec("staged_hodge_execution")
        , m_storage(
                  Kokkos::view_alloc(Kokkos::WithoutInitializing, "staged_hodge_star"),
                  Domain(metric.non_indices_domain(),
                         tensor::tensor_accessor_for_domain_t<
                                 hodge_star_domain_t<Indices1, Indices2>>::domain())
                          .size())
        , m_coefficients(
                  ddc::ChunkSpan<double, Domain, Kokkos::layout_right, Memory>(
                          m_storage.data(),
                          Domain(metric.non_indices_domain(),
                                 tensor::tensor_accessor_for_domain_t<
                                         hodge_star_domain_t<Indices1, Indices2>>::domain())))
    {
        m_exec() = exec;
        fill_discrete_hodge_star<
                Indices1,
                Indices2,
                Complex>(exec, m_coefficients, metric, position);
    }

    template <class NaturalElement>
    KOKKOS_FUNCTION double value(
            MetricType,
            PositionType,
            BatchElem element,
            NaturalElement natural_element) const
    {
        return m_coefficients[element].get(
                m_coefficients.accessor().access_element(natural_element));
    }

    template <class Output, class Input>
    KOKKOS_FUNCTION void operator()(
            Output output,
            Input input,
            MetricType,
            PositionType,
            BatchElem element) const
    {
        tensor::tensor_prod(output, input, m_coefficients[element]);
    }

    template <class Output, class Input>
    Output operator()(Output output, Input input) const
    {
        Coefficients const coefficients = m_coefficients;
        ddc::parallel_for_each(
                "similie_apply_staged_hodge_star",
                m_exec(),
                output.non_indices_domain(),
                KOKKOS_LAMBDA(BatchElem element) {
                    tensor::tensor_prod(output[element], input[element], coefficients[element]);
                });
        return output;
    }
};

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
