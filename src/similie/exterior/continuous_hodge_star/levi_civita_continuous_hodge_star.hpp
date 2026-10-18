// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#pragma once

#include <similie/exterior/continuous_hodge_star/continuous_hodge_star.hpp>
#include <similie/tensor/tensor_indices/levi_civita_tensor.hpp>

namespace sil::exterior {

/** Continuous Hodge star for an identity metric and the positive coordinate orientation.
 * \important This operator and documentation is fully AI-generated.
 * The Levi-Civita tensor has no stored coefficients. Contracting its source
 * indices with a k-form and dividing by k! gives the complementary (n-k)-form.
 * The factorial accounts for the permutations of the antisymmetric input in
 * the full natural-index contraction performed by tensor_prod.
 */
template <
        misc::Specialization<ddc::TypeSeq> Indices1,
        misc::Specialization<ddc::TypeSeq> Indices2,
        class MetricType,
        class BatchElem>
class LeviCivitaContinuousHodgeStar
{
    using Index = misc::convert_type_seq_to_t<
            tensor::TensorLeviCivitaIndex,
            ddc::type_seq_merge_t<Indices1, Indices2>>;
    using MetricIndex
            = ddc::type_seq_element_t<0, ddc::to_type_seq_t<typename MetricType::indices_domain_t>>;
    static_assert(misc::Specialization<MetricIndex, tensor::TensorIdentityIndex>);
    static_assert(Index::rank() > 0);
    static_assert(tensor::metric_index_1<MetricIndex>::size() == Index::rank());

    KOKKOS_FUNCTION static tensor::Tensor<
            double,
            ddc::DiscreteDomain<Index>,
            Kokkos::layout_right,
            typename MetricType::memory_space>
    levi_civita()
    {
        return tensor::Tensor(
                ddc::ChunkSpan<
                        double,
                        ddc::DiscreteDomain<Index>,
                        Kokkos::layout_right,
                        typename MetricType::
                                memory_space>(nullptr, tensor::TensorAccessor<Index>::domain()));
    }

public:
    using staged_storage_type
            = detail::ImplicitLocalOperatorStorage<LeviCivitaContinuousHodgeStar, MetricType>;

    template <
            misc::Specialization<tensor::Tensor> HodgeTensorType,
            misc::Specialization<tensor::Tensor> FormTensorType>
    KOKKOS_FUNCTION static void operator()(
            HodgeTensorType const& hodge_tensor,
            FormTensorType const& form_tensor,
            MetricType,
            BatchElem)
    {
        if constexpr (ddc::type_seq_size_v<Indices1> == 0) {
            // A scalar has no contracted natural index. Remove its storage tag
            // when passing its single value to the generic tensor product.
            tensor::Tensor scalar(
                    ddc::ChunkSpan<
                            double,
                            ddc::DiscreteDomain<>,
                            Kokkos::layout_right,
                            typename FormTensorType::memory_space>(
                            form_tensor.data_handle(),
                            ddc::DiscreteDomain<>()));
            tensor::tensor_prod(hodge_tensor, levi_civita(), scalar);
        } else {
            tensor::tensor_prod(hodge_tensor, levi_civita(), form_tensor);
        }
        ddc::device_for_each(hodge_tensor.domain(), [&](auto element) {
            hodge_tensor.mem(element) /= misc::factorial(ddc::type_seq_size_v<Indices1>);
        });
    }

    KOKKOS_FUNCTION static detail::LocalOperatorStencil<
            tensor::lower_t<reduction_index_t<Indices1>>,
            reduction_index_t<Indices2>>
    value(MetricType const& metric, BatchElem element)
    {
        using Source = tensor::lower_t<reduction_index_t<Indices1>>;
        using Target = reduction_index_t<Indices2>;
        detail::LocalOperatorStencil<Source, Target> result;
        auto unit = detail::make_stencil<typename MetricType::memory_space, Source>(
                ddc::DiscreteElement<>());
        auto output = detail::make_stencil<typename MetricType::memory_space, Target>(
                ddc::DiscreteElement<>());
        for (std::size_t source = 0; source < Source::mem_size(); ++source) {
            unit.mem(ddc::DiscreteElement<Source>(source)) = 1.0;
            operator()(
                    tensor::Tensor(output.span_view()),
                    tensor::Tensor(unit.span_view()),
                    metric,
                    element);
            for (std::size_t target = 0; target < Target::mem_size(); ++target)
                result.weights[target * Source::mem_size() + source]
                        = output.mem(ddc::DiscreteElement<Target>(target));
            unit.mem(ddc::DiscreteElement<Source>(source)) = 0.0;
        }
        return result;
    }

    template <class NaturalElement>
    KOKKOS_FUNCTION static double value(MetricType, BatchElem, NaturalElement natural)
    {
        return levi_civita().get(levi_civita().access_element(natural))
               / misc::factorial(ddc::type_seq_size_v<Indices1>);
    }
};

static_assert(ContinuousHodgeStarConcept<
              LeviCivitaContinuousHodgeStar<
                      ddc::TypeSeq<detail::DirectionConcept>,
                      ddc::TypeSeq<>,
                      detail::FieldConcept<detail::MetricIndexConcept>,
                      ddc::DiscreteElement<detail::GridConcept>>,
              detail::FieldConcept<detail::ScalarConcept>,
              detail::FieldConcept<detail::DirectionConcept>,
              detail::FieldConcept<detail::MetricIndexConcept>,
              ddc::DiscreteElement<detail::GridConcept>,
              ddc::DiscreteElement<detail::DirectionConcept>>);

} // namespace sil::exterior
