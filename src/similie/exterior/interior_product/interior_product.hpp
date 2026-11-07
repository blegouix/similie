// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#pragma once

#include <array>

#include <similie/exterior/form.hpp>
#include <similie/tensor/metric.hpp>
#include <similie/tensor/tensor_indices/full_tensor.hpp>
#include <similie/tensor/tensor_prod/tensor_prod.hpp>

namespace sil::exterior {

template <tensor::TensorNatIndex VelocityIndex, tensor::TensorIndex FormIndex>
struct InteriorProduct
{
    // Insert the metric dual of a velocity one-form. The caller supplies the
    // inverse metric, so this definition also applies in nonorthogonal frames.
    template <
            TensorFormType Output,
            TensorFormType Velocity,
            TensorFormType Input,
            misc::Specialization<tensor::Tensor> InverseMetric,
            class Element>
        requires(
                form_degree_v<Velocity> == 1 && form_degree_v<Output> + 1 == FormIndex::rank()
                && std::is_same_v<
                        typename Input::accessor_t::natural_domain_t,
                        tensor::natural_domain_t<FormIndex>>)
    KOKKOS_FUNCTION void operator()(
            Output const& output,
            Velocity const& velocity,
            Input const& input,
            InverseMetric const& inverse_metric,
            Element element) const
    {
        using VelocityStorageIndex = ddc::
                type_seq_element_t<0, ddc::to_type_seq_t<typename Velocity::indices_domain_t>>;
        using Contracted
                = tensor::Covariant<tensor::prime<tensor::uncharacterize_t<VelocityIndex>>>;
        std::array<double, VelocityIndex::mem_size()> storage {};
        tensor::Tensor vector(
                ddc::ChunkSpan<
                        double,
                        ddc::DiscreteDomain<tensor::TensorFullIndex<VelocityIndex>>,
                        Kokkos::layout_right,
                        typename Input::memory_space>(
                        storage.data(),
                        tensor::TensorAccessor<tensor::TensorFullIndex<VelocityIndex>>::domain()));
        tensor::tensor_prod(
                vector,
                tensor::relabelize_metric<VelocityIndex, tensor::upper_t<Contracted>>(
                        inverse_metric[element]),
                tensor::relabelize_index_of<VelocityStorageIndex, Contracted>(
                        velocity.coefficients()));
        operator()(output, vector, input);
    }

    template <
            TensorFormType Output,
            misc::Specialization<tensor::Tensor> Vector,
            TensorFormType Input>
        requires(
                form_degree_v<Output> + 1 == FormIndex::rank()
                && std::is_same_v<
                        typename Input::accessor_t::natural_domain_t,
                        tensor::natural_domain_t<FormIndex>>)
    KOKKOS_FUNCTION void operator()(Output const& output, Vector const& vector, Input const& input)
            const
    {
        static_assert(FormIndex::rank() > 0);
        static_assert(std::is_same_v<
                      typename Input::accessor_t::natural_domain_t,
                      tensor::natural_domain_t<FormIndex>>);
        static_assert([]<class... Index>(ddc::TypeSeq<Index...>) {
            return (Index::rank() + ... + 0) == FormIndex::rank() - 1;
        }(ddc::to_type_seq_t<typename Output::indices_domain_t>()));
        static_assert(Output::non_indices_domain_t::rank() == 0);
        static_assert(Input::non_indices_domain_t::rank() == 0);
        static_assert(Vector::non_indices_domain_t::rank() == 0);
        static_assert(
                std::is_same_v<typename VelocityIndex::character, tensor::ContravariantCharacter>);
        // A one-slot full index selects the characterized contraction overload
        // for one-forms as well as higher-degree forms.
        using VectorStorageIndex
                = ddc::type_seq_element_t<0, ddc::to_type_seq_t<typename Vector::indices_domain_t>>;
        tensor::relabelize_index_of_t<
                Vector,
                VectorStorageIndex,
                tensor::TensorFullIndex<VelocityIndex>> const insertion
                = tensor::relabelize_index_of<
                        VectorStorageIndex,
                        tensor::TensorFullIndex<VelocityIndex>>(vector);
        if constexpr (FormIndex::rank() == 1) {
            tensor::Tensor scalar(
                    ddc::ChunkSpan<
                            typename Output::element_type,
                            ddc::DiscreteDomain<>,
                            typename Output::layout_type,
                            typename Output::
                                    memory_space>(output.data_handle(), ddc::DiscreteDomain<>()));
            tensor::tensor_prod(scalar, insertion, input.coefficients());
        } else {
            tensor::tensor_prod(output.coefficients(), insertion, input.coefficients());
        }
    }
};

} // namespace sil::exterior
