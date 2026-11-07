// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#pragma once

#include <array>
#include <cassert>

#include <ddc/ddc.hpp>

#include <similie/exterior/cochain_indices.hpp>
#include <similie/misc/signed_discrete_element.hpp>
#include <similie/tensor/scalar_extrapolation_rules.hpp>
#include <similie/tensor/tensor_prod/tensor_prod.hpp>

namespace sil::exterior {

/** Interior product of a primal k-cochain with a primal velocity one-cochain.
 * On a uniform Cartesian mesh, the velocity one-cochain stores edge integrals
 * of the Euclidean velocity covector. Each component is divided by its edge
 * length and averaged over the incident edges to the vertex. Likewise, each
 * k-form coefficient is averaged over its 2^k incident cells after division
 * by their k-volume. At the common vertex we evaluate
 * (i_u alpha)_{a_2...a_k} = sum_b u^b alpha_{b a_2...a_k}.
 * Finally, averaging over the vertices of the output (k-1)-cell and multiplying
 * by its volume produces the output cochain. These two averages are adjoints
 * in the uniform Cartesian cochain inner product. In particular, for k=2,
 * the pairing of the velocity cochain with its interior product is zero.
 * This avoids multiplying edge and face coefficients at different positions.
 *
 * Axes of the tensor indices and spatial domain must have the same order.
 * Spacing contains the positive, constant Cartesian edge lengths; a curvilinear
 * metric is not supported. The supplied extrapolation rule defines boundary
 * samples; periodic closure preserves the adjoint identity at the seam.
 * Input and output storage must not alias. For vorticity transport, use
 * -d(i_u omega): i_u omega is minus u cross omega in three dimensions.
 * \important This operator and documentation is fully AI-generated.
 */
template <tensor::TensorNatIndex VelocityIndex, tensor::TensorIndex FormIndex>
struct InteriorProduct
{
private:
    template <
            class Index,
            class Local,
            class Field,
            class Vertex,
            class Spacing,
            class Extrapolation>
    KOKKOS_FUNCTION static void reconstruct(
            Local const& local,
            Field const& field,
            Vertex const& vertex,
            Spacing const& spacing,
            Extrapolation const& extrapolation)
    {
        ddc::device_for_each(local.domain(), [&](auto component) {
            auto const natural = local.accessor().canonical_natural_element(component);
            std::array const axes = detail::flat_natural_elem_ids(natural);
            double volume = 1;
            for (std::size_t axis : axes)
                volume *= spacing[axis];
            double coefficient = 0;
            for (std::size_t sample = 0; sample < (std::size_t(1) << Index::rank()); ++sample) {
                Vertex neighbor(vertex);
                for (std::size_t j = 0; j < axes.size(); ++j)
                    ddc::detail::array(neighbor)[axes[j]] -= (sample >> j) & 1;
                auto const sampled = extrapolation(field, neighbor);
                coefficient += sampled.get(sampled.access_element(natural));
            }
            local.mem(component) = coefficient / ((std::size_t(1) << Index::rank()) * volume);
        });
    }

    template <
            class Memory,
            class Vertex,
            class Velocity,
            class Form,
            class Spacing,
            class Extrapolation,
            class Contracted,
            class... Free>
    KOKKOS_FUNCTION static double contract(
            Vertex const& vertex,
            Velocity const& velocity,
            Form const& form,
            Spacing const& spacing,
            Extrapolation const& extrapolation,
            std::array<std::size_t, sizeof...(Free)> const& axes,
            ddc::DiscreteDomain<Contracted, Free...>)
    {
        std::array<double, VelocityIndex::mem_size()> velocity_storage {};
        std::array<double, FormIndex::mem_size()> form_storage {};
        std::array<double, (std::size_t(1) * ... * Free::mem_size())> output_storage {};
        tensor::Tensor const local_velocity(
                ddc::ChunkSpan<
                        double,
                        ddc::DiscreteDomain<VelocityIndex>,
                        Kokkos::layout_right,
                        Memory>(
                        velocity_storage.data(),
                        tensor::TensorAccessor<VelocityIndex>::domain()));
        tensor::Tensor const local_form(
                ddc::ChunkSpan<
                        double,
                        ddc::DiscreteDomain<FormIndex>,
                        Kokkos::layout_right,
                        Memory>(form_storage.data(), tensor::TensorAccessor<FormIndex>::domain()));
        tensor::Tensor const local_output(
                ddc::ChunkSpan<
                        double,
                        ddc::DiscreteDomain<Free...>,
                        Kokkos::layout_right,
                        Memory>(output_storage.data(), tensor::TensorAccessor<Free...>::domain()));
        reconstruct<VelocityIndex>(local_velocity, velocity, vertex, spacing, extrapolation);
        reconstruct<FormIndex>(local_form, form, vertex, spacing, extrapolation);
        // Edge integrals give covector components. The existing spacing-based
        // operator uses their Euclidean dual as the insertion vector.
        tensor::Tensor const insertion(
                ddc::ChunkSpan<
                        double,
                        ddc::DiscreteDomain<
                                tensor::TensorFullIndex<tensor::swap_character_t<Contracted>>>,
                        Kokkos::layout_right,
                        Memory>(
                        velocity_storage.data(),
                        tensor::TensorAccessor<tensor::TensorFullIndex<
                                tensor::swap_character_t<Contracted>>>::domain()));
        tensor::tensor_prod(local_output, insertion, local_form);
        return local_output.get(local_output.access_element(
                detail::natural_elem_from_flat_ids<ddc::DiscreteElement<Free...>>(axes)));
    }

public:
    template <
            class ExecSpace,
            class Output,
            class Velocity,
            class Form,
            class Extrapolation = tensor::ClampExtrapolationRule>
    Output operator()(
            ExecSpace const& exec,
            Output const& output,
            Velocity const& velocity,
            Form const& form,
            std::array<double, VelocityIndex::size()> const& spacing,
            Extrapolation const& extrapolation = {}) const
    {
        using OutputIndex
                = ddc::type_seq_element_t<0, ddc::to_type_seq_t<typename Output::indices_domain_t>>;
        using Element = typename Output::non_indices_domain_t::discrete_element_type;
        using SignedElement = misc::convert_type_seq_to_t<
                misc::SignedDiscreteElement,
                ddc::to_type_seq_t<typename Output::non_indices_domain_t>>;
        static_assert(FormIndex::rank() > 0);
        static_assert(OutputIndex::rank() + 1 == FormIndex::rank());
        static_assert(Output::non_indices_domain_t::rank() == VelocityIndex::size());
        static_assert(VelocityIndex::rank() == 1);
        for (double const length : spacing)
            assert(length > 0);
        assert(output.data_handle() != form.data_handle());
        assert(output.data_handle() != velocity.data_handle());
        ddc::parallel_for_each(
                "similie_interior_product",
                exec,
                output.domain(),
                KOKKOS_LAMBDA(typename Output::discrete_element_type element) {
                    auto const natural_element = output.accessor().canonical_natural_element(
                            ddc::DiscreteElement<OutputIndex>(element));
                    std::array<std::size_t, OutputIndex::rank()> axes {};
                    if constexpr (OutputIndex::rank() > 0)
                        axes = detail::flat_natural_elem_ids(natural_element);
                    double volume = 1;
                    for (std::size_t axis : axes)
                        volume *= spacing[axis];
                    double contraction = 0;
                    for (std::size_t corner = 0; corner < (std::size_t(1) << OutputIndex::rank());
                         ++corner) {
                        SignedElement vertex {Element(element)};
                        for (std::size_t j = 0; j < axes.size(); ++j)
                            ddc::detail::array(vertex)[axes[j]] += (corner >> j) & 1;
                        contraction += contract<typename Output::memory_space>(
                                vertex,
                                velocity,
                                form,
                                spacing,
                                extrapolation,
                                axes,
                                tensor::natural_domain_t<FormIndex>());
                    }
                    output.mem(element)
                            = volume * contraction / (std::size_t(1) << OutputIndex::rank());
                });
        return output;
    }
};

} // namespace sil::exterior
