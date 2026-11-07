// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#pragma once

#include <tuple>
#include <utility>

#include <similie/exterior/cochain.hpp>
#include <similie/exterior/form.hpp>

namespace sil::exterior {
namespace detail {
template <
        class Operator,
        class Output,
        class Forms,
        class Element,
        std::size_t... I,
        class Arguments>
KOKKOS_FUNCTION void apply_form_tuple(
        Operator const& operation,
        Output const& output,
        Forms const& forms,
        Element element,
        std::index_sequence<I...>,
        Arguments const& arguments)
{
    [&]<std::size_t... A>(std::index_sequence<A...>) {
        if constexpr (requires {
                          operation(
                                  output[element],
                                  std::get<I>(forms)[element]...,
                                  std::get<A>(arguments)...,
                                  element);
                      })
            operation(
                    output[element],
                    std::get<I>(forms)[element]...,
                    std::get<A>(arguments)...,
                    element);
        else
            operation(output[element], std::get<I>(forms)[element]..., std::get<A>(arguments)...);
    }(std::make_index_sequence<std::tuple_size_v<Arguments>>());
}
template <class Operator, class Output, class Buffers, class Arguments>
struct ApplyRepresentation
{
    Operator operation;
    Output output;
    Buffers buffers;
    Arguments arguments;

    KOKKOS_FUNCTION void operator()(
            typename Output::non_indices_domain_t::discrete_element_type element) const
    {
        apply_form_tuple(
                operation,
                output,
                buffers,
                element,
                std::make_index_sequence<std::tuple_size_v<Buffers>>(),
                arguments);
    }
};
} // namespace detail


// Transfer callables carry the geometry, complex and sampling rule. The adapter
// owns no hidden discretization and uses the caller's explicitly typed buffers.
template <class Operator>
class OperatorActingOnCochainInPlaceOfForm
{
    Operator m_operator;

public:
    KOKKOS_DEFAULTED_FUNCTION OperatorActingOnCochainInPlaceOfForm() = default;
    KOKKOS_FUNCTION explicit OperatorActingOnCochainInPlaceOfForm(Operator const& operation)
        : m_operator(operation)
    {
    }

    template <
            CochainType Output,
            CochainType Input,
            FormType Reconstructed,
            FormType Transformed,
            class Reconstruction,
            class Reduction,
            class... Arguments>
    KOKKOS_FUNCTION void operator()(
            Output const& output,
            Input const& input,
            Reconstructed const& reconstructed,
            Transformed const& transformed,
            Reconstruction const& reconstruct,
            Reduction const& reduce,
            Arguments const&... arguments) const
    {
        reconstruct(reconstructed, input);
        m_operator(transformed, reconstructed, arguments...);
        reduce(output, transformed);
    }
    // Value operators can return an owned result (including scalar forms).
    // Transfers return values, so no mutable-view convention is imposed here.
    template <class TransferIn, class TransferOut, class... Inputs>
        requires(sizeof...(Inputs) > 0
                 && ((CochainType<Inputs> || misc::Specialization<Inputs, Cosimplex>) && ...)
                 && requires(Operator const& operation, TransferIn const& transfer_in,
                             TransferOut const& transfer_out, Inputs const&... inputs) {
                     requires (FormType<decltype(transfer_in(inputs))> && ...);
                     requires FormType<decltype(operation(transfer_in(inputs)...))>;
                     requires (CochainType<decltype(transfer_out(operation(transfer_in(inputs)...)))>
                               || misc::Specialization<decltype(transfer_out(operation(transfer_in(inputs)...))), Cosimplex>);
                 })
    KOKKOS_FUNCTION auto operator()(
            TransferIn const& transfer_in, TransferOut const& transfer_out, Inputs const&... inputs) const
    {
        return transfer_out(m_operator(transfer_in(inputs)...));
    }

    // Multiple operands share one geometric transfer policy. Its overloads
    // select the appropriate form degree and placement for each operand.
    template <
            class Exec,
            CochainType Output,
            class... Inputs,
            class... Buffers,
            FormType Transformed,
            class TransferIn,
            class TransferOut,
            class... Arguments>
    Output operator()(
            Exec const& exec,
            Output const& output,
            std::tuple<Inputs...> const& inputs,
            std::tuple<Buffers...> const& buffers,
            Transformed const& transformed,
            TransferIn const& transfer_in,
            TransferOut const& transfer_out,
            Arguments const&... arguments) const
    {
        static_assert(sizeof...(Inputs) == sizeof...(Buffers));
        static_assert((CochainType<Inputs> && ...));
        static_assert((FormType<Buffers> && ...));
        [&]<std::size_t... I>(std::index_sequence<I...>) {
            (transfer_in(exec, std::get<I>(buffers), std::get<I>(inputs)), ...);
        }(std::index_sequence_for<Inputs...>());
        std::tuple<Arguments...> const argument_tuple(arguments...);
        ddc::parallel_for_each(
                "similie_apply_representation_adapter",
                exec,
                transformed.non_indices_domain(),
                detail::ApplyRepresentation<
                        Operator,
                        Transformed,
                        std::tuple<Buffers...>,
                        std::tuple<
                                Arguments...>> {m_operator, transformed, buffers, argument_tuple});
        transfer_out(exec, output, transformed);
        return output;
    }
};

template <class Operator>
class OperatorActingOnFormInPlaceOfCochain
{
    Operator m_operator;

public:
    static constexpr bool input_is_form = true;
    static constexpr bool output_is_form = true;
    KOKKOS_DEFAULTED_FUNCTION OperatorActingOnFormInPlaceOfCochain() = default;
    KOKKOS_FUNCTION explicit OperatorActingOnFormInPlaceOfCochain(Operator const& operation)
        : m_operator(operation)
    {
    }

    template <
            FormType Output,
            FormType Input,
            CochainType Reduced,
            CochainType Transformed,
            class Reduction,
            class Reconstruction,
            class... Arguments>
    KOKKOS_FUNCTION void operator()(
            Output const& output,
            Input const& input,
            Reduced const& reduced,
            Transformed const& transformed,
            Reduction const& reduce,
            Reconstruction const& reconstruct,
            Arguments const&... arguments) const
    {
        reduce(reduced, input);
        m_operator(transformed, reduced, arguments...);
        reconstruct(output, transformed);
    }
    // Value operators can return an owned result (including scalar forms).
    // Transfers return values, so no mutable-view convention is imposed here.
    template <class TransferIn, class TransferOut, class... Inputs>
        requires(sizeof...(Inputs) > 0
                 && (FormType<Inputs> && ...)
                 && requires(Operator const& operation, TransferIn const& transfer_in,
                             TransferOut const& transfer_out, Inputs const&... inputs) {
                     requires ((CochainType<decltype(transfer_in(inputs))>
                                || misc::Specialization<decltype(transfer_in(inputs)), Cosimplex>) && ...);
                     requires FormType<decltype(transfer_out(operation(transfer_in(inputs)...)))>;
                 })
    KOKKOS_FUNCTION auto operator()(
            TransferIn const& transfer_in, TransferOut const& transfer_out, Inputs const&... inputs) const
    {
        return transfer_out(m_operator(transfer_in(inputs)...));
    }

    // Multiple operands share one geometric transfer policy. Its overloads
    // select the appropriate form degree and placement for each operand.
    template <
            class Exec,
            FormType Output,
            class... Inputs,
            class... Buffers,
            CochainType Transformed,
            class TransferIn,
            class TransferOut,
            class... Arguments>
    Output operator()(
            Exec const& exec,
            Output const& output,
            std::tuple<Inputs...> const& inputs,
            std::tuple<Buffers...> const& buffers,
            Transformed const& transformed,
            TransferIn const& transfer_in,
            TransferOut const& transfer_out,
            Arguments const&... arguments) const
    {
        static_assert(sizeof...(Inputs) == sizeof...(Buffers));
        static_assert((FormType<Inputs> && ...));
        static_assert((CochainType<Buffers> && ...));
        [&]<std::size_t... I>(std::index_sequence<I...>) {
            (transfer_in(exec, std::get<I>(buffers), std::get<I>(inputs)), ...);
        }(std::index_sequence_for<Inputs...>());
        std::tuple<Arguments...> const argument_tuple(arguments...);
        ddc::parallel_for_each(
                "similie_apply_representation_adapter",
                exec,
                transformed.non_indices_domain(),
                detail::ApplyRepresentation<
                        Operator,
                        Transformed,
                        std::tuple<Buffers...>,
                        std::tuple<
                                Arguments...>> {m_operator, transformed, buffers, argument_tuple});
        transfer_out(exec, output, transformed);
        return output;
    }
};

} // namespace sil::exterior
