#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Baptiste Legouix
# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sympy import diff, solve, symbols
from sympy.printing.codeprinter import cxxcode


@dataclass(frozen=True)
class HamiltonianDefinition:
    namespace: str
    struct_name: str
    parameters: list[str]
    hamiltonian: object
    variables: list[object]
    input_variables: list[object] | None = None
    includes: list[str] | None = None
    value_computer_type: str | None = None


def _flatten_variable_entries(entries: list[object]) -> list[object]:
    flattened: list[object] = []
    for entry in entries:
        if isinstance(entry, (list, tuple)):
            flattened.extend(_flatten_variable_entries(list(entry)))
        else:
            flattened.append(entry)
    return flattened


def _replace_symbols(expression: str, replacements: dict[str, str]) -> str:
    for source, target in replacements.items():
        expression = expression.replace(source, target)
    return expression


def _render_members(parameters: list[tuple[str, str, bool]]) -> str:
    return "\n".join(
        f"    {'const ' if is_const else ''}double {name};"
        for name, _, is_const in parameters
    )


def _render_constructor_signature(struct_name: str, parameters: list[tuple[str, str, bool]]) -> str:
    params = ",\n            ".join(f"double {constructor_name}_" for _, constructor_name, _ in parameters)
    return f"    constexpr {struct_name}(\n            {params})"


def _render_constructor_initializers(parameters: list[tuple[str, str, bool]]) -> str:
    return ", ".join(f"{name}({constructor_name}_)" for name, constructor_name, _ in parameters)


def _render_indexed_method(
    method_name: str,
    argument_prefix: str,
    symbols_: list[str],
    expressions: list,
    replacements: dict[str, str],
) -> str:
    branches: list[str] = []
    for i, (symbol_name, expression) in enumerate(zip(symbols_, expressions, strict=True)):
        branches.append(
            f"""        if constexpr (I == {i}) {{
            double const {symbol_name} = {argument_prefix};
            return {_replace_symbols(cxxcode(expression), replacements)};
        }}"""
        )

    branches.append(
        """        else {
            static_assert(I < N, "Hamiltonian component index out of range");
        }"""
    )

    return f"""
    template <std::size_t I>
    constexpr double {method_name}(double {argument_prefix}) const
    {{
{chr(10).join(branches)}
    }}
"""


def _render_indexed_nonlocal_value_method(
    method_name: str,
    symbols_: list[str],
    expressions: list,
    replacements: dict[str, str],
) -> str:
    differentiated_expressions = [
        diff(expression, symbols(symbol_name))
        for symbol_name, expression in zip(symbols_, expressions, strict=True)
    ]
    branches: list[str] = []
    for i, expression in enumerate(differentiated_expressions):
        branches.append(
            f"""        if constexpr (I == {i}) {{
            auto value = moments_computer_value.template operator()<I>(chain, lower_chain, elem);
            value *= {_replace_symbols(cxxcode(expression), replacements)};
            return value;
        }}"""
        )
    branches.append(
        """        else {
            static_assert(I < N, "Hamiltonian component index out of range");
        }"""
    )
    return f"""
    template <std::size_t I, class ChainType, class LowerChainType, class Elem, class MomentsComputerValue>
    constexpr auto {method_name}(ChainType chain, LowerChainType lower_chain, Elem elem, MomentsComputerValue const& moments_computer_value) const
    {{
{chr(10).join(branches)}
    }}
"""


def write_cpp_hamiltonian_header(
    output_path: Path,
    namespace: str,
    struct_name: str,
    parameters: list[tuple[str, str, bool]],
    hamiltonian,
    derivative_symbols: list[str],
    derivative_expressions: list,
    includes: list[str] | None = None,
    value_computer_type: str | None = None,
    scalar_argument_name: str | None = None,
    input_argument_names: list[str] | None = None,
    scalar_derivative_expression=None,
    inverse_symbols: list[str] | None = None,
    inverse_expressions: list | None = None,
    nonlocal_pi_value_methods: bool = False,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    parameter_replacements = {
        constructor_name: member_name for member_name, constructor_name, _ in parameters
    }
    h_replacements = dict(parameter_replacements)
    input_argument_names = [] if input_argument_names is None else input_argument_names
    for i, input_argument_name in enumerate(input_argument_names):
        h_replacements[input_argument_name] = f"inputs[{i}]"
    for i, symbol_name in enumerate(derivative_symbols):
        h_replacements[symbol_name] = f"pi[{i}]"

    scalar_replacements = dict(parameter_replacements)
    if scalar_argument_name is not None:
        scalar_replacements[scalar_argument_name] = scalar_argument_name
    for i, input_argument_name in enumerate(input_argument_names):
        scalar_replacements[input_argument_name] = f"inputs[{i}]"

    if scalar_argument_name is not None:
        if input_argument_names:
            h_signature = (
                f"    constexpr double H(double {scalar_argument_name}, "
                f"std::span<double const, {len(input_argument_names)}> inputs, "
                "std::span<double const, N> pi) const"
            )
        else:
            h_signature = (
                f"    constexpr double H(double {scalar_argument_name}, std::span<double const, N> pi) const"
            )
    else:
        h_signature = f"    constexpr double H(std::span<double const, N> pi) const"

    scalar_method = ""
    if scalar_argument_name is not None and scalar_derivative_expression is not None:
        scalar_signature = f"double {scalar_argument_name}"
        if input_argument_names:
            scalar_signature += f", std::span<double const, {len(input_argument_names)}> inputs"
        scalar_method = f"""
    constexpr double dH_dphi({scalar_signature}) const
    {{
        return {_replace_symbols(cxxcode(scalar_derivative_expression), scalar_replacements)};
    }}
"""

    inverse_methods = ""
    if inverse_symbols is not None and inverse_expressions is not None:
        inverse_methods = _render_indexed_method(
            "pi",
            "dphi_dx",
            inverse_symbols,
            inverse_expressions,
            scalar_replacements,
        )

    nonlocal_value_methods = ""
    scalar_nonlocal_value_method = ""
    if nonlocal_pi_value_methods:
        nonlocal_value_methods = _render_indexed_nonlocal_value_method(
            "dH_dpi_value",
            derivative_symbols,
            derivative_expressions,
            scalar_replacements,
        )
        if scalar_argument_name is not None and scalar_derivative_expression is not None:
            scalar_value_components = []
            for input_argument_name in input_argument_names:
                scalar_value_components.append(
                    _replace_symbols(
                        cxxcode(diff(scalar_derivative_expression, symbols(input_argument_name))),
                        scalar_replacements,
                    )
                )
            scalar_value_return_type = (
                f"std::array<double, {len(input_argument_names)}>"
                if input_argument_names
                else "double"
            )
            scalar_value_return = (
                "{"
                + ", ".join(scalar_value_components)
                + "}"
                if input_argument_names
                else _replace_symbols(
                    cxxcode(diff(scalar_derivative_expression, symbols(scalar_argument_name))),
                    scalar_replacements,
                )
            )
            scalar_nonlocal_value_method = f"""
    template <class ChainType, class LowerChainType, class Elem, class MomentsComputerValue>
    constexpr {scalar_value_return_type} dH_dphi_value(
            [[maybe_unused]] ChainType chain,
            [[maybe_unused]] LowerChainType lower_chain,
            [[maybe_unused]] Elem elem,
            [[maybe_unused]] MomentsComputerValue const& moments_computer_value) const
    {{
        return {scalar_value_return};
    }}
"""

    rendered_includes = ""
    if includes:
        rendered_includes = "".join(f"#include {header}\n" for header in includes)

    rendered_value_computer_type = ""
    if value_computer_type is not None:
        rendered_value_computer_type = (
            f"    using value_computer_type = {value_computer_type};\n\n"
        )

    output_path.write_text(
        f"""\
// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <span>
{rendered_includes}

namespace {namespace} {{

struct {struct_name} {{
    static constexpr std::size_t N = {len(derivative_symbols)};

{rendered_value_computer_type}\
{_render_members(parameters)}

{_render_constructor_signature(struct_name, parameters)}
        : {_render_constructor_initializers(parameters)} {{}}

{h_signature}
    {{
        return {_replace_symbols(cxxcode(hamiltonian), h_replacements)};
    }}
{scalar_method}{scalar_nonlocal_value_method}{_render_indexed_method("dH_dpi", "pi", derivative_symbols, derivative_expressions, scalar_replacements)}{nonlocal_value_methods}
{inverse_methods}}};
}} // namespace {namespace}
"""
    )


def generate_cpp_hamiltonian(functor_class, output_path: Path, *args, **kwargs) -> None:
    definition = functor_class.__call__(*args, **kwargs)
    parameter_tuples = [(f"m_{name}", name, True) for name in definition.parameters]

    scalar_argument_name = None
    input_argument_names: list[str] = []
    scalar_derivative_expression = None
    inverse_symbols = None
    inverse_expressions = None
    derivative_state_variables = _flatten_variable_entries(list(definition.variables))

    if derivative_state_variables:
        scalar_state_variable = derivative_state_variables.pop(0)
        scalar_argument_name = str(scalar_state_variable)
        scalar_derivative_expression = diff(definition.hamiltonian, scalar_state_variable)

    input_variables = (
        []
        if definition.input_variables is None
        else _flatten_variable_entries(list(definition.input_variables))
    )
    input_argument_names = [str(symbol) for symbol in input_variables]
    if input_variables:
        derivative_state_variables = [
            symbol for symbol in derivative_state_variables if symbol not in input_variables
        ]

    derivative_symbols = [str(symbol) for symbol in derivative_state_variables]
    derivative_expressions = [
        diff(definition.hamiltonian, symbol) for symbol in derivative_state_variables
    ]

    if scalar_argument_name == "phi":
        dphi_dx_symbols = symbols(f"dphi_dx0:{len(derivative_state_variables)}")
        inverse_solution = solve(
            [
                dphi_dx_symbols[i] - derivative_expressions[i]
                for i in range(len(derivative_state_variables))
            ],
            derivative_state_variables,
            dict=True,
        )
        if inverse_solution:
            inverse_symbols = [str(symbol) for symbol in dphi_dx_symbols]
            inverse_expressions = [
                inverse_solution[0][derivative_state_variables[i]]
                for i in range(len(derivative_state_variables))
            ]

    write_cpp_hamiltonian_header(
        output_path=output_path,
        namespace=definition.namespace,
        struct_name=definition.struct_name,
        parameters=parameter_tuples,
        hamiltonian=definition.hamiltonian,
        derivative_symbols=derivative_symbols,
        derivative_expressions=derivative_expressions,
        includes=definition.includes,
        value_computer_type=definition.value_computer_type,
        scalar_argument_name=scalar_argument_name,
        input_argument_names=input_argument_names,
        scalar_derivative_expression=scalar_derivative_expression,
        inverse_symbols=inverse_symbols,
        inverse_expressions=inverse_expressions,
        nonlocal_pi_value_methods=(scalar_argument_name is not None and scalar_argument_name != "phi"),
    )
