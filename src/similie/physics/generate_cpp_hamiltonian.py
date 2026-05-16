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


def write_cpp_hamiltonian_header(
    output_path: Path,
    namespace: str,
    struct_name: str,
    parameters: list[tuple[str, str, bool]],
    hamiltonian,
    derivative_symbols: list[str],
    derivative_expressions: list,
    scalar_argument_name: str | None = None,
    scalar_derivative_expression=None,
    inverse_symbols: list[str] | None = None,
    inverse_expressions: list | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    parameter_replacements = {
        constructor_name: member_name for member_name, constructor_name, _ in parameters
    }
    h_replacements = dict(parameter_replacements)
    for i, symbol_name in enumerate(derivative_symbols):
        h_replacements[symbol_name] = f"pi[{i}]"

    scalar_replacements = dict(parameter_replacements)
    if scalar_argument_name is not None:
        scalar_replacements[scalar_argument_name] = scalar_argument_name

    h_signature = (
        f"    constexpr double H(double {scalar_argument_name}, std::span<double const, N> pi) const"
        if scalar_argument_name is not None
        else f"    constexpr double H(std::span<double const, N> pi) const"
    )

    scalar_method = ""
    if scalar_argument_name is not None and scalar_derivative_expression is not None:
        scalar_method = f"""
    constexpr double dH_dphi(double {scalar_argument_name}) const
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

    output_path.write_text(
        f"""\
// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: MIT

#pragma once

#include <cmath>
#include <cstddef>
#include <span>

namespace {namespace} {{

struct {struct_name} {{
    static constexpr std::size_t N = {len(derivative_symbols)};

{_render_members(parameters)}

{_render_constructor_signature(struct_name, parameters)}
        : {_render_constructor_initializers(parameters)} {{}}

{h_signature}
    {{
        return {_replace_symbols(cxxcode(hamiltonian), h_replacements)};
    }}
{scalar_method}{_render_indexed_method("dH_dpi", "pi", derivative_symbols, derivative_expressions, scalar_replacements)}
{inverse_methods}}};
}} // namespace {namespace}
"""
    )


def generate_cpp_hamiltonian(functor_class, output_path: Path, *args, **kwargs) -> None:
    definition = functor_class.__call__(*args, **kwargs)
    parameter_tuples = [(f"m_{name}", name, True) for name in definition.parameters]

    scalar_argument_name = None
    scalar_derivative_expression = None
    inverse_symbols = None
    inverse_expressions = None
    derivative_state_variables = list(definition.variables)

    if derivative_state_variables and str(derivative_state_variables[0]) == "phi":
        scalar_state_variable = derivative_state_variables.pop(0)
        scalar_argument_name = str(scalar_state_variable)
        scalar_derivative_expression = diff(definition.hamiltonian, scalar_state_variable)

    derivative_symbols = [str(symbol) for symbol in derivative_state_variables]
    derivative_expressions = [
        diff(definition.hamiltonian, symbol) for symbol in derivative_state_variables
    ]

    if scalar_argument_name is not None:
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
        scalar_argument_name=scalar_argument_name,
        scalar_derivative_expression=scalar_derivative_expression,
        inverse_symbols=inverse_symbols,
        inverse_expressions=inverse_expressions,
    )
