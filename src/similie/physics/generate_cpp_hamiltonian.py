#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Baptiste Legouix
# SPDX-License-Identifier: MIT

from __future__ import annotations

from pathlib import Path

from sympy.printing.codeprinter import cxxcode


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
    return (
        f"    constexpr {struct_name}(\n"
        f"            {params})"
    )


def _render_constructor_initializers(parameters: list[tuple[str, str, bool]]) -> str:
    return ", ".join(f"{name}({constructor_name}_)" for name, constructor_name, _ in parameters)


def _render_indexed_method(
    method_name: str,
    argument_prefix: str,
    symbols: list[str],
    expressions: list,
    replacements: dict[str, str],
) -> str:
    branches: list[str] = []
    for i, (symbol_name, expression) in enumerate(zip(symbols, expressions, strict=True)):
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
    h_expression,
    derivative_symbols: list[str],
    derivative_expressions: list,
    array_argument_name: str,
    scalar_argument_name: str | None = None,
    scalar_derivative_expression=None,
    inverse_symbols: list[str] | None = None,
    inverse_expressions: list | None = None,
    aliases: list[str] | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    parameter_replacements = {name: name for name, _, _ in parameters}
    h_replacements = dict(parameter_replacements)
    for i, symbol_name in enumerate(derivative_symbols):
        h_replacements[symbol_name] = f"{array_argument_name}[{i}]"

    scalar_replacements = dict(parameter_replacements)
    if scalar_argument_name is not None:
        scalar_replacements[scalar_argument_name] = scalar_argument_name

    h_signature = (
        f"    constexpr double H(double {scalar_argument_name}, std::span<double const, N> {array_argument_name}) const"
        if scalar_argument_name is not None
        else f"    constexpr double H(std::span<double const, N> {array_argument_name}) const"
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

    alias_block = ""
    if aliases:
        alias_block = "\n".join(f"using {alias} = {struct_name};" for alias in aliases)
        alias_block = f"\n{alias_block}\n"

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
        return {_replace_symbols(cxxcode(h_expression), h_replacements)};
    }}
{scalar_method}{_render_indexed_method("dH_dpi", "pi", derivative_symbols, derivative_expressions, scalar_replacements)}
{inverse_methods}}};
{alias_block}
}} // namespace {namespace}
"""
    )


def generate_cpp_hamiltonian(functor_class, output_path: Path, *args, **kwargs) -> None:
    definition = functor_class.__call__(*args, **kwargs)
    write_cpp_hamiltonian_header(
        output_path=output_path,
        namespace=definition.namespace,
        struct_name=definition.struct_name,
        parameters=definition.parameters,
        h_expression=definition.h_expression,
        derivative_symbols=definition.derivative_symbols,
        derivative_expressions=definition.derivative_expressions,
        array_argument_name=definition.array_argument_name,
        scalar_argument_name=definition.scalar_argument_name,
        scalar_derivative_expression=definition.scalar_derivative_expression,
        inverse_symbols=definition.inverse_symbols,
        inverse_expressions=definition.inverse_expressions,
        aliases=definition.aliases,
    )
