#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Baptiste Legouix
# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sympy import diff, solve, symbols
from sympy.printing.codeprinter import cxxcode


@dataclass(frozen=True)
class ConstitutiveLawDefinition:
    namespace: str
    class_name: str
    parameters: list[str]
    variables: list[str]
    output_variable: str
    constitutive_law: object


def _replace_symbols(expression: str, replacements: dict[str, str]) -> str:
    for source, target in replacements.items():
        expression = expression.replace(source, target)
    return expression


def _render_expression(
    expression, symbol, rendered_symbol: str, replacements: dict[str, str]
) -> str:
    return _replace_symbols(
        cxxcode(expression.subs(symbol, symbols(rendered_symbol))), replacements
    )


def write_cpp_constitutive_law_header(
    output_path: Path,
    namespace: str,
    class_name: str,
    parameters: list[tuple[str, str, bool]],
    variables: list[str],
    output_variable: str,
    constitutive_law,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    parameter_replacements = {
        constructor_name: member_name for member_name, constructor_name, _ in parameters
    }
    constitutive_law_symbols = {
        str(symbol): symbol
        for symbol in constitutive_law.free_symbols
        if str(symbol) not in parameter_replacements
    }
    if not variables:
        raise ValueError("Constitutive law must define at least one variable")
    if any(name not in constitutive_law_symbols for name in variables):
        raise ValueError("Constitutive law variables must match symbolic variables")

    state_variable_name = variables[-1]
    state_variable_symbol = constitutive_law_symbols[state_variable_name]
    constitutive_law_replacements = dict(parameter_replacements)
    output_symbol = symbols(output_variable)
    inverse_solution = solve(
        output_symbol - constitutive_law, state_variable_symbol, dict=True
    )
    if not inverse_solution:
        raise ValueError("Unable to invert constitutive law")
    inverse_expression = inverse_solution[0][state_variable_symbol]
    forward_value_expression = diff(constitutive_law, state_variable_symbol)
    inverse_value_expression = diff(inverse_expression, output_symbol)
    inverse_replacements = dict(parameter_replacements)

    parameter_members = "\n".join(
        f"    {'const ' if is_const else ''}double {member_name};"
        for member_name, _, is_const in parameters
    )
    constructor_signature = ",\n            ".join(
        f"double {constructor_name}_" for _, constructor_name, _ in parameters
    )
    constructor_initializers = ", ".join(
        f"{member_name}({constructor_name}_)"
        for member_name, constructor_name, _ in parameters
    )
    forward_arguments = ", ".join(f"double {name}" for name in variables)
    inverse_output_name = state_variable_name
    inverse_arguments = ", ".join(
        [
            *[f"double {name}" for name in variables if name != inverse_output_name],
            f"double {output_variable}",
        ]
    )

    output_path.write_text(
        f"""\
// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: MIT

#pragma once

#include <Kokkos_Core.hpp>

namespace {namespace} {{

class {class_name}
{{
{parameter_members}

public:
    constexpr explicit {class_name}(
            {constructor_signature})
        : {constructor_initializers}
    {{
    }}

    KOKKOS_FUNCTION constexpr double value({forward_arguments}) const
    {{
        return {_render_expression(forward_value_expression, state_variable_symbol, inverse_output_name, constitutive_law_replacements)};
    }}

    KOKKOS_FUNCTION constexpr double operator()({forward_arguments}) const
    {{
        return {_render_expression(constitutive_law, state_variable_symbol, inverse_output_name, constitutive_law_replacements)};
    }}

    KOKKOS_FUNCTION constexpr double inverse_value({inverse_arguments}) const
    {{
        return {_render_expression(inverse_value_expression, output_symbol, output_variable, inverse_replacements)};
    }}

    KOKKOS_FUNCTION constexpr double inverse({inverse_arguments}) const
    {{
        return {_render_expression(inverse_expression, output_symbol, output_variable, inverse_replacements)};
    }}
}};

}} // namespace {namespace}
"""
    )


def generate_cpp_constitutive_law(
    functor_class, output_path: Path, *args, **kwargs
) -> None:
    definition = functor_class.__call__(*args, **kwargs)
    parameter_tuples = [(f"m_{name}", name, True) for name in definition.parameters]
    write_cpp_constitutive_law_header(
        output_path=output_path,
        namespace=definition.namespace,
        class_name=definition.class_name,
        parameters=parameter_tuples,
        variables=definition.variables,
        output_variable=definition.output_variable,
        constitutive_law=definition.constitutive_law,
    )
