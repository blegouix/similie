#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Baptiste Legouix
# SPDX-License-Identifier: AGPL-3.0-or-later
# AI-GENERATED

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
    output_symbol = symbols(output_variable)
    inverse_solution = solve(
        output_symbol - constitutive_law, state_variable_symbol, dict=True
    )
    if not inverse_solution:
        raise ValueError("Unable to invert constitutive law")
    inverse_expression = inverse_solution[0][state_variable_symbol]
    forward_value_expression = diff(constitutive_law, state_variable_symbol)
    inverse_value_expression = diff(inverse_expression, output_symbol)

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
    context_members = "\n".join(
        f"        double {name} = 0.0;" for name in variables[:-1]
    )

    def render(expression, inverse: bool = False) -> str:
        argument_symbols = {
            constitutive_law_symbols[name]: symbols(f"__state_{i}")
            for i, name in enumerate(variables)
        }
        argument_replacements = {
            f"__state_{i}": f"state.{name}" for i, name in enumerate(variables[:-1])
        }
        argument_replacements[f"__state_{len(variables) - 1}"] = "state.value[0]"
        if inverse:
            argument_symbols[output_symbol] = symbols("__output")
            argument_replacements["__output"] = "state.value[0]"
        return _replace_symbols(
            _replace_symbols(
                cxxcode(expression.xreplace(argument_symbols)), parameter_replacements
            ),
            argument_replacements,
        )

    output_path.write_text(
        f"""\
// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#pragma once

#include <array>
#include <cstddef>
#include <Kokkos_Core.hpp>
#include <similie/physics/concepts.hpp>

namespace {namespace} {{

class {class_name}
{{
{parameter_members}

public:
    static constexpr std::size_t N = 1;
    struct State
    {{
        std::array<double, N> value {{}};
{context_members}
    }};

    KOKKOS_FUNCTION constexpr explicit {class_name}(
            {constructor_signature})
        : {constructor_initializers}
    {{
    }}

    template <class Elem>
    [[nodiscard]] KOKKOS_FUNCTION constexpr std::array<double, N * N> value(State const& state, Elem elem) const
    {{
        static_cast<void>(elem);
        return {{{render(forward_value_expression)}}};
    }}

    template <class Elem>
    [[nodiscard]] KOKKOS_FUNCTION constexpr std::array<double, N * N> jacobian(State const& state, Elem elem) const
    {{
        return value(state, elem);
    }}

    template <class Elem>
    [[nodiscard]] KOKKOS_FUNCTION constexpr std::array<double, N> operator()(State const& state, Elem elem) const
    {{
        static_cast<void>(elem);
        return {{{render(constitutive_law)}}};
    }}

    template <class Elem>
    [[nodiscard]] KOKKOS_FUNCTION constexpr std::array<double, N * N> inverse_value(State const& state, Elem elem) const
    {{
        static_cast<void>(elem);
        return {{{render(inverse_value_expression, True)}}};
    }}

    template <class Elem>
    [[nodiscard]] KOKKOS_FUNCTION constexpr std::array<double, N> inverse(State const& state, Elem elem) const
    {{
        static_cast<void>(elem);
        return {{{render(inverse_expression, True)}}};
    }}
}};

static_assert(similie::physics::ConstitutiveLawConcept<{class_name}, similie::physics::detail::ConceptElement>);

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
