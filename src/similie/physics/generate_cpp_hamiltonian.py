#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Baptiste Legouix
# SPDX-License-Identifier: AGPL-3.0-or-later
# AI-GENERATED

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sympy import Derivative, Subs, Symbol, diff, powsimp, solve, symbols
from sympy.printing.codeprinter import cxxcode


@dataclass(frozen=True)
class SymbolicFunctionDefinition:
    value_expression: str
    derivative_expressions: dict[int, str]


@dataclass(frozen=True)
class HamiltonianDefinition:
    namespace: str
    struct_name: str
    parameters: list[str]
    hamiltonian: object
    variables: list[object]
    includes: list[str] | None = None
    moments_computer: str | None = None
    template_parameters: list[str] | None = None
    parameter_types: dict[str, str] | None = None
    parameter_value_expressions: dict[str, str] | None = None
    symbolic_functions: dict[str, SymbolicFunctionDefinition] | None = None
    namespace_preamble: str = ""
    namespace_epilogue: str = ""
    is_linear: bool = False


def _flatten_variable_entries(entries: list[object]) -> list[object]:
    flattened: list[object] = []
    for entry in entries:
        if isinstance(entry, (list, tuple)):
            flattened.extend(_flatten_variable_entries(list(entry)))
        else:
            flattened.append(entry)
    return flattened


def _entry_symbols(entry: object) -> list[object]:
    if isinstance(entry, (list, tuple)):
        return _flatten_variable_entries(list(entry))
    return [entry]


def _entry_name(entry: object) -> str:
    symbols_ = _entry_symbols(entry)
    if len(symbols_) == 1:
        return str(symbols_[0])
    first_name = str(symbols_[0])
    prefix = first_name.rstrip("0123456789")
    return prefix if prefix else first_name


def _replace_symbols(expression: str, replacements: dict[str, str]) -> str:
    for source, target in sorted(
        replacements.items(), key=lambda replacement: len(replacement[0]), reverse=True
    ):
        expression = expression.replace(source, target)
    return expression


def _render_cxx_expression(
    expression,
    replacements: dict[str, str],
    symbolic_functions: dict[str, SymbolicFunctionDefinition] | None = None,
    expression_replacements: dict[object, str] | None = None,
) -> str:
    symbolic_functions = symbolic_functions or {}
    expression_replacements = expression_replacements or {}
    placeholder_replacements: dict[str, str] = {}

    def render_subexpression(subexpression) -> str:
        for source_expression, target_expression in expression_replacements.items():
            if subexpression == source_expression:
                return target_expression
        return _render_cxx_expression(
            subexpression,
            replacements,
            symbolic_functions,
            expression_replacements,
        )

    def placeholder(rendered_expression: str) -> Symbol:
        symbol = Symbol(f"__similie_symbolic_function_{len(placeholder_replacements)}")
        placeholder_replacements[str(symbol)] = rendered_expression
        return symbol

    for function_name, function_definition in symbolic_functions.items():
        expression = expression.replace(
            lambda node, name=function_name: (
                isinstance(node, Subs)
                and isinstance(node.expr, Derivative)
                and node.expr.expr.func.__name__ == name
            ),
            lambda node, definition=function_definition: placeholder(
                definition.derivative_expressions[node.expr.derivative_count].format(
                    argument=render_subexpression(node.point[0])
                )
            ),
        )
        expression = expression.replace(
            lambda node, name=function_name: node.func.__name__ == name,
            lambda node, definition=function_definition: placeholder(
                definition.value_expression.format(
                    argument=render_subexpression(node.args[0])
                )
            ),
        )

    for source_expression, target_expression in expression_replacements.items():
        expression = expression.replace(
            lambda node, source=source_expression: node == source,
            lambda node, target=target_expression: placeholder(target),
        )

    return _replace_symbols(
        _replace_symbols(cxxcode(expression), replacements),
        placeholder_replacements,
    )


def _render_members(parameters: list[tuple[str, str, bool, str]]) -> str:
    return "\n".join(
        f"    {'const ' if is_const else ''}{type_name} {name};"
        for name, _, is_const, type_name in parameters
    )


def _render_constructor_signature(
    struct_name: str, parameters: list[tuple[str, str, bool, str]]
) -> str:
    params = ",\n            ".join(
        f"{type_name} {constructor_name}_"
        for _, constructor_name, _, type_name in parameters
    )
    return f"    KOKKOS_FUNCTION constexpr {struct_name}(\n            {params})"


def _render_constructor_initializers(
    parameters: list[tuple[str, str, bool, str]],
) -> str:
    return ", ".join(
        f"{name}({constructor_name}_)" for name, constructor_name, _, _ in parameters
    )


def _render_state_component_method(
    name: str,
    expressions: list,
    replacements: dict[str, str],
    symbolic_functions: dict[str, SymbolicFunctionDefinition],
) -> str:
    cases = "\n".join(
        f"""        {"if" if i == 0 else "else if"} constexpr (Index::value == {i}) {{
            return {_render_cxx_expression(expression, replacements, symbolic_functions)};
        }}"""
        for i, expression in enumerate(expressions)
    )
    return f"""
    template <class Index, class Elem>
    [[nodiscard]] KOKKOS_FUNCTION constexpr double {name}(State const& state, Elem elem) const
    {{
        static_cast<void>(state);
        static_cast<void>(elem);
{cases}
        else {{
            static_assert(Index::value < {len(expressions)}, "unsupported component index");
            return 0.0;
        }}
    }}
"""


def write_cpp_hamiltonian_header(
    output_path: Path,
    namespace: str,
    struct_name: str,
    parameters: list[tuple[str, str, bool, str]],
    hamiltonian,
    variable_entries: list[dict[str, object]],
    includes: list[str] | None = None,
    moments_computer: str | None = None,
    inverse_symbols: list[str] | None = None,
    inverse_expressions: list | None = None,
    template_parameters: list[str] | None = None,
    parameter_value_expressions: dict[str, str] | None = None,
    definition: HamiltonianDefinition | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    parameter_value_expressions = parameter_value_expressions or {}
    symbolic_functions = (
        {} if definition is None else (definition.symbolic_functions or {})
    )
    replacements = {
        constructor_name: (
            f"({parameter_value_expressions[constructor_name]})"
            if constructor_name in parameter_value_expressions
            else member_name
        )
        for member_name, constructor_name, _, _ in parameters
    }
    state_members = []
    for i, entry in enumerate(variable_entries):
        field_name = "potential" if i == 0 else "moments" if i == 1 else entry["name"]
        state_members.append(
            f"        std::array<double, {len(entry['symbols'])}> {field_name} {{}};"
        )
        replacements.update(
            {
                str(symbol): f"state.{field_name}[{component}]"
                for component, symbol in enumerate(entry["symbols"])
            }
        )
    potential_symbols = variable_entries[0]["symbols"]
    moment_symbols = variable_entries[1]["symbols"]
    potential_method = _render_state_component_method(
        "dhamiltonian_dpotential",
        [powsimp(diff(hamiltonian, symbol)) for symbol in potential_symbols],
        replacements,
        symbolic_functions,
    )
    moment_method = _render_state_component_method(
        "dhamiltonian_dmoments",
        [powsimp(diff(hamiltonian, symbol)) for symbol in moment_symbols],
        replacements,
        symbolic_functions,
    )
    inverse_method = ""
    if inverse_expressions is not None:
        inverse_replacements = {
            **replacements,
            **{
                symbol: f"state.moments[{i}]"
                for i, symbol in enumerate(inverse_symbols)
            },
        }
        inverse_method = _render_state_component_method(
            "moments",
            inverse_expressions,
            inverse_replacements,
            symbolic_functions,
        )
    jacobian_cases = []
    for row, row_symbol in enumerate(moment_symbols):
        for column, column_symbol in enumerate(moment_symbols):
            expression = diff(diff(hamiltonian, row_symbol), column_symbol)
            jacobian_cases.append(
                f"""        {"if" if not jacobian_cases else "else if"} constexpr (RowIndex::value == {row} && ColumnIndex::value == {column}) {{
            return {_render_cxx_expression(expression, replacements, symbolic_functions)};
        }}"""
            )
    nonlocal_method = ""
    if moments_computer is not None:
        coefficients = [
            diff(diff(hamiltonian, symbol), symbol) for symbol in moment_symbols
        ]
        coefficient_method = _render_state_component_method(
            "moment_coefficient",
            coefficients,
            replacements,
            symbolic_functions,
        )
        nonlocal_method = f"""
    template <class Index, class Elem>
    [[nodiscard]] KOKKOS_FUNCTION constexpr auto dhamiltonian_dmoments_value(State const& state, Elem elem) const
    {{
        using Component = ddc::type_seq_element_t<Index::value, ddc::TypeSeq<SpatialIndex...>>;
        auto value = MomentsComputer::template forward_value<Component>(elem);
        value *= moment_coefficient<Index>(state, elem);
        return value;
    }}
"""
    else:
        coefficient_method = ""
    private_methods = coefficient_method
    template_prefix = (
        ""
        if not template_parameters
        else "template <" + ", ".join(template_parameters) + ">\n"
    )
    rendered_includes = "".join(f"#include {header}\n" for header in includes or [])
    initializers = _render_constructor_initializers(parameters)
    constructor = _render_constructor_signature(struct_name, parameters)
    if initializers:
        constructor += f"\n        : {initializers} {{}}"
    else:
        constructor += " {}"
    concept_type = struct_name
    concept_arguments = []
    spatial_arguments = []
    parameter_types = (
        set()
        if definition is None
        else set((definition.parameter_types or {}).values())
    )
    for i, parameter in enumerate(template_parameters or []):
        if parameter.split()[-1] in parameter_types:
            concept_arguments.append("similie::physics::detail::ConceptParameter")
        elif "..." in parameter:
            count = 2 if moments_computer is not None else 1
            spatial_arguments = [
                f"std::integral_constant<std::size_t, {i + j + 1}>"
                for j in range(count)
            ]
            concept_arguments.extend(spatial_arguments)
        else:
            concept_arguments.append(f"std::integral_constant<std::size_t, {i}>")
    if template_parameters:
        concept_type += "<" + ", ".join(concept_arguments) + ">"
    checks = f"static_assert(similie::physics::HamiltonianConcept<{concept_type}, similie::physics::detail::ConceptElement>);\n"
    if nonlocal_method and spatial_arguments:
        grid_arguments = ", ".join(
            "similie::physics::detail::ConceptGrid<" + arg + ">"
            for arg in spatial_arguments
        )
        checks += f"static_assert(similie::physics::NonlocalHamiltonianConcept<{concept_type}, ddc::DiscreteElement<{grid_arguments}>>);\n"
    output_path.write_text(
        f"""\
// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <span>
#include <type_traits>
#include <Kokkos_Core.hpp>
#include <similie/physics/concepts.hpp>
{rendered_includes}

namespace {namespace} {{

{"" if definition is None else definition.namespace_preamble}

{template_prefix}class {struct_name}
{{
{_render_members(parameters)}
{"" if moments_computer is None else "    using MomentsComputer = " + moments_computer + ";"}

public:
    static constexpr std::size_t N = {len(moment_symbols)};
    static constexpr bool IS_LINEAR = {"true" if definition is not None and definition.is_linear else "false"};

    struct State
    {{
{chr(10).join(state_members)}
    }};

{constructor}

    template <class Elem>
    [[nodiscard]] KOKKOS_FUNCTION constexpr double hamiltonian(State const& state, Elem elem) const
    {{
        static_cast<void>(elem);
        return {_render_cxx_expression(hamiltonian, replacements, symbolic_functions)};
    }}
{potential_method}{moment_method}{inverse_method}
    template <class RowIndex, class ColumnIndex, class Elem>
    [[nodiscard]] KOKKOS_FUNCTION constexpr double jacobian(State const& state, Elem elem) const
    {{
        static_cast<void>(state);
        static_cast<void>(elem);
{chr(10).join(jacobian_cases)}
        else {{
            static_assert(RowIndex::value < N && ColumnIndex::value < N, "unsupported moment indices");
            return 0.0;
        }}
    }}
{nonlocal_method}
private:
{private_methods}
}};

{checks}
{"" if definition is None else definition.namespace_epilogue}
}} // namespace {namespace}
"""
    )


def generate_cpp_hamiltonian(functor_class, output_path: Path, *args, **kwargs) -> None:
    definition = functor_class.__call__(*args, **kwargs)
    parameter_types = definition.parameter_types or {}
    parameter_tuples = [
        (f"m_{name}", name, True, parameter_types.get(name, "double"))
        for name in definition.parameters
    ]

    inverse_symbols = None
    inverse_expressions = None
    variable_entries = [
        {"name": _entry_name(entry), "symbols": _entry_symbols(entry)}
        for entry in definition.variables
    ]

    flat_variables = _flatten_variable_entries(list(definition.variables))
    if variable_entries[0]["name"] == "phi":
        dphi_dx_symbols = symbols(f"dphi_dx0:{len(variable_entries[1]['symbols'])}")
        inverse_solution = solve(
            [
                dphi_dx_symbols[i]
                - diff(definition.hamiltonian, variable_entries[1]["symbols"][i])
                for i in range(len(variable_entries[1]["symbols"]))
            ],
            variable_entries[1]["symbols"],
            dict=True,
        )
        if inverse_solution:
            inverse_symbols = [str(symbol) for symbol in dphi_dx_symbols]
            inverse_expressions = [
                inverse_solution[0][variable_entries[1]["symbols"][i]]
                for i in range(len(variable_entries[1]["symbols"]))
            ]

    write_cpp_hamiltonian_header(
        output_path=output_path,
        namespace=definition.namespace,
        struct_name=definition.struct_name,
        parameters=parameter_tuples,
        hamiltonian=definition.hamiltonian,
        variable_entries=variable_entries,
        includes=definition.includes,
        moments_computer=definition.moments_computer,
        inverse_symbols=inverse_symbols,
        inverse_expressions=inverse_expressions,
        template_parameters=definition.template_parameters,
        parameter_value_expressions=definition.parameter_value_expressions,
        definition=definition,
    )
