#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Baptiste Legouix
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sympy import Derivative, Subs, Symbol, diff, solve, symbols
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
    moments_object_component_expression: str | None = None
    moments_object_norm2_expression: str | None = None
    generate_moments_jacobian: bool = False
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


def _generalize_component_expression(
    symbol_name: str,
    expression,
    replacements: dict[str, str],
    component_argument_name: str,
    symbolic_functions: dict[str, SymbolicFunctionDefinition] | None = None,
    expression_replacements: dict[object, str] | None = None,
) -> str:
    return _render_cxx_expression(
        expression,
        {**replacements, symbol_name: component_argument_name},
        symbolic_functions,
        expression_replacements,
    )


def _all_same(expressions: list[str]) -> bool:
    return all(expression == expressions[0] for expression in expressions)


def _has_temporal_index(template_parameters: list[str] | None) -> bool:
    return (
        template_parameters is not None and "class TemporalIndex" in template_parameters
    )


def _render_indexed_method(
    method_name: str,
    argument_prefix: str,
    symbols_: list[str],
    expressions: list,
    replacements: dict[str, str],
    template_parameters: list[str] | None,
    symbolic_functions: dict[str, SymbolicFunctionDefinition] | None = None,
    expression_replacements: dict[object, str] | None = None,
) -> str:
    component_expressions = [
        _generalize_component_expression(
            symbol_name,
            expression,
            replacements,
            argument_prefix,
            symbolic_functions,
            expression_replacements,
        )
        for symbol_name, expression in zip(symbols_, expressions, strict=True)
    ]
    if _all_same(component_expressions):
        body = f"        return {component_expressions[0]};"
    elif _has_temporal_index(template_parameters):
        body = f"""        if constexpr (std::is_same_v<Index, TemporalIndex>) {{
            return {component_expressions[0]};
        }} else {{
            return {component_expressions[1]};
        }}"""
    else:
        raise ValueError(
            f"{method_name} cannot be generated from a scalar component without tag-specific "
            "expressions"
        )

    return f"""
    template <class Index>
    KOKKOS_FUNCTION constexpr double {method_name}(double {argument_prefix}) const
    {{
{body}
    }}
"""


def _render_indexed_elem_method(
    method_name: str,
    argument_prefix: str,
    symbols_: list[str],
    expressions: list,
    replacements: dict[str, str],
    template_parameters: list[str] | None,
    symbolic_functions: dict[str, SymbolicFunctionDefinition] | None = None,
    expression_replacements: dict[object, str] | None = None,
) -> str:
    component_expressions = [
        _generalize_component_expression(
            symbol_name,
            expression,
            replacements,
            argument_prefix,
            symbolic_functions,
            expression_replacements,
        )
        for symbol_name, expression in zip(symbols_, expressions, strict=True)
    ]
    if _all_same(component_expressions):
        body = f"        return {component_expressions[0]};"
    elif _has_temporal_index(template_parameters):
        body = f"""        if constexpr (std::is_same_v<Index, TemporalIndex>) {{
            return {component_expressions[0]};
        }} else {{
            return {component_expressions[1]};
        }}"""
    else:
        raise ValueError(
            f"{method_name} cannot be generated from a scalar component without tag-specific "
            "expressions"
        )

    return f"""
    template <class Index, class Elem>
    KOKKOS_FUNCTION constexpr double {method_name}(double {argument_prefix}, Elem elem) const
    {{
{body}
    }}
"""


def _render_indexed_nonlocal_value_method(
    method_name: str,
    symbols_: list[str],
    expressions: list,
    replacements: dict[str, str],
    symbolic_functions: dict[str, SymbolicFunctionDefinition] | None = None,
) -> str:
    differentiated_expressions = [
        diff(expression, symbols(symbol_name))
        for symbol_name, expression in zip(symbols_, expressions, strict=True)
    ]
    generalized_expressions = [
        _render_cxx_expression(expression, replacements, symbolic_functions)
        for expression in differentiated_expressions
    ]
    if _all_same(generalized_expressions):
        body = f"""        auto value = MomentsComputer::template forward_value<Index>(elem);
        value *= {generalized_expressions[0]};
        return value;"""
    else:
        raise ValueError(
            f"{method_name} cannot be generated without tag-specific nonlocal expressions"
        )

    return f"""
    template <class Index, class Elem>
    KOKKOS_FUNCTION constexpr auto {method_name}(Elem elem) const
    {{
{body}
    }}
"""


def _render_moments_object_method(
    symbols_: list[str],
    expressions: list,
    replacements: dict[str, str],
    symbolic_functions: dict[str, SymbolicFunctionDefinition],
    expression_replacements: dict[object, str],
    component_expression: str,
) -> str:
    component_expressions = [
        _generalize_component_expression(
            symbol_name,
            expression,
            replacements,
            component_expression.format(index="Index", moments="moments"),
            symbolic_functions,
            expression_replacements,
        )
        for symbol_name, expression in zip(symbols_, expressions, strict=True)
    ]
    if not _all_same(component_expressions):
        raise ValueError(
            "dhamiltonian_dmoments object method cannot be generated without "
            "tag-independent component expressions"
        )

    return f"""
    template <class Index, class Moments, class Elem>
    KOKKOS_FUNCTION constexpr double dhamiltonian_dmoments(Moments moments, Elem elem) const
    {{
        static_cast<void>(elem);
        return {component_expressions[0]};
    }}

    template <class Index, class Moments, class Elem>
    KOKKOS_FUNCTION constexpr double dpotential_dt(Moments moments, Elem elem) const
    {{
        return dhamiltonian_dmoments<Index>(moments, elem);
    }}
"""


def _render_moments_object_jacobian_method(
    symbols_: list[str],
    hamiltonian,
    replacements: dict[str, str],
    symbolic_functions: dict[str, SymbolicFunctionDefinition],
    expression_replacements: dict[object, str],
    component_expression: str,
) -> str:
    row_component = component_expression.format(index="RowIndex", moments="moments")
    column_component = component_expression.format(
        index="ColumnIndex", moments="moments"
    )
    diagonal_expressions = [
        _render_cxx_expression(
            diff(diff(hamiltonian, symbol), symbol),
            {**replacements, str(symbol): row_component},
            symbolic_functions,
            expression_replacements,
        )
        for symbol in symbols_
    ]
    off_diagonal_expressions = [
        _render_cxx_expression(
            diff(diff(hamiltonian, symbols_[0]), symbol),
            {
                **replacements,
                str(symbols_[0]): row_component,
                str(symbol): column_component,
            },
            symbolic_functions,
            expression_replacements,
        )
        for symbol in symbols_[1:]
    ]
    if not _all_same(diagonal_expressions) or not _all_same(off_diagonal_expressions):
        raise ValueError(
            "moments jacobian cannot be generated without tag-independent expressions"
        )

    return f"""
    template <class RowIndex, class ColumnIndex, class Moments, class Elem>
    KOKKOS_FUNCTION constexpr double jacobian(Moments moments, Elem elem) const
    {{
        static_cast<void>(elem);
        if constexpr (std::is_same_v<RowIndex, ColumnIndex>) {{
            return {diagonal_expressions[0]};
        }} else {{
            return {off_diagonal_expressions[0]};
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
    parameter_replacements = {}
    for member_name, constructor_name, _, _ in parameters:
        parameter_replacements[constructor_name] = parameter_value_expressions.get(
            constructor_name, member_name
        )
    h_replacements = dict(parameter_replacements)
    requires_elem = any(
        "elem" in replacement for replacement in parameter_replacements.values()
    )
    argument_signature_parts: list[str] = []
    arguments_call_parts: list[str] = []
    for entry in variable_entries:
        entry_name = entry["name"]
        entry_symbols = entry["symbols"]
        if len(entry_symbols) == 1:
            argument_signature_parts.append(f"double {entry_name}")
            arguments_call_parts.append(entry_name)
            h_replacements[str(entry_symbols[0])] = entry_name
        else:
            argument_signature_parts.append(
                f"std::span<double const, {len(entry_symbols)}> {entry_name}"
            )
            arguments_call_parts.append(entry_name)
            for i, symbol in enumerate(entry_symbols):
                h_replacements[str(symbol)] = f"{entry_name}[{i}]"

    if requires_elem:
        h_signature = (
            f"    template <class Elem>\n"
            f"    KOKKOS_FUNCTION constexpr double hamiltonian({', '.join(argument_signature_parts)}, Elem elem) const"
        )
    else:
        h_signature = f"    KOKKOS_FUNCTION constexpr double hamiltonian({', '.join(argument_signature_parts)}) const"

    potential_entry = variable_entries[0]
    moments_entry = variable_entries[1]
    potential_symbols = potential_entry["symbols"]
    moments_symbols = moments_entry["symbols"]

    potential_replacements = dict(h_replacements)
    moments_replacements = dict(parameter_replacements)

    potential_derivative_expressions = [
        diff(hamiltonian, symbol) for symbol in potential_symbols
    ]
    moments_derivative_expressions = [
        diff(hamiltonian, symbol) for symbol in moments_symbols
    ]

    potential_argument_entries = [potential_entry, *variable_entries[2:]]
    potential_signature_parts: list[str] = []
    for entry in potential_argument_entries:
        entry_name = entry["name"]
        entry_symbols = entry["symbols"]
        if len(entry_symbols) == 1:
            potential_signature_parts.append(f"double {entry_name}")
        else:
            potential_signature_parts.append(
                f"std::span<double const, {len(entry_symbols)}> {entry_name}"
            )

    if len(potential_symbols) == 1:
        potential_method_signature = ", ".join(potential_signature_parts)
        if requires_elem:
            potential_method = f"""
    template <class Elem>
    KOKKOS_FUNCTION constexpr double dhamiltonian_dpotential({potential_method_signature}, Elem elem) const
    {{
        return {_render_cxx_expression(potential_derivative_expressions[0], potential_replacements, symbolic_functions)};
    }}
"""
        else:
            potential_method = f"""
    KOKKOS_FUNCTION constexpr double dhamiltonian_dpotential({potential_method_signature}) const
    {{
        return {_render_cxx_expression(potential_derivative_expressions[0], potential_replacements, symbolic_functions)};
    }}
"""
    else:
        potential_method = ""

    inverse_methods = ""
    if inverse_symbols is not None and inverse_expressions is not None:
        inverse_methods = _render_indexed_method(
            "moments",
            "dpotential_dx",
            inverse_symbols,
            inverse_expressions,
            potential_replacements,
            template_parameters,
            symbolic_functions,
        )

    use_moments_object = (
        len(moments_symbols) > 1
        and definition is not None
        and definition.moments_object_component_expression is not None
    )

    if not use_moments_object and (
        len(moments_symbols) == 1 or moments_computer is None
    ):
        moments_method = _render_indexed_method(
            "dhamiltonian_dmoments",
            "moments",
            [str(symbol) for symbol in moments_symbols],
            moments_derivative_expressions,
            moments_replacements,
            template_parameters,
            symbolic_functions,
        )
    else:
        moments_method = ""
    generic_moments_method = ""
    if len(moments_symbols) > 1 and moments_computer is not None:
        generic_moments_replacements = {
            **parameter_replacements,
            **{str(symbol): "moments" for symbol in moments_symbols},
        }
        if requires_elem:
            generic_moments_method = _render_indexed_elem_method(
                "dhamiltonian_dmoments",
                "moments",
                [str(symbol) for symbol in moments_symbols],
                moments_derivative_expressions,
                generic_moments_replacements,
                template_parameters,
                symbolic_functions,
            )
        else:
            generic_moments_method = _render_indexed_method(
                "dhamiltonian_dmoments",
                "moments",
                [str(symbol) for symbol in moments_symbols],
                moments_derivative_expressions,
                generic_moments_replacements,
                template_parameters,
                symbolic_functions,
            )

    moments_object_method = ""
    moments_jacobian_method = ""
    if use_moments_object:
        object_expression_replacements = {}
        if definition.moments_object_norm2_expression is not None:
            object_expression_replacements[
                sum(symbol**2 for symbol in moments_symbols)
            ] = definition.moments_object_norm2_expression.format(moments="moments")
        moments_object_method = _render_moments_object_method(
            [str(symbol) for symbol in moments_symbols],
            moments_derivative_expressions,
            moments_replacements,
            symbolic_functions,
            object_expression_replacements,
            definition.moments_object_component_expression,
        )
        if definition.generate_moments_jacobian:
            moments_jacobian_method = _render_moments_object_jacobian_method(
                moments_symbols,
                hamiltonian,
                moments_replacements,
                symbolic_functions,
                object_expression_replacements,
                definition.moments_object_component_expression,
            )

    nonlocal_value_methods = ""
    if moments_computer is not None:
        nonlocal_value_methods = _render_indexed_nonlocal_value_method(
            "dhamiltonian_dmoments_value",
            [str(symbol) for symbol in moments_symbols],
            moments_derivative_expressions,
            moments_replacements,
            symbolic_functions,
        )
    rendered_includes = ""
    if includes:
        rendered_includes += "".join(f"#include {header}\n" for header in includes)

    rendered_moments_computer = ""
    if moments_computer is not None:
        rendered_moments_computer = (
            f"    using MomentsComputer = {moments_computer};\n\n"
        )
    template_prefix = ""
    if template_parameters:
        template_prefix = "template <" + ", ".join(template_parameters) + ">\n"

    is_linear = False if definition is None else definition.is_linear

    output_path.write_text(
        f"""\
// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <Kokkos_Core.hpp>
#include <span>
#include <type_traits>
#include <utility>
{rendered_includes}

namespace {namespace} {{

{"" if definition is None else definition.namespace_preamble}

{template_prefix}struct {struct_name} {{
    static constexpr std::size_t N = {len(moments_symbols)};
    static constexpr bool IS_LINEAR = {"true" if is_linear else "false"};

{rendered_moments_computer}\
{_render_members(parameters)}

{_render_constructor_signature(struct_name, parameters)}
        : {_render_constructor_initializers(parameters)} {{}}

{h_signature}
    {{
        return {_render_cxx_expression(hamiltonian, h_replacements, symbolic_functions)};
    }}
{potential_method}{moments_method}{generic_moments_method}{moments_object_method}{moments_jacobian_method}{nonlocal_value_methods}
{inverse_methods}}};

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
