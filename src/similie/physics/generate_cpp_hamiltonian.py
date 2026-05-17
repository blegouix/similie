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
    includes: list[str] | None = None
    moments_computer: str | None = None
    template_parameters: list[str] | None = None
    parameter_types: dict[str, str] | None = None
    parameter_value_expressions: dict[str, str] | None = None


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
    for source, target in replacements.items():
        expression = expression.replace(source, target)
    return expression


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
    return f"    constexpr {struct_name}(\n            {params})"


def _render_constructor_initializers(parameters: list[tuple[str, str, bool, str]]) -> str:
    return ", ".join(
        f"{name}({constructor_name}_)" for name, constructor_name, _, _ in parameters
    )


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


def _render_indexed_span_method(
    method_name: str,
    argument_name: str,
    symbols_: list[str],
    expressions: list,
    replacements: dict[str, str],
) -> str:
    branches: list[str] = []
    for i, expression in enumerate(expressions):
        branches.append(
            f"""        if constexpr (I == {i}) {{
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
    constexpr double {method_name}(std::span<double const, {len(symbols_)}> {argument_name}) const
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
            auto value = MomentsComputer::template forward_value<I>(elem);
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
    template <std::size_t I, class Elem>
    constexpr auto {method_name}(Elem elem) const
    {{
{chr(10).join(branches)}
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
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    parameter_value_expressions = parameter_value_expressions or {}
    parameter_replacements = {}
    for member_name, constructor_name, _, _ in parameters:
        parameter_replacements[constructor_name] = parameter_value_expressions.get(
            constructor_name, member_name
        )
    h_replacements = dict(parameter_replacements)
    requires_elem = any("elem" in replacement for replacement in parameter_replacements.values())
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
            f"    constexpr double H({', '.join(argument_signature_parts)}, Elem elem) const"
        )
    else:
        h_signature = f"    constexpr double H({', '.join(argument_signature_parts)}) const"

    potential_entry = variable_entries[0]
    moments_entry = variable_entries[1]
    potential_name = potential_entry["name"]
    potential_symbols = potential_entry["symbols"]
    moments_name = moments_entry["name"]
    moments_symbols = moments_entry["symbols"]

    potential_replacements = dict(h_replacements)
    moments_replacements = dict(parameter_replacements)

    potential_derivative_expressions = [diff(hamiltonian, symbol) for symbol in potential_symbols]
    moments_derivative_expressions = [diff(hamiltonian, symbol) for symbol in moments_symbols]

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
    constexpr double dH_d{potential_name}({potential_method_signature}, Elem elem) const
    {{
        return {_replace_symbols(cxxcode(potential_derivative_expressions[0]), potential_replacements)};
    }}

    template <class Elem>
    constexpr double dH_dpotential(double potential, Elem elem) const
    {{
        return dH_d{potential_name}(potential, elem);
    }}
"""
        else:
            potential_method = f"""
    constexpr double dH_d{potential_name}({potential_method_signature}) const
    {{
        return {_replace_symbols(cxxcode(potential_derivative_expressions[0]), potential_replacements)};
    }}

    constexpr double dH_dpotential(double potential) const
    {{
        return dH_d{potential_name}(potential);
    }}
"""
    else:
        branches: list[str] = []
        for i, expression in enumerate(potential_derivative_expressions):
            branches.append(
                f"""        if constexpr (I == {i}) {{
            return {_replace_symbols(cxxcode(expression), potential_replacements)};
        }}"""
            )
        branches.append(
            """        else {
            static_assert(I < N, "Hamiltonian component index out of range");
        }"""
        )
        potential_method_signature = ", ".join(potential_signature_parts)
        if requires_elem:
            potential_method = f"""
    template <std::size_t I, class Elem>
    constexpr double dH_d{potential_name}({potential_method_signature}, Elem elem) const
    {{
{chr(10).join(branches)}
    }}
"""
        else:
            potential_method = f"""
    template <std::size_t I>
    constexpr double dH_d{potential_name}({potential_method_signature}) const
    {{
{chr(10).join(branches)}
    }}
"""

    inverse_methods = ""
    if inverse_symbols is not None and inverse_expressions is not None:
        inverse_methods = _render_indexed_method(
            "pi",
            "dphi_dx",
            inverse_symbols,
            inverse_expressions,
            potential_replacements,
        )

    if len(moments_symbols) == 1 or moments_computer is None:
        moments_method = _render_indexed_method(
            f"dH_d{moments_name}",
            moments_name,
            [str(symbol) for symbol in moments_symbols],
            moments_derivative_expressions,
            moments_replacements,
        )
    else:
        if requires_elem:
            branches: list[str] = []
            for i, expression in enumerate(moments_derivative_expressions):
                branches.append(
                    f"""        if constexpr (I == {i}) {{
            return {_replace_symbols(cxxcode(expression), h_replacements)};
        }}"""
                )
            branches.append(
                """        else {
            static_assert(I < N, "Hamiltonian component index out of range");
        }"""
            )
            moments_method = f"""
    template <std::size_t I, class Elem>
    constexpr double dH_d{moments_name}(std::span<double const, {len(moments_symbols)}> {moments_name}, Elem elem) const
    {{
{chr(10).join(branches)}
    }}
"""
        else:
            moments_method = _render_indexed_span_method(
                f"dH_d{moments_name}",
                moments_name,
                [str(symbol) for symbol in moments_symbols],
                moments_derivative_expressions,
                h_replacements,
            )
    generic_moments_branches: list[str] = []
    generic_moments_replacements = {
        **parameter_replacements,
        **{str(symbol): "moments" for symbol in moments_symbols},
    }
    for i, expression in enumerate(moments_derivative_expressions):
        generic_moments_branches.append(
            f"""        if constexpr (I == {i}) {{
            return {_replace_symbols(cxxcode(expression), generic_moments_replacements)};
        }}"""
        )
    generic_moments_branches.append(
        """        else {
            static_assert(I < N, "Hamiltonian component index out of range");
        }"""
    )
    if requires_elem:
        generic_moments_method = f"""
    template <std::size_t I, class Elem>
    constexpr double dH_dmoments(double moments, Elem elem) const
    {{
{chr(10).join(generic_moments_branches)}
    }}
"""
    else:
        generic_moments_method = f"""
    template <std::size_t I>
    constexpr double dH_dmoments(double moments) const
    {{
{chr(10).join(generic_moments_branches)}
    }}
"""

    nonlocal_value_methods = ""
    if moments_computer is not None:
        nonlocal_value_methods = _render_indexed_nonlocal_value_method(
            f"dH_d{moments_name}_value",
            [str(symbol) for symbol in moments_symbols],
            moments_derivative_expressions,
            moments_replacements,
        )
        nonlocal_value_methods += """
    template <std::size_t I, class Elem>
    constexpr auto dH_dmoments_value(Elem elem) const
    {
        return dH_d"""
        nonlocal_value_methods += moments_name
        nonlocal_value_methods += """_value<I>(elem);
    }
"""

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

    output_path.write_text(
        f"""\
// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <span>
#include <utility>
{rendered_includes}

namespace {namespace} {{

{template_prefix}struct {struct_name} {{
    static constexpr std::size_t N = {len(moments_symbols)};

{rendered_moments_computer}\
{_render_members(parameters)}

{_render_constructor_signature(struct_name, parameters)}
        : {_render_constructor_initializers(parameters)} {{}}

{h_signature}
    {{
        return {_replace_symbols(cxxcode(hamiltonian), h_replacements)};
    }}
{potential_method}{moments_method}{generic_moments_method}{nonlocal_value_methods}
{inverse_methods}}};
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
                dphi_dx_symbols[i] - diff(definition.hamiltonian, variable_entries[1]["symbols"][i])
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
    )
