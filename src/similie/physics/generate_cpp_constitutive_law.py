#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Baptiste Legouix
# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sympy import solve, symbols
from sympy.printing.codeprinter import cxxcode


@dataclass(frozen=True)
class ConstitutiveLawDefinition:
    namespace: str
    class_name: str
    parameters: list[str]
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
    constitutive_law,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    parameter_replacements = {
        constructor_name: member_name for member_name, constructor_name, _ in parameters
    }
    constitutive_law_symbols = [
        symbol for symbol in constitutive_law.free_symbols if str(symbol) not in parameter_replacements
    ]
    if len(constitutive_law_symbols) != 1:
        raise ValueError("Constitutive law must depend on exactly one state variable")

    induction_symbol = constitutive_law_symbols[0]
    constitutive_law_expression = _replace_symbols(
        cxxcode(constitutive_law),
        {
            **parameter_replacements,
            str(induction_symbol): "b",
        },
    )
    magnetic_field_symbol = symbols("h")
    inverse_solution = solve(magnetic_field_symbol - constitutive_law, induction_symbol, dict=True)
    if not inverse_solution:
        raise ValueError("Unable to invert constitutive law")
    inverse_expression = _replace_symbols(
        cxxcode(inverse_solution[0][induction_symbol]),
        {
            **parameter_replacements,
            str(magnetic_field_symbol): "h",
        },
    )

    parameter_members = "\n".join(
        f"    {'const ' if is_const else ''}double {member_name};"
        for member_name, _, is_const in parameters
    )
    constructor_signature = ",\n            ".join(
        [*[f"double {constructor_name}_" for _, constructor_name, _ in parameters],
         "std::array<double, 3> hodge_star = {1.0, 1.0, 1.0}"]
    )
    constructor_initializers = ",\n        ".join(
        [f"{member_name}({constructor_name}_)" for member_name, constructor_name, _ in parameters]
        + ["m_hodge_star(hodge_star)"]
    )

    output_path.write_text(
        f"""\
// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: MIT

#pragma once

#include <array>

#include <similie/misc/specialization.hpp>
#include <similie/physics/magnetostatics/magnetostatics_indices.hpp>
#include <similie/tensor/tensor.hpp>

namespace {namespace} {{

class {class_name}
{{
{parameter_members}
    std::array<double, 3> m_hodge_star;

public:
    constexpr explicit {class_name}(
            {constructor_signature})
        : {constructor_initializers}
    {{
    }}

    template <
            sil::misc::Specialization<sil::tensor::Tensor> MagneticInductionTensorType,
            sil::misc::Specialization<sil::tensor::Tensor> MagneticFieldTensorType>
    KOKKOS_FUNCTION void inverse(
            MagneticInductionTensorType magnetic_induction,
            MagneticFieldTensorType magnetic_field) const
    {{
        double const hx = magnetic_field(magnetic_field.template access_element<X>());
        double const bx = {inverse_expression.replace("h", "hx")};
        magnetic_induction(magnetic_induction.template access_element<Y, Z>()) = bx * m_hodge_star[0];
        double const hy = magnetic_field(magnetic_field.template access_element<Y>());
        double const by = {inverse_expression.replace("h", "hy")};
        magnetic_induction(magnetic_induction.template access_element<X, Z>()) = -by * m_hodge_star[1];
        double const hz = magnetic_field(magnetic_field.template access_element<Z>());
        double const bz = {inverse_expression.replace("h", "hz")};
        magnetic_induction(magnetic_induction.template access_element<X, Y>()) = bz * m_hodge_star[2];
    }}

    template <
            sil::misc::Specialization<sil::tensor::Tensor> MagneticFieldTensorType,
            sil::misc::Specialization<sil::tensor::Tensor> MagneticInductionTensorType>
    KOKKOS_FUNCTION void forward(
            MagneticFieldTensorType magnetic_field,
            MagneticInductionTensorType magnetic_induction) const
    {{
        double const bx = magnetic_induction(magnetic_induction.template access_element<Y, Z>()) / m_hodge_star[0];
        magnetic_field(magnetic_field.template access_element<X>()) = {constitutive_law_expression.replace("b", "bx")};
        double const by = -magnetic_induction(magnetic_induction.template access_element<X, Z>()) / m_hodge_star[1];
        magnetic_field(magnetic_field.template access_element<Y>()) = {constitutive_law_expression.replace("b", "by")};
        double const bz = magnetic_induction(magnetic_induction.template access_element<X, Y>()) / m_hodge_star[2];
        magnetic_field(magnetic_field.template access_element<Z>()) = {constitutive_law_expression.replace("b", "bz")};
    }}
}};

}} // namespace {namespace}
"""
    )


def generate_cpp_constitutive_law(functor_class, output_path: Path, *args, **kwargs) -> None:
    definition = functor_class.__call__(*args, **kwargs)
    parameter_tuples = [(f"m_{name}", name, True) for name in definition.parameters]
    write_cpp_constitutive_law_header(
        output_path=output_path,
        namespace=definition.namespace,
        class_name=definition.class_name,
        parameters=parameter_tuples,
        constitutive_law=definition.constitutive_law,
    )
