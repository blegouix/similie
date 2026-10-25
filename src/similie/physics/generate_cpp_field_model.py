#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Baptiste Legouix
# SPDX-License-Identifier: AGPL-3.0-or-later
# AI-GENERATED

"""Render field-model APIs with constitutive laws differentiated from H(D, B)."""

from pathlib import Path

from sympy import Symbol, diff
from sympy.printing.codeprinter import cxxcode


def generate_field_model(model, template, output):
    replacements = {}
    for attribute, parameter in (
        ("displacement", "displacement"),
        ("induction", "induction"),
    ):
        for index, variable in enumerate(getattr(model, attribute, ())):
            replacements[variable] = f"{parameter}[{index}]"
    for parameter in ("permittivity", "permeability"):
        if hasattr(model, parameter):
            replacements[getattr(model, parameter)] = parameter

    def expression(value):
        # Replace symbols before printing so component names remain unambiguous.
        return cxxcode(
            value.xreplace(
                {symbol: Symbol(name) for symbol, name in replacements.items()}
            )
        )

    values = {"energy": expression(model.hamiltonian)}
    for attribute, field in (
        ("displacement", "electric_field"),
        ("induction", "magnetic_field"),
    ):
        if hasattr(model, attribute):
            values[field] = (
                "{"
                + ", ".join(
                    expression(diff(model.hamiltonian, variable))
                    for variable in getattr(model, attribute)
                )
                + "}"
            )
    text = Path(template).read_text()
    for name, value in values.items():
        text = text.replace(f"@{name}@", value)
    if "@" in text:
        raise ValueError("Unexpanded field-model template expression")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text)
