#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2024 Baptiste Legouix
# SPDX-License-Identifier: MIT

from __future__ import annotations

import argparse
from pathlib import Path

from sympy import diff, expand, symbols
from sympy.printing.codeprinter import cxxcode

parser = argparse.ArgumentParser(
    description="Generate a constexpr functor for the scalar field Hamiltonian derivative."
)
parser.add_argument(
    "--output",
    type=Path,
    required=True,
    help="Path to write the generated header.",
)
args = parser.parse_args()
args.output.parent.mkdir(parents=True, exist_ok=True)

x, y = symbols("x y")
hamiltonian = 0.5 * (x**2 + y**2) + 0.25 * (x**2 + y**2) ** 2
hamiltonian_derivative = expand(diff(hamiltonian, x))
hamiltonian_cxx = cxxcode(hamiltonian)
hamiltonian_derivative_cxx = cxxcode(hamiltonian_derivative)
cxx = f"""\
// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: MIT

#pragma once

struct ScalarFieldHamiltonian
{{
    static constexpr float operator()(float x, float y)
    {{
        return static_cast<float>({hamiltonian_cxx});
    }}

    static constexpr float grad()(float x, float y)
    {{
        return std::make_pair(static_cast<float>({hamiltonian_derivative_cxx}), static_cast<float>({hamiltonian_derivative_cxx}));
    }}

}};
"""
args.output.write_text(cxx)
