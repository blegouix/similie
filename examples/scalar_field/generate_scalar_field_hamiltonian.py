#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2024 Baptiste Legouix
# SPDX-License-Identifier: MIT

from __future__ import annotations

import argparse
from pathlib import Path

from sympy import diff, expand, symbols
from sympy.printing.cxxcode import cxxcode


def build_derivative_expression() -> str:
    x, y = symbols("x y")
    hamiltonian = 0.5 * (x**2 + y**2) + 0.25 * (x**2 + y**2) ** 2
    derivative = expand(diff(hamiltonian, x))
    return cxxcode(derivative, standard="c++11")


def write_header(output_path: Path) -> None:
    expression = build_derivative_expression()
    header = f"""// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: MIT

#pragma once

struct ScalarFieldHamiltonianDerivative
{{
    static constexpr float operator()(float x, float y)
    {{
        return static_cast<float>({expression});
    }}
}};
"""
    output_path.write_text(header)


def main() -> None:
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
    write_header(args.output)


if __name__ == "__main__":
    main()
