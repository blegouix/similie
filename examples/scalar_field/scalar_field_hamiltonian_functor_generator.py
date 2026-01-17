#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2024 Baptiste Legouix
# SPDX-License-Identifier: MIT

from __future__ import annotations

import sys
from pathlib import Path

from sympy import symbols, Matrix, diff
from sympy.printing.codeprinter import cxxcode

output_path = Path(sys.argv[1])
output_path.parent.mkdir(parents=True, exist_ok=True)

x, y = symbols("x y")
hamiltonian = 0.5 * (x**2 + y**2) + 0.25 * (x**2 + y**2) ** 2 # TODO params in class
hamiltonian_grad = Matrix([diff(hamiltonian, x), diff(hamiltonian, y)])
hamiltonian_cxx = cxxcode(hamiltonian)
cxx = f"""\
// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: MIT

#pragma once

struct ScalarFieldHamiltonian
{{
    static constexpr double value(double x, double y)
    {{
        return {hamiltonian_cxx};
    }}

    static constexpr std::pair<double, double> grad(double x, double y)
    {{
        return std::make_pair({cxxcode(hamiltonian_grad[0])}, {cxxcode(hamiltonian_grad[1])});
    }}

}};
"""
output_path.write_text(cxx)
