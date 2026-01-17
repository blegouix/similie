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

phi, pi_x, pi_y = symbols("phi pi_x pi_y")
mass = symbols("mass")
hamiltonian = 0.5 * (pi_x**2 + pi_y**2) + 0.5 * mass**2 * phi**2
hamiltonian_grad = Matrix(
    [diff(hamiltonian, phi), diff(hamiltonian, pi_x), diff(hamiltonian, pi_y)]
)

cxx = f"""\
// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: MIT

#pragma once

struct ScalarFieldHamiltonian {{
    const double mass;

    ScalarFieldHamiltonian(const double mass_) : mass(mass_) {{}}

    constexpr double value(double phi, double pi_x, double pi_y)
    {{
        return {cxxcode(hamiltonian)};
    }}

    constexpr std::tuple<double, double, double> d(double phi, double pi_x, double pi_y)
    {{
        return std::make_tuple({cxxcode(hamiltonian_grad[0])}, {cxxcode(hamiltonian_grad[1])}, {cxxcode(hamiltonian_grad[2])});
    }}

}};
"""
output_path.write_text(cxx)
