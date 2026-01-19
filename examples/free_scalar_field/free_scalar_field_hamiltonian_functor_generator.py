#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2024 Baptiste Legouix
# SPDX-License-Identifier: MIT

from __future__ import annotations

import sys
from pathlib import Path

from sympy import symbols, Matrix, diff, solve
from sympy.printing.codeprinter import cxxcode

N = int(sys.argv[1])  # Number of dimensions
output_path = Path(sys.argv[2])  # Path of the output file
output_path.parent.mkdir(parents=True, exist_ok=True)

mass = symbols("mass")
phi = symbols("phi")
pi = symbols(f"pi0:{N}")

hamiltonian = 0.5 * (sum(pi_**2 for pi_ in pi) + mass**2 * phi**2)
hamiltonian_diff = Matrix(
    [diff(hamiltonian, phi), *[diff(hamiltonian, pi_) for pi_ in pi]]
)

dphi_dx = symbols(f"dphi_dx0:{N}")
pi_from_dphi_dx = solve(
    [dphi_dx[i] + hamiltonian_diff[i + 1] for i in range(N)], list(pi), dict=True
)[0]
if not pi_from_dphi_dx:
    raise RuntimeError("Could not solve for pi in terms of dphi/dx.")


def preprocess_cxx(expr: str) -> str:
    for i in range(N):
        expr = expr.replace(f"pi{i}", f"pi[{i}]")
    return expr


output_path.write_text(f"""\
// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: MIT

#pragma once

#include <span>

struct FreeScalarFieldHamiltonian {{
    static constexpr std::size_t N = {N};

    const double mass;

    explicit FreeScalarFieldHamiltonian(double mass_) : mass(mass_) {{}}

    constexpr double H(double phi, const std::span<const double, N>& pi) const
    {{
        return {preprocess_cxx(cxxcode(hamiltonian))};
    }}

    constexpr double dH_dphi(double phi) const
    {{
        return {cxxcode(hamiltonian_diff[0])};
    }}
{
    "".join(
        f'''
    constexpr double dH_dpi{i}(const double pi{i}) const
    {{
        return {cxxcode(hamiltonian_diff[i + 1])};
    }}
'''
        for i in range(N)
    )
}{
    "".join(
        f'''
    constexpr double pi{i}(const double dphi_dx{i}) const
    {{
        return {cxxcode(pi_from_dphi_dx[pi[i]])};
    }}
'''
        for i in range(N)
    )
}
}};
""")
