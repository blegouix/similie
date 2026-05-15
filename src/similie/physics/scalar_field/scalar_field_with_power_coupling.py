#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Baptiste Legouix
# SPDX-License-Identifier: MIT

from __future__ import annotations

import sys
from pathlib import Path

from sympy import Matrix, diff, gamma, solve, symbols
from sympy.printing.codeprinter import cxxcode


N = int(sys.argv[1])
output_path = Path(sys.argv[2])
output_path.parent.mkdir(parents=True, exist_ok=True)

mass = symbols("mass")
coupling_constant = symbols("coupling_constant")
coupling_power = symbols("coupling_power")
phi = symbols("phi")
pi = symbols(f"pi0:{N}")

metric_sign = [-1] + [1] * (N - 1)
hamiltonian = 0.5 * (
    -(mass**2) * phi**2 + sum(metric_sign[i] * pi[i] ** 2 for i in range(N))
) - coupling_constant * phi**coupling_power / gamma(coupling_power + 1)

hamiltonian_diff = Matrix(
    [diff(hamiltonian, phi), *[diff(hamiltonian, pi_) for pi_ in pi]]
)

dphi_dx = symbols(f"dphi_dx0:{N}")
pi_from_dphi_dx = solve(
    [dphi_dx[i] - metric_sign[i] * hamiltonian_diff[i + 1] for i in range(N)],
    list(pi),
    dict=True,
)[0]
if not pi_from_dphi_dx:
    raise RuntimeError("Could not solve for pi in terms of dphi/dx.")


def preprocess_cxx(expr: str) -> str:
    for i in range(N):
        expr = expr.replace(f"pi{i}", f"pi[{i}]")
    return expr


output_path.write_text(
    f"""\
// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: MIT

#pragma once

#include <cmath>
#include <cstddef>
#include <span>

namespace similie::physics::scalar_field {{

struct ScalarFieldWithPowerCoupling {{
    static constexpr std::size_t N = {N};

    const double mass;
    const double coupling_constant;
    const double coupling_power;

    constexpr ScalarFieldWithPowerCoupling(
            double mass_,
            double coupling_constant_,
            double coupling_power_)
        : mass(mass_), coupling_constant(coupling_constant_), coupling_power(coupling_power_) {{}}

    constexpr double H(double phi, std::span<double const, N> pi) const
    {{
        return {preprocess_cxx(cxxcode(hamiltonian))};
    }}

    constexpr double dH_dphi(double phi) const
    {{
        return {cxxcode(hamiltonian_diff[0])};
    }}
{''.join(
f'''
    constexpr double dH_dpi{i}(double pi{i}) const
    {{
        return {cxxcode(hamiltonian_diff[i + 1])};
    }}
'''
for i in range(N)
)}
{''.join(
f'''
    constexpr double pi{i}(double dphi_dx{i}) const
    {{
        return {cxxcode(pi_from_dphi_dx[pi[i]])};
    }}
'''
for i in range(N)
)}
}};

using ScalarFieldHamiltonian = ScalarFieldWithPowerCoupling;

}} // namespace similie::physics::scalar_field
"""
)
