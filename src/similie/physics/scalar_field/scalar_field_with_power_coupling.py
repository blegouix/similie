#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Baptiste Legouix
# SPDX-License-Identifier: MIT

from __future__ import annotations

import sys
from pathlib import Path

from sympy import Matrix, diff, gamma, solve, symbols

from similie.physics.generate_cpp_hamiltonian import write_cpp_hamiltonian_header


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

write_cpp_hamiltonian_header(
    output_path=output_path,
    namespace="similie::physics::scalar_field",
    struct_name="ScalarFieldWithPowerCouplingHamiltonian",
    parameters=[
        ("mass", "mass", True),
        ("coupling_constant", "coupling_constant", True),
        ("coupling_power", "coupling_power", True),
    ],
    h_expression=hamiltonian,
    derivative_symbols=[f"pi{i}" for i in range(N)],
    derivative_expressions=[hamiltonian_diff[i + 1] for i in range(N)],
    array_argument_name="pi",
    scalar_argument_name="phi",
    scalar_derivative_expression=hamiltonian_diff[0],
    inverse_symbols=[f"dphi_dx{i}" for i in range(N)],
    inverse_expressions=[pi_from_dphi_dx[pi[i]] for i in range(N)],
    aliases=["ScalarFieldWithPowerCoupling", "ScalarFieldHamiltonian"],
)
