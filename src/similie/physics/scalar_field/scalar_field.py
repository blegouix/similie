#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Baptiste Legouix
# SPDX-License-Identifier: MIT

from __future__ import annotations

from similie_generate_cpp_hamiltonian import HamiltonianDefinition
from sympy import gamma, symbols


class ScalarFieldHamiltonian:
    @staticmethod
    def __call__(dimension: int) -> HamiltonianDefinition:
        mass = symbols("mass")
        coupling_constant = symbols("coupling_constant")
        coupling_power = symbols("coupling_power")
        phi = symbols("phi")
        pi = symbols(f"pi0:{dimension}")

        metric_sign = [-1] + [1] * (dimension - 1)
        hamiltonian = 0.5 * (
            -(mass**2) * phi**2 + sum(metric_sign[i] * pi[i] ** 2 for i in range(dimension))
        ) - coupling_constant * phi**coupling_power / gamma(coupling_power + 1)

        return HamiltonianDefinition(
            namespace="similie::physics::scalar_field",
            struct_name="ScalarFieldWithPowerCouplingHamiltonian",
            parameters=["mass", "coupling_constant", "coupling_power"],
            hamiltonian=hamiltonian,
            state_variables=[phi, *list(pi)],
        )
