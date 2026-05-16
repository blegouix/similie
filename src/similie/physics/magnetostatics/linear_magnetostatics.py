#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Baptiste Legouix
# SPDX-License-Identifier: MIT

from __future__ import annotations

from similie_generate_cpp_constitutive_law import ConstitutiveLawDefinition
from similie_generate_cpp_hamiltonian import HamiltonianDefinition
from sympy import symbols


class LinearMagnetostaticsHamiltonian:
    @staticmethod
    def __call__() -> HamiltonianDefinition:
        b0, b1, b2, mu = symbols("b0 b1 b2 mu")
        hamiltonian = (b0**2 + b1**2 + b2**2) / (2 * mu)

        return HamiltonianDefinition(
            namespace="similie::physics::magnetostatics",
            struct_name="LinearMagnetostaticsHamiltonian",
            parameters=["mu"],
            hamiltonian=hamiltonian,
            variables=[b0, b1, b2],
        )


class LinearMagneticInductionToMagneticFieldConstitutiveLaw:
    @staticmethod
    def __call__() -> ConstitutiveLawDefinition:
        hodge_star, b, mu = symbols("hodge_star b mu")
        return ConstitutiveLawDefinition(
            namespace="similie::physics::magnetostatics",
            class_name="LinearMagneticInductionToMagneticField",
            parameters=["mu"],
            variables=["hodge_star", "b"],
            output_variable="h",
            constitutive_law=b / mu / hodge_star,
        )
