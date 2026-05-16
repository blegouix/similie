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
        A, B0, B1, B2, mu, current_density = symbols("A B0 B1 B2 mu current_density")
        hamiltonian = (B0**2 + B1**2 + B2**2) / (2 * mu) - current_density * A

        return HamiltonianDefinition(
            namespace="similie::physics::magnetostatics",
            struct_name="LinearMagnetostaticsHamiltonian",
            parameters=["mu", "current_density"],
            hamiltonian=hamiltonian,
            variables=[A, B0, B1, B2],
            includes=["<similie/physics/magnetostatics/magnetostatics_quantities.hpp>"],
            value_computer_type="MagneticInductionValueFromPotential",
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
            constitutive_law=hodge_star * b / mu,
        )
