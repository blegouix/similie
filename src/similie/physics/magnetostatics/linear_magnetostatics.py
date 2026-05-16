#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Baptiste Legouix
# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import dataclass

from sympy import symbols


@dataclass(frozen=True)
class HamiltonianDefinition:
    namespace: str
    struct_name: str
    parameters: list[tuple[str, str, bool]]
    h_expression: object
    derivative_symbols: list[str]
    derivative_expressions: list[object]
    array_argument_name: str
    scalar_argument_name: str | None = None
    scalar_derivative_expression: object | None = None
    inverse_symbols: list[str] | None = None
    inverse_expressions: list[object] | None = None
    aliases: list[str] | None = None


@dataclass(frozen=True)
class ConstitutiveLawDefinition:
    namespace: str
    class_name: str
    parameters: list[tuple[str, str, bool]]
    hodge_star_size: int
    induction_components: list[tuple[str, str, int]]


class LinearMagnetostaticsHamiltonian:
    @staticmethod
    def __call__() -> HamiltonianDefinition:
        b0, b1, b2, mu = symbols("b0 b1 b2 mu")
        hamiltonian = (b0**2 + b1**2 + b2**2) / (2 * mu)

        return HamiltonianDefinition(
            namespace="similie::physics::magnetostatics",
            struct_name="LinearMagnetostaticsHamiltonian",
            parameters=[("mu", "mu", False)],
            h_expression=hamiltonian,
            derivative_symbols=["b0", "b1", "b2"],
            derivative_expressions=[hamiltonian.diff(b0), hamiltonian.diff(b1), hamiltonian.diff(b2)],
            array_argument_name="pi",
        )


class LinearMagneticInductionToMagneticFieldConstitutiveLaw:
    @staticmethod
    def __call__() -> ConstitutiveLawDefinition:
        return ConstitutiveLawDefinition(
            namespace="similie::physics::magnetostatics",
            class_name="LinearMagneticInductionToMagneticField",
            parameters=[("m_mu", "mu", False)],
            hodge_star_size=3,
            induction_components=[
                ("Y, Z", "X", 0),
                ("X, Z", "Y", 1),
                ("X, Y", "Z", 2),
            ],
        )
