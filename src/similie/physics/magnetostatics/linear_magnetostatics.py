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
        A = symbols("A0:3")
        B = symbols("B0:3")
        j = symbols("j0:3")
        mu = symbols("mu")
        hamiltonian = sum(B[i] ** 2 / (2 * mu) - j[i] * A[i] for i in range(3))

        return HamiltonianDefinition(
            namespace="similie::physics::magnetostatics",
            struct_name="LinearMagnetostaticsHamiltonian",
            parameters=["mu"],
            hamiltonian=hamiltonian,
            variables=[A, B, j],
            includes=["<similie/physics/magnetostatics/magnetostatics_quantities.hpp>"],
            moments_computer="MagneticVectorPotentialToMagneticInduction",
            template_parameters=["class MuTensor"],
            parameter_types={"mu": "MuTensor"},
            parameter_value_expressions={
                "mu": "m_mu(elem, ddc::DiscreteElement<sil::tensor::Covariant<sil::tensor::ScalarIndex>>(0))"
            },
            is_linear=True,
        )


class LinearMagneticInductionToMagneticField:
    @staticmethod
    def __call__() -> ConstitutiveLawDefinition:
        hodge_star, b, h, mu = symbols("hodge_star b h mu")
        return ConstitutiveLawDefinition(
            namespace="similie::physics::magnetostatics",
            class_name="LinearMagneticInductionToMagneticField",
            parameters=["mu"],
            variables=[str(hodge_star), str(b)],
            output_variable=str(h),
            constitutive_law=hodge_star * b / mu,
        )
