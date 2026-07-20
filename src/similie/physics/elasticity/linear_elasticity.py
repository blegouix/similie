#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Baptiste Legouix
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

from similie_generate_cpp_constitutive_law import ConstitutiveLawDefinition
from similie_generate_cpp_hamiltonian import HamiltonianDefinition
from sympy import symbols


def _lame_coefficients(dimension: int):
    young_modulus, poisson_ratio = symbols("young_modulus poisson_ratio")
    shear_modulus = young_modulus / (2 * (1 + poisson_ratio))
    lame_lambda = young_modulus * poisson_ratio / (
        (1 + poisson_ratio) * (1 - (dimension - 1) * poisson_ratio)
    )
    return young_modulus, poisson_ratio, shear_modulus, lame_lambda


def _strain_indices(dimension: int) -> list[tuple[int, int]]:
    return [(i, j) for i in range(dimension) for j in range(i, dimension)]


def _strain_component_index(index: tuple[int, int]) -> str:
    return f"StrainTensorIndex<{index[0]}, {index[1]}>"


class LinearElasticityHamiltonian:
    @staticmethod
    def __call__(dimension: int) -> HamiltonianDefinition:
        if dimension < 1:
            raise ValueError("dimension must be at least 1")

        displacement = symbols(f"u0:{dimension}")
        body_force = symbols(f"f0:{dimension}")
        strain_indices = _strain_indices(dimension)
        strain = symbols(f"strain0:{len(strain_indices)}")
        _, _, shear_modulus, lame_lambda = _lame_coefficients(dimension)

        strain_by_index = {
            tensor_index: component
            for tensor_index, component in zip(strain_indices, strain, strict=True)
        }
        trace_strain = sum(strain_by_index[i, i] for i in range(dimension))
        strain_double_contraction = sum(
            component**2 if i == j else 2 * component**2
            for (i, j), component in strain_by_index.items()
        )
        strain_energy = (
            shear_modulus * strain_double_contraction
            + lame_lambda * trace_strain**2 / 2
        )
        body_force_potential = sum(
            force_component * displacement_component
            for force_component, displacement_component in zip(
                body_force,
                displacement,
                strict=True,
            )
        )
        hamiltonian = strain_energy - body_force_potential

        return HamiltonianDefinition(
            namespace="similie::physics::elasticity",
            struct_name="LinearElasticityHamiltonian",
            parameters=["young_modulus", "poisson_ratio"],
            hamiltonian=hamiltonian,
            variables=[displacement, strain, body_force],
            includes=["<similie/physics/elasticity/elasticity_quantities.hpp>"],
            template_parameters=["class... SpatialIndex"],
            moments_object_component_expression="moments.template get<{index}>()",
            component_indices=[
                _strain_component_index(strain_index) for strain_index in strain_indices
            ],
            is_linear=True,
        )


class LinearElasticStrainToStress:
    @staticmethod
    def __call__() -> ConstitutiveLawDefinition:
        trace_strain, strain, stress, stiffness, trace_coupling = symbols(
            "trace_strain strain stress stiffness trace_coupling"
        )
        return ConstitutiveLawDefinition(
            namespace="similie::physics::elasticity",
            class_name="LinearElasticStrainToStress",
            parameters=["stiffness", "trace_coupling"],
            variables=[str(trace_strain), str(strain)],
            output_variable=str(stress),
            constitutive_law=stiffness * strain + trace_coupling * trace_strain,
        )
