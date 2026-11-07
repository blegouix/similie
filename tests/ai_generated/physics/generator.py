# SPDX-FileCopyrightText: 2026 Baptiste Legouix
# SPDX-License-Identifier: AGPL-3.0-or-later
# AI-GENERATED

import sys
from pathlib import Path

from sympy import symbols

source_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(source_root / "src/similie/physics"))

from generate_cpp_hamiltonian import HamiltonianDefinition, generate_cpp_hamiltonian


class SpatialScalarHamiltonian:
    @staticmethod
    def __call__():
        stiffness, phi, pi = symbols("stiffness phi pi")
        return HamiltonianDefinition(
            namespace="test",
            struct_name="SpatialScalarHamiltonian",
            parameters=["stiffness"],
            hamiltonian=stiffness * (phi**2 + pi**2) / 2,
            variables=[phi, pi],
            parameter_value_expressions={"stiffness": "m_stiffness * (1 + elem)"},
            is_linear=True,
        )


if len(sys.argv) != 2:
    raise SystemExit(f"Usage: {Path(sys.argv[0]).name} OUTPUT_HEADER")

generate_cpp_hamiltonian(SpatialScalarHamiltonian, Path(sys.argv[1]))
