# SPDX-FileCopyrightText: 2026 Baptiste Legouix
# SPDX-License-Identifier: AGPL-3.0-or-later
# AI-GENERATED

import subprocess
import sys
import tempfile
from pathlib import Path

from sympy import symbols

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src/similie/physics"))

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


with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    generate_cpp_hamiltonian(SpatialScalarHamiltonian, root / "hamiltonian.hpp")
    # The full library tests compile generated headers with Kokkos. This small
    # syntax check isolates the generator's element-dependent expressions.
    (root / "Kokkos_Core.hpp").write_text("#define KOKKOS_FUNCTION\n")
    (root / "test.cpp").write_text(
        """
#include "hamiltonian.hpp"
struct Index;
constexpr test::SpatialScalarHamiltonian model(2.0);
static_assert(model.N == 1);
static_assert(model.IS_LINEAR);
static_assert(model.dhamiltonian_dmoments<Index>(3.0, 4) == 30.0);
static_assert(model.dhamiltonian_dpotential(3.0, 4) == 30.0);
static_assert(model.moments<Index>(30.0, 4) == 3.0);
"""
    )
    subprocess.run(
        [
            sys.argv[1],
            "-std=c++23",
            "-fsyntax-only",
            f"-I{root}",
            str(root / "test.cpp"),
        ],
        check=True,
    )
