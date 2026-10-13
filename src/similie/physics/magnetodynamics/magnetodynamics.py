#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Baptiste Legouix
# SPDX-License-Identifier: AGPL-3.0-or-later
# AI-GENERATED

"""Magnetic-only Hamiltonian sector and ideal/resistive MHD coupling.

H(B)=B.B/(2 mu), H_field=delta H/delta B. Faraday evolves B from
an Ohm-law electric field; no independent electric displacement is stored.
"""

import argparse
import importlib.util
import sys
from pathlib import Path

from sympy import Matrix, diff, symbols


class MagnetodynamicsHamiltonian:
    @staticmethod
    def __call__():
        # Use the project's Hamiltonian generator rather than a fluid Hamiltonian.
        from similie_generate_cpp_hamiltonian import HamiltonianDefinition

        potential = symbols("potential")
        induction = symbols("B0:3")
        permeability = symbols("permeability", positive=True)
        return HamiltonianDefinition(
            namespace="similie::physics::magnetodynamics",
            struct_name="MagnetodynamicsHamiltonian",
            parameters=["permeability"],
            hamiltonian=sum(b**2 / (2 * permeability) for b in induction),
            variables=[potential, induction],
            is_linear=True,
        )


class Magnetodynamics:
    """Local expressions; exterior supplies every curl/divergence operation."""

    def __init__(self):
        self.induction = Matrix(symbols("B0:3"))
        self.velocity = Matrix(symbols("u0:3"))
        self.current = Matrix(symbols("J0:3"))
        self.permeability = symbols("mu", positive=True)
        self.hamiltonian = self.induction.dot(self.induction) / (2 * self.permeability)
        self.magnetic_field = Matrix(
            [diff(self.hamiltonian, b) for b in self.induction]
        )
        self.magnetic_energy = self.hamiltonian
        self.ohm_electric_field = -self.velocity.cross(self.induction)
        self.lorentz_force = self.current.cross(self.induction)
        self.momentum_flux = (
            self.magnetic_energy * Matrix.eye(3)
            - self.induction * self.magnetic_field.T
        )
        self.ohmic_heating = 0

    @property
    def poynting_flux(self):
        return self.ohm_electric_field.cross(self.magnetic_field)

    @property
    def fluid_energy_source(self):
        return self.current.dot(self.ohm_electric_field)


class ResistiveMagnetodynamics(Magnetodynamics):
    def __init__(self):
        super().__init__()
        self.resistivity = symbols("eta", nonnegative=True)
        self.ohm_electric_field += self.resistivity * self.current
        self.ohmic_heating = self.resistivity * self.current.dot(self.current)


def generate(output):
    generator_path = Path(__file__).resolve().parents[1] / "generate_cpp_hamiltonian.py"
    spec = importlib.util.spec_from_file_location(
        "similie_generate_cpp_hamiltonian", generator_path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    module.generate_cpp_hamiltonian(MagnetodynamicsHamiltonian, output)
    # The generic generator predates the annotation rule; annotate this output.
    # REUSE-IgnoreStart
    text = output.read_text().replace(
        "// SPDX-License-Identifier: AGPL-3.0-or-later\n",
        "// SPDX-License-Identifier: AGPL-3.0-or-later\n// AI-GENERATED\n",
        1,
    )
    # REUSE-IgnoreEnd
    text = text.replace(
        "struct MagnetodynamicsHamiltonian",
        "/** Magnetic energy density; derivatives give H_field.\n"
        " * Faraday's native exterior derivative and Ohm law supply evolution.\n"
        " * \\important This operator and documentation is fully AI-generated.\n"
        " */\nstruct MagnetodynamicsHamiltonian",
        1,
    )
    output.write_text(text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    generate(parser.parse_args().output)
