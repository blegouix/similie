#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Baptiste Legouix
# SPDX-License-Identifier: AGPL-3.0-or-later
# AI-GENERATED

"""Electric-only electrohydrodynamics with electrostatic field reconstruction.

H(D)=D.D/(2 epsilon), E=H_D=-grad(phi), div D=charge.
Charge evolves by -div J. Ideal charge is advected, J=charge*u;
finite resistivity adds j=E/eta. No magnetic field or Maxwell-wave state.
"""

import argparse
import importlib.util
import sys
from pathlib import Path

from sympy import Matrix, diff, symbols


class ElectrodynamicsHamiltonian:
    @staticmethod
    def __call__():
        from similie_generate_cpp_hamiltonian import HamiltonianDefinition

        phi = symbols("potential")
        displacement = symbols("D0:3")
        permittivity = symbols("permittivity", positive=True)
        return HamiltonianDefinition(
            namespace="similie::physics::electrodynamics",
            struct_name="ElectrodynamicsHamiltonian",
            parameters=["permittivity"],
            hamiltonian=sum(d * d / (2 * permittivity) for d in displacement),
            variables=[phi, displacement],
            is_linear=True,
        )


class Electrodynamics:
    def __init__(self):
        self.displacement = Matrix(symbols("D0:3"))
        self.velocity = Matrix(symbols("u0:3"))
        self.permittivity = symbols("epsilon", positive=True)
        self.hamiltonian = self.displacement.dot(self.displacement) / (
            2 * self.permittivity
        )
        self.electric_field = Matrix(
            [diff(self.hamiltonian, d) for d in self.displacement]
        )
        self.charge_density = symbols("charge", real=True)
        self.conduction_current = Matrix.zeros(3, 1)

    @property
    def current(self):
        return self.charge_density * self.velocity + self.conduction_current

    @property
    def electric_force(self):
        return self.charge_density * self.electric_field

    @property
    def fluid_energy_source(self):
        return self.current.dot(self.electric_field)

    @property
    def ohmic_heating(self):
        return self.conduction_current.dot(self.electric_field)


class ResistiveElectrodynamics(Electrodynamics):
    def __init__(self):
        super().__init__()
        self.resistivity = symbols("eta", positive=True)
        self.conduction_current = self.electric_field / self.resistivity


def generate(output):
    path = Path(__file__).resolve().parents[1] / "generate_cpp_hamiltonian.py"
    spec = importlib.util.spec_from_file_location(
        "similie_generate_cpp_hamiltonian", path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    output = Path(output)
    module.generate_cpp_hamiltonian(ElectrodynamicsHamiltonian, output)
    text = output.read_text().replace(
        "// SPDX-License-Identifier: AGPL-3.0-or-later\n",
        "// SPDX-License-Identifier: AGPL-3.0-or-later\n// AI-GENERATED\n",
        1,
    )
    text = text.replace(
        "struct ElectrodynamicsHamiltonian",
        "/** Electric energy density; the charge-constrained field uses Gauss's law.\n"
        " * The potential solve is separate from charge time integration.\n"
        " * \\important This operator and documentation is fully AI-generated.\n"
        " */\nstruct ElectrodynamicsHamiltonian",
        1,
    )
    output.write_text(text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    generate(parser.parse_args().output)
