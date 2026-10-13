#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Baptiste Legouix
# SPDX-License-Identifier: AGPL-3.0-or-later
# AI-GENERATED

"""Hamiltonian electromagnetic sector and local ideal/resistive MHD couplings.

H(D,B)=|D|^2/(2 epsilon)+|B|^2/(2 mu). Its variational derivatives
are E and H. Maxwell's curl Poisson operator supplies D_t=curl H-J,
B_t=-curl E. MHD uses the quasistatic magnetic sector, J=curl H,
and closes E by an ideal or resistive Ohm law instead of evolving D.
"""

import argparse
import importlib.util
import sys
from pathlib import Path

from sympy import Matrix, Symbol, diff, simplify, symbols
from sympy.printing.codeprinter import cxxcode


class MagnetodynamicsHamiltonian:
    @staticmethod
    def __call__():
        # Use the project's Hamiltonian generator rather than a fluid Hamiltonian.
        from similie_generate_cpp_hamiltonian import HamiltonianDefinition

        displacement = symbols("D0:3")
        induction = symbols("B0:3")
        permittivity, permeability = symbols("permittivity permeability", positive=True)
        return HamiltonianDefinition(
            namespace="similie::physics::magnetodynamics",
            struct_name="MagnetodynamicsHamiltonian",
            parameters=["permittivity", "permeability"],
            hamiltonian=sum(d**2 / (2 * permittivity) for d in displacement)
            + sum(b**2 / (2 * permeability) for b in induction),
            variables=[displacement, induction],
            is_linear=True,
        )


class Magnetodynamics:
    """Local expressions; exterior supplies every curl/divergence operation."""

    def __init__(self):
        self.displacement = Matrix(symbols("D0:3"))
        self.induction = Matrix(symbols("B0:3"))
        self.velocity = Matrix(symbols("u0:3"))
        self.current = Matrix(symbols("J0:3"))
        self.permittivity, self.permeability = symbols("epsilon mu", positive=True)
        self.hamiltonian = self.displacement.dot(self.displacement) / (
            2 * self.permittivity
        ) + self.induction.dot(self.induction) / (2 * self.permeability)
        self.magnetic_field = Matrix(
            [diff(self.hamiltonian, b) for b in self.induction]
        )
        self.electric_field = Matrix(
            [diff(self.hamiltonian, d) for d in self.displacement]
        )
        self.magnetic_energy = self.hamiltonian.subs(
            dict.fromkeys(self.displacement, 0)
        )
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
    text = output.read_text().replace(
        "// SPDX-License-Identifier: AGPL-3.0-or-later\n",
        "// SPDX-License-Identifier: AGPL-3.0-or-later\n// AI-GENERATED\n",
        1,
    )
    text = text.replace(
        "struct MagnetodynamicsHamiltonian",
        "/** Electromagnetic energy density in D and B; derivatives give E and H.\n"
        " * Maxwell's spatial curl Poisson operator supplies the evolution.\n"
        " * \\important This operator and documentation is fully AI-generated.\n"
        " */\nstruct MagnetodynamicsHamiltonian",
        1,
    )
    # The generic generator emits derivatives of the second variable block.
    # Add the electric derivative of the first, vector-valued block from SymPy.
    definition = MagnetodynamicsHamiltonian.__call__()
    displacement = definition.variables[0]
    derivative = diff(definition.hamiltonian, displacement[0])
    for component in displacement[1:]:
        if (
            simplify(
                diff(definition.hamiltonian, component).subs(component, displacement[0])
                - derivative
            )
            != 0
        ):
            raise ValueError(
                "Electric derivative requires a component-aware constitutive law"
            )
    parameters = {str(symbol): symbol for symbol in derivative.free_symbols}
    derivative = derivative.xreplace(
        {
            displacement[0]: Symbol("displacement"),
            parameters["permittivity"]: Symbol("m_permittivity"),
        }
    )
    method = "\n    KOKKOS_FUNCTION constexpr double dhamiltonian_ddisplacement(double displacement) const\n"
    method += "    { return " + cxxcode(derivative) + "; }\n"
    location = text.rfind("};")
    text = text[:location] + method + text[location:]
    output.write_text(text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    generate(parser.parse_args().output)
