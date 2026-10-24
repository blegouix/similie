#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Baptiste Legouix
# SPDX-License-Identifier: AGPL-3.0-or-later
# AI-GENERATED

"""Derive local electromagnetic exchange from the minimal-coupling Hamiltonian.

H(x,p,t) = |p-q A(x,t)|^2/(2m) + q phi(x,t).
Hamilton's equations evolve canonical momentum p. The force on mechanical
momentum m v = p-q A additionally subtracts q dA/dt along the trajectory.
After introducing E=-grad(phi)-partial_t A and B=curl(A), this gives the
electric and magnetic forces. Summing carriers replaces q by charge density
and q v by total current density; no common carrier velocity is assumed.

These are exchange laws, not extra stored energies to add to fluid+field H.
Magnetic work vanishes for each carrier. All fluid energy transfer is J.E,
including conversion into internal energy for a conducting fluid.
"""

import argparse
from pathlib import Path

from sympy import Function, Matrix, Symbol, diff, simplify, symbols
from sympy.printing.codeprinter import cxxcode


class ElectromagneticInteraction:
    def __init__(self):
        self.position = Matrix(symbols("x0:3", real=True))
        self.time = symbols("t", real=True)
        self.momentum = Matrix(symbols("p0:3", real=True))
        self.velocity = Matrix(symbols("v0:3", real=True))
        self.charge, self.mass = symbols("q m", nonzero=True, real=True)
        self.potential = Function("phi")(*self.position, self.time)
        self.vector_potential = Matrix(
            [Function(f"A{i}")(*self.position, self.time) for i in range(3)]
        )
        mechanical = self.momentum - self.charge * self.vector_potential
        self.hamiltonian = mechanical.dot(mechanical) / (2 * self.mass)
        self.hamiltonian += self.charge * self.potential
        self.canonical_velocity = self.hamiltonian.diff(self.momentum)
        self.canonical_force = -self.hamiltonian.diff(self.position)
        mechanical_force = self.canonical_force - self.charge * (
            self.vector_potential.diff(self.time)
            + self.vector_potential.jacobian(self.position) * self.canonical_velocity
        )
        mechanical_force = mechanical_force.subs(
            dict(
                zip(
                    self.momentum,
                    self.mass * self.velocity + self.charge * self.vector_potential,
                )
            ),
            simultaneous=True,
        )

        self.electric_field = Matrix(symbols("E0:3", real=True))
        self.induction = Matrix(symbols("B0:3", real=True))
        substitutions = {
            diff(self.potential, self.position[i]): -self.electric_field[i]
            - diff(self.vector_potential[i], self.time)
            for i in range(3)
        }
        # Rewrite the three antisymmetric derivative pairs using B=curl A.
        # The remaining symmetric derivatives cancel, without choosing a gauge.
        for i in range(3):
            j, k = (i + 1) % 3, (i + 2) % 3
            substitutions[diff(self.vector_potential[k], self.position[j])] = (
                self.induction[i] + diff(self.vector_potential[j], self.position[k])
            )
        self.mechanical_force = mechanical_force.subs(
            substitutions, simultaneous=True
        ).applyfunc(simplify)
        self.electric_force = self.mechanical_force.subs(
            dict.fromkeys(self.velocity, 0)
        )
        self.magnetic_force = self.mechanical_force - self.electric_force
        self.electric_power = self.velocity.dot(self.electric_force)
        self.magnetic_power = simplify(self.velocity.dot(self.magnetic_force))
        self.charge_density = symbols("charge_density", real=True)
        self.current = Matrix(symbols("J0:3", real=True))
        carrier_sum = {
            self.velocity[i]: self.current[i] / self.charge for i in range(3)
        }
        self.electric_force_density = self.electric_force.subs(
            self.charge, self.charge_density
        )
        self.magnetic_force_density = self.magnetic_force.subs(carrier_sum).applyfunc(
            simplify
        )
        self.fluid_energy_source = simplify(self.electric_power.subs(carrier_sum))


def generate(output_directory):
    model = ElectromagneticInteraction()
    replacements = {model.charge_density: Symbol("state.charge_density")}
    for variables, member in (
        (model.electric_field, "electric_field"),
        (model.induction, "induction"),
        (model.current, "current"),
    ):
        replacements.update(
            {
                symbol: Symbol(f"state.{member}[{i}]")
                for i, symbol in enumerate(variables)
            }
        )

    def expression(value):
        return cxxcode(value.xreplace(replacements))

    template = Path(__file__).with_suffix(".hpp.in").read_text()
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    for name, filename, force, power in (
        (
            "ElectricCouplingPhysics",
            "electric_coupling_physics.hpp",
            model.electric_force_density,
            model.fluid_energy_source,
        ),
        (
            "MagneticCouplingPhysics",
            "magnetic_coupling_physics.hpp",
            model.magnetic_force_density,
            model.magnetic_power,
        ),
    ):
        text = template.replace("@class@", name)
        text = text.replace(
            "@force@",
            "{" + ", ".join(expression(component) for component in force) + "}",
        )
        text = text.replace("@power@", expression(power))
        text = text.replace("@field_power@", expression(-power))
        (output_directory / filename).write_text(text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, required=True)
    generate(parser.parse_args().output_directory)
