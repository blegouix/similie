#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Baptiste Legouix
# SPDX-License-Identifier: AGPL-3.0-or-later
# AI-GENERATED

"""Electric-only Hamiltonian, Poisson reconstruction, and EHD power checks."""

import tempfile
import unittest
from pathlib import Path

from electrodynamics import Electrodynamics, ResistiveElectrodynamics, generate
from sympy import Matrix, cos, diff, oo, simplify, sin, symbols


class ElectrodynamicsTests(unittest.TestCase):
    def test_ideal_advection_and_electric_force(self):
        model = Electrodynamics()
        self.assertEqual(model.current, model.charge_density * model.velocity)
        self.assertEqual(
            model.electric_force, model.charge_density * model.electric_field
        )
        self.assertEqual(model.ohmic_heating, 0)
        self.assertFalse(hasattr(model, "induction"))

    def test_electric_hamiltonian(self):
        model = Electrodynamics()
        for a in range(3):
            self.assertEqual(
                diff(model.hamiltonian, model.displacement[a]), model.electric_field[a]
            )
            self.assertEqual(
                diff(model.hamiltonian, model.displacement[a], 2),
                1 / model.permittivity,
            )

    def test_work_and_heat(self):
        for model in (Electrodynamics(), ResistiveElectrodynamics()):
            self.assertEqual(
                simplify(
                    model.fluid_energy_source
                    - model.velocity.dot(model.electric_force)
                    - model.ohmic_heating
                ),
                0,
            )
        model = ResistiveElectrodynamics()
        self.assertEqual(
            simplify(
                model.ohmic_heating
                - model.electric_field.dot(model.electric_field) / model.resistivity
            ),
            0,
        )
        self.assertEqual(
            simplify(
                model.ohmic_heating
                - model.resistivity
                * model.conduction_current.dot(model.conduction_current)
            ),
            0,
        )

    def test_nonconducting_limit(self):
        ideal, resistive = Electrodynamics(), ResistiveElectrodynamics()
        self.assertEqual(
            resistive.current.subs(resistive.resistivity, oo), ideal.current
        )
        self.assertEqual(resistive.ohmic_heating.subs(resistive.resistivity, oo), 0)

    def test_poisson_sign_and_charge_relaxation(self):
        x, k, eta, epsilon = symbols("x k eta epsilon", positive=True)
        potential = cos(k * x)
        electric = -diff(potential, x)
        charge = -epsilon * diff(potential, x, 2)
        self.assertEqual(simplify(diff(epsilon * electric, x) - charge), 0)
        self.assertEqual(
            simplify(-diff(electric / eta, x) + charge / (eta * epsilon)), 0
        )
        self.assertEqual(simplify(electric - k * sin(k * x)), 0)

    def test_electric_energy_flux_identity(self):
        # div(J+D_t)=0 follows by differentiating Gauss and charge continuity.
        x, k, epsilon, u = symbols("x k epsilon u", positive=True)
        potential = cos(k * x)
        electric = -diff(potential, x)
        displacement = epsilon * electric
        charge = diff(displacement, x)
        current = charge * u
        potential_rate = -u * diff(potential, x)
        displacement_rate = -epsilon * diff(potential_rate, x)
        energy_rate = electric * displacement_rate
        flux = potential * (current + displacement_rate)
        self.assertEqual(simplify(diff(displacement_rate + current, x)), 0)
        self.assertEqual(simplify(energy_rate + diff(flux, x) + current * electric), 0)

    def test_energy_partition(self):
        model = Electrodynamics()
        fluid_energy = symbols("fluid_energy")
        combined = fluid_energy + model.hamiltonian
        self.assertEqual(simplify(combined - fluid_energy - model.hamiltonian), 0)
        self.assertEqual(model.conduction_current, Matrix.zeros(3, 1))

    def test_generated_electric_only_hamiltonian(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "electrodynamics_hamiltonian.hpp"
            generate(output)
            text = output.read_text()
            self.assertIn("ElectrodynamicsHamiltonian", text)
            self.assertIn("dhamiltonian_dmoments", text)
            self.assertIn("AI-GENERATED", text)
            self.assertNotIn("permeability", text)
            self.assertNotIn("Derivative(", text)


if __name__ == "__main__":
    unittest.main()
