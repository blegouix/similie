#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Baptiste Legouix
# SPDX-License-Identifier: AGPL-3.0-or-later
# AI-GENERATED

"""Independent Hamiltonian, MHD coupling, and resistive-energy identities."""

import tempfile
import unittest
from pathlib import Path

from magnetodynamics import Magnetodynamics, ResistiveMagnetodynamics, generate
from sympy import Matrix, diff, simplify, symbols


class MagnetodynamicsTests(unittest.TestCase):
    def test_hamiltonian_derivatives(self):
        model = Magnetodynamics()
        self.assertFalse(hasattr(model, "displacement"))
        self.assertEqual(model.magnetic_field, model.induction / model.permeability)
        self.assertEqual(
            model.magnetic_energy,
            model.induction.dot(model.induction) / (2 * model.permeability),
        )
        for a in range(3):
            self.assertEqual(
                diff(model.hamiltonian, model.induction[a], 2), 1 / model.permeability
            )

    def test_ideal_poynting_and_lorentz_exchange(self):
        model = Magnetodynamics()
        expected = (
            model.induction.dot(model.induction) * model.velocity
            - model.velocity.dot(model.induction) * model.induction
        ) / model.permeability
        self.assertEqual(simplify(model.poynting_flux - expected), Matrix.zeros(3, 1))
        self.assertEqual(
            simplify(
                model.fluid_energy_source - model.velocity.dot(model.lorentz_force)
            ),
            0,
        )
        self.assertEqual(model.momentum_flux, model.momentum_flux.T)

    def test_resistive_power_and_zero_resistivity(self):
        ideal, resistive = Magnetodynamics(), ResistiveMagnetodynamics()
        expected = (
            resistive.velocity.dot(resistive.lorentz_force) + resistive.ohmic_heating
        )
        self.assertEqual(simplify(resistive.fluid_energy_source - expected), 0)
        self.assertEqual(
            simplify(
                resistive.poynting_flux
                - ideal.poynting_flux
                - resistive.resistivity
                * resistive.current.cross(resistive.induction)
                / resistive.permeability
            ),
            Matrix.zeros(3, 1),
        )
        self.assertEqual(
            resistive.ohm_electric_field.subs(resistive.resistivity, 0),
            ideal.ohm_electric_field,
        )
        self.assertEqual(resistive.ohmic_heating.subs(resistive.resistivity, 0), 0)

    def test_total_energy_and_pressure_mapping(self):
        model = Magnetodynamics()
        fluid_energy, pressure = symbols("fluid_energy p")
        total = fluid_energy + model.magnetic_energy
        ideal_energy_flux = (
            fluid_energy + pressure
        ) * model.velocity + model.poynting_flux
        expected = (total + pressure + model.magnetic_energy) * model.velocity
        expected -= model.velocity.dot(model.induction) * model.magnetic_field
        self.assertEqual(simplify(ideal_energy_flux - expected), Matrix.zeros(3, 1))
        self.assertEqual(simplify(total - model.magnetic_energy - fluid_energy), 0)

    def test_magnetic_energy_and_faraday_sign(self):
        model = Magnetodynamics()
        curl_e = Matrix(symbols("curlE0:3"))
        energy_rate = -model.magnetic_field.dot(curl_e)
        poynting_divergence = model.magnetic_field.dot(
            curl_e
        ) - model.ohm_electric_field.dot(model.current)
        self.assertEqual(
            simplify(energy_rate + poynting_divergence + model.fluid_energy_source), 0
        )

    def test_resistive_diffusion_sign(self):
        x, y, z, k, eta, mu = symbols("x y z k eta mu", positive=True)
        from sympy import cos

        def curl(field):
            return Matrix(
                [
                    diff(field[2], y) - diff(field[1], z),
                    diff(field[0], z) - diff(field[2], x),
                    diff(field[1], x) - diff(field[0], y),
                ]
            )

        field = Matrix([0, cos(k * x), 0])
        rate = -curl(eta * curl(field / mu))
        self.assertEqual(simplify(rate + eta / mu * k * k * field), Matrix.zeros(3, 1))

    def test_generated_hamiltonian(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "magnetodynamics_hamiltonian.hpp"
            generate(output)
            text = output.read_text()
            self.assertIn("AI-GENERATED", text)
            self.assertNotIn("permittivity", text)
            self.assertIn("dhamiltonian_dmoments", text)
            self.assertNotIn("Derivative(", text)


if __name__ == "__main__":
    unittest.main()
