#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Baptiste Legouix
# SPDX-License-Identifier: AGPL-3.0-or-later
# AI-GENERATED

"""Full Maxwell charge, energy, force, and resistive coupling identities."""

import unittest

from electromagnetism import Electromagnetism, ResistiveElectromagnetism
from sympy import Matrix, diff, oo, simplify, symbols


class ElectromagnetismTests(unittest.TestCase):
    def test_combined_hamiltonian(self):
        model = Electromagnetism()
        self.assertEqual(
            model.hamiltonian, model.electric.hamiltonian + model.magnetic.hamiltonian
        )
        for a in range(3):
            self.assertEqual(
                diff(model.hamiltonian, model.displacement[a]), model.electric_field[a]
            )
            self.assertEqual(
                diff(model.hamiltonian, model.induction[a]), model.magnetic_field[a]
            )

    def test_ideal_advected_charge(self):
        model = Electromagnetism()
        self.assertEqual(model.current, model.charge_density * model.velocity)
        self.assertEqual(model.ohmic_heating, 0)
        self.assertEqual(
            simplify(
                model.fluid_energy_source - model.velocity.dot(model.lorentz_force)
            ),
            0,
        )

    def test_resistive_power_identity(self):
        model = ResistiveElectromagnetism()
        self.assertEqual(
            simplify(
                model.fluid_energy_source
                - model.velocity.dot(model.lorentz_force)
                - model.ohmic_heating
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
        self.assertEqual(
            simplify(
                model.ohmic_heating
                - model.rest_frame_electric_field.dot(model.rest_frame_electric_field)
                / model.resistivity
            ),
            0,
        )

    def test_nonconducting_limit(self):
        ideal, resistive = Electromagnetism(), ResistiveElectromagnetism()
        self.assertEqual(
            resistive.current.subs(resistive.resistivity, oo), ideal.current
        )
        self.assertEqual(resistive.ohmic_heating.subs(resistive.resistivity, oo), 0)

    def test_maxwell_energy_exchange(self):
        for model in (Electromagnetism(), ResistiveElectromagnetism()):
            curl_h = Matrix(symbols("curlH0:3"))
            curl_e = Matrix(symbols("curlE0:3"))
            field_power = model.electric_field.dot(
                curl_h - model.current
            ) - model.magnetic_field.dot(curl_e)
            flux_divergence = model.magnetic_field.dot(
                curl_e
            ) - model.electric_field.dot(curl_h)
            self.assertEqual(
                simplify(field_power + flux_divergence + model.fluid_energy_source), 0
            )

    def test_gauss_constraint_and_faraday(self):
        x, y, z = symbols("x y z")
        h = Matrix([x * y, y * z, z * x])
        e = Matrix([x * x * y, y * y * z, z * z * x])
        current = Matrix([x * x, y * y, z * z])

        def curl(field):
            return Matrix(
                [
                    diff(field[2], y) - diff(field[1], z),
                    diff(field[0], z) - diff(field[2], x),
                    diff(field[1], x) - diff(field[0], y),
                ]
            )

        def div(field):
            return diff(field[0], x) + diff(field[1], y) + diff(field[2], z)

        self.assertEqual(simplify(div(curl(h) - current) + div(current)), 0)
        self.assertEqual(simplify(div(-curl(e))), 0)

    def test_total_energy_partition(self):
        model = Electromagnetism()
        fluid_energy = symbols("fluid_energy")
        total = fluid_energy + model.hamiltonian
        self.assertEqual(simplify(total - model.hamiltonian - fluid_energy), 0)
        self.assertNotEqual(model.hamiltonian, model.magnetic.hamiltonian)
        self.assertNotEqual(model.hamiltonian, model.electric.hamiltonian)

    def test_maxwell_wave_speed(self):
        epsilon, mu, k, omega = symbols("epsilon mu k omega", positive=True)
        # D_y(t,x), B_z(t,x): D_t=-B_x/mu, B_t=-D_x/epsilon.
        wave_matrix = Matrix([[omega, -k / mu], [-k / epsilon, omega]])
        self.assertEqual(
            simplify(wave_matrix.det() - omega**2 + k * k / (epsilon * mu)), 0
        )


if __name__ == "__main__":
    unittest.main()
