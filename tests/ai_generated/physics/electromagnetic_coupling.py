# SPDX-FileCopyrightText: 2026 Baptiste Legouix
# SPDX-License-Identifier: AGPL-3.0-or-later
# AI-GENERATED

import sys
import unittest
from pathlib import Path

from sympy import Matrix, diff, simplify

sys.path.insert(
    0, str(Path(__file__).resolve().parents[3] / "src/similie/physics/coupling")
)

from electromagnetic_coupling import ElectromagneticInteraction


class MinimalCouplingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = ElectromagneticInteraction()

    def test_canonical_velocity(self):
        model = self.model
        expected = (model.momentum - model.charge * model.vector_potential) / model.mass
        self.assertEqual(
            (model.canonical_velocity - expected).applyfunc(simplify),
            Matrix.zeros(3, 1),
        )

    def test_time_dependent_potential_produces_lorentz_force(self):
        model = self.model
        expected = model.charge * (
            model.electric_field + model.velocity.cross(model.induction)
        )
        self.assertEqual(
            (model.mechanical_force - expected).applyfunc(simplify), Matrix.zeros(3, 1)
        )

    def test_current_is_independent_of_bulk_charge_advection(self):
        model = self.model
        self.assertEqual(
            model.electric_force_density, model.charge_density * model.electric_field
        )
        self.assertEqual(
            model.magnetic_force_density, model.current.cross(model.induction)
        )
        self.assertEqual(
            simplify(
                model.fluid_energy_source - model.current.dot(model.electric_field)
            ),
            0,
        )
        self.assertEqual(model.magnetic_power, 0)

    def test_gauge_transformation_leaves_mechanical_velocity_unchanged(self):
        model = self.model
        # A nonuniform, time-dependent gauge changes canonical momentum as well
        # as A. Omitting that momentum transformation would change the physics.
        gauge = model.position[0] * model.position[1] * model.time
        gradient = Matrix([diff(gauge, coordinate) for coordinate in model.position])
        substitutions = dict(
            zip(model.momentum, model.momentum + model.charge * gradient)
        )
        substitutions.update(
            dict(zip(model.vector_potential, model.vector_potential + gradient))
        )
        transformed = model.canonical_velocity.subs(substitutions, simultaneous=True)
        self.assertEqual(
            (transformed - model.canonical_velocity).applyfunc(simplify),
            Matrix.zeros(3, 1),
        )


if __name__ == "__main__":
    unittest.main()
