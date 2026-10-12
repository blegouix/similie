# SPDX-FileCopyrightText: 2026 Baptiste Legouix
# SPDX-License-Identifier: AGPL-3.0-or-later
# // AI-GENERATED

"""Regression tests for the physical reference checks."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from check_fields import check, check_exports, compare


class ReferenceChecks(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.similie = Path(self.directory.name) / "similie.pos"
        self.reference = Path(self.directory.name) / "reference.pos"

    def write_potential(self, path, values, offset=0):
        path.write_text(
            'View "potential" {\n'
            f"SQ({offset},0,0,{offset + 1},0,0,{offset + 1},1,0,{offset},1,0)"
            "{" + ",".join(map(str, values)) + "};\n};\n"
        )

    def test_constant_potential_offset_preserves_physics(self):
        self.write_potential(self.reference, [0, 2, 5, 3])
        self.write_potential(self.similie, [7, 9, 12, 10])
        metrics = compare(self.similie, self.reference)
        self.assertEqual(metrics["velocity_relative_area_l2"], 0)
        self.assertEqual(metrics["pressure_relative_area_l2"], 0)
        check(metrics)

    def test_corner_corruption_is_rejected(self):
        self.write_potential(self.reference, [0, 2, 5, 3])
        self.write_potential(self.similie, [0, 2, 5, 30])
        metrics = compare(self.similie, self.reference)
        # Verify the local guard independently of the global error guards.
        metrics["velocity_relative_area_l2"] = 0
        metrics["pressure_relative_area_l2"] = 0
        with self.assertRaisesRegex(ValueError, "corner"):
            check(metrics)

    def test_exported_vertex_fields_follow_the_potential(self):
        def write_export(pressure):
            self.write_potential(self.similie, [0, 2, 5, 3])
            with self.similie.open("a") as stream:
                stream.write(
                    'View "SimiLie velocity" {\n'
                    "VQ(0,0,0,1,0,0,1,1,0,0,1,0){2,3,0,2,3,0,2,3,0,2,3,0};\n};\n"
                    'View "SimiLie pressure difference" {\n'
                    "SQ(0,0,0,1,0,0,1,1,0,0,1,0){"
                    + ",".join([str(pressure)] * 4)
                    + "};\n};\n"
                )

        write_export(-7.9625)
        check_exports(self.similie)
        write_export(0)
        with self.assertRaisesRegex(ValueError, "Bernoulli"):
            check_exports(self.similie)

    def test_alternating_nodal_error_is_visible(self):
        self.write_potential(self.reference, [0, 2, 5, 3])
        self.write_potential(self.similie, [1, 1, 6, 2])
        metrics = compare(self.similie, self.reference)
        self.assertGreater(metrics["velocity_relative_area_l2"], 0.1)
        self.assertGreater(metrics["pressure_relative_area_l2"], 0.1)
        with self.assertRaises(ValueError):
            check(metrics)

    def test_nonfinite_potential_is_rejected(self):
        self.write_potential(self.reference, [0, 2, 5, 3])
        self.write_potential(self.similie, [0, 2, float("nan"), 3])
        with self.assertRaisesRegex(ValueError, "non-finite"):
            compare(self.similie, self.reference)

    def test_different_mesh_is_rejected(self):
        self.write_potential(self.reference, [0, 2, 5, 3])
        self.write_potential(self.similie, [0, 2, 5, 3], offset=1)
        with self.assertRaisesRegex(ValueError, "identical cells"):
            compare(self.similie, self.reference)

    def test_zero_reference_mass_flow_is_compared(self):
        log = Path(self.directory.name) / "similie.log"
        log.write_text(
            "SimiLie potential-flow diagnostics: circulation=1, mass flow rate=1\n"
        )
        self.reference.write_text("0 1\n0 0\n")
        result = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("check_reference.py")),
                str(log),
                str(self.reference),
            ],
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("mass-flow difference", result.stderr)

    def test_imposed_circulation_has_unreported_reference_mass_flow(self):
        log = Path(self.directory.name) / "similie.log"
        log.write_text(
            "SimiLie potential-flow diagnostics: circulation=1, mass flow rate=1\n"
        )
        for mass_flow, expected_status in [("0", 0), ("nan", 1), ("1", 1)]:
            self.reference.write_text(f"0 1\n0 {mass_flow}\n")
            result = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).with_name("check_reference.py")),
                    str(log),
                    str(self.reference),
                    "--imposed-circulation",
                    "1",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, expected_status)

    def test_nonfinite_reference_diagnostic_is_rejected(self):
        log = Path(self.directory.name) / "similie.log"
        log.write_text(
            "SimiLie potential-flow diagnostics: circulation=1, mass flow rate=1\n"
        )
        for reference in ["0 nan\n0 1\n", "0 1\n0 nan\n"]:
            self.reference.write_text(reference)
            result = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).with_name("check_reference.py")),
                    str(log),
                    str(self.reference),
                ],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("non-finite", result.stderr)


if __name__ == "__main__":
    unittest.main()
