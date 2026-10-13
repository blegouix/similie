# SPDX-FileCopyrightText: 2026 Baptiste Legouix
# SPDX-License-Identifier: AGPL-3.0-or-later
# AI-GENERATED
"""Validate CI performance evidence and reject incomplete/mislabeled results."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
from ci_campaign import (
    PERFORMANCE_REPETITIONS,
    PERFORMANCE_STEPS,
    validate_results,
    write_report,
)


class CampaignEvidenceTests(unittest.TestCase):
    def fixture(self, directory, reference=False):
        repeats = 1 if reference else PERFORMANCE_REPETITIONS
        times = [0, 8, 10] if reference else np.linspace(0, 0.02, PERFORMANCE_STEPS + 1)
        steps = len(times) - 1
        for repeat in range(repeats):
            base = directory / f"tgv_run{repeat}"
            performance = {
                "configuration": "compressible_ideal_hd",
                "N": 64,
                "mode": "reference" if reference else "performance",
                "steps": steps,
                "final_time": float(times[-1]),
                "solver_seconds": 0.01 * steps,
                "mean_step_seconds": 0.01,
                "cell_updates_per_second": 64**3 / 0.01,
            }
            base.with_suffix(".performance.json").write_text(json.dumps(performance))
            with h5py.File(base.with_suffix(".h5"), "w") as data:
                monitors = np.zeros((steps + 1, 14))
                monitors[:, 0] = times
                monitors[:, 1] = 0.125
                monitors[:, 8] = 1
                monitors[:, 11:13] = 1
                data["monitors"] = monitors
                constraints = np.zeros((steps + 1, 5))
                constraints[:, 0] = times
                data["reconstruction_monitors"] = constraints
                if reference:
                    for index, time in enumerate(times):
                        data.create_group(f"snapshot_{index}")["time"] = [time]
        if not reference:
            (directory / "benchmark.json").write_text(
                json.dumps(
                    {
                        "benchmarks": [
                            {
                                "run_type": "iteration",
                                "iterations": steps,
                                "steps": steps,
                            }
                            for _ in range(repeats)
                        ]
                    }
                )
            )

    def test_completed_performance_repetitions(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.fixture(directory)
            results = validate_results(directory, "compressible_ideal_hd", 64, False)
            self.assertEqual(len(results), PERFORMANCE_REPETITIONS)

    def test_wrong_iteration_accounting_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.fixture(directory)
            report = directory / "benchmark.json"
            data = json.loads(report.read_text())
            data["benchmarks"][0]["iterations"] = PERFORMANCE_STEPS + 1
            report.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "wrong number of steps"):
                validate_results(directory, "compressible_ideal_hd", 64, False)

    def test_reference_requires_complete_history(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.fixture(directory, reference=True)
            with patch("ci_campaign.process") as post_process:
                results = validate_results(directory, "compressible_ideal_hd", 64, True)
                self.assertEqual(results[0]["final_time"], 10)
                post_process.assert_called_once()
            with h5py.File(directory / "tgv_run0.h5", "a") as data:
                del data["snapshot_1"]
            with self.assertRaisesRegex(ValueError, "snapshots 0,8,10"):
                validate_results(directory, "compressible_ideal_hd", 64, True)

    def test_wrong_physics_and_nonfinite_timing_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.fixture(directory)
            with self.assertRaisesRegex(ValueError, "Wrong configuration"):
                validate_results(directory, "incompressible_ideal_hd", 64, False)
            path = directory / "tgv_run0.performance.json"
            record = json.loads(path.read_text())
            record["solver_seconds"] = float("nan")
            path.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, "Invalid solver_seconds"):
                validate_results(directory, "compressible_ideal_hd", 64, False)

    def test_report_mean_and_variation(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            records = [
                {
                    "mean_step_seconds": duration,
                    "steps": 5,
                    "final_time": 0.02,
                    "cell_updates_per_second": 2e6,
                }
                for duration in (0.001, 0.003)
            ]
            manifest = {
                "configuration": "compressible_ideal_hd",
                "runs": [
                    {
                        "mode": "performance",
                        "N": 64,
                        "status": "passed",
                        "performance": records,
                    }
                ],
            }
            write_report(directory, manifest)
            report = (directory / "report.md").read_text()
            self.assertIn(
                "| performance | 64 | 5 | 2 | 0.02 | 2 | 1 | 2 | passed |", report
            )
            self.assertEqual(
                json.loads((directory / "manifest.json").read_text()), manifest
            )


if __name__ == "__main__":
    unittest.main()
