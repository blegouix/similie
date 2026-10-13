# SPDX-FileCopyrightText: 2026 Baptiste Legouix
# SPDX-License-Identifier: AGPL-3.0-or-later
# AI-GENERATED
"""Check Google Benchmark iteration accounting and evolving solver state in Docker."""

import argparse
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
from post_process import process


class BenchmarkIterationTests(unittest.TestCase):
    binary: Path

    def execute(self, directory, fluid, steps, repetitions):
        report = directory / "benchmark.json"
        output = directory / "fields"
        subprocess.run(
            [
                str(self.binary),
                "--N=8",
                f"--{fluid}",
                f"--benchmark_min_time={steps}x",
                f"--benchmark_repetitions={repetitions}",
                f"--benchmark_out={report}",
                "--benchmark_out_format=json",
                f"--output-dir={output}",
            ],
            cwd=directory,
            check=True,
            capture_output=True,
            text=True,
            timeout=180,
        )
        records = json.loads(report.read_text())["benchmarks"]
        measurements = [r for r in records if r.get("run_type") == "iteration"]
        self.assertEqual(len(measurements), repetitions)
        for measurement in measurements:
            self.assertFalse(measurement.get("error_occurred", False))
            self.assertEqual(measurement["iterations"], steps)
            self.assertEqual(measurement["steps"], steps)
            self.assertGreater(measurement["simulation_time"], 0)
        files = sorted(output.glob("*.h5"))
        self.assertEqual(len(files), repetitions)
        histories = []
        for path in files:
            performance = json.loads(path.with_suffix(".performance.json").read_text())
            self.assertEqual(performance["steps"], steps)
            self.assertAlmostEqual(
                performance["mean_step_seconds"] * steps, performance["solver_seconds"]
            )
            with h5py.File(path, "r") as data:
                monitors = data["monitors"][:]
                self.assertEqual(len(monitors), steps + 1)
                self.assertEqual(monitors[0, 0], 0)
                self.assertTrue(np.all(np.diff(monitors[:, 0]) > 0))
                snapshots = sorted(
                    (name for name in data if name.startswith("snapshot_")),
                    key=lambda name: float(data[name]["time"][0]),
                )
                final = data[snapshots[-1]]
                self.assertAlmostEqual(final["time"][0], monitors[-1, 0])
                self.assertAlmostEqual(performance["final_time"], monitors[-1, 0])
                self.assertGreater(
                    np.max(
                        np.abs(final["velocity"][:] - data[snapshots[0]]["velocity"][:])
                    ),
                    1e-12,
                )
                histories.append(monitors)
            process(path, vtk=False, plots=False)
        return histories

    def test_step_counts_continuation_and_repetition_reset(self):
        for fluid in ("incompressible", "compressible"):
            with self.subTest(fluid=fluid), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                one_directory = root / "one"
                three_directory = root / "three"
                one_directory.mkdir()
                three_directory.mkdir()
                one = self.execute(one_directory, fluid, steps=1, repetitions=1)
                three = self.execute(three_directory, fluid, steps=3, repetitions=2)
                # Both runs start at the same state; a longer run continues after its first step.
                np.testing.assert_allclose(
                    one[0][:, :13], three[0][:2, :13], rtol=1e-9, atol=1e-11
                )
                # Separate repetitions restart, unlike iterations within a repetition.
                np.testing.assert_allclose(
                    three[0][:, :13], three[1][:, :13], rtol=1e-9, atol=1e-11
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    arguments = parser.parse_args()
    BenchmarkIterationTests.binary = arguments.binary.resolve()
    unittest.main(argv=[__file__])
