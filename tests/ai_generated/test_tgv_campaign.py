# SPDX-FileCopyrightText: 2026 Baptiste Legouix
# SPDX-License-Identifier: AGPL-3.0-or-later
# AI-GENERATED
"""Check that the campaign detects changed physics while ignoring timing noise."""

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np

sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "benchmarks/taylor_green_vortex")
)
from ci_campaign import (
    STRATEGIES,
    campaign,
    compare_vorticity_slice,
    reference_history,
    compare_strategies,
    validate_trajectory,
    write_report,
)


class CampaignConfiguration(unittest.TestCase):
    def test_summary_omits_the_solver_timing_table(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            write_report(output, "performance", [])
            report = (output / "report.md").read_text()
            self.assertNotIn("Solver timings exclude", report)
            self.assertNotIn("| N | Flags | Steps |", report)

    def test_codifferential_strategy_precedes_the_staged_laplacian(self):
        self.assertEqual(len(STRATEGIES), 7)
        self.assertEqual(
            STRATEGIES[-2],
            ("--staged-codifferential",) + STRATEGIES[-1][1:],
        )

    def test_reference_benchmark_names_include_explicit_iterations_and_repeats(self):
        for fluid in ("incompressible", "compressible"):
            for suffix in ("/iterations:1/repeats:1", "/iterations:2/repeats:1"):
                with (
                    self.subTest(fluid=fluid, suffix=suffix),
                    tempfile.TemporaryDirectory() as temporary,
                ):
                    output = Path(temporary)
                    directory = output / "validation"
                    directory.mkdir()
                    rows = []
                    for n in (64, 128):
                        case = directory / f"N{n}"
                        case.mkdir()
                        (case / "trajectory.performance.json").touch()
                        rows.append(
                            {
                                "run_name": f"TaylorGreen/{fluid}_ideal_hd/N{n}/"
                                + "+".join(STRATEGIES[3])
                                + suffix,
                                "repetitions": 1,
                            }
                        )
                    (directory / "benchmark.json").write_text(
                        json.dumps({"benchmarks": rows})
                    )
                    with (
                        patch("ci_campaign.reference_history"),
                        patch("ci_campaign.execute"),
                        patch(
                            "ci_campaign.validate_trajectory",
                            side_effect=RuntimeError("Reached trajectory checks"),
                        ) as validate,
                    ):
                        if suffix == "/iterations:1/repeats:1":
                            with self.assertRaisesRegex(
                                RuntimeError, "Reached trajectory checks"
                            ):
                                campaign(
                                    Path("unused"),
                                    output,
                                    fluid,
                                    "ideal",
                                    "hd",
                                    "validation",
                                )
                            validate.assert_called_once()
                        else:
                            with self.assertRaisesRegex(
                                ValueError, "Unexpected or duplicate"
                            ):
                                campaign(
                                    Path("unused"),
                                    output,
                                    fluid,
                                    "ideal",
                                    "hd",
                                    "validation",
                                )
                            validate.assert_not_called()

    def test_last_strategy_has_the_requested_ordered_operator_chain(self):
        self.assertEqual(
            STRATEGIES[-1],
            (
                "--staged-laplacian",
                "--weighted-levi-civita-discrete-hodge-star",
                "--diagonal-reduction",
                "--levi-civita-continuous-hodge-star",
                "--diagonal-reconstruction",
            ),
        )


class ReferenceSnapshots(unittest.TestCase):
    def test_dense_reference_schedule_and_missing_exports(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "trajectory.performance.json"
            path.write_text(
                json.dumps(
                    {
                        "configuration": "compressible_ideal_hd",
                        "N": 8,
                        "mode": "reference",
                        "steps": 1,
                        "solver_seconds": 1,
                        "mean_step_seconds": 1,
                        "cell_updates_per_second": 512,
                        "final_time": 10,
                    }
                )
            )
            hdf = Path(temporary) / "trajectory.h5"
            times = sorted([i * 0.033 for i in range(304)] + [8.0, 10.0])
            with h5py.File(hdf, "w") as data:
                monitors = np.zeros((2, 14))
                monitors[:, 0] = [0, 10]
                monitors[:, 1] = 0.125
                monitors[:, 11:13] = 1
                data["monitors"] = monitors
                data["reconstruction_monitors"] = np.zeros((2, 4))
                for i, time in enumerate(times):
                    data[f"snapshot_{i}/time"] = [time]
            xmf = hdf.with_suffix(".xmf")
            grids = [f'<Grid><Time Value="{time:.17g}"/></Grid>' for time in times]
            xmf.write_text(
                "<Xdmf><Domain><Grid>" + "".join(grids) + "</Grid></Domain></Xdmf>"
            )
            validate_trajectory(path, "compressible_ideal_hd", 8, reference=True)
            xmf.write_text(
                "<Xdmf><Domain><Grid>" + "".join(grids[:-1]) + "</Grid></Domain></Xdmf>"
            )
            with self.assertRaisesRegex(ValueError, "XDMF time series is incomplete"):
                validate_trajectory(path, "compressible_ideal_hd", 8, reference=True)
            with h5py.File(hdf, "r+") as data:
                del data["snapshot_1"]
            with self.assertRaisesRegex(ValueError, "snapshots every 0.033 s"):
                validate_trajectory(path, "compressible_ideal_hd", 8, reference=True)


class StrategyComparison(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.baseline = Path(self.directory.name) / "baseline.h5"
        self.candidate = Path(self.directory.name) / "candidate.h5"
        with h5py.File(self.baseline, "w") as data:
            data["final_state"] = np.ones((3, 32))
            data["final_electromagnetic_state"] = np.ones((7, 32))
            monitors = np.ones((6, 14))
            monitors[:, 0] = np.arange(6) * 0.01
            data["monitors"] = monitors
            data["electromagnetic_monitors"] = np.ones((6, 9))
        shutil.copyfile(self.baseline, self.candidate)

    def test_timing_noise_and_roundoff_do_not_fail_but_changed_time_does(self):
        with h5py.File(self.candidate, "r+") as data:
            data["monitors"][:, 13] = 1000
            data["final_state"][0, 0] += 1e-10
        self.assertTrue(
            all(
                item["passed"]
                for item in compare_strategies(self.baseline, self.candidate).values()
            )
        )
        with h5py.File(self.candidate, "r+") as data:
            data["monitors"][3, 0] += 1e-5
        self.assertFalse(
            compare_strategies(self.baseline, self.candidate)["monitors"]["passed"]
        )

    def test_changed_field_sector_is_detected_even_with_identical_fluid(self):
        with h5py.File(self.candidate, "r+") as data:
            data["final_electromagnetic_state"][1, 7] += 1e-3
        results = compare_strategies(self.baseline, self.candidate)
        self.assertTrue(results["final_state"]["passed"])
        self.assertFalse(results["final_electromagnetic_state"]["passed"])

    def test_nonfinite_state_cannot_pass(self):
        with h5py.File(self.candidate, "r+") as data:
            data["final_state"][0, 0] = np.nan
        with self.assertRaises(ValueError):
            compare_strategies(self.baseline, self.candidate)


class NumericalReference(unittest.TestCase):
    def test_original_inputs_and_matching_periodic_slice_nodes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference_directory = root / "reference"
            history = reference_history(reference_directory)
            self.assertEqual(history.shape, (2000, 4))
            self.assertEqual(history[0, 1], 0.125)
            self.assertEqual(history[0, 3], 0.375)
            self.assertEqual(len(list(reference_directory.iterdir())), 4)
            reference = np.fromfile(
                reference_directory / "wn_slice_x0_08000.out", dtype="<f8"
            ).reshape(512, 512)
            # Use a deliberately asymmetric reference to detect a mistaken
            # transpose in the binary storage/coordinate interpretation.
            reference = reference + np.arange(512)[:, None] / 512
            reference.astype("<f8").tofile(
                reference_directory / "wn_slice_x0_08000.out"
            )
            n = 8
            grid = -np.pi + np.arange(n) * (2 * np.pi / n)
            position = np.zeros((1, n, n, 3))
            position[..., 0] = -np.pi
            position[0, ..., 1] = grid[:, None]
            position[0, ..., 2] = grid[None, :]
            vorticity = np.zeros((1, n, n, 3))
            vorticity[0, ..., 0] = reference[::64, ::64]
            hdf = root / "trajectory.h5"
            with h5py.File(hdf, "w") as data:
                data["position"] = position
                data["snapshot_0/time"] = [8.0]
                data["snapshot_0/vorticity"] = vorticity
            metrics = compare_vorticity_slice(hdf, reference_directory, tolerance=0.01)
            self.assertEqual(metrics["relative_l2_error"], 0)
            self.assertTrue(metrics["passed"])
            self.assertEqual(metrics["samples"], n * n)
            self.assertTrue(hdf.with_suffix(".vorticity-comparison.png").is_file())
            with h5py.File(hdf, "r+") as data:
                data["snapshot_0/vorticity"][0, 1, 3, 0] += 50
            metrics = compare_vorticity_slice(hdf, reference_directory, tolerance=0.01)
            self.assertFalse(metrics["passed"])
            self.assertEqual(metrics["max_absolute_error"], 50)
            with h5py.File(hdf, "r+") as data:
                data["snapshot_0/time"][0] = 7.9999
            with self.assertRaisesRegex(ValueError, "exactly one t=8"):
                compare_vorticity_slice(hdf, reference_directory)

    def test_reference_checksum_rejects_changed_original_data(self):
        source = (
            Path(__file__).resolve().parents[2]
            / "benchmarks/taylor_green_vortex/reference_data"
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shutil.copytree(source, root / "reference_data")
            (root / "reference_data/spectral_Re1600_512.gdiag").write_text("corrupted")
            with patch("ci_campaign.__file__", str(root / "ci_campaign.py")):
                with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                    reference_history(root / "artifact")


if __name__ == "__main__":
    unittest.main()
