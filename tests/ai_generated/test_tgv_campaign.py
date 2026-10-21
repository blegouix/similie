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

import h5py
import numpy as np

sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "benchmarks/taylor_green_vortex")
)
from ci_campaign import STRATEGIES, compare_spectrum, compare_strategies


class CampaignConfiguration(unittest.TestCase):
    def test_last_strategy_has_the_requested_ordered_operator_chain(self):
        self.assertEqual(
            STRATEGIES[-1],
            (
                "--staged-laplacian",
                "--levi-civita-discrete-hodge-star",
                "--diagonal-reduction",
                "--levi-civita-continuous-hodge-star",
                "--diagonal-reconstruction",
            ),
        )


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

    def test_image_reference_is_not_extrapolated_to_unshown_low_modes(self):
        source = (
            Path(__file__).resolve().parents[2]
            / "benchmarks/taylor_green_vortex/reference_data"
        )
        reference = np.loadtxt(source / "spectrum_t8.csv", delimiter=",", skiprows=4)
        wave_number = np.arange(33)
        spectrum = np.ones(33)
        selected = wave_number >= reference[0, 0]
        spectrum[selected] = 10 ** np.interp(
            np.log10(wave_number[selected]),
            np.log10(reference[:, 0]),
            np.log10(reference[:, 1]),
        )
        # These low modes are not present in the PDF and must not affect its error.
        spectrum[:5] = 1e20
        metrics = compare_spectrum(
            self.baseline,
            wave_number,
            spectrum,
            Path(self.directory.name) / "reference",
        )
        self.assertEqual(metrics["wave_number_range"], [5, 32])
        self.assertLess(metrics["relative_l2_error"], 1e-12)
        self.assertTrue(self.baseline.with_suffix(".spectrum-comparison.png").is_file())
        self.assertEqual(
            json.loads(
                self.baseline.with_suffix(".spectrum-comparison.json").read_text()
            )["time"],
            8,
        )


if __name__ == "__main__":
    unittest.main()
