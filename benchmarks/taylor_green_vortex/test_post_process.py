# SPDX-FileCopyrightText: 2026 Baptiste Legouix
# SPDX-License-Identifier: AGPL-3.0-or-later
# AI-GENERATED
"""Independent normalization and output-schema checks; run inside Docker."""

import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
from post_process import (
    COLUMNS,
    EM_COLUMNS,
    EM_FIELDS,
    energy_spectrum,
    process,
    write_vtk,
)


class PostProcessingTests(unittest.TestCase):
    def setUp(self):
        self.n = 8
        self.x = np.arange(self.n) * 2 * np.pi / self.n - np.pi
        x, y, z = np.meshgrid(self.x, self.x, self.x, indexing="ij")
        self.position = np.stack((x, y, z), axis=-1)
        self.velocity = np.stack(
            (
                np.sin(x) * np.cos(y) * np.cos(z),
                -np.cos(x) * np.sin(y) * np.cos(z),
                np.zeros_like(x),
            ),
            axis=-1,
        )
        self.density = np.ones((self.n,) * 3)

    def test_taylor_green_shell_and_parseval(self):
        k, spectrum, outside = energy_spectrum(self.velocity, self.density)
        # Eight modes (±1,±1,±1) lie in shell round(sqrt(3))=2.
        self.assertEqual(k[2], 2)
        self.assertAlmostEqual(spectrum[2], 1 / 8, places=13)
        self.assertAlmostEqual(spectrum.sum(), 1 / 8, places=13)
        self.assertLess(outside, 1e-28)

    def test_density_weighting(self):
        density = self.density * 1.3
        _, spectrum, outside = energy_spectrum(self.velocity, density)
        self.assertAlmostEqual(spectrum.sum() + outside, 1.3 / 8, places=13)

    def test_cutoff_corner_energy_is_reported(self):
        index = np.indices((self.n,) * 3).sum(axis=0)
        velocity = np.zeros_like(self.velocity)
        velocity[..., 0] = (-1.0) ** index
        _, spectrum, outside = energy_spectrum(velocity, self.density)
        self.assertLess(spectrum.sum(), 1e-28)
        self.assertAlmostEqual(outside, 0.5, places=13)

    def test_vtk_x_fast_order(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "grid.vtk"
            write_vtk(path, self.position, {"density": self.density}, time=8)
            lines = path.read_text().splitlines()
            start = (
                next(i for i, line in enumerate(lines) if line.startswith("POINTS "))
                + 1
            )
            points = np.loadtxt(lines[start : start + self.n**3])
            np.testing.assert_allclose(points[0], self.position[0, 0, 0])
            np.testing.assert_allclose(points[1], self.position[1, 0, 0])
            np.testing.assert_allclose(points[self.n], self.position[0, 1, 0])
            np.testing.assert_allclose(points[self.n**2], self.position[0, 0, 1])
            self.assertIn("time 1 1 double", lines)

    def test_hdf_schema_slice_and_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.h5"
            path.with_suffix(".dat").write_text(
                "# code name: SimiLie\n# mesh resolution: 8 8 8\n"
            )
            with h5py.File(path, "w") as data:
                data.attrs["monitor_columns"] = ",".join(COLUMNS)
                data["position"] = self.position
                monitors = np.zeros((3, len(COLUMNS)))
                monitors[:, 0] = (0, 8, 10)
                monitors[:, 1] = 1 / 8
                data["monitors"] = monitors
                constraints = np.zeros((len(monitors), 5))
                constraints[:, 0] = monitors[:, 0]
                data["reconstruction_monitors"] = constraints
                electromagnetic = np.zeros((len(monitors), len(EM_COLUMNS)))
                electromagnetic[:, 0] = monitors[:, 0]
                electromagnetic[:, 1] = 0.03
                data["electromagnetic_monitors"] = electromagnetic
                for index, time in enumerate((0, 8, 10)):
                    snapshot = data.create_group(f"snapshot_{index}")
                    snapshot["time"] = [time]
                    snapshot["velocity"] = self.velocity
                    snapshot["vector_potential"] = self.velocity
                    snapshot["density"] = self.density
                    snapshot["pressure"] = self.density
                    snapshot["temperature"] = self.density
                    snapshot["heat_flux"] = np.zeros_like(self.velocity)
                    for field in EM_FIELDS:
                        snapshot[field] = (
                            self.velocity
                            if field
                            in (
                                "induction",
                                "displacement",
                                "electric_field",
                                "current",
                                "lorentz_force",
                            )
                            else self.density
                        )
                    vorticity = np.zeros_like(self.velocity)
                    vorticity[0, ..., 0] = 2  # Distinguish the x=-pi face.
                    vorticity[1:, ..., 0] = 7
                    snapshot["vorticity"] = vorticity
            process(path, vtk=True, plots=False)
            spectrum = np.loadtxt(path.with_suffix(".spectrum.dat"))
            self.assertAlmostEqual(spectrum[:, 1].sum(), 1 / 8, places=13)
            face = np.loadtxt(path.with_suffix(".vorticity_slice.dat"))
            np.testing.assert_allclose(face[:, 2], 2)
            np.testing.assert_allclose(
                face[:, :2], self.position[0, ..., 1:].reshape(-1, 2)
            )
            for index in range(3):
                self.assertTrue(path.with_suffix(f".snapshot_{index}.vtk").exists())
            self.assertTrue(path.with_suffix(".monitors.dat").exists())
            np.testing.assert_allclose(
                np.loadtxt(path.with_suffix(".reconstruction.dat")), constraints
            )
            np.testing.assert_allclose(
                np.loadtxt(path.with_suffix(".electromagnetic.dat")), electromagnetic
            )
            for field in EM_FIELDS:
                self.assertIn(
                    field + " ", path.with_suffix(".snapshot_0.vtk").read_text()
                )
            self.assertIn(
                "vector_potential 3",
                path.with_suffix(".snapshot_0.vtk").read_text(),
            )

    def evolving_fixture(self, path, times):
        path.with_suffix(".dat").write_text("# code name: SimiLie\n")
        with h5py.File(path, "w") as data:
            data.attrs["monitor_columns"] = ",".join(COLUMNS)
            data["position"] = self.position
            monitors = np.zeros((len(times), len(COLUMNS)))
            monitors[:, 0] = times
            data["monitors"] = monitors
            for index, time in enumerate(times):
                snapshot = data.create_group(f"snapshot_{index}")
                snapshot["time"] = [time]
                snapshot["velocity"] = (1 - 0.1 * index) * self.velocity
                snapshot["vorticity"] = np.zeros_like(self.velocity)
                snapshot["density"] = self.density
                snapshot["pressure"] = self.density
                snapshot["temperature"] = self.density
                snapshot["heat_flux"] = np.zeros_like(self.velocity)

    def test_iteration_limited_trajectories_and_spectrum_selection(self):
        for times, selected in (((0, 0.1, 0.2), 2), ((0, 8, 10, 12), 1)):
            with self.subTest(times=times), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "evolving.h5"
                self.evolving_fixture(path, times)
                monitors, _, spectrum = process(path, vtk=True, plots=False)
                self.assertEqual(monitors[-1, 0], times[-1])
                self.assertAlmostEqual(spectrum.sum(), (1 - 0.1 * selected) ** 2 / 8)
                header = path.with_suffix(".spectrum.dat").read_text()
                self.assertIn(f"t/tc: {times[selected]:.17g};", header)
                self.assertEqual(len(list(Path(directory).glob("*.vtk"))), len(times))

    def test_missing_final_snapshot_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "truncated.h5"
            self.evolving_fixture(path, (0, 0.1, 0.2))
            with h5py.File(path, "a") as data:
                del data["snapshot_2"]
            with self.assertRaisesRegex(ValueError, "Missing final snapshot"):
                process(path, vtk=False, plots=False)


if __name__ == "__main__":
    unittest.main()
