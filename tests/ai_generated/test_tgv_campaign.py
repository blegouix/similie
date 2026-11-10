# SPDX-FileCopyrightText: 2026 Baptiste Legouix
# SPDX-License-Identifier: AGPL-3.0-or-later
# AI-GENERATED
"""Check that the campaign detects changed physics while ignoring timing noise."""

import io
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
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
    ReferenceValidationError,
    campaign,
    compare_vorticity_slice,
    reference_history,
    compare_strategies,
    execute,
    main,
    profile_performance,
    validate_trajectory,
    validate_reference_comparisons,
    write_report,
)


class IsolatedSystemsCaptures(unittest.TestCase):
    def test_cases_are_validated_separately_and_merged(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            configuration = "incompressible_ideal_hd"
            names = [
                f"TaylorGreen/{configuration}/N{n}/{'+'.join(flags) or 'default'}"
                for flags in STRATEGIES
                for n in (64, 128)
            ]
            captures = []

            def capture(command, output, environment, stream_output=True):
                self.assertFalse(stream_output)
                output.mkdir(parents=True, exist_ok=True)
                name = names[len(captures)]
                (output / "run.log").write_text(
                    "Profiler chatter\nBenchmark Time CPU Iterations\n"
                    f"\x1b[0;32m{name} 1 s 1 s 5\x1b[m\nCollecting data...\n"
                )
                pattern = re.escape(name).replace(r"\-", "-")
                self.assertIn(f"--benchmark_filter=^{pattern}(/|$)", command)
                self.assertIn("--benchmark_color=true", command)
                self.assertNotIn(r"\-", command[-1])
                captures.append(output)
                (directory / "benchmark.json").write_text(
                    json.dumps({"context": {}, "benchmarks": [{"run_name": name}]})
                )

            def summarize(output, expected):
                self.assertEqual(list(expected.values()), [5])
                return {"cases": {case: {"regions": []} for case in expected}}

            with (
                patch(
                    "ci_campaign.profile_command",
                    side_effect=lambda cmd, out: (cmd, {}),
                ),
                patch("ci_campaign.execute", side_effect=capture),
                patch("sys.stdout", new_callable=io.StringIO) as console,
                patch(
                    "ci_campaign.summarize_profile", side_effect=summarize
                ) as summary,
            ):
                (directory / "nsys").mkdir()
                profile_performance(
                    ["binary", "--strategy-suite"], directory, configuration
                )
            self.assertEqual(
                console.getvalue().count("Benchmark Time CPU Iterations"), 1
            )
            self.assertEqual(
                sum(
                    re.sub(r"\x1b\[[0-9;]*m", "", line).startswith("TaylorGreen/")
                    for line in console.getvalue().splitlines()
                ),
                14,
            )
            self.assertNotIn("Profiler chatter", console.getvalue())
            self.assertNotIn("Collecting data", console.getvalue())
            self.assertIn("\x1b[0;32m", console.getvalue())
            self.assertEqual(len(set(captures)), 14)
            self.assertEqual(summary.call_count, 14)
            self.assertEqual(
                [
                    row["run_name"]
                    for row in json.loads((directory / "benchmark.json").read_text())[
                        "benchmarks"
                    ]
                ],
                names,
            )
            self.assertEqual(
                len(json.loads((directory / "nsys/timings.json").read_text())["cases"]),
                14,
            )

    def test_failed_summary_stops_before_publishing_combined_timings(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "benchmark.json").write_text(json.dumps({"benchmarks": []}))

            def capture(command, output, environment, stream_output=True):
                output.mkdir(parents=True, exist_ok=True)
                (output / "run.log").write_text("")

            with (
                patch("ci_campaign.profile_command", return_value=(["nsys"], {})),
                patch("ci_campaign.execute", side_effect=capture) as execute_capture,
                patch(
                    "ci_campaign.summarize_profile",
                    side_effect=ValueError("missing kernels"),
                ),
            ):
                with self.assertRaisesRegex(ValueError, "missing kernels"):
                    profile_performance(
                        ["binary"], directory, "incompressible_ideal_hd"
                    )
            self.assertEqual(execute_capture.call_count, 14)
            self.assertFalse((directory / "nsys/timings.json").exists())


class ReferenceAccuracyGate(unittest.TestCase):
    def test_cli_reports_accuracy_failure_without_a_traceback(self):
        message = "HD reference validation failed: N=64 vorticity_norm_t8"
        with (
            patch(
                "sys.argv",
                [
                    "ci_campaign.py",
                    "--binary=unused",
                    "--output=unused",
                    "--fluid=incompressible",
                    "--transport=ideal",
                    "--coupling=hd",
                    "--phase=validation",
                ],
            ),
            patch(
                "ci_campaign.campaign", side_effect=ReferenceValidationError(message)
            ),
            self.assertRaises(SystemExit) as failure,
        ):
            main()
        self.assertEqual(failure.exception.code, message)
        self.assertTrue(failure.exception.__suppress_context__)

    def comparisons(self):
        return {
            f"N{n}": {
                name: {
                    "relative_l2_error": error / 4
                    if name == "kinetic_energy"
                    else error,
                    "max_absolute_error": 0.01,
                }
                for name in (
                    "kinetic_energy",
                    "dissipation",
                    "enstrophy",
                    "vorticity_norm_t8",
                )
            }
            for n, error in ((64, 0.2), (128, 0.1))
        }

    def test_consistent_refinement_passes(self):
        comparisons = self.comparisons()
        validate_reference_comparisons(comparisons)
        self.assertTrue(
            all(
                metric["passed"]
                for metrics in comparisons.values()
                for metric in metrics.values()
            )
        )

    def test_observed_ci_failure_is_rejected_without_explicit_tolerances(self):
        comparisons = self.comparisons()
        for n, energy, dissipation, vorticity in (
            (64, 0.599826, 2.47418, 1.37982),
            (128, 0.803107, 3.75165, 0.981704),
        ):
            for name, value in (
                ("kinetic_energy", energy),
                ("dissipation", dissipation),
                ("enstrophy", dissipation),
                ("vorticity_norm_t8", vorticity),
            ):
                comparisons[f"N{n}"][name]["relative_l2_error"] = value
        with self.assertRaisesRegex(ValueError, "HD reference validation failed"):
            validate_reference_comparisons(comparisons)

    def test_refinement_must_improve_even_below_error_ceiling(self):
        comparisons = self.comparisons()
        comparisons["N128"]["kinetic_energy"]["relative_l2_error"] = 0.075
        with self.assertRaisesRegex(ValueError, "worsens with refinement"):
            validate_reference_comparisons(comparisons)

    def test_small_reference_model_floor_is_allowed(self):
        comparisons = self.comparisons()
        comparisons["N64"]["kinetic_energy"]["relative_l2_error"] = 0.001
        comparisons["N128"]["kinetic_energy"]["relative_l2_error"] = 0.004
        validate_reference_comparisons(comparisons)

    def test_slice_at_the_error_limit_fails(self):
        comparisons = self.comparisons()
        comparisons["N128"]["vorticity_norm_t8"]["relative_l2_error"] = 1.49
        validate_reference_comparisons(comparisons)
        comparisons["N128"]["vorticity_norm_t8"]["relative_l2_error"] = 1.5
        with self.assertRaisesRegex(ValueError, "vorticity_norm_t8"):
            validate_reference_comparisons(comparisons)

    def test_energy_has_a_stricter_guard_than_under_resolved_dissipation(self):
        comparisons = self.comparisons()
        comparisons["N64"]["kinetic_energy"]["relative_l2_error"] = 0.15
        with self.assertRaisesRegex(ValueError, "kinetic_energy"):
            validate_reference_comparisons(comparisons)

    def test_nonfinite_or_negative_metrics_fail(self):
        for key in ("relative_l2_error", "max_absolute_error"):
            for value in (float("nan"), float("inf"), -1.0):
                with self.subTest(key=key, value=value):
                    comparisons = self.comparisons()
                    comparisons["N64"]["dissipation"][key] = value
                    with self.assertRaises(ValueError):
                        validate_reference_comparisons(comparisons)

    def test_incomplete_comparisons_fail(self):
        comparisons = self.comparisons()
        del comparisons["N128"]["enstrophy"]
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            validate_reference_comparisons(comparisons)
        del comparisons["N128"]
        with self.assertRaisesRegex(ValueError, "Missing"):
            validate_reference_comparisons(comparisons)

    def test_invalid_tolerances_fail(self):
        for value in (0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    validate_reference_comparisons(self.comparisons(), value)
                with self.assertRaises(ValueError):
                    validate_reference_comparisons(
                        self.comparisons(), slice_tolerance=value
                    )
                with self.assertRaises(ValueError):
                    validate_reference_comparisons(
                        self.comparisons(), energy_tolerance=value
                    )


class CampaignExecution(unittest.TestCase):
    def test_dead_profiler_with_inherited_stdout_fails_and_cleans_up(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            script = directory / "profiler.py"
            script.write_text(
                "import os, signal, time\n"
                "child = os.fork()\n"
                "if child == 0:\n"
                "    signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
                "    while True: time.sleep(1)\n"
                "open('orphan.pid', 'w').write(str(child))\n"
                "print('profiler diagnostic', flush=True)\n"
                "os.kill(os.getpid(), signal.SIGKILL)\n"
            )
            # An outer timeout turns the original open-pipe hang into a bounded
            # regression failure rather than blocking the entire test suite.
            try:
                runner = subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        "from ci_campaign import execute; from pathlib import Path; "
                        f"execute([{sys.executable!r}, {str(script)!r}], Path({str(directory)!r}))",
                    ],
                    env=dict(os.environ, PYTHONPATH=sys.path[0]),
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                orphan = int((directory / "orphan.pid").read_text())
                status = Path(f"/proc/{orphan}/stat")
                deadline = time.monotonic() + 3
                while True:
                    try:
                        state = status.read_text().split()[2]
                    except FileNotFoundError:
                        break
                    if state == "Z":
                        break
                    self.assertLess(time.monotonic(), deadline, "Orphan is still alive")
                    time.sleep(0.01)
            finally:
                pid_file = directory / "orphan.pid"
                if pid_file.exists():
                    try:
                        os.kill(int(pid_file.read_text()), signal.SIGKILL)
                    except ProcessLookupError:
                        pass
            self.assertNotEqual(runner.returncode, 0)
            self.assertIn("status -9", runner.stderr)
            self.assertIn("profiler diagnostic", runner.stdout)
            self.assertEqual(
                (directory / "run.log").read_text(), "profiler diagnostic\n"
            )

    def test_successful_command_preserves_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            execute([sys.executable, "-c", "print('completed')"], directory)
            self.assertEqual((directory / "run.log").read_text(), "completed\n")

    def test_timeout_stops_the_command(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            started = time.monotonic()
            with self.assertRaises(subprocess.TimeoutExpired):
                execute(
                    [sys.executable, "-c", "import time; time.sleep(10)"],
                    directory,
                    timeout=0.1,
                )
            self.assertLess(time.monotonic() - started, 5)


class CampaignConfiguration(unittest.TestCase):
    def test_filtered_slice_reaches_validation_gate_and_failure_report(self):
        for corrupted in (False, True):
            with (
                self.subTest(corrupted=corrupted),
                tempfile.TemporaryDirectory() as temporary,
            ):
                output = Path(temporary)
                directory = output / "validation"
                flags = "+".join(STRATEGIES[3])

                def synthetic_reference(destination):
                    history = reference_history(destination)
                    grid = -np.pi + np.arange(512) * (2 * np.pi / 512)
                    # This mode aliases at both simulation resolutions but
                    # integrates to zero in their centered filtering cells.
                    norm = np.broadcast_to(
                        10 + 3 * np.cos(128 * grid[:, None]), (512, 512)
                    )
                    norm.astype("<f8").tofile(destination / "wn_slice_x0_08000.out")
                    return history

                def synthetic_benchmark(command, destination):
                    self.assertIn("--reference-suite", command)
                    history = np.loadtxt(
                        destination / "reference-data/spectral_Re1600_512.gdiag"
                    )
                    history = history[history[:, 0] <= 10]
                    rows = []
                    times = np.sort(np.append(np.arange(0, 10, 0.033), [8, 10]))
                    for n in (64, 128):
                        case = destination / f"N{n}"
                        case.mkdir()
                        record = {
                            "configuration": "incompressible_ideal_hd",
                            "N": n,
                            "mode": "reference",
                            "strategy": 3,
                            "flags": flags,
                            "steps": len(history) - 1,
                            "solver_seconds": 1,
                            "mean_step_seconds": 1,
                            "cell_updates_per_second": n**3,
                            "final_time": 10,
                        }
                        (case / "trajectory.performance.json").write_text(
                            json.dumps(record)
                        )
                        with h5py.File(case / "trajectory.h5", "w") as data:
                            data.create_dataset(
                                "final_state", (3, n**3), dtype="f8", fillvalue=0
                            )
                            monitors = np.zeros((len(history), 14))
                            monitors[:, 0] = history[:, 0]
                            monitors[:, 1] = history[:, 1]
                            monitors[0, 1] = (
                                0.125 * np.sinc(1 / n) ** 2 * np.cos(np.pi / n) ** 4
                            )
                            monitors[:, 3:5] = history[:, 2:4]
                            monitors[:, 8] = 1
                            monitors[:, 11:13] = 1
                            data["monitors"] = monitors
                            data["reconstruction_monitors"] = np.zeros(
                                (len(history), 5)
                            )
                            grid = -np.pi + np.arange(n) * (2 * np.pi / n)
                            position = np.zeros((1, n, n, 3))
                            position[..., 0] = -np.pi
                            position[0, ..., 1] = grid[:, None]
                            position[0, ..., 2] = grid[None, :]
                            data["position"] = position
                            for index, time_value in enumerate(times):
                                snapshot = data.create_group(f"snapshot_{index}")
                                snapshot["time"] = [time_value]
                                if time_value == 8:
                                    vorticity = np.zeros((1, n, n, 3))
                                    vorticity[..., 0] = (
                                        30 if corrupted and n == 128 else 10
                                    )
                                    snapshot["vorticity"] = vorticity
                        grids = "".join(
                            f'<Grid><Time Value="{t}"/></Grid>' for t in times
                        )
                        (case / "trajectory.xmf").write_text(
                            f"<Xdmf><Domain><Grid>{grids}</Grid></Domain></Xdmf>"
                        )
                        rows.append(
                            {
                                "run_name": f"TaylorGreen/incompressible_ideal_hd/N{n}/{flags}/iterations:1/repeats:1",
                                "repetitions": 1,
                                "iterations": 1,
                            }
                        )
                    (destination / "benchmark.json").write_text(
                        json.dumps({"benchmarks": rows})
                    )

                # Replace the expensive solver and unrelated field rendering;
                # trajectory checks, comparisons, filtering and gates are real.
                with (
                    patch(
                        "ci_campaign.reference_history", side_effect=synthetic_reference
                    ),
                    patch("ci_campaign.execute", side_effect=synthetic_benchmark),
                    patch("ci_campaign.process"),
                ):
                    if corrupted:
                        with self.assertRaisesRegex(
                            ReferenceValidationError, "N=128 vorticity_norm_t8"
                        ):
                            campaign(
                                Path("synthetic"),
                                output,
                                "incompressible",
                                "ideal",
                                "hd",
                                "validation",
                                reference_slice_relative_tolerance=0.01,
                            )
                    else:
                        campaign(
                            Path("synthetic"),
                            output,
                            "incompressible",
                            "ideal",
                            "hd",
                            "validation",
                            reference_slice_relative_tolerance=0.01,
                        )
                gate = json.loads((directory / "reference-validation.json").read_text())
                self.assertEqual(gate["passed"], not corrupted)
                metrics = json.loads(
                    (directory / "reference-comparison.json").read_text()
                )
                self.assertLess(
                    metrics["N64"]["vorticity_norm_t8"]["relative_l2_error"], 1e-12
                )
                fine = metrics["N128"]["vorticity_norm_t8"]
                self.assertEqual(fine["passed"], not corrupted)
                self.assertAlmostEqual(fine["relative_l2_error"], 2 if corrupted else 0)
                report = (directory / "report.md").read_text()
                self.assertEqual("**FAILED:**" in report, corrupted)
                self.assertIn("| N | Reference observable |", report)
                self.assertNotIn("# Taylor–Green validation", report)
                self.assertNotIn("HD accuracy uses", report)
                self.assertNotIn("Reference accuracy gate:", report)
                self.assertNotIn("HDF5 snapshots", report)
                manifest = json.loads((directory / "manifest.json").read_text())
                self.assertEqual(manifest["error"] is not None, corrupted)

    def test_default_validation_gate_controls_campaign_status_and_report(self):
        for inconsistent in (False, True):
            with (
                self.subTest(inconsistent=inconsistent),
                tempfile.TemporaryDirectory() as temporary,
            ):
                output = Path(temporary)
                directory = output / "validation"
                directory.mkdir()
                flags = "+".join(STRATEGIES[3])
                rows = []
                trajectories = []
                histories = []
                for n, error in ((64, 0.05), (128, 0.025)):
                    case = directory / f"N{n}"
                    case.mkdir()
                    (case / "trajectory.performance.json").touch()
                    rows.append(
                        {
                            "run_name": f"TaylorGreen/incompressible_ideal_hd/N{n}/{flags}/iterations:1/repeats:1",
                            "repetitions": 1,
                            "iterations": 1,
                        }
                    )
                    trajectories.append(
                        ({"strategy": 3, "flags": flags}, case / "trajectory.h5")
                    )
                    histories.append(
                        {
                            name: {
                                "relative_l2_error": 0.8 if inconsistent else error,
                                "max_absolute_error": 0.01,
                            }
                            for name in ("kinetic_energy", "dissipation", "enstrophy")
                        }
                    )
                (directory / "benchmark.json").write_text(
                    json.dumps({"benchmarks": rows})
                )
                with (
                    patch("ci_campaign.reference_history"),
                    patch("ci_campaign.execute"),
                    patch("ci_campaign.validate_trajectory", side_effect=trajectories),
                    patch("ci_campaign.process"),
                    patch("ci_campaign.compare_reference", side_effect=histories),
                    patch(
                        "ci_campaign.compare_vorticity_slice",
                        return_value={
                            "relative_l2_error": 0.2,
                            "max_absolute_error": 0.01,
                        },
                    ),
                ):
                    if inconsistent:
                        with self.assertRaisesRegex(
                            ValueError, "HD reference validation failed"
                        ):
                            campaign(
                                Path("unused"),
                                output,
                                "incompressible",
                                "ideal",
                                "hd",
                                "validation",
                            )
                    else:
                        campaign(
                            Path("unused"),
                            output,
                            "incompressible",
                            "ideal",
                            "hd",
                            "validation",
                        )
                gate = json.loads((directory / "reference-validation.json").read_text())
                self.assertEqual(gate["passed"], not inconsistent)
                self.assertEqual(
                    "**FAILED:**" in (directory / "report.md").read_text(), inconsistent
                )
                manifest = json.loads((directory / "manifest.json").read_text())
                self.assertEqual(manifest["error"] is not None, inconsistent)

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


class InitialDiscretization(unittest.TestCase):
    def test_performance_without_monitoring_still_checks_final_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "trajectory.performance.json"
            record = {
                "configuration": "incompressible_ideal_hd",
                "N": 8,
                "mode": "performance",
                "monitoring_enabled": False,
                "steps": 5,
                "solver_seconds": 1,
                "mean_step_seconds": 0.2,
                "cell_updates_per_second": 512,
                "final_time": 1,
            }
            path.write_text(json.dumps(record))
            hdf = path.with_name("trajectory.h5")
            with h5py.File(hdf, "w") as data:
                data["final_state"] = np.ones((3, 8**3))
            actual, actual_hdf = validate_trajectory(path, "incompressible_ideal_hd", 8)
            self.assertEqual(actual, record)
            self.assertEqual(actual_hdf, hdf)
            with h5py.File(hdf, "r+") as data:
                data["final_state"][0, 0] = np.nan
            with self.assertRaisesRegex(ValueError, "Invalid final state"):
                validate_trajectory(path, "incompressible_ideal_hd", 8)
            record["mode"] = "reference"
            path.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, "requires monitoring"):
                validate_trajectory(path, "incompressible_ideal_hd", 8, reference=True)

    def test_potential_based_energy_includes_vertex_quadrature_projection(self):
        for fluid in ("incompressible", "compressible"):
            for n in (8, 16):
                with (
                    self.subTest(fluid=fluid, n=n),
                    tempfile.TemporaryDirectory() as temporary,
                ):
                    path = Path(temporary) / "trajectory.performance.json"
                    configuration = f"{fluid}_ideal_hd"
                    path.write_text(
                        json.dumps(
                            {
                                "configuration": configuration,
                                "N": n,
                                "mode": "performance",
                                "steps": 5,
                                "solver_seconds": 1,
                                "mean_step_seconds": 1,
                                "cell_updates_per_second": n**3,
                                "final_time": 1,
                            }
                        )
                    )
                    h = 2 * np.pi / n
                    expected = (
                        0.125 * (np.sin(h / 2) / (h / 2)) ** 2 * np.cos(h / 2) ** 4
                        if fluid == "incompressible"
                        else 0.125
                    )
                    with h5py.File(path.with_name("trajectory.h5"), "w") as data:
                        components = 3 if fluid == "incompressible" else 5
                        data["final_state"] = np.ones((components, n**3))
                        monitors = np.zeros((6, 14))
                        monitors[:, 0] = np.linspace(0, 1, 6)
                        monitors[:, 1] = expected
                        monitors[:, 11:13] = 1
                        data["monitors"] = monitors
                        data["reconstruction_monitors"] = np.zeros((6, 5))
                    validate_trajectory(path, configuration, n)
                    with h5py.File(path.with_name("trajectory.h5"), "r+") as data:
                        data["monitors"][0, 1] = expected + 0.001
                    with self.assertRaisesRegex(ValueError, "Incorrect initial"):
                        validate_trajectory(path, configuration, n)


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
                data["final_state"] = np.ones((5, 8**3))
                monitors = np.zeros((2, 14))
                monitors[:, 0] = [0, 10]
                monitors[:, 1] = 0.125
                monitors[:, 11:13] = 1
                data["monitors"] = monitors
                data["reconstruction_monitors"] = np.zeros((2, 5))
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

    def test_final_time_is_compared_without_monitor_histories(self):
        for path in (self.baseline, self.candidate):
            with h5py.File(path, "r+") as data:
                del data["monitors"]
                del data["electromagnetic_monitors"]
        baseline = {"final_time": 1.0}
        candidate = {"final_time": 1.0}
        results = compare_strategies(self.baseline, self.candidate, baseline, candidate)
        self.assertNotIn("monitors", results)
        self.assertTrue(all(metric["passed"] for metric in results.values()))
        candidate["final_time"] += 1e-3
        results = compare_strategies(self.baseline, self.candidate, baseline, candidate)
        self.assertTrue(results["final_state"]["passed"])
        self.assertFalse(results["final_time"]["passed"])

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
    def test_original_inputs_and_centered_periodic_slice_filter(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference_directory = root / "reference"
            history = reference_history(reference_directory)
            self.assertEqual(history.shape, (2000, 4))
            self.assertEqual(history[0, 1], 0.125)
            self.assertEqual(history[0, 3], 0.375)
            self.assertEqual(len(list(reference_directory.iterdir())), 4)
            # Distinct low modes detect swapped axes or a shifted filter.
            # Unresolved modes average to zero over the coarse cell, but
            # alias into a nonzero constant if the reference is only sampled.
            fine_grid = -np.pi + np.arange(512) * (2 * np.pi / 512)
            reference = (
                10
                + np.cos(11 * fine_grid[:, None])
                + 2 * np.sin(7 * fine_grid[None, :])
                + 3 * np.cos(128 * fine_grid[:, None])
                + 2 * np.cos(128 * fine_grid[None, :])
            )
            reference.astype("<f8").tofile(
                reference_directory / "wn_slice_x0_08000.out"
            )
            n = 128
            grid = -np.pi + np.arange(n) * (2 * np.pi / n)
            position = np.zeros((1, n, n, 3))
            position[..., 0] = -np.pi
            position[0, ..., 1] = grid[:, None]
            position[0, ..., 2] = grid[None, :]
            vorticity = np.zeros((1, n, n, 3))

            # Exact trapezoidal integral of a Fourier mode over a centered
            # coarse cell on the fine grid (including half-weight endpoints).
            def gain(k):
                return np.sin(k * np.pi / n) / ((512 // n) * np.tan(k * np.pi / 512))

            expected = (
                10
                + gain(11) * np.cos(11 * grid[:, None])
                + 2 * gain(7) * np.sin(7 * grid[None, :])
            )
            vorticity[0, ..., 0] = expected
            hdf = root / "trajectory.h5"
            with h5py.File(hdf, "w") as data:
                data["position"] = position
                data["snapshot_0/time"] = [8.0]
                data["snapshot_0/vorticity"] = vorticity
            metrics = compare_vorticity_slice(hdf, reference_directory, tolerance=0.01)
            self.assertLess(metrics["relative_l2_error"], 1e-13)
            self.assertTrue(metrics["passed"])
            self.assertEqual(metrics["samples"], n * n)
            self.assertEqual(metrics["reference_filter"]["center"], "simulation node")
            self.assertEqual(metrics["reference_filter"]["reference_grid_stride"], 4)
            compared = np.loadtxt(
                hdf.with_suffix(".vorticity-comparison.csv"), delimiter=",", skiprows=1
            )
            np.testing.assert_allclose(
                compared[:, 3].reshape(n, n), expected, atol=1e-12
            )
            self.assertTrue(hdf.with_suffix(".vorticity-comparison.png").is_file())
            with h5py.File(hdf, "r+") as data:
                data["snapshot_0/vorticity"][0, 1, 3, 0] += 50
            metrics = compare_vorticity_slice(hdf, reference_directory, tolerance=0.01)
            self.assertFalse(metrics["passed"])
            self.assertAlmostEqual(metrics["max_absolute_error"], 50)
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
