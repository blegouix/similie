# SPDX-FileCopyrightText: 2026 Baptiste Legouix
# SPDX-License-Identifier: AGPL-3.0-or-later
# AI-GENERATED
"""Run one performance suite or the separate HD reference-validation suite."""

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import h5py
import numpy as np
from post_process import process
from nsys_report import profile_command, summarize_profile

MESH_SIZES = (64, 128)
PERFORMANCE_STEPS = 5
STRATEGY_RTOL = 1e-7
STRATEGY_ATOL = 1e-9
STRATEGIES = (
    (),
    ("--levi-civita-continuous-hodge-star",),
    (
        "--weighted-levi-civita-discrete-hodge-star",
        "--diagonal-reduction",
        "--levi-civita-continuous-hodge-star",
        "--diagonal-reconstruction",
    ),
    (
        "--staged-weighted-levi-civita-discrete-hodge-star",
        "--diagonal-reduction",
        "--levi-civita-continuous-hodge-star",
        "--diagonal-reconstruction",
    ),
    (
        "--staged-discrete-hodge-star",
        "--diagonal-reduction",
        "--levi-civita-continuous-hodge-star",
        "--diagonal-reconstruction",
    ),
    (
        "--staged-codifferential",
        "--weighted-levi-civita-discrete-hodge-star",
        "--diagonal-reduction",
        "--levi-civita-continuous-hodge-star",
        "--diagonal-reconstruction",
    ),
    (
        "--staged-laplacian",
        "--weighted-levi-civita-discrete-hodge-star",
        "--diagonal-reduction",
        "--levi-civita-continuous-hodge-star",
        "--diagonal-reconstruction",
    ),
)
REFERENCE_URL = "https://cfd.ku.edu/hiocfd/data.tgz"
REFERENCE_FILE = "spectral_Re1600_512.gdiag"
REFERENCE_SLICE = "wn_slice_x0_08000.out"
REFERENCE_CHECKSUMS = {
    REFERENCE_FILE: "8e01c0024ae2dde20df8dc77a20fa7fb41c3abef2fc528a2ac684afed96d6429",
    REFERENCE_SLICE: "a350f7627ecb6bd80caa03832d057227be43cbd0dbd67a7f85c8ffbbdbc094a4",
    "read_and_plot_w.py": "7d8112dfb171e71a5b9e9e05857466f31c5db6863ea6057aff42f0395714e557",
}


def execute(command, directory, environment=None, timeout=None):
    """Preserve the complete Google Benchmark output without campaign chatter."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "command.json").write_text(
        json.dumps(list(map(str, command)), indent=2)
    )
    with (
        (directory / "run.log").open("w") as log,
        (directory / "run.log").open() as output,
    ):
        child = subprocess.Popen(
            list(map(str, command)),
            cwd=directory,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        started = time.monotonic()
        try:
            while True:
                if timeout is not None and time.monotonic() - started >= timeout:
                    raise subprocess.TimeoutExpired(child.args, timeout)
                try:
                    status = child.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    status = None
                sys.stdout.write(output.read())
                sys.stdout.flush()
                if status is not None:
                    break
        finally:
            # A profiler can die while its injected application remains alive.
            # Monitor the profiler itself rather than waiting for pipe EOF,
            # and release its descendants on failure or cancellation.
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait()
        if status:
            raise RuntimeError(
                f"Benchmark exited with status {status}; see {directory / 'run.log'}"
            )


def validate_trajectory(path, configuration, n, reference=False):
    record = json.loads(path.read_text())
    if record["configuration"] != configuration or record["N"] != n:
        raise ValueError(f"Wrong configuration or resolution: {path}")
    if record["mode"] != ("reference" if reference else "performance"):
        raise ValueError(f"Wrong trajectory mode: {path}")
    if record["steps"] <= 0 or (not reference and record["steps"] != PERFORMANCE_STEPS):
        raise ValueError(f"Incorrect step count: {path}")
    monitoring_enabled = record.get("monitoring_enabled", True)
    if reference and not monitoring_enabled:
        raise ValueError(f"Reference validation requires monitoring: {path}")
    for key in (
        "solver_seconds",
        "mean_step_seconds",
        "cell_updates_per_second",
        "final_time",
    ):
        if not np.isfinite(record[key]) or record[key] <= 0:
            raise ValueError(f"Invalid {key}: {path}")
    hdf = path.with_name(path.name.removesuffix(".performance.json") + ".h5")
    with h5py.File(hdf, "r") as data:
        components = 4 if configuration.startswith("compressible") else 3
        state = data["final_state"][:]
        if state.shape != (components, n**3) or not np.all(np.isfinite(state)):
            raise ValueError(f"Invalid final state: {hdf}")
        if "final_electromagnetic_state" in data and not np.all(
            np.isfinite(data["final_electromagnetic_state"][:])
        ):
            raise ValueError(f"Invalid final electromagnetic state: {hdf}")
        if monitoring_enabled:
            monitors = data["monitors"][:]
            if monitors.shape != (record["steps"] + 1, 14) or not np.all(
                np.isfinite(monitors)
            ):
                raise ValueError(f"Invalid monitor history: {hdf}")
            if monitors[0, 0] != 0 or np.any(np.diff(monitors[:, 0]) <= 0):
                raise ValueError(f"Time did not advance monotonically: {hdf}")
            if monitors[-1, 0] != record["final_time"] or np.any(
                monitors[:, 11:13] <= 0
            ):
                raise ValueError(f"Invalid final time, density, or pressure: {hdf}")
            initial_energy = (
                0.125 * np.sinc(1 / n) ** 2
                if configuration.startswith("incompressible")
                else 0.125
            )
            if abs(monitors[0, 1] - initial_energy) > 1e-10:
                raise ValueError(f"Incorrect initial Taylor-Green kinetic energy: {hdf}")
            if np.max(np.abs(monitors[:, 8] - monitors[0, 8])) > 1e-8:
                raise ValueError(f"Mass conservation exceeded tolerance: {hdf}")
            if (
                configuration.startswith("incompressible")
                and np.max(monitors[:, 10]) > 1e-8
            ):
                raise ValueError(f"Incompressibility constraint exceeded tolerance: {hdf}")
            constraints = data["reconstruction_monitors"][:]
            if not np.all(np.isfinite(constraints)) or np.max(
                constraints[:, 1:]
            ) > 1e-8:
                raise ValueError(f"Reconstruction constraint exceeded tolerance: {hdf}")
            if "electromagnetic_monitors" in data:
                electromagnetic = data["electromagnetic_monitors"][:]
                if not np.all(np.isfinite(electromagnetic)):
                    raise ValueError(f"Nonfinite electromagnetic monitors: {hdf}")
                if np.max(electromagnetic[:, 5:7]) > 1e-8:
                    raise ValueError(
                        f"Gauss or magnetic divergence constraint exceeded tolerance: {hdf}"
                    )
                if np.max(np.abs(electromagnetic[:, 8])) > 1e-8:
                    raise ValueError(f"Inconsistent field/fluid energy exchange: {hdf}")
        if reference:
            expected_times = np.sort(np.append(np.arange(0, 10, 0.033), [8, 10]))
            times = sorted(
                float(data[name]["time"][0])
                for name in data
                if name.startswith("snapshot_")
            )
            if (
                record["final_time"] != 10
                or len(times) != len(expected_times)
                or not np.allclose(times, expected_times, rtol=0, atol=1e-12)
            ):
                raise ValueError(
                    f"Reference must finish at t=10 with snapshots every 0.033 s, including t=8: {hdf}"
                )
            xmf = hdf.with_suffix(".xmf")
            try:
                grids = ET.parse(xmf).getroot().findall("./Domain/Grid/Grid")
            except (ET.ParseError, OSError) as error:
                raise ValueError(
                    f"Missing or invalid XDMF time series: {xmf}"
                ) from error
            times = [float(grid.find("Time").attrib["Value"]) for grid in grids]
            if len(times) != len(expected_times) or not np.allclose(
                times, expected_times, rtol=0, atol=1e-12
            ):
                raise ValueError(f"XDMF time series is incomplete: {xmf}")
    return record, hdf


def compare_strategies(
    baseline, candidate, baseline_record=None, candidate_record=None
):
    """Compare final states and, when collected, evolving monitors.

    All cases advance exactly five steps from the same initial state. Compare
    times as well as fields when trajectory records are supplied.
    Timing columns and Krylov residuals are deliberately excluded from equality.
    """
    results = {}
    with h5py.File(baseline, "r") as expected, h5py.File(candidate, "r") as actual:
        names = ["final_state"]
        if "monitors" in expected and "monitors" in actual:
            names.append("monitors")
        if "final_electromagnetic_state" in expected:
            names.append("final_electromagnetic_state")
        if ("monitors" in expected) != ("monitors" in actual):
            raise ValueError(f"Monitoring datasets differ: {candidate}")
        if (
            "electromagnetic_monitors" in expected
            and "electromagnetic_monitors" in actual
        ):
            names.append("electromagnetic_monitors")
        if ("electromagnetic_monitors" in expected) != (
            "electromagnetic_monitors" in actual
        ):
            raise ValueError(f"Electromagnetic monitoring datasets differ: {candidate}")
        for name in names:
            left, right = expected[name][:], actual[name][:]
            if name == "monitors":
                left, right = left[:, :13], right[:, :13]
            if left.shape != right.shape or not np.all(np.isfinite(right)):
                raise ValueError(
                    f"Invalid strategy comparison dataset {name}: {candidate}"
                )
            difference = np.abs(left - right)
            results[name] = {
                "max_absolute_error": float(np.max(difference)),
                "relative_l2_error": float(
                    np.linalg.norm(difference.ravel())
                    / max(np.linalg.norm(left.ravel()), STRATEGY_ATOL)
                ),
                "passed": bool(
                    np.all(difference <= STRATEGY_ATOL + STRATEGY_RTOL * np.abs(left))
                ),
            }
    if baseline_record is not None and candidate_record is not None:
        difference = abs(
            baseline_record["final_time"] - candidate_record["final_time"]
        )
        results["final_time"] = {
            "max_absolute_error": difference,
            "relative_l2_error": difference
            / max(abs(baseline_record["final_time"]), STRATEGY_ATOL),
            "passed": difference
            <= STRATEGY_ATOL
            + STRATEGY_RTOL * abs(baseline_record["final_time"]),
        }
    return results


def reference_history(directory):
    """Use the checked-in workshop inputs and keep their provenance in artifacts."""
    directory.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).parent / "reference_data"
    for name, expected in REFERENCE_CHECKSUMS.items():
        path = source / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Workshop reference checksum mismatch: {name}")
        shutil.copyfile(path, directory / name)
    (directory / "source.json").write_text(
        json.dumps(
            {
                "url": REFERENCE_URL,
                "description": "UCL dealiased pseudo-spectral 512^3 reference, HiOCFD C3.5 (previous edition of HOW4 BS1)",
                "sha256": REFERENCE_CHECKSUMS,
                "columns": [
                    "time",
                    "kinetic_energy",
                    "positive_dissipation",
                    "enstrophy",
                ],
                "vorticity_slice": {
                    "file": REFERENCE_SLICE,
                    "time": 8,
                    "plane": "x/L=-pi",
                    "shape": [512, 512],
                    "dtype": "little-endian float64",
                    "storage_axes": ["y", "z"],
                    "grid": "-pi + 2*pi*i/512, i=0,...,511; periodic endpoint excluded",
                    "quantity": "vorticity norm * L/V0",
                },
            },
            indent=2,
        )
        + "\n"
    )
    history = np.loadtxt(directory / REFERENCE_FILE)
    if (
        history.ndim != 2
        or history.shape[1] != 4
        or not np.all(np.isfinite(history))
        or history[0, 0] != 0
        or history[-1, 0] < 10
        or np.any(np.diff(history[:, 0]) <= 0)
    ):
        raise ValueError("Invalid workshop reference history")
    return history


def text_plot(
    time, simulation, reference, title, x_label="t", reference_label="UCL512"
):
    """A text plot works directly in GitHub logs; PNGs accompany the artifact."""
    width, height = 60, 10
    sample_time = np.linspace(time[0], time[-1], width)
    values = [
        np.interp(sample_time, time, series) for series in (simulation, reference)
    ]
    low, high = min(v.min() for v in values), max(v.max() for v in values)
    grid = [[" " for _ in range(width)] for _ in range(height)]
    for symbol, series in zip(("S", "R"), values, strict=True):
        rows = np.rint((high - series) / max(high - low, 1e-30) * (height - 1)).astype(
            int
        )
        for column, row in enumerate(rows):
            grid[row][column] = "+" if grid[row][column] != " " else symbol
    print(f"{title}: S=SimiLie R={reference_label} +=overlap")
    for row in grid:
        print("|" + "".join(row) + "|")
    print(f"{x_label}={time[0]:g}" + " " * 42 + f"{x_label}={time[-1]:g}")


def compare_reference(hdf, reference, tolerance=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with h5py.File(hdf, "r") as data:
        monitors = data["monitors"][:]
    sample = reference[(reference[:, 0] >= 0) & (reference[:, 0] <= monitors[-1, 0])]
    metrics = {}
    figure, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for axis, (name, column, ref_column) in zip(
        axes,
        (("kinetic_energy", 1, 1), ("dissipation", 3, 2), ("enstrophy", 4, 3)),
        strict=True,
    ):
        simulation = np.interp(sample[:, 0], monitors[:, 0], monitors[:, column])
        expected = sample[:, ref_column]
        error = float(np.linalg.norm(simulation - expected) / np.linalg.norm(expected))
        metrics[name] = {
            "relative_l2_error": error,
            "max_absolute_error": float(np.max(np.abs(simulation - expected))),
        }
        if tolerance is not None:
            metrics[name]["passed"] = error <= tolerance
        print(
            f"Reference comparison {hdf.stem}: {name} relative_L2_error={error:.6g}"
            + (
                f" tolerance={tolerance:g} {'PASS' if error <= tolerance else 'FAIL'}"
                if tolerance is not None
                else " (resolution-study metric)"
            )
        )
        axis.plot(sample[:, 0], expected, label="UCL spectral 512³")
        axis.plot(monitors[:, 0], monitors[:, column], label="SimiLie")
        axis.set(xlabel="t/tc", ylabel=name)
        axis.legend()
        text_plot(sample[:, 0], simulation, expected, name)
    figure.savefig(hdf.with_suffix(".reference-comparison.png"), dpi=160)
    plt.close(figure)
    hdf.with_suffix(".reference-comparison.json").write_text(
        json.dumps(metrics, indent=2) + "\n"
    )
    return metrics


def compare_vorticity_slice(hdf, reference_directory, tolerance=None):
    """Compare the t=8 periodic-face norm at the simulation's matching nodes."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    reference = np.fromfile(reference_directory / REFERENCE_SLICE, dtype="<f8")
    if (
        reference.size != 512 * 512
        or not np.all(np.isfinite(reference))
        or np.any(reference < 0)
    ):
        raise ValueError("Invalid workshop vorticity slice")
    # The supplied reader loops over its first coordinate outside its second.
    # These are y and z on the x=-pi face; z is contiguous in the binary file.
    reference = reference.reshape(512, 512)
    with h5py.File(hdf, "r") as data:
        snapshots = [
            data[name]
            for name in data
            if name.startswith("snapshot_")
            and np.isclose(float(data[name]["time"][0]), 8, rtol=0, atol=1e-12)
        ]
        if len(snapshots) != 1:
            raise ValueError(
                "Expected exactly one t=8 snapshot for vorticity comparison"
            )
        simulation = np.linalg.norm(snapshots[0]["vorticity"][0], axis=-1)
        position = data["position"][0]
    n = simulation.shape[0]
    coordinates = -np.pi + np.arange(n) * (2 * np.pi / n)
    if (
        simulation.shape != (n, n)
        or not np.all(np.isfinite(simulation))
        or n > 512
        or 512 % n
        or position.shape != (n, n, 3)
        or not np.allclose(position[..., 0], -np.pi, rtol=0, atol=1e-12)
        or not np.allclose(position[..., 1], coordinates[:, None], rtol=0, atol=1e-12)
        or not np.allclose(position[..., 2], coordinates[None, :], rtol=0, atol=1e-12)
    ):
        raise ValueError(
            "Vorticity comparison requires a matching periodic grid dividing 512"
        )
    # N=64 and N=128 are exact subsets of the 512-node reference: no
    # interpolation or axis rotation is needed to compare the same points.
    expected = reference[:: 512 // n, :: 512 // n]
    difference = simulation - expected
    metrics = {
        "source": REFERENCE_URL,
        "file": REFERENCE_SLICE,
        "time": 8,
        "plane": "x/L=-pi",
        "samples": int(simulation.size),
        "relative_l2_error": float(
            np.linalg.norm(difference) / np.linalg.norm(expected)
        ),
        "max_absolute_error": float(np.max(np.abs(difference))),
        "rms_absolute_error": float(np.sqrt(np.mean(difference**2))),
    }
    if tolerance is not None:
        metrics["passed"] = metrics["relative_l2_error"] <= tolerance
    np.savetxt(
        hdf.with_suffix(".vorticity-comparison.csv"),
        np.column_stack(
            (
                position[..., 1].ravel(),
                position[..., 2].ravel(),
                simulation.ravel(),
                expected.ravel(),
                difference.ravel(),
            )
        ),
        delimiter=",",
        header="y,z,simulation_vorticity_norm,reference_vorticity_norm,error",
        comments="",
    )
    hdf.with_suffix(".vorticity-comparison.json").write_text(
        json.dumps(metrics, indent=2) + "\n"
    )
    figure, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    reference_coordinates = -np.pi + np.arange(512) * (2 * np.pi / 512)
    maximum = max(float(reference.max()), float(simulation.max()))
    for axis, grid, norm, label in (
        (axes[0], reference_coordinates, reference, "UCL spectral 512³"),
        (axes[1], coordinates, simulation, f"SimiLie N={n}"),
    ):
        image = axis.pcolormesh(
            grid, grid, norm.T, shading="auto", vmin=0, vmax=maximum
        )
        levels = [
            level for level in (1, 5, 10, 20, 30) if norm.min() < level < norm.max()
        ]
        if levels:
            axis.contour(
                grid, grid, norm.T, levels=levels, colors="black", linewidths=0.4
            )
        axis.set(title=label, xlabel="y/L", ylabel="z/L", aspect="equal")
    figure.colorbar(image, ax=axes[:2], label="Vorticity norm * L/V0")
    bound = max(metrics["max_absolute_error"], 1e-15)
    image = axes[2].pcolormesh(
        coordinates,
        coordinates,
        difference.T,
        shading="auto",
        cmap="coolwarm",
        vmin=-bound,
        vmax=bound,
    )
    axes[2].set(
        title="SimiLie minus reference", xlabel="y/L", ylabel="z/L", aspect="equal"
    )
    figure.colorbar(image, ax=axes[2], label="Vorticity norm error")
    figure.suptitle("Taylor–Green: x/L=-pi, t/tc=8")
    figure.savefig(hdf.with_suffix(".vorticity-comparison.png"), dpi=160)
    plt.close(figure)
    print(
        f"Vorticity slice comparison {hdf.stem}: relative_L2_error={metrics['relative_l2_error']:.6g}, max_absolute_error={metrics['max_absolute_error']:.6g}"
    )
    return metrics


def write_report(output, phase, entries, error=None):
    lines = [
        f"# Taylor–Green {phase}",
    ]
    if phase == "performance":
        lines.extend(
            (
                "",
                f"Final fluid/field states and final times are compared with the default strategy using rtol={STRATEGY_RTOL:g}, atol={STRATEGY_ATOL:g}. Monitoring can be enabled separately for validation.",
            )
        )
        comparison_path = output / "strategy-comparison.json"
        if comparison_path.is_file():
            comparison_data = json.loads(comparison_path.read_text())
            comparisons = comparison_data["comparisons"]
            lines.extend(
                (
                    "",
                    f"Overall strategy equivalence: {'PASS' if comparison_data['passed'] else 'FAIL'}.",
                )
            )
            lines.extend(
                (
                    "| N | Strategy | Dataset | Max absolute error | Relative L2 error | Result |",
                    "| ---: | ---: | --- | ---: | ---: | --- |",
                )
            )
            for case, datasets in comparisons.items():
                strategy = int(case.split("/")[0].split("-")[-1])
                n = int(case.rsplit("N", 1)[-1])
                for name, metric in datasets.items():
                    lines.append(
                        f"| {n} | {strategy} | {name} | {metric['max_absolute_error']:.6e} | {metric['relative_l2_error']:.6e} | {'PASS' if metric['passed'] else 'FAIL'} |"
                    )
    else:
        lines.extend(
            (
                "",
                "HD accuracy uses the original UCL 512³ pseudo-spectral data from https://cfd.ku.edu/hiocfd/data.tgz: kinetic energy, dissipation and enstrophy histories, plus the vorticity norm on x/L=-pi at t/tc=8. Slice errors compare matching nodes of the 512² reference and the simulation grid. Inputs and SHA-256 provenance are included in reference-data; comparison CSVs and PNGs accompany the results. Relative L2 errors are resolution-study metrics; --reference-relative-tolerance optionally gates all four observables.",
                "",
                "HDF5 snapshots and their XDMF time series are saved every 0.033 seconds of simulation time, with additional snapshots at t=8 and t=10.",
            )
        )
    if phase == "validation":
        lines.extend(
            (
                "",
                "| N | Reference observable | Relative L2 error | Max absolute error |",
                "| ---: | --- | ---: | ---: |",
            )
        )
        for path in sorted(output.glob("N*/*.reference-comparison.json")):
            for name, metric in json.loads(path.read_text()).items():
                lines.append(
                    f"| {path.parent.name[1:]} | {name} | {metric['relative_l2_error']:.3%} | {metric['max_absolute_error']:.6g} |"
                )
        for path in sorted(output.glob("N*/*.vorticity-comparison.json")):
            metric = json.loads(path.read_text())
            lines.append(
                f"| {path.parent.name[1:]} | vorticity norm at t=8, x/L=-pi | {metric['relative_l2_error']:.3%} | {metric['max_absolute_error']:.6g} |"
            )
    if error:
        lines.extend(("", f"**FAILED:** {error}"))
    (output / "report.md").write_text("\n".join(lines) + "\n")
    (output / "manifest.json").write_text(
        json.dumps({"phase": phase, "runs": entries, "error": error}, indent=2) + "\n"
    )


def campaign(
    binary,
    output,
    fluid,
    transport,
    coupling,
    phase="performance",
    reference_relative_tolerance=None,
):
    if phase == "validation" and coupling != "hd":
        raise ValueError("The long workshop validation is only defined for HD")
    directory = output / phase
    directory.mkdir(parents=True, exist_ok=True)
    entries = []
    configuration = f"{fluid}_{transport}_{coupling}"
    command = [
        binary,
        f"--{fluid}",
        f"--{transport}",
        f"--{coupling}",
        f"--output-dir={directory}",
        f"--benchmark_out={directory / 'benchmark.json'}",
        "--benchmark_out_format=json",
    ]
    if phase == "performance":
        command += [
            "--strategy-suite",
            "--no-snapshots",
            f"--benchmark_min_time={PERFORMANCE_STEPS}x",
            "--benchmark_repetitions=1",
        ]
    else:
        command += ["--reference-suite"]
    try:
        if phase == "validation":
            reference = reference_history(directory / "reference-data")
        environment = None
        if phase == "performance":
            command, environment = profile_command(command, directory / "nsys")
        execute(command, directory, environment)
        benchmark = json.loads((directory / "benchmark.json").read_text())
        rows = [
            row
            for row in benchmark["benchmarks"]
            if row.get("run_type", "iteration") == "iteration"
        ]
        if len(rows) != (
            len(STRATEGIES) * len(MESH_SIZES)
            if phase == "performance"
            else len(MESH_SIZES)
        ) or any(row.get("error_occurred") or row.get("skipped") for row in rows):
            raise ValueError("Incomplete Google Benchmark suite")
        benchmark_rows = {}
        expected_cases = [
            (strategy, n)
            for strategy in (range(len(STRATEGIES)) if phase == "performance" else (3,))
            for n in MESH_SIZES
        ]
        expected_names = {
            f"TaylorGreen/{configuration}/N{n}/{'+'.join(STRATEGIES[strategy]) or 'default'}"
            + ("/iterations:1/repeats:1" if phase == "validation" else ""): (
                strategy,
                n,
            )
            for strategy, n in expected_cases
        }
        for row in rows:
            key = expected_names.get(row.get("run_name"))
            if key is None or key in benchmark_rows:
                raise ValueError(
                    f"Unexpected or duplicate Google Benchmark case: {row.get('run_name')}"
                )
            if row.get("repetitions") != 1 or (
                phase == "performance" and row.get("repeats") != 1
            ):
                raise ValueError(f"Unexpected repeat count in {row.get('run_name')}")
            if phase == "performance" and not np.isfinite(
                row.get("setup_seconds", np.nan)
            ):
                raise ValueError(f"Missing setup timing in {row.get('run_name')}")
            benchmark_rows[key] = row
        if set(benchmark_rows) != set(expected_names.values()):
            raise ValueError(
                "Google Benchmark suite does not contain the expected cases"
            )
        files = {}
        records = {}
        for strategy in range(len(STRATEGIES)) if phase == "performance" else (3,):
            for n in MESH_SIZES:
                case = (
                    directory / f"strategy-{strategy + 1}" / f"N{n}"
                    if phase == "performance"
                    else directory / f"N{n}"
                )
                trajectories = sorted(case.glob("*.performance.json"))
                if len(trajectories) != 1:
                    raise ValueError(f"Expected one trajectory in {case}")
                record, hdf = validate_trajectory(
                    trajectories[0], configuration, n, phase == "validation"
                )
                if record["strategy"] != strategy or record["flags"] != (
                    "+".join(STRATEGIES[strategy]) or "default"
                ):
                    raise ValueError(f"Incorrect operator strategy in {case}")
                entries.append(record)
                row = benchmark_rows[strategy, n]
                record["repeats"] = int(row.get("repeats", row["repetitions"]))
                record["benchmark_iterations"] = int(row["iterations"])
                if "setup_seconds" in row:
                    record["setup_seconds"] = float(row["setup_seconds"])
                files[strategy, n] = hdf
                records[strategy, n] = record
        if phase == "performance":
            comparisons = {
                f"strategy-{strategy + 1}/N{n}": compare_strategies(
                    files[0, n],
                    files[strategy, n],
                    records[0, n],
                    records[strategy, n],
                )
                for strategy in range(1, len(STRATEGIES))
                for n in MESH_SIZES
            }
            equivalence_passed = all(
                result["passed"]
                for comparison in comparisons.values()
                for result in comparison.values()
            )
            (directory / "strategy-comparison.json").write_text(
                json.dumps(
                    {
                        "rtol": STRATEGY_RTOL,
                        "atol": STRATEGY_ATOL,
                        "passed": equivalence_passed,
                        "comparisons": comparisons,
                    },
                    indent=2,
                )
                + "\n"
            )
            summarize_profile(
                directory / "nsys",
                {
                    f"TaylorGreen/N={n}/{'+'.join(flags) or 'default'}": PERFORMANCE_STEPS
                    for n in MESH_SIZES
                    for flags in STRATEGIES
                },
            )
            if not equivalence_passed:
                raise ValueError(
                    "Operator strategies disagree; see strategy-comparison.json"
                )
        else:
            comparisons = {}
            for n in MESH_SIZES:
                hdf = files[3, n]
                process(hdf, plots=True)
                comparisons[f"N{n}"] = compare_reference(
                    hdf, reference, reference_relative_tolerance
                )
                comparisons[f"N{n}"]["vorticity_norm_t8"] = compare_vorticity_slice(
                    hdf, directory / "reference-data", reference_relative_tolerance
                )
                print(
                    f"Trajectory checks N={n}: PASS (mass, positivity, reconstruction and incompressibility where applicable)"
                )
            (directory / "reference-comparison.json").write_text(
                json.dumps(comparisons, indent=2) + "\n"
            )
            if reference_relative_tolerance is not None and not all(
                metric["passed"]
                for comparison in comparisons.values()
                for metric in comparison.values()
            ):
                raise ValueError(
                    "HD reference accuracy exceeded the requested tolerance"
                )
    except Exception as error:
        write_report(directory, phase, entries, str(error))
        raise
    write_report(directory, phase, entries)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--fluid", choices=("incompressible", "compressible"), required=True
    )
    parser.add_argument("--transport", choices=("ideal", "resistive"), required=True)
    parser.add_argument(
        "--coupling", choices=("hd", "ehd", "mhd", "fmhd"), required=True
    )
    parser.add_argument(
        "--phase", choices=("performance", "validation"), default="performance"
    )
    parser.add_argument(
        "--reference-relative-tolerance",
        type=float,
        help="Optional relative L2 accuracy gate against UCL DNS; no workshop threshold is prescribed",
    )
    args = parser.parse_args()
    if args.reference_relative_tolerance is not None and (
        not np.isfinite(args.reference_relative_tolerance)
        or args.reference_relative_tolerance <= 0
    ):
        parser.error("Reference relative tolerance must be finite and positive")
    campaign(
        args.binary.resolve(),
        args.output.resolve(),
        args.fluid,
        args.transport,
        args.coupling,
        args.phase,
        args.reference_relative_tolerance,
    )


if __name__ == "__main__":
    main()
