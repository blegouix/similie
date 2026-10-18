# SPDX-FileCopyrightText: 2026 Baptiste Legouix
# SPDX-License-Identifier: AGPL-3.0-or-later
# AI-GENERATED
"""Run the fixed reference mesh campaign and report solver performance in CI."""

import argparse
import json
import math
import statistics
import subprocess
from pathlib import Path

import h5py
import numpy as np
from post_process import process

MESH_SIZES = (64, 128, 256)
PERFORMANCE_STEPS = 5
PERFORMANCE_REPETITIONS = 3
STRATEGY_MIN_TIME = "0.5"
LAPLACIAN_STRATEGIES = (
    ("general", ()),
    ("levi-civita", ("--levi-civita-continuous-hodge-star",)),
    (
        "levi-civita-diagonal",
        (
            "--levi-civita-continuous-hodge-star",
            "--weighted-levi-civita-discrete-hodge-star",
            "--diagonal-reduction",
            "--diagonal-reconstruction",
        ),
    ),
    ("staged-laplacian", ("--staged-laplacian",)),
)


def execute(command, directory, log):
    directory.mkdir(parents=True, exist_ok=True)
    with log.open("w") as output:
        output.write("Command: " + " ".join(map(str, command)) + "\n")
        output.flush()
        result = subprocess.run(
            command, cwd=directory, stdout=output, stderr=subprocess.STDOUT, check=False
        )
    if result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}); see {log}")


def staged_laplacian_bytes(n):
    # Radius-two 3D stencil: 37 points. The HD benchmark caches scalar and
    # three-component two-form Laplacians, each with dense input/output rows.
    return n**3 * 37 * 8 * (1 + 3 * 3)


def validate_results(
    directory,
    configuration,
    n,
    reference,
    expected_steps=PERFORMANCE_STEPS,
    repetitions=PERFORMANCE_REPETITIONS,
):
    files = sorted(directory.glob("*.performance.json"))
    expected = 1 if reference else repetitions
    if expected is not None and len(files) != expected:
        raise ValueError(
            f"Expected {expected} completed trajectories, found {len(files)}"
        )
    if not files:
        raise ValueError("No completed performance trajectories were written")
    if not reference:
        benchmark_report = json.loads((directory / "benchmark.json").read_text())
        measurements = [
            r
            for r in benchmark_report["benchmarks"]
            if r.get("run_type") == "iteration"
        ]
        if len(measurements) != (1 if repetitions is None else repetitions):
            raise ValueError("Incorrect Google Benchmark result count")
        for measurement in measurements:
            if measurement.get("error_occurred"):
                raise ValueError("Google Benchmark reported an error")
            if expected_steps is None:
                if measurement["iterations"] <= 0:
                    raise ValueError("Google Benchmark selected no iterations")
            elif measurement["iterations"] != expected_steps:
                raise ValueError("Google Benchmark measured the wrong number of steps")
            if measurement["steps"] != measurement["iterations"]:
                raise ValueError(
                    "Benchmark iterations do not correspond to solver steps"
                )
    selected_steps = expected_steps
    if not reference and expected_steps is None:
        selected_steps = measurements[0]["iterations"]
    if selected_steps is not None:
        files = [
            path
            for path in files
            if json.loads(path.read_text())["steps"] == selected_steps
        ]
    if not reference and repetitions is None and not files:
        raise ValueError(
            "No trajectory matches the automatically selected iteration count"
        )
    if not reference and repetitions is None:
        files = files[-1:]
    records = []
    for path in files:
        record = json.loads(path.read_text())
        if record["configuration"] != configuration or record["N"] != n:
            raise ValueError("Wrong configuration or resolution in performance output")
        if record["mode"] != ("reference" if reference else "performance"):
            raise ValueError("Wrong trajectory mode")
        if record["steps"] <= 0 or (
            not reference
            and selected_steps is not None
            and record["steps"] != selected_steps
        ):
            raise ValueError("Incorrect iteration count")
        for key in (
            "solver_seconds",
            "mean_step_seconds",
            "cell_updates_per_second",
            "final_time",
        ):
            if not math.isfinite(record[key]) or record[key] <= 0:
                raise ValueError(f"Invalid {key}")
        hdf = path.with_name(path.name.removesuffix(".performance.json") + ".h5")
        with h5py.File(hdf, "r") as data:
            monitors = data["monitors"][:]
            if monitors.shape != (record["steps"] + 1, 14) or not np.all(
                np.isfinite(monitors)
            ):
                raise ValueError("Invalid monitor history")
            if monitors[0, 0] != 0 or np.any(np.diff(monitors[:, 0]) <= 0):
                raise ValueError("Time did not advance monotonically")
            if monitors[-1, 0] != record["final_time"] or np.any(
                monitors[:, 11:13] <= 0
            ):
                raise ValueError("Invalid final time, density, or pressure")
            if abs(monitors[0, 1] - 0.125) > 1e-10:
                raise ValueError("Initial Taylor-Green kinetic energy is incorrect")
            if np.max(np.abs(monitors[:, 8] - monitors[0, 8])) > 1e-8:
                raise ValueError("Mass conservation exceeded tolerance")
            if (
                configuration.startswith("incompressible")
                and np.max(monitors[:, 10]) > 1e-8
            ):
                raise ValueError("Incompressibility constraint exceeded tolerance")
            constraints = data["reconstruction_monitors"][:]
            if (
                not np.all(np.isfinite(constraints))
                or np.max(constraints[:, 1:]) > 1e-8
            ):
                raise ValueError("Fluid reconstruction constraint exceeded tolerance")
            if "electromagnetic_monitors" in data:
                electromagnetic = data["electromagnetic_monitors"][:]
                if not np.all(np.isfinite(electromagnetic)):
                    raise ValueError("Nonfinite electromagnetic monitors")
                if np.max(electromagnetic[:, 5:7]) > 1e-8:
                    raise ValueError(
                        "Gauss or magnetic divergence constraint exceeded tolerance"
                    )
                if np.max(np.abs(electromagnetic[:, 8])) > 1e-8:
                    raise ValueError(
                        "Field/fluid spatial energy exchange is inconsistent"
                    )
            if reference:
                times = sorted(
                    float(data[name]["time"][0])
                    for name in data
                    if name.startswith("snapshot_")
                )
                if record["final_time"] != 10 or times != [0, 8, 10]:
                    raise ValueError(
                        "Reference trajectory must finish at t=10 with snapshots 0,8,10"
                    )
        if reference:
            process(hdf, vtk=False, plots=True)
        records.append(record)
    return records


def write_report(output, manifest):
    lines = [
        f"# Taylor–Green: {manifest['configuration']}",
        "",
        (
            "CUDA, Release. One iteration is one evolving SSPRK3 step. "
            "Strategy rows use automatic iteration counts with a 0.5 s minimum-time target. "
            "Reference rows are complete t=0→10 HD trajectories."
        ),
        "",
        "| Mode | N | Steps/run | Runs | Final time | Mean step (ms) | Std (ms) | Mcell updates/s | Status |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    if manifest.get("validation_error"):
        lines.extend(
            ["", "N=8 solver validation failed: " + manifest["validation_error"], ""]
        )
    for entry in manifest["runs"]:
        records = entry.get("performance", [])
        if records:
            step_times = [r["mean_step_seconds"] * 1000 for r in records]
            steps = statistics.mean(r["steps"] for r in records)
            time = statistics.mean(r["final_time"] for r in records)
            throughput = (
                statistics.mean(r["cell_updates_per_second"] for r in records) / 1e6
            )
            lines.append(
                f"| {entry['mode']} | {entry['N']} | {steps:g} | {len(records)} | {time:.6g} | "
                f"{statistics.mean(step_times):.6g} | {statistics.pstdev(step_times):.3g} | "
                f"{throughput:.6g} | {entry['status']} |"
            )
        else:
            lines.append(
                f"| {entry['mode']} | {entry['N']} | — | — | — | — | — | — | {entry['status']} |"
            )
        if entry.get("error"):
            lines.extend(
                [
                    "",
                    f"Failure for {entry['mode']} N={entry['N']}: {entry['error']}",
                    "",
                ]
            )
        elif entry.get("note"):
            lines.extend(
                ["", f"Note for {entry['mode']} N={entry['N']}: {entry['note']}", ""]
            )
    lines.extend(
        [
            "",
            (
                "Raw Google Benchmark JSON, hardware context, solver/monitor/export "
                "timings, Krylov counts, logs, and monitor histories are included in the artifact. "
                "Full HD cases additionally include HDF5/XDMF fields, spectra, and plots."
            ),
        ]
    )
    (output / "report.md").write_text("\n".join(lines) + "\n")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def campaign(binary, output, fluid, transport, coupling):
    configuration = f"{fluid}_{transport}_{coupling}"
    flags = [f"--{fluid}", f"--{transport}", f"--{coupling}"]
    if coupling == "hd" and transport != "ideal":
        raise ValueError("Resistive HD is not a supported physics configuration")
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"configuration": configuration, "mesh_sizes": MESH_SIZES, "runs": []}
    write_report(output, manifest)
    validation_flags = (
        [
            "--levi-civita-continuous-hodge-star",
            "--weighted-levi-civita-discrete-hodge-star",
            "--diagonal-reduction",
            "--diagonal-reconstruction",
        ]
        if coupling == "hd"
        else []
    )
    if coupling == "hd":
        manifest["validation"] = "full reference trajectories at N=64,128,256"
    else:
        try:
            execute(
                [str(binary), "--N=8", *flags, "--validate"],
                output,
                output / "validation.log",
            )
            manifest["validation"] = "passed"
        except RuntimeError as error:
            manifest["validation"] = "failed"
            manifest["validation_error"] = str(error)
            write_report(output, manifest)
            raise
    modes = ("performance", "reference") if coupling == "hd" else ("performance",)
    for mode in modes:
        for n in MESH_SIZES:
            directory = output / mode / f"N{n}"
            directory.mkdir(parents=True, exist_ok=True)
            entry = {"mode": mode, "N": n, "status": "running"}
            manifest["runs"].append(entry)
            write_report(output, manifest)
            command = [str(binary), f"--N={n}", *flags, f"--output-dir={directory}"]
            if mode == "reference":
                command.extend(("--reference", *validation_flags))
            else:
                command.extend(
                    [
                        "--no-snapshots",
                        f"--benchmark_min_time={PERFORMANCE_STEPS}x",
                        f"--benchmark_repetitions={PERFORMANCE_REPETITIONS}",
                        f"--benchmark_out={directory / 'benchmark.json'}",
                        "--benchmark_out_format=json",
                    ]
                )
            print(f"Running {configuration}: {mode}, N={n}", flush=True)
            try:
                execute(command, directory, directory / "run.log")
                entry["performance"] = validate_results(
                    directory, configuration, n, mode == "reference"
                )
                entry["status"] = "passed"
            except (RuntimeError, ValueError, OSError, KeyError) as error:
                entry["status"] = "failed"
                entry["error"] = str(error)
                print(str(error), flush=True)
            write_report(output, manifest)
    if coupling == "hd":
        for strategy, strategy_flags in LAPLACIAN_STRATEGIES:
            for n in MESH_SIZES:
                directory = output / "laplacian-strategies" / strategy / f"N{n}"
                directory.mkdir(parents=True, exist_ok=True)
                entry = {
                    "mode": f"strategy-{strategy}",
                    "strategy": strategy,
                    "N": n,
                    "status": "running",
                }
                manifest["runs"].append(entry)
                write_report(output, manifest)
                if strategy == "staged-laplacian" and n == 256:
                    required = staged_laplacian_bytes(n)
                    entry["status"] = "skipped"
                    entry["note"] = (
                        f"Estimated staged stencil storage is {required / 1024**3:.2f} GiB; "
                        "the N=256 staged-Laplacian comparison is skipped because it exceeds "
                        "the available CI GPU memory."
                    )
                    write_report(output, manifest)
                    continue
                command = [
                    str(binary),
                    f"--N={n}",
                    *flags,
                    *strategy_flags,
                    "--no-snapshots",
                    f"--output-dir={directory}",
                    f"--benchmark_min_time={STRATEGY_MIN_TIME}",
                    f"--benchmark_out={directory / 'benchmark.json'}",
                    "--benchmark_out_format=json",
                ]
                print(
                    f"Running {configuration}: strategy={strategy}, N={n}", flush=True
                )
                try:
                    execute(command, directory, directory / "run.log")
                    entry["performance"] = validate_results(
                        directory,
                        configuration,
                        n,
                        False,
                        expected_steps=None,
                        repetitions=None,
                    )
                    entry["status"] = "passed"
                except (RuntimeError, ValueError, OSError, KeyError) as error:
                    entry["status"] = "failed"
                    entry["error"] = str(error)
                    print(str(error), flush=True)
                write_report(output, manifest)
    if any(entry["status"] == "failed" for entry in manifest["runs"]):
        raise RuntimeError(
            "One or more Taylor–Green CI cases failed; see report.md and logs"
        )


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
    args = parser.parse_args()
    campaign(
        args.binary.resolve(),
        args.output.resolve(),
        args.fluid,
        args.transport,
        args.coupling,
    )


if __name__ == "__main__":
    main()
