# SPDX-FileCopyrightText: 2026 Baptiste Legouix
# SPDX-License-Identifier: AGPL-3.0-or-later
# AI-GENERATED
"""Run one performance suite or the separate HD reference-validation suite."""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

import h5py
import numpy as np
from post_process import process

MESH_SIZES = (64, 128)
PERFORMANCE_STEPS = 5
STRATEGY_RTOL = 1e-7
STRATEGY_ATOL = 1e-9
STRATEGIES = (
    (),
    ("--levi-civita-continuous-hodge-star",),
    (
        "--levi-civita-continuous-hodge-star",
        "--weighted-levi-civita-discrete-hodge-star",
        "--diagonal-reduction",
        "--diagonal-reconstruction",
    ),
    (
        "--levi-civita-continuous-hodge-star",
        "--staged-weighted-levi-civita-discrete-hodge-star",
        "--diagonal-reduction",
        "--diagonal-reconstruction",
    ),
    ("--staged-laplacian",),
)
REFERENCE_URL = "https://cfd.ku.edu/hiocfd/data.tgz"
REFERENCE_FILE = "spectral_Re1600_512.gdiag"
REFERENCE_SHA256 = "8e01c0024ae2dde20df8dc77a20fa7fb41c3abef2fc528a2ac684afed96d6429"


def execute(command, directory):
    """Preserve the complete Google Benchmark output without campaign chatter."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "command.json").write_text(
        json.dumps(list(map(str, command)), indent=2)
    )
    with (directory / "run.log").open("w") as log:
        child = subprocess.Popen(
            list(map(str, command)),
            cwd=directory,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in child.stdout:
            log.write(line)
            log.flush()
            sys.stdout.write(line)
            sys.stdout.flush()
        if child.wait():
            raise RuntimeError(f"Benchmark failed; see {directory / 'run.log'}")


def validate_trajectory(path, configuration, n, reference=False):
    record = json.loads(path.read_text())
    if record["configuration"] != configuration or record["N"] != n:
        raise ValueError(f"Wrong configuration or resolution: {path}")
    if record["mode"] != ("reference" if reference else "performance"):
        raise ValueError(f"Wrong trajectory mode: {path}")
    if record["steps"] <= 0 or (not reference and record["steps"] != PERFORMANCE_STEPS):
        raise ValueError(f"Incorrect step count: {path}")
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
        monitors = data["monitors"][:]
        if monitors.shape != (record["steps"] + 1, 14) or not np.all(
            np.isfinite(monitors)
        ):
            raise ValueError(f"Invalid monitor history: {hdf}")
        if monitors[0, 0] != 0 or np.any(np.diff(monitors[:, 0]) <= 0):
            raise ValueError(f"Time did not advance monotonically: {hdf}")
        if monitors[-1, 0] != record["final_time"] or np.any(monitors[:, 11:13] <= 0):
            raise ValueError(f"Invalid final time, density, or pressure: {hdf}")
        if abs(monitors[0, 1] - 0.125) > 1e-10:
            raise ValueError(f"Incorrect initial Taylor-Green kinetic energy: {hdf}")
        if np.max(np.abs(monitors[:, 8] - monitors[0, 8])) > 1e-8:
            raise ValueError(f"Mass conservation exceeded tolerance: {hdf}")
        if (
            configuration.startswith("incompressible")
            and np.max(monitors[:, 10]) > 1e-8
        ):
            raise ValueError(f"Incompressibility constraint exceeded tolerance: {hdf}")
        constraints = data["reconstruction_monitors"][:]
        if not np.all(np.isfinite(constraints)) or np.max(constraints[:, 1:]) > 1e-8:
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
            times = sorted(
                float(data[name]["time"][0])
                for name in data
                if name.startswith("snapshot_")
            )
            if record["final_time"] != 10 or times != list(range(11)):
                raise ValueError(
                    f"Reference must finish at t=10 with snapshots every tc: {hdf}"
                )
            for index in range(11):
                if not hdf.with_suffix(f".snapshot_{index}.vtk").is_file():
                    raise ValueError(f"Missing live PDI VTK export {index}: {hdf}")
    return record, hdf


def compare_strategies(baseline, candidate):
    """Compare complete final states and evolving monitors after all timing stops.

    All cases advance exactly five steps from the same initial state. Compare
    times as well as fields: different adaptive time steps cannot silently pass.
    Timing columns and Krylov residuals are deliberately excluded from equality.
    """
    results = {}
    with h5py.File(baseline, "r") as expected, h5py.File(candidate, "r") as actual:
        names = ["final_state", "monitors"]
        if "final_electromagnetic_state" in expected:
            names.extend(("final_electromagnetic_state", "electromagnetic_monitors"))
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
    return results


def reference_history(directory):
    """Keep the exact workshop input and its provenance in the artifact."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / REFERENCE_FILE
    if not path.exists():
        archive_path = directory / "data.tgz"
        with (
            urllib.request.urlopen(REFERENCE_URL, timeout=120) as response,
            archive_path.open("wb") as out,
        ):
            while block := response.read(1024 * 1024):
                out.write(block)
        with tarfile.open(archive_path) as archive:
            members = [
                member
                for member in archive.getmembers()
                if Path(member.name).name == REFERENCE_FILE and member.isfile()
            ]
            if len(members) != 1:
                raise ValueError("Workshop archive has an ambiguous reference history")
            path.write_bytes(archive.extractfile(members[0]).read())
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    if checksum != REFERENCE_SHA256:
        raise ValueError("Workshop reference checksum mismatch")
    (directory / "source.json").write_text(
        json.dumps(
            {
                "url": REFERENCE_URL,
                "description": "UCL dealiased pseudo-spectral 512^3 reference, HiOCFD C3.5 (previous edition of HOW4 BS1)",
                "sha256": checksum,
                "columns": [
                    "time",
                    "kinetic_energy",
                    "positive_dissipation",
                    "enstrophy",
                ],
            },
            indent=2,
        )
        + "\n"
    )
    return np.loadtxt(path)


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


def compare_spectrum(hdf, wave_number, spectrum, reference_directory):
    """Compare only the resolved, digitized range; never extrapolate the image."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    source = Path(__file__).parent / "reference_data"
    reference_directory.mkdir(parents=True, exist_ok=True)
    for path in source.iterdir():
        if path.is_file():
            shutil.copyfile(path, reference_directory / path.name)
    csv = reference_directory / "spectrum_t8.csv"
    provenance = json.loads((reference_directory / "spectrum_t8.json").read_text())
    if hashlib.sha256(csv.read_bytes()).hexdigest() != provenance["csv_sha256"]:
        raise ValueError("Digitized spectrum reference checksum mismatch")
    reference = np.loadtxt(csv, delimiter=",", skiprows=4)
    selected = (wave_number >= reference[0, 0]) & (wave_number <= reference[-1, 0])
    k, energy = wave_number[selected], spectrum[selected]
    if len(k) < 2 or not np.all(np.isfinite(energy)) or np.any(energy <= 0):
        raise ValueError("No positive spectrum overlapping the digitized reference")
    expected = 10 ** np.interp(
        np.log10(k), np.log10(reference[:, 0]), np.log10(reference[:, 1])
    )
    metrics = {
        "source": provenance["source"],
        "image_derived_reference": True,
        "time": 8,
        "wave_number_range": [float(k[0]), float(k[-1])],
        "relative_l2_error": float(
            np.linalg.norm(energy - expected) / np.linalg.norm(expected)
        ),
        "rms_log10_error": float(np.sqrt(np.mean(np.log10(energy / expected) ** 2))),
        "digitization_relative_energy_bound": provenance["relative_energy_bound"],
        "digitization_relative_wave_number_bound": provenance[
            "relative_wave_number_bound"
        ],
    }
    np.savetxt(
        hdf.with_suffix(".spectrum-comparison.csv"),
        np.column_stack((k, energy, expected, energy / expected)),
        delimiter=",",
        header="wave_number,simulation_energy,digitized_reference_energy,ratio",
        comments="",
    )
    hdf.with_suffix(".spectrum-comparison.json").write_text(
        json.dumps(metrics, indent=2) + "\n"
    )
    figure, axis = plt.subplots(figsize=(7, 5), constrained_layout=True)
    in_range = reference[:, 0] <= wave_number[-1]
    axis.loglog(
        reference[in_range, 0],
        reference[in_range, 1],
        label="HOW4 Spectral (digitized Figure 3)",
    )
    axis.fill_between(
        reference[in_range, 0],
        reference[in_range, 4],
        reference[in_range, 5],
        alpha=0.25,
        label="±2-pixel energy bound (about 7%)",
    )
    positive = (wave_number > 0) & (spectrum > 0)
    axis.loglog(wave_number[positive], spectrum[positive], label="SimiLie")
    axis.set(
        xlabel="Wave number kL",
        ylabel="Shell energy / V0²",
        title="Taylor–Green spectrum at t/tc=8",
    )
    axis.legend()
    figure.savefig(hdf.with_suffix(".spectrum-comparison.png"), dpi=160)
    plt.close(figure)
    print(
        f"Spectrum comparison {hdf.stem}: relative_L2_error={metrics['relative_l2_error']:.6g}, RMS_log10_error={metrics['rms_log10_error']:.6g}; digitized reference energy bound about 7%, k bound about 1.3%"
    )
    text_plot(
        np.log10(k),
        np.log10(energy),
        np.log10(expected),
        "Log10 spectrum at t=8",
        "log10(kL)",
        "digitized HOW4 Figure 3",
    )
    return metrics


def write_report(output, phase, entries, error=None):
    lines = [
        f"# Taylor–Green {phase}",
        "",
        "Solver timings exclude monitoring, export, strategy comparisons and reference post-processing.",
        "",
        "| N | Flags | Steps | Solver step (s) | Mcell updates/s |",
        "| ---: | --- | ---: | ---: | ---: |",
    ]
    for record in entries:
        lines.append(
            f"| {record['N']} | `{record['flags']}` | {record['steps']} | {record['mean_step_seconds']:.6g} | {record['cell_updates_per_second'] / 1e6:.6g} |"
        )
    if phase == "performance":
        lines.extend(
            (
                "",
                f"Final fluid/field states and monitor histories are compared with the default strategy using rtol={STRATEGY_RTOL:g}, atol={STRATEGY_ATOL:g}; see strategy-comparison.json.",
            )
        )
    else:
        lines.extend(
            (
                "",
                "HD accuracy is compared with the UCL 512³ pseudo-spectral workshop history. Relative L2 errors are resolution-study metrics, not a workshop-prescribed acceptance threshold. An explicit --reference-relative-tolerance enables an accuracy gate. See reference-comparison.json and PNG plots. Spectra at t=8 are compared with the Spectral curve digitized from page 3, Figure 3 of the HOW4 BS1 PDF. The CSV, provenance, extraction overlay and pixel uncertainty bounds are included in reference-data; this is an image-derived reference.",
                "",
                "Live PDI HDF5/VTK snapshots are saved every convective time unit, including t=0,8,10.",
            )
        )
    if phase == "validation":
        lines.extend(
            (
                "",
                "| N | Reference observable | Relative L2 error |",
                "| ---: | --- | ---: |",
            )
        )
        for path in sorted(output.glob("N*/*.reference-comparison.json")):
            for name, metric in json.loads(path.read_text()).items():
                lines.append(
                    f"| {path.parent.name[1:]} | {name} | {metric['relative_l2_error']:.3%} |"
                )
        for path in sorted(output.glob("N*/*.spectrum-comparison.json")):
            metric = json.loads(path.read_text())
            lines.append(
                f"| {path.parent.name[1:]} | spectrum at t=8 (digitized) | {metric['relative_l2_error']:.3%} |"
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
        execute(command, directory)
        benchmark = json.loads((directory / "benchmark.json").read_text())
        rows = [
            row
            for row in benchmark["benchmarks"]
            if row.get("run_type", "iteration") == "iteration"
        ]
        if len(rows) != (10 if phase == "performance" else 2) or any(
            row.get("error_occurred") or row.get("skipped") for row in rows
        ):
            raise ValueError("Incomplete Google Benchmark suite")
        files = {}
        for strategy in range(5) if phase == "performance" else (3,):
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
                files[strategy, n] = hdf
        if phase == "performance":
            comparisons = {
                f"strategy-{strategy + 1}/N{n}": compare_strategies(
                    files[0, n], files[strategy, n]
                )
                for strategy in range(1, 5)
                for n in MESH_SIZES
            }
            (directory / "strategy-comparison.json").write_text(
                json.dumps(
                    {
                        "rtol": STRATEGY_RTOL,
                        "atol": STRATEGY_ATOL,
                        "comparisons": comparisons,
                    },
                    indent=2,
                )
                + "\n"
            )
            if not all(
                result["passed"]
                for comparison in comparisons.values()
                for result in comparison.values()
            ):
                raise ValueError(
                    "Operator strategies disagree; see strategy-comparison.json"
                )
        else:
            comparisons = {}
            for n in MESH_SIZES:
                hdf = files[3, n]
                _, wave_number, spectrum = process(hdf, vtk=False, plots=True)
                compare_spectrum(
                    hdf, wave_number, spectrum, directory / "reference-data"
                )
                comparisons[f"N{n}"] = compare_reference(
                    hdf, reference, reference_relative_tolerance
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
