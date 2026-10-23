# SPDX-FileCopyrightText: 2026 Baptiste Legouix
# SPDX-License-Identifier: AGPL-3.0-or-later
# AI-GENERATED
"""Collect Compute diagnostics and join them with Systems timings for each case."""

import argparse
import csv
import importlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

from ci_campaign import MESH_SIZES, STRATEGIES, execute
from nsys_report import markdown, profiling_environment


METRICS = {
    "SM (%)": "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "Memory (%)": "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
    "Achieved occupancy (%)": "sm__warps_active.avg.pct_of_peak_sustained_active",
    "Theoretical occupancy (%)": "sm__maximum_warps_per_active_cycle_pct",
    "Active warps/SM": "sm__warps_active.avg.per_cycle_active",
    "Registers/thread": "launch__registers_per_thread",
}


def case_specs():
    for strategy, flags in enumerate(STRATEGIES, 1):
        for n in MESH_SIZES:
            yield (
                f"TaylorGreen/N={n}/{'+'.join(flags) or 'default'}",
                f"strategy-{strategy}/N{n}",
                n,
                flags,
            )


def nvtx_filter(case):
    # Slashes and '+' are literal characters in our full NVTX range names.
    name = case + "/iteration"
    return "".join("\\" + char if char in "@,[]/*+" else char for char in name) + "/"


def compute_command(binary, directory, case, n, flags, fluid, transport, coupling):
    return [
        "ncu",
        "--nvtx",
        "--nvtx-include=" + nvtx_filter(case),
        "--filter-mode=global",
        "--kernel-name-base=mangled",
        "--rename-kernels=off",
        "--kernel-id=::regex:.*:1",
        "--section=SpeedOfLight",
        "--section=LaunchStats",
        "--section=Occupancy",
        "--apply-rules=yes",
        "--force-overwrite",
        f"--export={directory / 'kernels'}",
        str(binary),
        f"--{fluid}",
        f"--{transport}",
        f"--{coupling}",
        f"--N={n}",
        *flags,
        "--no-snapshots",
        f"--output-dir={directory}",
        "--benchmark_min_time=1x",
        "--benchmark_repetitions=1",
        "--benchmark_min_warmup_time=0",
        f"--benchmark_out={directory / 'benchmark.json'}",
        "--benchmark_out_format=json",
    ]


def report_api(directory):
    executable = shutil.which("ncu")
    if executable is None:
        raise FileNotFoundError("Nsight Compute (ncu) is missing from the CUDA image")
    version = subprocess.check_output([executable, "--version"], text=True)
    (directory / "version.txt").write_text(version)
    release = re.search(r"Version (\d+\.\d+\.\d+)", version)
    roots = [
        Path(executable).resolve().parent,
        Path(executable).resolve().parent.parent,
    ]
    if release:
        roots += sorted(Path("/opt/nvidia/nsight-compute").glob(release[1] + "*"))
    candidates = [root / "extras/python" for root in roots]
    if os.environ.get("SIMILIE_NCU_PYTHON_PATH"):
        candidates.insert(0, Path(os.environ["SIMILIE_NCU_PYTHON_PATH"]))
    for path in candidates:
        if (path / "ncu_report.py").is_file():
            sys.path.insert(0, str(path))
            return importlib.import_module("ncu_report")
    raise FileNotFoundError(
        "Cannot locate the matching Nsight Compute extras/python API"
    )


def read_compute_report(path, api):
    report = api.load_report(str(path))
    kernels = {}
    for index in range(report.num_ranges()):
        section = report.range_by_idx(index)
        for action_index in range(section.num_actions()):
            action = section.action_by_idx(action_index)
            symbol = action.name(api.IAction.NameBase_MANGLED)
            if symbol in kernels:
                raise ValueError(f"Compute sampled a kernel more than once: {symbol}")
            metrics = {}
            for label, name in METRICS.items():
                metric = action.metric_by_name(name)
                value = metric.as_double() if metric and metric.has_value() else None
                metrics[label] = (
                    value if value is not None and math.isfinite(value) else None
                )
            rules = []
            for rule in action.rule_results():
                if rule.has_rule_message():
                    message = rule.rule_message()
                    rules.append(
                        dict(
                            name=rule.name(),
                            section=rule.section_identifier(),
                            title=message["title"],
                            message=message["message"],
                        )
                    )
            kernels[symbol] = dict(
                display_name=action.name(api.IAction.NameBase_DEMANGLED),
                metrics=metrics,
                rules=rules,
            )
    if not kernels:
        raise ValueError(
            f"Compute captured no kernels inside the iteration NVTX region: {path}"
        )
    return kernels


def profile_cases(binary, output, fluid, transport, coupling):
    directory = output / "ncu"
    directory.mkdir(parents=True, exist_ok=True)
    results = dict(configuration=f"{fluid}_{transport}_{coupling}", cases={}, errors=[])
    try:
        api = report_api(directory)
        environment = profiling_environment()
        for case, relative, n, flags in case_specs():
            case_directory = directory / relative
            case_directory.mkdir(parents=True, exist_ok=True)
            try:
                print(f"\nNsight Compute: {case}", flush=True)
                execute(
                    compute_command(
                        binary,
                        case_directory,
                        case,
                        n,
                        flags,
                        fluid,
                        transport,
                        coupling,
                    ),
                    case_directory,
                    environment,
                )
                benchmark = json.loads((case_directory / "benchmark.json").read_text())
                rows = [
                    row
                    for row in benchmark["benchmarks"]
                    if row.get("run_type", "iteration") == "iteration"
                ]
                if (
                    len(rows) != 1
                    or rows[0].get("iterations") != 1
                    or rows[0].get("error_occurred")
                    or rows[0].get("skipped")
                ):
                    raise ValueError(
                        "Compute diagnostic must execute exactly one benchmark iteration"
                    )
                # Recent releases default to compressed .ncu-repz reports.
                reports = [
                    path
                    for path in case_directory.glob("kernels.ncu-rep*")
                    if path.suffix in (".ncu-rep", ".ncu-repz")
                ]
                if len(reports) != 1:
                    raise FileNotFoundError(
                        f"Expected one Compute report in {case_directory}"
                    )
                results["cases"][case] = read_compute_report(reports[0], api)
            except Exception as error:
                results["errors"].append(f"{case}: {error}")
                print(results["errors"][-1], flush=True)
            finally:
                (directory / "results.json").write_text(
                    json.dumps(results, indent=2) + "\n"
                )
    except Exception as error:
        results["errors"].append(str(error))
        raise
    finally:
        (directory / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    if results["errors"]:
        raise RuntimeError(
            "Some Compute cases failed; see ncu/results.json and per-case run.log"
        )


def diagnostic_text(sample):
    if sample is None:
        return "Not sampled in the single-iteration Compute run"
    messages = []
    for rule in sample["rules"]:
        # Nsight's internal report links have no target in a GitHub table.
        text = re.sub(r"@section:[^:]+:([^@]+)@", r"\1", rule["message"])
        text = re.sub(r"@url:([^:]+):[^@]+@", r"\1", text)
        title = rule["title"] or rule["name"]
        messages.append(f"{title}: {text}")
    return "\n".join(messages) or "No Nsight Compute rule messages"


def format_number(value, scale=1):
    return "—" if value is None else f"{value / scale:.6g}"


def write_combined_report(output):
    systems_path = output / "performance/nsys/timings.json"
    compute_path = output / "ncu/results.json"
    systems = json.loads(systems_path.read_text()) if systems_path.is_file() else {}
    compute = json.loads(compute_path.read_text()) if compute_path.is_file() else {}
    errors = list(compute.get("errors", []))
    if not systems:
        errors.append("Missing Nsight Systems timings")
    if not compute:
        errors.append("Missing Nsight Compute diagnostics")
    lines = [
        "# Taylor–Green Nsight Systems and Compute report",
        "",
        "Times and call counts come from the Systems performance run, including setup. "
        "Region host time and projected GPU span are inclusive; nested regions overlap "
        "and must not be added. Region kernel time sums contained kernel executions. "
        "Kernel rows aggregate each full kernel symbol once per case.",
        "",
        "Compute metrics and rule messages describe the first eligible invocation of each "
        "full kernel symbol within the iteration region of a separate one-iteration run. "
        "They are diagnostic samples, not averages over the Systems run. Setup-only kernels "
        "have no Compute sample. The artifact retains full symbols, metrics, messages, "
        "CSV/JSON tables, the Systems timeline and each Compute report. "
        "SM and Memory percentages measure throughput relative to peak sustained throughput.",
    ]
    if compute.get("configuration"):
        lines.insert(2, "Configuration: " + markdown(compute["configuration"]) + "\n")
    if errors:
        lines += ["", "Collection errors: " + markdown("; ".join(errors))]
    columns = [
        "Kind",
        "NVTX region / kernel",
        "Calls",
        "Host (ms)",
        "GPU span (ms)",
        "Kernel time (ms)",
        "Contained kernel calls",
        "Mean host/kernel (µs)",
        *METRICS,
        "Nsight Compute analysis",
    ]
    records = []
    for case, _, _, _ in case_specs():
        lines += [
            "",
            f"## {markdown(case.removeprefix('TaylorGreen/'))}",
            "",
            "| " + " | ".join(columns) + " |",
            "| " + " | ".join("---" for _ in columns) + " |",
        ]
        regions = sorted(
            (row for row in systems.get("case_regions", []) if row["case"] == case),
            key=lambda row: row["name"],
        )
        kernels = sorted(
            (row for row in systems.get("case_kernels", []) if row["case"] == case),
            key=lambda row: row["total_ns"],
            reverse=True,
        )
        samples = compute.get("cases", {}).get(case, {})
        rows = []
        for index, region in enumerate(regions, 1):
            rows.append(
                dict(
                    case=case,
                    kind="Region",
                    name=region["name"],
                    display_name=f"R{index}: "
                    + region["name"].removeprefix(case + "/"),
                    calls=region["calls"],
                    host_ns=region["host_ns"],
                    gpu_span_ns=region["gpu_span_ns"],
                    kernel_ns=region["kernel_ns"],
                    kernel_calls=region["kernel_calls"],
                    mean_us=region["host_ns"] / region["calls"] / 1e3
                    if region["calls"]
                    else None,
                    sample=None,
                    analysis="—",
                )
            )
        symbols = {kernel["kernel"] for kernel in kernels}
        for symbol in sorted(set(samples) - symbols):
            kernels.append(dict(kernel=symbol, calls=None, total_ns=None))
        for index, kernel in enumerate(kernels, 1):
            symbol = kernel["kernel"]
            sample = samples.get(symbol)
            rows.append(
                dict(
                    case=case,
                    kind="Kernel",
                    name=symbol,
                    display_name=f"K{index}: "
                    + (
                        "; ".join(kernel["labels"])
                        if kernel.get("labels")
                        else (sample["display_name"] if sample else symbol)
                    ),
                    calls=kernel["calls"],
                    host_ns=None,
                    gpu_span_ns=None,
                    kernel_ns=kernel["total_ns"],
                    kernel_calls=None,
                    mean_us=kernel["total_ns"] / kernel["calls"] / 1e3
                    if kernel["calls"]
                    else None,
                    sample=sample,
                    analysis=diagnostic_text(sample),
                )
            )
        if not rows:
            lines.append(
                "| "
                + " | ".join(
                    ["Unavailable", "No collected data"] + ["—"] * (len(columns) - 2)
                )
                + " |"
            )
        for row in rows:
            label = row.get("display_name", row["name"]).removeprefix(case + "/")
            if len(label) > 160:
                label = label[:157] + "..."
            cells = [
                row["kind"],
                label,
                format_number(row["calls"]),
                format_number(row["host_ns"], 1e6),
                format_number(row["gpu_span_ns"], 1e6),
                format_number(row["kernel_ns"], 1e6),
                format_number(row["kernel_calls"]),
                format_number(row["mean_us"]),
            ]
            cells += [
                format_number(row["sample"]["metrics"].get(metric))
                if row["sample"]
                else "—"
                for metric in METRICS
            ]
            cells += [row["analysis"]]
            lines.append(
                "| "
                + " | ".join(
                    "<br>".join(markdown(part) for part in cell.split("\n"))
                    for cell in cells
                )
                + " |"
            )
            record = {key: value for key, value in row.items() if key != "sample"}
            record.update(row["sample"]["metrics"] if row["sample"] else {})
            record["rules"] = row["sample"]["rules"] if row["sample"] else []
            records.append(record)
    directory = output / "nsight"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "report.md").write_text("\n".join(lines) + "\n")
    (directory / "report.json").write_text(
        json.dumps(dict(rows=records, errors=errors), indent=2) + "\n"
    )
    with (directory / "report.csv").open("w", newline="") as destination:
        writer = csv.DictWriter(
            destination,
            fieldnames=[
                "case",
                "kind",
                "name",
                "display_name",
                "calls",
                "host_ns",
                "gpu_span_ns",
                "kernel_ns",
                "kernel_calls",
                "mean_us",
                *METRICS,
                "analysis",
            ],
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(records)
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("compute", "report"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--fluid", choices=("incompressible", "compressible"))
    parser.add_argument("--transport", choices=("ideal", "resistive"))
    parser.add_argument("--coupling", choices=("hd", "ehd", "mhd", "fmhd"))
    args = parser.parse_args()
    if args.phase == "compute":
        if not all((args.binary, args.fluid, args.transport, args.coupling)):
            parser.error(
                "compute requires --binary, --fluid, --transport and --coupling"
            )
        profile_cases(
            args.binary.resolve(),
            args.output.resolve(),
            args.fluid,
            args.transport,
            args.coupling,
        )
    elif write_combined_report(args.output.resolve()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
