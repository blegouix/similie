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
import time

from ci_campaign import MESH_SIZES, STRATEGIES, execute
from nsys_report import (
    iteration_case,
    iteration_profiles,
    markdown,
    profiling_environment,
)


METRICS = {
    "SM (%)": "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "Memory (%)": "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
    "L1/TEX (%)": "l1tex__throughput.avg.pct_of_peak_sustained_elapsed",
    "L2 (%)": "lts__throughput.avg.pct_of_peak_sustained_elapsed",
    "DRAM (%)": "dram__throughput.avg.pct_of_peak_sustained_elapsed",
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
    name = re.escape(case + "/iteration").replace("/", r"\/")
    return "regex:^" + name + "$/"


def compute_groups(plan, max_launches=32):
    if max_launches < 1:
        raise ValueError("Compute batches must contain at least one launch")
    batches = []
    for group in plan["sample_groups"]:
        if "kernel_invocations" in plan:
            for symbol in group["symbols"]:
                if max(group["invocations"]) > plan["kernel_invocations"].get(
                    symbol, 0
                ):
                    raise ValueError(
                        f"Missing first-iteration launch counts for {symbol}"
                    )
        # Keep the exact invocation pattern for each set of symbols. Taking
        # their union also samples unrequested calls and makes NCU retain much
        # larger reports. Bound each process while preserving every target.
        for start in range(0, len(group["invocations"]), max_launches):
            invocations = group["invocations"][start : start + max_launches]
            symbols_per_batch = max_launches // len(invocations)
            for offset in range(0, len(group["symbols"]), symbols_per_batch):
                symbols = group["symbols"][offset : offset + symbols_per_batch]
                batches.append(
                    dict(
                        symbols=symbols,
                        invocations=invocations,
                        launches=len(symbols) * len(invocations),
                    )
                )
    return batches


def compute_command(
    binary, directory, case, n, flags, fluid, transport, coupling, sample_ranges, group
):
    symbols = "|".join(re.escape(symbol) for symbol in group["symbols"])
    invocations = "|".join(map(str, group["invocations"]))
    count = group.get("launches", len(group["symbols"]) * len(group["invocations"]))
    return [
        "ncu",
        "--nvtx",
        "--nvtx-include=" + nvtx_filter(case),
        "--filter-mode=global",
        "--kernel-name-base=mangled",
        "--rename-kernels=off",
        "--range-filter=::^(" + "|".join(map(str, sample_ranges)) + ")$",
        f"--kernel-id=::regex:^({symbols})$:^({invocations})$",
        f"--launch-count={count}",
        "--kill=yes",
        # Kernel replay backs up device allocations for each sampled launch.
        # At N=128 those backups spill into host memory; application replay
        # recreates the state instead and preserves every requested metric.
        "--replay-mode=application",
        "--app-replay-mode=strict",
        # Each application replay recreates the cache history before a sample.
        # Flushing caches at every sampled launch adds work and discards that
        # history, making the counters less representative of the Systems run.
        "--cache-control=none",
        # These reports use hardware counters and launch/occupancy attributes,
        # not instruction or source analysis. Importing the large templated
        # CUDA cubins into every report adds unnecessary processing and I/O.
        "--import-sass=no",
        "--section=SpeedOfLight",
        "--section=LaunchStats",
        "--section=Occupancy",
        "--metrics="
        + ",".join(METRICS[name] for name in ("L1/TEX (%)", "L2 (%)", "DRAM (%)")),
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


def read_compute_report(path, api, expected_launches=None):
    report = api.load_report(str(path))
    kernels = {}
    launches = 0
    for index in range(report.num_ranges()):
        section = report.range_by_idx(index)
        for action_index in range(section.num_actions()):
            action = section.action_by_idx(action_index)
            symbol = action.name(api.IAction.NameBase_MANGLED)
            state = action.nvtx_state()
            stack = []
            if state is not None:
                for domain_id in state.domains():
                    domain = state.domain_by_id(domain_id)
                    if domain.name() in ("", "<default domain>"):
                        stack = list(domain.push_pop_ranges())
            regions = [name for name in stack if iteration_case(name)]
            if not regions:
                continue
            label = "" if iteration_case(stack[-1]) else stack[-1]
            launches += 1
            metrics = {}
            for metric_label, name in METRICS.items():
                metric = action.metric_by_name(name)
                value = metric.as_double() if metric and metric.has_value() else None
                metrics[metric_label] = (
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
            for region in regions:
                # Shared ancestors retain their first sample; sibling regions
                # retain distinct samples even when they execute the same symbol.
                kernels.setdefault(
                    (region, label, symbol),
                    dict(
                        region=region,
                        label=label,
                        symbol=symbol,
                        metrics=metrics,
                        rules=rules,
                    ),
                )
    if expected_launches is not None and launches != expected_launches:
        raise ValueError(
            f"Compute captured {launches} launches; expected {expected_launches} first invocations: {path}"
        )
    if not kernels:
        raise ValueError(
            f"Compute captured no kernels inside the iteration NVTX region: {path}"
        )
    return list(kernels.values())


def read_compute_report_isolated(path, directory, expected_launches):
    # The native report reader can retain allocations after its Python objects
    # are gone. Exit after each report so they cannot accumulate in the campaign
    # process while the next GPU application is running.
    return json.loads(
        subprocess.check_output(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "read-compute",
                "--output",
                str(directory),
                "--compute-report",
                str(path),
                "--expected-launches",
                str(expected_launches),
            ],
            text=True,
        )
    )


def profile_cases(binary, output, fluid, transport, coupling):
    directory = output / "ncu"
    directory.mkdir(parents=True, exist_ok=True)
    results = dict(configuration=f"{fluid}_{transport}_{coupling}", cases={}, errors=[])
    try:
        environment = profiling_environment()
        systems = json.loads((output / "performance/nsys/timings.json").read_text())[
            "cases"
        ]
        database = output / "performance/nsys/tgv.sqlite"
        if database.is_file() and any(
            "kernel_invocations" not in plan for plan in systems.values()
        ):
            # Older captures can supply the counts without rerunning the
            # performance suite or changing its saved timing report.
            systems = iteration_profiles(database)
        for case, relative, n, flags in case_specs():
            case_directory = directory / relative
            case_directory.mkdir(parents=True, exist_ok=True)
            try:
                plan = systems[case]
                if not plan["sample_ranges"]:
                    results["cases"][case] = []
                    continue
                groups = compute_groups(plan)
                count = sum(group["launches"] for group in groups)
                print(
                    f"\nNsight Compute: {case}: {count} kernel calls covering all first samples; "
                    f"{len(groups)} profiler processes (each repeats application setup)",
                    flush=True,
                )
                results["cases"][case] = []
                for index, group in enumerate(groups, 1):
                    batch_directory = case_directory / f"batch-{index}"
                    expected = group["launches"]
                    print(
                        f"  Profiler process {index}/{len(groups)}: "
                        f"capture {expected} launches, then stop",
                        flush=True,
                    )
                    started = time.monotonic()
                    execute(
                        compute_command(
                            binary,
                            batch_directory,
                            case,
                            n,
                            flags,
                            fluid,
                            transport,
                            coupling,
                            plan["sample_ranges"],
                            group,
                        ),
                        batch_directory,
                        environment,
                    )
                    print(
                        f"  Profiler process {index}/{len(groups)} finished in "
                        f"{time.monotonic() - started:.1f} s",
                        flush=True,
                    )
                    # --kill ends each replay pass after the requested calls.
                    # Strict matching rejects inconsistent launches between
                    # passes. No timings or completed trajectory are used.
                    reports = [
                        path
                        for path in batch_directory.glob("kernels.ncu-rep*")
                        if path.suffix in (".ncu-rep", ".ncu-repz")
                    ]
                    if len(reports) != 1:
                        raise FileNotFoundError(
                            f"Expected one Compute report in {batch_directory}"
                        )
                    results["cases"][case].extend(
                        read_compute_report_isolated(reports[0], directory, expected)
                    )
                    (directory / "results.json").write_text(
                        json.dumps(results, indent=2) + "\n"
                    )
                sampled = {
                    (row["region"], row["label"], row["symbol"])
                    for row in results["cases"][case]
                }
                missing = set(map(tuple, plan["targets"])) - sampled
                if missing:
                    raise ValueError(
                        f"Missing Compute samples for {len(missing)} first-iteration region/kernel pairs"
                    )
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
            "Some Compute cases failed; see ncu/results.json and per-batch run.log"
        )


def diagnostic_text(sample):
    if sample is None:
        return "Not sampled"
    diagnostics = []
    seen = set()
    for rule in sample["rules"]:
        title = rule["title"] or rule["name"]
        detail = rule["message"].strip()
        diagnostic = f"{title}: {detail}" if detail else title
        if diagnostic not in seen:
            diagnostics.append(diagnostic)
            seen.add(diagnostic)
    return "\n".join(diagnostics) or "—"


def format_number(value, scale=1):
    return "—" if value is None else f"{value / scale:.6g}"


def kernel_display_label(kernel):
    label = kernel["label"]
    if label.startswith("similie_"):
        return label
    name = kernel.get("name") or kernel["symbol"]
    return f"{name} ({label})" if label else name


def limiting_metrics(metrics):
    """Highlight near-peak throughput, without inferring limits from occupancy."""
    values = {
        name: metrics[name]
        for name in ("SM (%)", "L1/TEX (%)", "L2 (%)", "DRAM (%)")
        if metrics.get(name) is not None
    }
    peak = max(values.values(), default=0)
    return {
        name: 32 if name == "SM (%)" else 31
        for name, value in values.items()
        if value >= 85 and peak - value <= 5
    }


def combined_rows(case, profile, samples):
    indexed = {(row["region"], row["label"], row["symbol"]): row for row in samples}
    for region in profile.get("regions", []):
        common = dict(case=case, region=region["name"])
        yield dict(
            **common,
            kind="region",
            label="Total",
            symbol="",
            calls=region["calls"],
            nvtx_ns=region["host_ns"],
            total_ns=region["gpu_ns"],
            mean_ns=None,
            metrics={},
            rules=[],
            analysis="—",
        )
        counts = {}
        indices = {}
        kernels = sorted(
            region["kernels"], key=lambda kernel: kernel["total_ns"], reverse=True
        )
        for kernel in kernels:
            label = kernel_display_label(kernel)
            counts[label] = counts.get(label, 0) + 1
        for kernel in kernels:
            sample = indexed.get((region["name"], kernel["label"], kernel["symbol"]))
            label = kernel_display_label(kernel)
            indices[label] = indices.get(label, 0) + 1
            display_label = label
            if counts[label] > 1:
                display_label += f" [{indices[label]}]"
            yield dict(
                **common,
                kind="kernel",
                label=kernel["label"],
                display_label=display_label,
                symbol=kernel["symbol"],
                calls=kernel["calls"],
                nvtx_ns=None,
                total_ns=kernel["total_ns"],
                mean_ns=kernel["total_ns"] / kernel["calls"],
                metrics=sample["metrics"] if sample else {},
                rules=sample["rules"] if sample else [],
                analysis=diagnostic_text(sample),
            )
        yield dict(
            **common,
            kind="others",
            label="Others",
            symbol="",
            calls=None,
            nvtx_ns=region["others_host_ns"],
            total_ns=region["others_ns"],
            mean_ns=None,
            metrics={},
            rules=[],
            analysis="—",
        )


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
        "# TGV iteration profiles",
        "",
        "Systems supplies all timings/counts; Compute supplies only metrics. "
        "NVTX durations are host times; GPU totals span each region's first to last GPU activity. "
        "Regions include nested scopes and all CUDA kernels, including library kernels. "
        "Others is time outside all reported kernels, including transfers and gaps; overlaps count once. "
        "L1/TEX, L2 and DRAM SOL are throughput percentages of each unit's sustained peak, not cache hit rates. "
        "Near-peak values (at least 85%, within 5 percentage points of the highest SM/memory throughput) "
        "are bold in the summary and colored in the logs: red for memory, green for SM. "
        "Nsight Compute rule analysis is shown per kernel; full symbols and metrics are in the artifacts.",
    ]
    if compute.get("configuration"):
        lines += ["", markdown(compute["configuration"])]
    if errors:
        lines += ["", "Collection errors: " + markdown("; ".join(errors))]
    visible_metrics = {
        "SM SOL (%)": "SM (%)",
        "L1/TEX SOL (%)": "L1/TEX (%)",
        "L2 SOL (%)": "L2 (%)",
        "DRAM SOL (%)": "DRAM (%)",
        "Occupancy (%)": "Achieved occupancy (%)",
        "Warps/SM": "Active warps/SM",
    }
    columns = [
        "Region",
        "Kernel",
        "GPU total (ms)",
        "Calls",
        "Mean kernel (µs)",
        *visible_metrics,
        "Analysis",
    ]
    records = []
    ansi_rows = {}
    for case, _, _, _ in case_specs():
        lines += [
            "",
            "## " + markdown(case.removeprefix("TaylorGreen/")),
            "",
            "| " + " | ".join(columns) + " |",
            "| " + " | ".join("---" for _ in columns) + " |",
        ]
        profile = systems.get("cases", {}).get(case, {})
        samples = compute.get("cases", {}).get(case, [])
        rows = list(combined_rows(case, profile, samples))
        if not rows:
            lines.append(
                "| "
                + " | ".join(["No iteration data"] + ["—"] * (len(columns) - 1))
                + " |"
            )
        last_region = None
        for row in rows:
            if row["kind"] == "region" and last_region is not None:
                # An empty row gives each NVTX section visual breathing room
                # while keeping one table per benchmark case.
                lines.append("| " + " | ".join("" for _ in columns) + " |")
            region = (
                row["region"].removeprefix(case + "/")
                if row["kind"] == "region"
                else ""
            )
            cells = [
                region,
                row.get("display_label", row["label"]),
                format_number(row["total_ns"], 1e6),
                format_number(row["calls"]),
                format_number(row["mean_ns"], 1e3),
            ]
            cells += [
                format_number(row["metrics"].get(name))
                for name in visible_metrics.values()
            ]
            cells += [row["analysis"]]
            cells = [markdown(cell) for cell in cells]
            ansi_cells = cells.copy()
            highlights = limiting_metrics(row["metrics"])
            for index, name in enumerate(visible_metrics.values(), 5):
                if name in highlights:
                    ansi_cells[index] = (
                        f"\x1b[1;{highlights[name]}m{cells[index]}\x1b[0m"
                    )
                    cells[index] = f"**{cells[index]}**"
            ansi_rows[len(lines)] = "| " + " | ".join(ansi_cells) + " |"
            lines.append("| " + " | ".join(cells) + " |")
            if row["kind"] == "region":
                last_region = row["region"]
        records.extend(rows)
    directory = output / "nsight"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "report.md").write_text("\n".join(lines) + "\n")
    (directory / "report.ansi").write_text(
        "\n".join(ansi_rows.get(index, line) for index, line in enumerate(lines)) + "\n"
    )
    (directory / "report.json").write_text(
        json.dumps(dict(rows=records, errors=errors), indent=2) + "\n"
    )
    with (directory / "report.csv").open("w", newline="") as destination:
        writer = csv.DictWriter(
            destination,
            fieldnames=[
                "case",
                "region",
                "kind",
                "label",
                "symbol",
                "calls",
                "nvtx_ns",
                "total_ns",
                "mean_ns",
                *METRICS,
                "analysis",
                "rules",
            ],
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(
            dict(row, **row["metrics"], rules=json.dumps(row["rules"]))
            for row in records
        )
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("compute", "report", "read-compute"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--fluid", choices=("incompressible", "compressible"))
    parser.add_argument("--transport", choices=("ideal", "resistive"))
    parser.add_argument("--coupling", choices=("hd", "ehd", "mhd", "fmhd"))
    parser.add_argument("--compute-report", type=Path)
    parser.add_argument("--expected-launches", type=int)
    args = parser.parse_args()
    if args.phase == "read-compute":
        if args.compute_report is None or args.expected_launches is None:
            parser.error(
                "read-compute requires --compute-report and --expected-launches"
            )
        print(
            json.dumps(
                read_compute_report(
                    args.compute_report, report_api(args.output), args.expected_launches
                )
            )
        )
    elif args.phase == "compute":
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
