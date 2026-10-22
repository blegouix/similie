# SPDX-FileCopyrightText: 2026 Baptiste Legouix
# SPDX-License-Identifier: AGPL-3.0-or-later
# AI-GENERATED
"""Capture CUDA/NVTX timelines and summarize TGV operations and their kernels."""

import argparse
import csv
import html
import json
import math
import os
from pathlib import Path
import subprocess


def profile_command(command, directory):
    directory.mkdir(parents=True, exist_ok=True)
    connector = Path(
        os.environ.get(
            "SIMILIE_KOKKOS_NVTX_CONNECTOR", "/usr/local/lib/libkp_nvtx_connector.so"
        )
    )
    if not connector.is_file():
        raise FileNotFoundError(f"Missing Kokkos NVTX connector: {connector}")
    environment = os.environ.copy()
    environment["KOKKOS_TOOLS_LIBS"] = str(connector)
    # GPU attribution uses CUDA launch correlation, without adding global fences.
    environment["KOKKOS_TOOLS_GLOBALFENCES"] = "0"
    return [
        "nsys",
        "profile",
        "--trace=cuda,nvtx",
        "--sample=none",
        "--cpuctxsw=none",
        "--force-overwrite=true",
        f"--output={directory / 'tgv'}",
        *map(str, command),
    ], environment


def number(row, label):
    key = next(key for key in row if key.split(" (", 1)[0] == label)
    value = float(row[key])
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"Invalid nsys statistic {key}: {row[key]}")
    return value


def region_name(row, column):
    name = row[column]
    start = name.find("TaylorGreen/N=")
    return name[start:] if start >= 0 else None


def read_csv(path):
    with path.open(newline="") as source:
        return list(csv.DictReader(source))


def markdown(value):
    return (
        html.escape(str(value), quote=False).replace("|", "&#124;").replace("\n", " ")
    )


def summarize_profile(directory, expected_cases=None):
    """Use NVIDIA's launch-correlated reports; host scopes need not fence the GPU.

    Host ranges, GPU spans, and kernel durations are inclusive. A kernel belongs
    to all its ancestor ranges, so rows at different depths must never be added.
    """
    report = directory / "tgv.nsys-rep"
    if not report.is_file():
        raise FileNotFoundError(f"Missing nsys capture: {report}")
    reports = ("nvtx_pushpop_sum", "nvtx_gpu_proj_sum", "nvtx_kern_sum")
    with (directory / "statistics.log").open("w") as log:
        subprocess.run(
            [
                "nsys",
                "stats",
                "--force-export=true",
                "--force-overwrite=true",
                "--report=" + ",".join(reports),
                "--format=csv",
                f"--output={directory / 'statistics'}",
                str(report),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    rows = {name: read_csv(directory / f"statistics_{name}.csv") for name in reports}
    operations = {}
    for row in rows["nvtx_pushpop_sum"]:
        name = region_name(row, "Range")
        if name:
            operations[name] = {
                "name": name,
                "calls": int(number(row, "Instances")),
                "host_ns": number(row, "Total Time"),
                "gpu_span_ns": 0,
                "gpu_operations": 0,
                "kernel_ns": 0,
                "kernel_calls": 0,
            }
    for row in rows["nvtx_gpu_proj_sum"]:
        name = region_name(row, "Range")
        if name in operations:
            operations[name]["gpu_span_ns"] += number(row, "Total Proj Time")
            operations[name]["gpu_operations"] += int(number(row, "Total GPU Ops"))
    kernels = []
    for row in rows["nvtx_kern_sum"]:
        name = region_name(row, "NVTX Range")
        if name not in operations:
            continue
        duration = number(row, "Total Time")
        calls = int(number(row, "Kern Inst"))
        operations[name]["kernel_ns"] += duration
        operations[name]["kernel_calls"] += calls
        kernels.append(
            {
                "operation": name,
                "kernel": row["Kernel Name"],
                "calls": calls,
                "total_ns": duration,
            }
        )
    if not operations or not kernels:
        raise ValueError(
            "nsys captured no TGV regions or no kernels within TGV regions"
        )
    if any(operation["host_ns"] <= 0 for operation in operations.values()):
        raise ValueError("nsys captured an empty TGV host region")
    cases = sorted({"/".join(name.split("/")[:3]) for name in operations})
    if expected_cases is not None:
        if set(cases) != set(expected_cases):
            raise ValueError(f"Unexpected profiled cases: {cases}")
        for case, iterations in expected_cases.items():
            for suffix, count in (("setup", 1), ("iteration", iterations)):
                operation = operations.get(f"{case}/{suffix}")
                if not operation or operation["calls"] != count:
                    raise ValueError(
                        f"Incorrect profiling range count for {case}/{suffix}"
                    )
                if operation["gpu_span_ns"] <= 0 or operation["kernel_calls"] <= 0:
                    raise ValueError(f"No GPU timings for {case}/{suffix}")
    # Kokkos' innermost NVTX ranges provide readable kernel labels. Restrict each
    # report to its case so identical kernel labels from different strategies
    # cannot be merged. Keep full symbols in the operation/kernel CSV above.
    case_kernels = []
    with (directory / "statistics.log").open("a") as log:
        for index, case in enumerate(cases):
            prefix = directory / f"case-{index + 1}"
            subprocess.run(
                [
                    "nsys",
                    "stats",
                    "--force-overwrite=true",
                    "--report=cuda_gpu_kern_sum:nvtx-name:base",
                    "--format=csv",
                    f"--filter-nvtx={case}",
                    f"--output={prefix}",
                    str(report),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
            # Nsight Systems includes report arguments such as ``nvtx-name:base``
            # in the CSV filename on some versions. Match the report stem instead
            # of assuming that only the report's base name is used.
            kernel_reports = sorted(
                directory.glob(prefix.name + "_cuda_gpu_kern_sum*.csv")
            )
            if len(kernel_reports) != 1:
                raise FileNotFoundError(
                    f"Expected one case kernel report for {case}, found "
                    f"{[path.name for path in kernel_reports]}; see "
                    f"{directory / 'statistics.log'}"
                )
            for row in read_csv(kernel_reports[0]):
                case_kernels.append(
                    {
                        "case": case,
                        "kernel": row["Name"],
                        "calls": int(number(row, "Instances")),
                        "total_ns": number(row, "Total Time"),
                    }
                )
            if not any(kernel["case"] == case for kernel in case_kernels):
                raise ValueError(f"No labeled CUDA kernels for {case}")
    data = {
        "operations": list(operations.values()),
        "kernels": kernels,
        "case_kernels": case_kernels,
    }
    (directory / "timings.json").write_text(json.dumps(data, indent=2) + "\n")
    lines = [
        "## Nsight Systems operation timings",
        "",
        "Host time is the NVTX range duration. GPU span runs from the first to the last "
        "GPU operation launched within the range, including gaps. Kernel time sums actual "
        "kernel execution durations. All three are inclusive: nested rows overlap and "
        "must not be added. No profiling fences are inserted.",
        "",
        "Equation regions measure their differential operators. Shared flux and state "
        "assembly kernels remain fused and appear in their enclosing/shared regions.",
        "",
        "The artifact contains the full `.nsys-rep` timeline, all CSV statistics, "
        "and `nsys/timings.json` with every operation/kernel pair. "
        "Tables below show every TGV operation and the 20 most expensive kernels per case.",
    ]
    for case in cases:
        lines += [
            "",
            f"<details><summary>{markdown(case.removeprefix('TaylorGreen/'))}</summary>",
            "",
            "| Operation | Calls | Host (ms) | GPU span (ms) | Kernels (ms) | Launches |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        selected = sorted(
            (
                value
                for name, value in operations.items()
                if name.startswith(case + "/")
            ),
            key=lambda value: value["name"],
        )
        for value in selected:
            operation = value["name"][len(case) + 1 :]
            lines.append(
                f"| {markdown(operation)} | {value['calls']} | "
                f"{value['host_ns'] / 1e6:.3f} | {value['gpu_span_ns'] / 1e6:.3f} | "
                f"{value['kernel_ns'] / 1e6:.3f} | {value['kernel_calls']} |"
            )
        # Case-filtered summaries count kernels once, unlike nested operation rows.
        selected_kernels = sorted(
            (value for value in case_kernels if value["case"] == case),
            key=lambda value: value["total_ns"],
            reverse=True,
        )
        lines += [
            "",
            "| CUDA kernel | Launches | Total (ms) | Mean (µs) |",
            "| --- | ---: | ---: | ---: |",
        ]
        for kernel in selected_kernels[:20]:
            label = kernel["kernel"].replace(case + "/", "")
            if len(label) > 180:
                label = label[:177] + "..."
            lines.append(
                f"| {markdown(label)} | {kernel['calls']} | "
                f"{kernel['total_ns'] / 1e6:.3f} | "
                f"{kernel['total_ns'] / kernel['calls'] / 1e3:.3f} |"
            )
        lines += ["", "</details>"]
    text = "\n".join(lines) + "\n"
    (directory / "report.md").write_text(text)
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directory", type=Path, help="Directory containing tgv.nsys-rep"
    )
    args = parser.parse_args()
    summarize_profile(args.directory.resolve())
