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


def profiling_environment():
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
    return environment


def profile_command(command, directory):
    directory.mkdir(parents=True, exist_ok=True)
    return [
        "nsys",
        "profile",
        "--trace=cuda,nvtx",
        "--sample=none",
        "--cpuctxsw=none",
        "--force-overwrite=true",
        f"--output={directory / 'tgv'}",
        *map(str, command),
    ], profiling_environment()


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


def read_statistics(prefix, report):
    # Report arguments are included in filenames by recent Nsight versions.
    paths = sorted(
        prefix.parent.glob(prefix.name + "_" + report.split(":")[0] + "*.csv")
    )
    if len(paths) != 1:
        raise FileNotFoundError(
            f"Expected one {report} CSV for {prefix}, found "
            f"{[path.name for path in paths]}; see {prefix.parent / 'statistics.log'}"
        )
    return read_csv(paths[0])


def summarize_regions(rows):
    regions = {}
    for row in rows["nvtx_pushpop_sum"]:
        name = row["Range"].removeprefix(":")
        region = regions.setdefault(
            name,
            dict(
                name=name,
                calls=0,
                host_ns=0,
                gpu_span_ns=0,
                kernel_ns=0,
                kernel_calls=0,
            ),
        )
        region["calls"] += int(number(row, "Instances"))
        region["host_ns"] += number(row, "Total Time")
    for row in rows["nvtx_gpu_proj_sum"]:
        name = row["Range"].removeprefix(":")
        if name in regions:
            regions[name]["gpu_span_ns"] += number(row, "Total Proj Time")
    for row in rows["nvtx_kern_sum:mangled"]:
        name = row["NVTX Range"].removeprefix(":")
        if name in regions:
            region = regions[name]
            region["kernel_ns"] += number(row, "Total Time")
            region["kernel_calls"] += int(number(row, "Kern Inst"))
    return list(regions.values())


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
    # Collect every region within each case, including Kokkos kernel labels.
    # Mangled symbols preserve template specializations for the Compute join.
    case_kernels = []
    case_regions = []
    case_reports = (
        "nvtx_pushpop_sum",
        "nvtx_gpu_proj_sum",
        "nvtx_kern_sum:mangled",
        "cuda_gpu_kern_sum:mangled",
    )
    # Reuse the first export directly; checking .nsys-rep/SQLite timestamps on
    # every stats call can otherwise reject an export as older than its capture.
    database = report.with_suffix(".sqlite")
    with (directory / "statistics.log").open("a") as log:
        for index, case in enumerate(cases):
            prefix = directory / f"case-{index + 1}"
            subprocess.run(
                [
                    "nsys",
                    "stats",
                    "--force-overwrite=true",
                    "--report=" + ",".join(case_reports),
                    "--format=csv",
                    f"--filter-nvtx={case}",
                    f"--output={prefix}",
                    str(database),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
            case_rows = {name: read_statistics(prefix, name) for name in case_reports}
            case_regions.extend(
                dict(region, case=case) for region in summarize_regions(case_rows)
            )
            labels = {}
            for row in case_rows["nvtx_kern_sum:mangled"]:
                name = row["NVTX Range"].removeprefix(":")
                if not name.startswith(case):
                    labels.setdefault(row["Kernel Name"], set()).add(name)
            for row in case_rows["cuda_gpu_kern_sum:mangled"]:
                case_kernels.append(
                    {
                        "case": case,
                        "kernel": row["Name"],
                        "calls": int(number(row, "Instances")),
                        "total_ns": number(row, "Total Time"),
                        "labels": sorted(labels.get(row["Name"], ())),
                    }
                )
            if not any(kernel["case"] == case for kernel in case_kernels):
                raise ValueError(f"No labeled CUDA kernels for {case}")
    data = {
        "operations": list(operations.values()),
        "kernels": kernels,
        "case_kernels": case_kernels,
        "case_regions": case_regions,
    }
    (directory / "timings.json").write_text(json.dumps(data, indent=2) + "\n")
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directory", type=Path, help="Directory containing tgv.nsys-rep"
    )
    args = parser.parse_args()
    summarize_profile(args.directory.resolve())
