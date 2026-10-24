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
import sqlite3
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


def iteration_case(name):
    parts = name.split("/")
    if len(parts) >= 4 and parts[0] == "TaylorGreen" and parts[3] == "iteration":
        return "/".join(parts[:3])
    return None


def covered_duration(intervals):
    end = None
    duration = 0
    for start, stop in sorted(intervals):
        duration += max(0, stop - max(start, end if end is not None else start))
        end = max(stop, end if end is not None else stop)
    return duration


def iteration_profiles(database):
    """Correlate GPU executions with the host NVTX stack at their launch.

    Region totals include nested scopes. Others uses the union of SimiLie GPU
    intervals, so overlapping kernels never produce a negative remainder.
    """
    with sqlite3.connect(database) as connection:
        connection.row_factory = sqlite3.Row
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        ranges = {}
        threads = {}
        cases = {}
        first_iterations = {}
        ordinals = {}
        launches = {}
        for row in connection.execute("""
            SELECT n.rowid AS id, n.start, n.end, n.globalTid AS tid,
                   coalesce(s.value, n.text, '') AS name
            FROM NVTX_EVENTS n LEFT JOIN StringIds s ON s.id=n.textId
            WHERE n.eventType IN (59,70) AND n.end IS NOT NULL
              AND coalesce(n.domainId,0)=0
            ORDER BY n.globalTid,n.start,n.end DESC
        """):
            scope = dict(row)
            ranges[scope["id"]] = scope
            state = threads.setdefault(scope["tid"], dict(ranges=[], stack=[]))
            stack = state["stack"]
            while stack and stack[-1]["end"] <= scope["start"]:
                stack.pop()
            case = iteration_case(scope["name"])
            if case and scope["name"] == case + "/iteration":
                scope["iteration"] = scope["id"]
                first_iterations.setdefault(case, scope["id"])
            elif stack:
                scope["iteration"] = stack[-1].get("iteration")
            if scope.get("iteration"):
                root = ranges[scope["iteration"]]
                scope["case"] = iteration_case(root["name"])
                profile = cases.setdefault(
                    scope["case"],
                    dict(
                        regions={},
                        sample_ranges=set(),
                        targets=set(),
                    ),
                )
                if case:
                    scope["devices"] = {}
                    region = profile["regions"].setdefault(
                        scope["name"],
                        dict(
                            name=scope["name"],
                            calls=0,
                            host_ns=0,
                            gpu_ns=0,
                            others_ns=0,
                            others_host_ns=0,
                            kernels={},
                        ),
                    )
                    region["calls"] += 1
                    region["host_ns"] += scope["end"] - scope["start"]
                if scope["name"].startswith("similie_"):
                    root_id = scope["iteration"]
                    ordinals[root_id] = ordinals.get(root_id, 0) + 1
                    scope["ordinal"] = ordinals[root_id]
            stack.append(scope)
            state["ranges"].append(scope)
        api_tables = [
            name
            for name in ("CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_DRIVER")
            if name in tables
        ]
        api_query = " UNION ALL ".join(
            f"SELECT start,globalTid,correlationId FROM {name}" for name in api_tables
        )
        gpu_queries = []
        for table in (
            "CUPTI_ACTIVITY_KIND_KERNEL",
            "CUPTI_ACTIVITY_KIND_MEMCPY",
            "CUPTI_ACTIVITY_KIND_MEMSET",
        ):
            if table not in tables:
                continue
            symbol = "coalesce(m.value,d.value)" if table.endswith("KERNEL") else "NULL"
            names = (
                "LEFT JOIN StringIds m ON m.id=g.mangledName LEFT JOIN StringIds d ON d.id=g.demangledName"
                if table.endswith("KERNEL")
                else ""
            )
            gpu_queries.append(f"""
                SELECT a.start AS launch, a.globalTid AS tid, g.start, g.end,
                       g.deviceId AS device, {symbol} AS symbol
                FROM {table} g JOIN apis a ON a.correlationId=g.correlationId
                  AND (a.globalTid & -16777216)=g.globalPid AND a.rank=1
                {names}
            """)
        if not api_query or not gpu_queries:
            raise ValueError("Systems export has no CUDA launch correlation data")
        query = (
            "WITH apis AS (SELECT *,row_number() OVER (PARTITION BY correlationId,(globalTid & -16777216) ORDER BY start DESC) AS rank FROM ("
            + api_query
            + ")) "
            + " UNION ALL ".join(gpu_queries)
            + " ORDER BY tid,launch"
        )
        for state in threads.values():
            state.update(position=0, active=[])
        for activity in connection.execute(query):
            state = threads.get(activity["tid"])
            if state is None:
                continue
            scopes = state["ranges"]
            active = state["active"]
            while (
                state["position"] < len(scopes)
                and scopes[state["position"]]["start"] <= activity["launch"]
            ):
                scope = scopes[state["position"]]
                while active and active[-1]["end"] <= scope["start"]:
                    active.pop()
                active.append(scope)
                state["position"] += 1
            while active and active[-1]["end"] <= activity["launch"]:
                active.pop()
            if not active or not active[-1].get("case"):
                continue
            profile = cases[active[-1]["case"]]
            # Mirror Compute's top-of-stack label filter. A library launch
            # nested in a SimiLie scope must still contribute to Others.
            label = active[-1] if active[-1]["name"].startswith("similie_") else None
            sample = False
            for scope in active:
                if "devices" not in scope:
                    continue
                device = scope["devices"].setdefault(
                    activity["device"],
                    dict(start=activity["start"], end=activity["end"], similie=[]),
                )
                device["start"] = min(device["start"], activity["start"])
                device["end"] = max(device["end"], activity["end"])
                if label is None or activity["symbol"] is None:
                    continue
                device["similie"].append((activity["start"], activity["end"]))
                region = profile["regions"][scope["name"]]
                key = (label["name"], activity["symbol"])
                kernel = region["kernels"].setdefault(
                    key, dict(label=key[0], symbol=key[1], calls=0, total_ns=0)
                )
                kernel["calls"] += 1
                kernel["total_ns"] += activity["end"] - activity["start"]
                if label["iteration"] == first_iterations[label["case"]]:
                    target = (scope["name"], *key)
                    if target not in profile["targets"]:
                        profile["sample_ranges"].add(label["ordinal"])
                        profile["targets"].add(target)
                        sample = True
            if (
                label is not None
                and activity["symbol"] is not None
                and label["iteration"] == first_iterations[label["case"]]
            ):
                launches.setdefault(label["case"], []).append(
                    (label["ordinal"], activity["symbol"], sample)
                )
        for scope in ranges.values():
            if "devices" not in scope:
                continue
            region = cases[scope["case"]]["regions"][scope["name"]]
            host_intervals = []
            for device in scope["devices"].values():
                duration = device["end"] - device["start"]
                region["gpu_ns"] += duration
                region["others_ns"] += duration - covered_duration(device["similie"])
                host_intervals.extend(
                    (max(start, scope["start"]), min(end, scope["end"]))
                    for start, end in device["similie"]
                    if end > scope["start"] and start < scope["end"]
                )
            region["others_host_ns"] += (
                scope["end"] - scope["start"] - covered_duration(host_intervals)
            )
        for case, profile in cases.items():
            invocations = {}
            samples = {}
            # Compute numbers each symbol's invocations after NVTX/range
            # filtering. A selected NVTX instance may contain repeated CUDA
            # launches; retain only the launch that first supplied a target.
            for ordinal, symbol, sample in launches.get(case, []):
                if ordinal not in profile["sample_ranges"]:
                    continue
                invocations[symbol] = invocations.get(symbol, 0) + 1
                if sample:
                    samples.setdefault(symbol, []).append(invocations[symbol])
            groups = {}
            # NCU accepts one invocation pattern per run. Group symbols with
            # the same indices to minimize the number of application launches.
            for symbol, indices in samples.items():
                groups.setdefault(tuple(indices), []).append(symbol)
            profile["sample_groups"] = [
                dict(symbols=sorted(symbols), invocations=list(indices))
                for indices, symbols in sorted(groups.items())
            ]
            profile["regions"] = sorted(
                profile["regions"].values(), key=lambda region: region["name"]
            )
            profile["sample_ranges"] = sorted(profile["sample_ranges"])
            profile["targets"] = sorted(profile["targets"])
            for region in profile["regions"]:
                region["kernels"] = sorted(
                    region["kernels"].values(),
                    key=lambda kernel: kernel["total_ns"],
                    reverse=True,
                )
        return cases


def markdown(value):
    return (
        html.escape(str(value), quote=False)
        .replace("|", "&#124;")
        .replace("\n", "<br>")
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
    data = dict(cases=iteration_profiles(report.with_suffix(".sqlite")))
    (directory / "timings.json").write_text(json.dumps(data, indent=2) + "\n")
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directory", type=Path, help="Directory containing tgv.nsys-rep"
    )
    args = parser.parse_args()
    summarize_profile(args.directory.resolve())
