# SPDX-FileCopyrightText: 2026 Baptiste Legouix
# SPDX-License-Identifier: AGPL-3.0-or-later
# AI-GENERATED
"""Check region-specific sampling, timing attribution and compact reports."""

import copy
import csv
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import textwrap
from types import SimpleNamespace
import unittest

sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "benchmarks/taylor_green_vortex")
)
from nsight_report import (
    METRICS,
    case_specs,
    combined_rows,
    compute_command,
    limiting_metrics,
    read_compute_report,
    write_combined_report,
)
from nsys_report import iteration_profiles

CASE = "TaylorGreen/N=64/default"
SYMBOL = "_Zprobe_full_symbol"


def capture_fixture(path):
    tid = (1 << 24) + 7
    with sqlite3.connect(path) as connection:
        connection.executescript("""
            CREATE TABLE StringIds(id INTEGER PRIMARY KEY,value TEXT);
            CREATE TABLE NVTX_EVENTS(start INTEGER,end INTEGER,eventType INTEGER,text TEXT,globalTid INTEGER,textId INTEGER,domainId INTEGER);
            CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INTEGER,end INTEGER,globalTid INTEGER,correlationId INTEGER);
            CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER,end INTEGER,deviceId INTEGER,correlationId INTEGER,globalPid INTEGER,mangledName INTEGER,demangledName INTEGER);
        """)
        connection.executemany(
            "INSERT INTO StringIds VALUES(?,?)",
            [(1, SYMBOL), (2, "void Kokkos::generic_kernel()")],
        )
        correlation = 0

        def region(start, end, name):
            connection.execute(
                "INSERT INTO NVTX_EVENTS VALUES(?,?,59,?,?,NULL,0)",
                (start, end, name, tid),
            )

        def launch(host, start, end):
            nonlocal correlation
            correlation += 1
            connection.execute(
                "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(?,?,?,?)",
                (host, host + 1, tid, correlation),
            )
            connection.execute(
                "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(?,?,0,?,?,1,2)",
                (start, end, correlation, 1 << 24),
            )

        region(0, 90, CASE + "/setup")
        region(10, 40, "similie_setup")
        launch(20, 50, 60)
        for base, gpu in ((100, 1000), (500, 2000)):
            region(base, base + 300, CASE + "/iteration")
            region(base + 10, base + 100, CASE + "/iteration/A")
            region(base + 20, base + 40, "similie_probe")
            launch(base + 25, gpu, gpu + 10)
            if base == 100:
                region(base + 50, base + 70, "similie_probe")
                launch(base + 55, gpu + 5, gpu + 15)
                region(base + 80, base + 90, "library_probe")
                launch(base + 85, gpu + 15, gpu + 25)
            region(base + 110, base + 200, CASE + "/iteration/B")
            region(base + 120, base + 140, "similie_probe")
            launch(
                base + 125,
                gpu + 30 if base == 100 else gpu + 20,
                gpu + 40 if base == 100 else gpu + 30,
            )
    return iteration_profiles(path)


class IterationAttribution(unittest.TestCase):
    def test_library_and_unlabelled_kernels_join_compute_metrics(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "capture.sqlite"
            capture_fixture(path)
            with sqlite3.connect(path) as connection:
                connection.executescript("""
                    INSERT INTO StringIds VALUES(3,'generic_kernel');
                    ALTER TABLE CUPTI_ACTIVITY_KIND_KERNEL ADD COLUMN shortName INTEGER;
                    UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET shortName=3;
                """)
                connection.execute(
                    "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(175,176,?,100)",
                    ((1 << 24) + 7,),
                )
                connection.execute(
                    "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(1025,1030,0,100,?,1,2,3)",
                    (1 << 24,),
                )
            profile = iteration_profiles(path)[CASE]

        def action(stack):
            domain = SimpleNamespace(
                name=lambda: "<default domain>", push_pop_ranges=lambda: stack
            )
            state = SimpleNamespace(domains=lambda: (0,), domain_by_id=lambda _: domain)
            values = {
                METRICS[key]: value
                for key, value in (("L1/TEX (%)", 11), ("L2 (%)", 22), ("DRAM (%)", 33))
            }

            def metric(name):
                if name not in values:
                    return None
                return SimpleNamespace(
                    has_value=lambda: True, as_double=lambda: values[name]
                )

            return SimpleNamespace(
                name=lambda _: SYMBOL,
                nvtx_state=lambda: state,
                metric_by_name=metric,
                rule_results=lambda: (),
            )

        actions = [
            action(stack)
            for stack in (
                (CASE + "/setup", "library_probe"),
                (CASE + "/iteration", CASE + "/iteration/A"),
                (CASE + "/iteration", CASE + "/iteration/A", "library_probe"),
            )
        ]
        section = SimpleNamespace(
            num_actions=lambda: len(actions), action_by_idx=lambda i: actions[i]
        )
        report = SimpleNamespace(num_ranges=lambda: 1, range_by_idx=lambda _: section)
        api = SimpleNamespace(
            load_report=lambda _: report, IAction=SimpleNamespace(NameBase_MANGLED=1)
        )
        samples = read_compute_report(Path("unused"), api, expected_launches=2)
        self.assertEqual(len(samples), 4)
        rows = list(combined_rows(CASE, profile, samples))
        direct = next(
            row
            for row in rows
            if row["kind"] == "kernel"
            and row["label"] == ""
            and row["region"].endswith("/A")
        )
        library = next(
            row
            for row in rows
            if row["kind"] == "kernel"
            and row["label"] == "library_probe"
            and row["region"].endswith("/A")
        )
        self.assertEqual(direct["display_label"], "generic_kernel")
        self.assertEqual(direct["calls"], 1)
        self.assertEqual(direct["total_ns"], 5)
        self.assertEqual(library["display_label"], "generic_kernel (library_probe)")
        for row in (direct, library):
            self.assertEqual(
                [row["metrics"][key] for key in ("L1/TEX (%)", "L2 (%)", "DRAM (%)")],
                [11, 22, 33],
            )
        self.assertEqual(
            profile["sample_groups"], [dict(symbols=[SYMBOL], invocations=[1, 3, 4, 5])]
        )

    def test_repeated_launches_in_one_nvtx_range_are_sampled_only_once(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "capture.sqlite"
            capture_fixture(path)
            with sqlite3.connect(path) as connection:
                connection.execute("INSERT INTO StringIds VALUES(3,'_Zother_kernel')")
                for correlation, host, symbol in ((100, 126, 1), (101, 127, 3)):
                    connection.execute(
                        "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(?,?,?,?)",
                        (host, host + 1, (1 << 24) + 7, correlation),
                    )
                    connection.execute(
                        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(1002,1008,0,?,?,?,2)",
                        (correlation, 1 << 24, symbol),
                    )
            profile = iteration_profiles(path)[CASE]
            self.assertEqual(profile["sample_ranges"], [1])
            groups = {
                tuple(group["invocations"]): group["symbols"]
                for group in profile["sample_groups"]
            }
            # Every first-iteration call advances the symbol's counter.
            # Repeats stay unselected; the library and sibling B retain
            # their first samples. Setup never advances these counters.
            self.assertEqual(groups, {(1,): ["_Zother_kernel"], (1, 4, 5): [SYMBOL]})
            group = dict(symbols=[SYMBOL], invocations=[1, 4, 5])
            command = compute_command(
                Path("binary"),
                Path(temporary),
                CASE,
                64,
                (),
                "incompressible",
                "ideal",
                "ehd",
                profile["sample_ranges"],
                group,
            )
            self.assertIn(f"--kernel-id=::regex:^({SYMBOL})$:^(1|4|5)$", command)
            self.assertIn("--range-filter=::^(1)$", command)
            self.assertIn(
                "--nvtx-include=regex:^TaylorGreen\\/N=64\\/default\\/iteration$/",
                command,
            )
            self.assertIn("--launch-count=3", command)
            self.assertIn("--kill=yes", command)
            self.assertIn(
                "--metrics="
                + ",".join(
                    METRICS[key] for key in ("L1/TEX (%)", "L2 (%)", "DRAM (%)")
                ),
                command,
            )

    def test_library_launch_nested_in_similie_scope_is_reported_separately(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "capture.sqlite"
            capture_fixture(path)
            with sqlite3.connect(path) as connection:
                connection.execute(
                    "INSERT INTO NVTX_EVENTS VALUES(175,195,59,'similie_outer',?,NULL,0)",
                    ((1 << 24) + 7,),
                )
            profile = iteration_profiles(path)[CASE]
            region = next(
                row for row in profile["regions"] if row["name"].endswith("/A")
            )
            self.assertEqual(region["kernels"][0]["calls"], 3)
            self.assertEqual(len(region["kernels"]), 2)
            self.assertEqual(region["kernels"][1]["label"], "library_probe")
            self.assertEqual(region["others_ns"], 0)
            self.assertEqual(profile["sample_ranges"], [1])

    def test_same_label_keeps_different_kernel_specializations(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "capture.sqlite"
            capture_fixture(path)
            with sqlite3.connect(path) as connection:
                connection.execute(
                    "INSERT INTO StringIds VALUES(3,'_Zother_specialization')"
                )
                connection.execute(
                    "UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET mangledName=3 WHERE correlationId=3"
                )
            profile = iteration_profiles(path)[CASE]
            region = next(
                row for row in profile["regions"] if row["name"].endswith("/A")
            )
            self.assertEqual(len(region["kernels"]), 3)
            self.assertEqual(profile["sample_ranges"], [1])

    def test_setup_library_calls_and_overlap_do_not_corrupt_region_timings(self):
        with tempfile.TemporaryDirectory() as temporary:
            profiles = capture_fixture(Path(temporary) / "capture.sqlite")
        profile = profiles[CASE]
        self.assertEqual(profile["sample_ranges"], [1])
        self.assertEqual(len(profile["targets"]), 5)
        regions = {region["name"]: region for region in profile["regions"]}
        self.assertEqual(
            set(regions),
            {CASE + "/iteration", CASE + "/iteration/A", CASE + "/iteration/B"},
        )
        root = regions[CASE + "/iteration"]
        self.assertEqual(root["calls"], 2)
        self.assertEqual(root["host_ns"], 600)
        self.assertEqual(root["gpu_ns"], 70)
        self.assertEqual(root["others_ns"], 15)
        self.assertEqual(root["others_host_ns"], 600)
        self.assertEqual(root["kernels"][0]["calls"], 5)
        a = regions[CASE + "/iteration/A"]
        self.assertEqual(a["others_ns"], 0)
        self.assertEqual(a["gpu_ns"], 35)
        self.assertEqual(
            a["kernels"],
            [
                dict(
                    label="similie_probe",
                    symbol=SYMBOL,
                    name="void Kokkos::generic_kernel()",
                    calls=3,
                    total_ns=30,
                ),
                dict(
                    label="library_probe",
                    symbol=SYMBOL,
                    name="void Kokkos::generic_kernel()",
                    calls=1,
                    total_ns=10,
                ),
            ],
        )

    def test_same_symbol_keeps_distinct_compute_samples_in_sibling_regions(self):
        def action(region, value):
            domain = SimpleNamespace(
                name=lambda: "<default domain>",
                push_pop_ranges=lambda: (CASE + "/iteration", region, "similie_probe"),
            )
            state = SimpleNamespace(domains=lambda: (0,), domain_by_id=lambda _: domain)

            def metric(name):
                number = float("nan") if "registers" in name else value
                return SimpleNamespace(has_value=lambda: True, as_double=lambda: number)

            return SimpleNamespace(
                name=lambda _: SYMBOL,
                nvtx_state=lambda: state,
                metric_by_name=metric,
                rule_results=lambda: (),
            )

        actions = [
            action(CASE + "/iteration/A", 10),
            action(CASE + "/iteration/B", 90),
            action(CASE + "/iteration/A", 30),
        ]
        section = SimpleNamespace(
            num_actions=lambda: len(actions), action_by_idx=lambda index: actions[index]
        )
        report = SimpleNamespace(num_ranges=lambda: 1, range_by_idx=lambda _: section)
        api = SimpleNamespace(
            load_report=lambda _: report, IAction=SimpleNamespace(NameBase_MANGLED=1)
        )
        samples = read_compute_report(Path("unused"), api)
        with self.assertRaisesRegex(ValueError, "captured 3 launches; expected 2"):
            read_compute_report(Path("unused"), api, expected_launches=2)
        by_region = {sample["region"]: sample for sample in samples}
        self.assertEqual(len(by_region), 3)
        self.assertEqual(by_region[CASE + "/iteration"]["metrics"]["SM (%)"], 10)
        self.assertEqual(by_region[CASE + "/iteration/A"]["metrics"]["SM (%)"], 10)
        self.assertEqual(by_region[CASE + "/iteration/B"]["metrics"]["SM (%)"], 90)
        self.assertIsNone(
            by_region[CASE + "/iteration/A"]["metrics"]["Registers/thread"]
        )


class CombinedReport(unittest.TestCase):
    def test_only_near_peak_competing_throughputs_are_highlighted(self):
        self.assertEqual(limiting_metrics({}), {})
        self.assertEqual(limiting_metrics({"SM (%)": 40, "DRAM (%)": 70}), {})
        self.assertEqual(
            limiting_metrics(
                {"SM (%)": 35, "L1/TEX (%)": 96, "L2 (%)": 85, "DRAM (%)": 93}
            ),
            {"L1/TEX (%)": 31, "DRAM (%)": 31},
        )
        self.assertEqual(
            limiting_metrics({"SM (%)": 98, "DRAM (%)": 90}), {"SM (%)": 32}
        )
        self.assertEqual(
            limiting_metrics({"SM (%)": 90, "L2 (%)": 95, "DRAM (%)": None}),
            {"SM (%)": 32, "L2 (%)": 31},
        )
        self.assertEqual(
            limiting_metrics({"Achieved occupancy (%)": 20, "SM (%)": 84}), {}
        )

    def test_tables_use_systems_timings_and_region_specific_compute_metrics(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            template = capture_fixture(output / "capture.sqlite")[CASE]
            profiles = {}
            samples = {}
            for case, _, _, _ in case_specs():
                profile = copy.deepcopy(template)
                for region in profile["regions"]:
                    region["name"] = region["name"].replace(CASE, case)
                    region["kernels"].reverse()
                profiles[case] = profile
                samples[case] = [
                    dict(
                        region=region["name"],
                        label=kernel["label"],
                        symbol=kernel["symbol"],
                        metrics={
                            label: 90 if region["name"].endswith("/B") else 10
                            for label in METRICS
                        },
                        rules=[
                            dict(
                                name="Bottleneck",
                                section="SpeedOfLight",
                                title="Latency",
                                message="Long diagnostic detail that belongs only in artifacts.",
                            ),
                            dict(
                                name="Occupancy",
                                section="Occupancy",
                                title="Occupancy analysis",
                                message="The original Nsight rule explanation.",
                            ),
                        ],
                    )
                    for region in profile["regions"]
                    for kernel in region["kernels"]
                ]
            (output / "performance/nsys").mkdir(parents=True)
            (output / "ncu").mkdir()
            (output / "performance/nsys/timings.json").write_text(
                json.dumps(dict(cases=profiles))
            )
            (output / "ncu/results.json").write_text(
                json.dumps(dict(cases=samples, errors=[]))
            )
            self.assertEqual(write_combined_report(output), [])
            report = (output / "nsight/report.md").read_text()
            ansi = (output / "nsight/report.ansi").read_text()
            self.assertNotIn("\x1b", report)
            self.assertIn("**90**", report)
            self.assertIn("\x1b[1;31m90\x1b[0m", ansi)
            self.assertIn("\x1b[1;32m90\x1b[0m", ansi)
            self.assertNotIn("**90**", ansi)
            self.assertEqual(
                report.count(
                    "| Region | Kernel | GPU total (ms) | Calls | Mean kernel (µs) |"
                ),
                14,
            )
            self.assertNotIn("NVTX (ms)", report)
            self.assertEqual(report.count("|  | similie_probe |"), 14 * 3)
            self.assertEqual(report.count("|  | Others |"), 14 * 3)
            self.assertNotIn("similie_setup", report)
            self.assertEqual(
                report.count("|  | void Kokkos::generic_kernel() (library_probe) |"),
                14 * 2,
            )
            for column in ("L1/TEX SOL (%)", "L2 SOL (%)", "DRAM SOL (%)"):
                self.assertIn(column, report)
            self.assertNotIn("Mem SOL (%)", report)
            self.assertNotIn(SYMBOL, report)
            self.assertIn("Latency: Long diagnostic detail", report)
            self.assertIn(
                "Occupancy analysis: The original Nsight rule explanation.", report
            )
            self.assertIn("<br>", report)
            self.assertIn("|  |  |  |  |  |  |  |  |  |", report)
            records = json.loads((output / "nsight/report.json").read_text())["rows"]
            for region in profiles[CASE]["regions"]:
                durations = [
                    row["total_ns"]
                    for row in records
                    if row["case"] == CASE
                    and row["region"] == region["name"]
                    and row["kind"] == "kernel"
                ]
                self.assertEqual(durations, sorted(durations, reverse=True))
            kernels = {
                row["region"]: row
                for row in records
                if row["case"] == CASE
                and row["kind"] == "kernel"
                and row["label"] == "similie_probe"
            }
            self.assertEqual(kernels[CASE + "/iteration/A"]["calls"], 3)
            self.assertEqual(kernels[CASE + "/iteration/A"]["mean_ns"], 10)
            self.assertEqual(kernels[CASE + "/iteration/A"]["total_ns"], 30)
            self.assertEqual(kernels[CASE + "/iteration/A"]["metrics"]["SM (%)"], 10)
            self.assertEqual(kernels[CASE + "/iteration/B"]["metrics"]["SM (%)"], 90)
            self.assertIn(
                "Long diagnostic detail",
                kernels[CASE + "/iteration/A"]["rules"][0]["message"],
            )
            with (output / "nsight/report.csv").open() as source:
                self.assertEqual(len(list(csv.DictReader(source))), 14 * 11)

    def test_missing_compute_data_still_reports_systems_times(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            profiles = capture_fixture(output / "capture.sqlite")
            (output / "performance/nsys").mkdir(parents=True)
            (output / "performance/nsys/timings.json").write_text(
                json.dumps(dict(cases=profiles))
            )
            errors = write_combined_report(output)
            self.assertIn("Missing Nsight Compute diagnostics", errors)
            report = (output / "nsight/report.md").read_text()
            self.assertEqual(report.count("| Region | Kernel | GPU total (ms) |"), 14)
            self.assertEqual(report.count("No iteration data"), 13)
            self.assertIn("Not sampled", report)


class WorkflowPublication(unittest.TestCase):
    def test_exit_traps_publish_reports_without_hiding_step_failure(self):
        workflow = (
            Path(__file__).resolve().parents[2]
            / ".github/workflows/benchmark_taylor_green_vortex.yml"
        ).read_text()
        for name, phase in (
            ("publish_performance_report", "performance"),
            ("publish_nsight_report", "nsight"),
            ("publish_validation_report", "validation"),
        ):
            start = workflow.index(f"        {name}()")
            end = workflow.index(f"        trap {name} EXIT", start) + len(
                f"        trap {name} EXIT"
            )
            script = "set -euo pipefail\n" + textwrap.dedent(workflow[start:end])
            for status, report_exists, colored in (
                (0, True, False),
                (7, True, False),
                (7, False, False),
                (7, True, True),
            ):
                with (
                    self.subTest(name=name, status=status, report_exists=report_exists),
                    tempfile.TemporaryDirectory() as temporary,
                ):
                    root = Path(temporary)
                    summary = root / "summary.md"
                    report = root / "results" / phase / "report.md"
                    if report_exists:
                        report.parent.mkdir(parents=True)
                        report.write_text("Collected report\n")
                    colored_report = "\x1b[1;31m95\x1b[0m\n"
                    if report_exists and colored and phase == "nsight":
                        report.with_suffix(".ansi").write_text(colored_report)
                    result = subprocess.run(
                        ["bash", "-c", script + f"\nexit {status}\n"],
                        env=dict(
                            os.environ,
                            CI_ROOT=str(root),
                            GITHUB_STEP_SUMMARY=str(summary),
                        ),
                        capture_output=True,
                        text=True,
                    )
                    self.assertEqual(result.returncode, status, result.stderr)
                    if report_exists:
                        self.assertEqual(summary.read_text(), "Collected report\n")
                        self.assertEqual(
                            result.stdout,
                            colored_report
                            if colored and phase == "nsight"
                            else "Collected report\n",
                        )
                    else:
                        self.assertFalse(summary.exists())


if __name__ == "__main__":
    unittest.main()
