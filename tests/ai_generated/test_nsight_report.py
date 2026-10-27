# SPDX-FileCopyrightText: 2026 Baptiste Legouix
# SPDX-License-Identifier: AGPL-3.0-or-later
# AI-GENERATED
"""Check region-specific sampling, timing attribution and compact reports."""

import copy
import csv
import json
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
import tempfile
import textwrap
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "benchmarks/taylor_green_vortex")
)
from nsight_report import (
    CORE_METRICS,
    METRICS,
    case_specs,
    combined_rows,
    compute_command,
    compute_groups,
    hot_groups,
    hottest_kernels,
    limiting_metrics,
    metric_cells,
    merge_hot_samples,
    profile_cases,
    read_compute_report,
    write_combined_report,
)
from nsys_report import iteration_profiles

CASE = "TaylorGreen/N=64/default"
SYMBOL = "_Zprobe_full_symbol"


class ComputeSamplingPlan(unittest.TestCase):
    def test_exact_filters_cover_targets_without_extra_calls(self):
        plan = dict(
            sample_groups=[
                dict(symbols=[SYMBOL], invocations=[1, 4, 5]),
                dict(symbols=["_Zshort_kernel"], invocations=[1]),
            ],
            kernel_invocations={SYMBOL: 6, "_Zshort_kernel": 1},
        )
        groups = compute_groups(plan)
        self.assertEqual(len(groups), 2)
        group = groups[0]
        self.assertEqual(group["invocations"], [1, 4, 5])
        self.assertEqual(group["launches"], 3)
        captured = {
            (symbol, index)
            for group in groups
            for symbol in group["symbols"]
            for index in group["invocations"]
            if index <= plan["kernel_invocations"][symbol]
        }
        requested = {
            (symbol, index)
            for original in plan["sample_groups"]
            for symbol in original["symbols"]
            for index in original["invocations"]
        }
        self.assertEqual(requested, captured)
        command = compute_command(
            Path("binary"),
            Path("output"),
            CASE,
            64,
            (),
            "incompressible",
            "ideal",
            "hd",
            [1],
            groups[0],
        )
        self.assertIn("--launch-count=3", command)

    def test_large_groups_split_without_losing_or_adding_samples(self):
        for symbols, invocations in (
            ([f"_Zkernel_{index}" for index in range(71)], [1, 4, 7]),
            ([SYMBOL], list(range(1, 75))),
        ):
            with self.subTest(symbols=len(symbols), invocations=len(invocations)):
                groups = compute_groups(
                    dict(sample_groups=[dict(symbols=symbols, invocations=invocations)])
                )
                captured = []
                for group in groups:
                    pairs = [
                        (symbol, index)
                        for symbol in group["symbols"]
                        for index in group["invocations"]
                    ]
                    self.assertEqual(group["launches"], len(pairs))
                    self.assertLessEqual(len(pairs), 32)
                    captured.extend(pairs)
                requested = {
                    (symbol, index) for symbol in symbols for index in invocations
                }
                self.assertEqual(set(captured), requested)
                self.assertEqual(len(captured), len(requested))

    def test_empty_plan_has_no_batches(self):
        self.assertEqual(compute_groups(dict(sample_groups=[])), [])

    def test_core_union_covers_samples_and_counts_only_existing_calls(self):
        plan = dict(
            sample_groups=[
                dict(symbols=[SYMBOL], invocations=[1, 4, 5]),
                dict(symbols=["_Zshort_kernel"], invocations=[1]),
            ],
            kernel_invocations={SYMBOL: 6, "_Zshort_kernel": 1},
        )
        groups = compute_groups(plan, combine=True)
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["launches"], 4)
        captured = {
            (symbol, index)
            for symbol in groups[0]["symbols"]
            for index in groups[0]["invocations"]
            if index <= plan["kernel_invocations"][symbol]
        }
        self.assertEqual(
            captured, {(SYMBOL, 1), (SYMBOL, 4), (SYMBOL, 5), ("_Zshort_kernel", 1)}
        )

    def test_failed_capture_stops_before_starting_another_case(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            directory = output / "performance/nsys"
            directory.mkdir(parents=True)
            plan = dict(
                sample_groups=[dict(symbols=[SYMBOL], invocations=[1])],
                kernel_invocations={SYMBOL: 1},
                sample_ranges=[1],
            )
            (directory / "timings.json").write_text(
                json.dumps(dict(cases={CASE: plan, "next": plan}))
            )
            with (
                patch("nsight_report.profiling_environment", return_value={}),
                patch(
                    "nsight_report.case_specs",
                    return_value=[
                        (CASE, "first", 64, ()),
                        ("next", "second", 64, ()),
                    ],
                ),
                patch(
                    "nsight_report.execute", side_effect=RuntimeError("capture failed")
                ) as execute,
                self.assertRaisesRegex(RuntimeError, "capture failed"),
            ):
                profile_cases(
                    Path("binary"),
                    output,
                    "incompressible",
                    "ideal",
                    "hd",
                    detail="core",
                )
            self.assertEqual(execute.call_count, 1)
            result = json.loads((output / "ncu/results.json").read_text())
            self.assertEqual(result["cases"], {CASE: []})
            self.assertEqual(result["errors"], [CASE + ": capture failed"])

    def test_invalid_batch_size_is_rejected(self):
        with self.assertRaises(ValueError):
            compute_groups(dict(sample_groups=[]), max_launches=0)

    def test_missing_launch_count_rejects_a_truncated_plan(self):
        with self.assertRaises(ValueError):
            compute_groups(
                dict(
                    sample_groups=[dict(symbols=[SYMBOL], invocations=[1, 4])],
                    kernel_invocations={SYMBOL: 3},
                )
            )

    def test_older_captures_keep_their_exact_groups(self):
        groups = [
            dict(symbols=[SYMBOL], invocations=[1, 4]),
            dict(symbols=["_Zshort_kernel"], invocations=[1]),
        ]
        self.assertEqual(
            compute_groups(dict(sample_groups=groups)),
            [dict(groups[0], launches=2), dict(groups[1], launches=1)],
        )


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
            region(base + 5, base + 205, CASE + "/iteration/solver")
            region(base + 10, base + 100, CASE + "/iteration/solver/A")
            region(base + 20, base + 40, "similie_probe")
            launch(base + 25, gpu, gpu + 10)
            if base == 100:
                region(base + 50, base + 70, "similie_probe")
                launch(base + 55, gpu + 5, gpu + 15)
                region(base + 80, base + 90, "library_probe")
                launch(base + 85, gpu + 15, gpu + 25)
            region(base + 110, base + 200, CASE + "/iteration/solver/B")
            region(base + 120, base + 140, "similie_probe")
            launch(
                base + 125,
                gpu + 30 if base == 100 else gpu + 20,
                gpu + 40 if base == 100 else gpu + 30,
            )
    return iteration_profiles(path)


class HotKernelSelection(unittest.TestCase):
    def plan(self):
        with tempfile.TemporaryDirectory() as temporary:
            return capture_fixture(Path(temporary) / "capture.sqlite")[CASE]

    def test_hot_ranking_uses_root_cumulative_time_without_counting_nested_ranges(self):
        plan = self.plan()
        root = next(
            region
            for region in plan["regions"]
            if region["name"] == CASE + "/iteration/solver"
        )
        library = next(
            kernel for kernel in root["kernels"] if kernel["label"] == "library_probe"
        )
        library["total_ns"] = 30
        for region in plan["regions"]:
            if region is not root:
                for kernel in region["kernels"]:
                    kernel["total_ns"] = 10**9
        selected = hottest_kernels(CASE, plan, 1)
        self.assertEqual(selected[0]["label"], "similie_probe")
        self.assertEqual(selected[0]["invocation"], 1)
        selected = hottest_kernels(CASE, plan, 20)
        self.assertEqual(len(selected), 2)
        self.assertEqual(selected[1]["label"], "library_probe")
        self.assertEqual(selected[1]["invocation"], 3)
        groups = hot_groups(plan, selected)
        self.assertEqual(
            groups, [dict(symbols=[SYMBOL], invocations=[1, 3], launches=2)]
        )
        self.assertEqual(hottest_kernels(CASE, plan, 0), [])

    def test_large_hot_filter_stays_bounded_and_keeps_every_requested_sample(self):
        selected = [
            dict(symbol=f"_Zkernel_{index}", invocation=index + 1)
            for index in range(20)
        ]
        plan = dict(kernel_invocations={kernel["symbol"]: 30 for kernel in selected})
        groups = hot_groups(plan, selected)
        self.assertTrue(all(group["launches"] <= 32 for group in groups))
        captured = {
            (symbol, index)
            for group in groups
            for symbol in group["symbols"]
            for index in group["invocations"]
        }
        self.assertTrue(
            {(kernel["symbol"], kernel["invocation"]) for kernel in selected}
            <= captured
        )

    def test_detailed_merge_preserves_core_counters_and_does_not_reuse_another_invocation(
        self,
    ):
        plan = self.plan()
        selected = hottest_kernels(CASE, plan, 2)
        samples = [
            dict(
                region=region,
                label=label,
                symbol=symbol,
                metrics=dict.fromkeys(CORE_METRICS, 42),
                rules=[],
                replay_passes=1,
            )
            for region, label, symbol in plan["targets"]
        ]
        detailed = [
            dict(row, metrics=dict.fromkeys(METRICS, 90), replay_passes=8)
            for row in samples
        ]
        merge_hot_samples(plan, samples, detailed, selected)
        indexed = {row["region"] + "/" + row["label"]: row for row in samples}
        hot = indexed[CASE + "/iteration/solver/similie_probe"]
        self.assertEqual(hot["metrics"]["SM (%)"], 90)
        self.assertEqual(hot["metrics"]["DRAM (%)"], 42)
        self.assertEqual(hot["replay_passes"], 1)
        self.assertEqual(hot["detailed_replay_passes"], 8)
        self.assertNotIn(
            "SM (%)", indexed[CASE + "/iteration/solver/B/similie_probe"]["metrics"]
        )
        with self.assertRaisesRegex(ValueError, "Missing detailed results"):
            merge_hot_samples(plan, samples, [], selected)

    def test_hybrid_collects_core_then_only_selected_hot_samples(self):
        plan = self.plan()
        base = [
            dict(
                region=region,
                label=label,
                symbol=symbol,
                metrics=dict.fromkeys(CORE_METRICS, 42),
                rules=[],
                replay_passes=1,
            )
            for region, label, symbol in plan["targets"]
        ]
        detailed = [
            dict(row, metrics=dict.fromkeys(METRICS, 90), replay_passes=8)
            for row in base
        ]
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / "performance/nsys").mkdir(parents=True)
            (output / "performance/nsys/timings.json").write_text(
                json.dumps(dict(cases={CASE: plan}))
            )
            with (
                patch("nsight_report.profiling_environment", return_value={}),
                patch(
                    "nsight_report.case_specs", return_value=[(CASE, "first", 64, ())]
                ),
                patch(
                    "nsight_report.capture_groups", side_effect=[base, detailed]
                ) as capture,
            ):
                profile_cases(
                    Path("binary"),
                    output,
                    "incompressible",
                    "ideal",
                    "hd",
                    hot_kernels=1,
                )
            self.assertEqual(
                [call.args[10] for call in capture.call_args_list], ["core", "full"]
            )
            result = json.loads((output / "ncu/results.json").read_text())
            self.assertEqual(result["detail"], "hybrid")
            self.assertEqual(len(result["hot_kernels"][CASE]), 1)
            self.assertEqual(result["errors"], [])

    def test_hybrid_keeps_n128_core_metrics_when_multipass_is_unsupported(self):
        case = CASE.replace("N=64", "N=128")
        plan = self.plan()

        def replace_case(value):
            if isinstance(value, str):
                return value.replace(CASE, case)
            if isinstance(value, list):
                return [replace_case(item) for item in value]
            if isinstance(value, dict):
                return {key: replace_case(item) for key, item in value.items()}
            return value

        plan = replace_case(plan)
        base = [
            dict(
                region=region,
                label=label,
                symbol=symbol,
                metrics=dict.fromkeys(CORE_METRICS, 42),
                rules=[],
                replay_passes=1,
            )
            for region, label, symbol in plan["targets"]
        ]
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / "performance/nsys").mkdir(parents=True)
            (output / "performance/nsys/timings.json").write_text(
                json.dumps(dict(cases={case: plan}))
            )
            with (
                patch("nsight_report.profiling_environment", return_value={}),
                patch(
                    "nsight_report.case_specs",
                    return_value=[(case, "only", 128, ())],
                ),
                patch("nsight_report.capture_groups", return_value=base) as capture,
            ):
                profile_cases(
                    Path("binary"),
                    output,
                    "incompressible",
                    "ideal",
                    "hd",
                    hot_kernels=1,
                )
            self.assertEqual(capture.call_count, 1)
            self.assertEqual(capture.call_args.args[10], "core")
            result = json.loads((output / "ncu/results.json").read_text())
            self.assertEqual(result["errors"], [])
            self.assertEqual(result["hot_kernels"], {})
            self.assertIn("device-memory snapshot", result["hot_kernel_skips"][case])


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
                (CASE + "/iteration", CASE + "/iteration/solver", CASE + "/iteration/solver/A"),
                (CASE + "/iteration", CASE + "/iteration/solver", CASE + "/iteration/solver/A", "library_probe"),
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
            self.assertEqual(
                profile["kernel_invocations"], {SYMBOL: 5, "_Zother_kernel": 1}
            )
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
                "--nvtx-include=regex:^TaylorGreen\\/N=64\\/default\\/iteration\\/solver$/",
                command,
            )
            self.assertIn("--launch-count=3", command)
            self.assertIn("--kill=yes", command)
            self.assertIn("--replay-mode=kernel", command)
            self.assertFalse(
                any(option.startswith("--app-replay-mode") for option in command)
            )
            self.assertIn("--cache-control=none", command)
            self.assertIn("--import-sass=no", command)
            self.assertIn(
                "--metrics=" + ",".join(CORE_METRICS.values()),
                command,
            )
            self.assertNotIn("--section=SpeedOfLight", command)
            self.assertIn("--section=LaunchStats", command)
            self.assertIn("--section=Occupancy", command)
            self.assertIn("--apply-rules=yes", command)
            full = compute_command(
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
                detail="full",
            )
            self.assertIn("--section=SpeedOfLight", full)
            self.assertIn("--section=Occupancy", full)
            self.assertIn("--cache-control=all", full)
            self.assertIn(
                "--metrics=" + ",".join({**METRICS, **CORE_METRICS}.values()), full
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
            {CASE + "/iteration", CASE + "/iteration/solver", CASE + "/iteration/solver/A", CASE + "/iteration/solver/B"},
        )
        root = regions[CASE + "/iteration"]
        self.assertEqual(root["calls"], 2)
        self.assertEqual(root["host_ns"], 600)
        self.assertEqual(root["gpu_ns"], 70)
        self.assertEqual(root["others_ns"], 15)
        self.assertEqual(root["others_host_ns"], 600)
        self.assertEqual(root["kernels"][0]["calls"], 5)
        a = regions[CASE + "/iteration/solver/A"]
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
                push_pop_ranges=lambda: (CASE + "/iteration", CASE + "/iteration/solver", region, "similie_probe"),
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
            action(CASE + "/iteration/solver/A", 10),
            action(CASE + "/iteration/solver/B", 90),
            action(CASE + "/iteration/solver/A", 30),
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
        self.assertEqual(by_region[CASE + "/iteration/solver"]["metrics"]["SM (%)"], 10)
        self.assertEqual(by_region[CASE + "/iteration/solver/A"]["metrics"]["SM (%)"], 10)
        self.assertEqual(by_region[CASE + "/iteration/solver/B"]["metrics"]["SM (%)"], 90)
        self.assertIsNone(
            by_region[CASE + "/iteration/solver/A"]["metrics"]["Registers/thread"]
        )


class CombinedReport(unittest.TestCase):
    def test_nvtx_ranges_are_sorted_by_gpu_duration(self):
        with tempfile.TemporaryDirectory() as temporary:
            plan = capture_fixture(Path(temporary) / "capture.sqlite")[CASE]
        for region, duration in zip(plan["regions"], (10, 100, 50)):
            region["gpu_ns"] = duration
        rows = list(combined_rows(CASE, plan, []))
        self.assertEqual(
            [row["total_ns"] for row in rows if row["kind"] == "region"], [100, 50, 10]
        )

    def test_summary_keeps_every_case_under_github_limit_when_detail_is_large(self):
        profiles, samples = {}, {}
        for case, _, _, _ in case_specs():
            region = case + "/iteration/solver"
            profiles[case] = dict(
                regions=[
                    dict(
                        name=region,
                        calls=1,
                        host_ns=100,
                        gpu_ns=100,
                        others_ns=0,
                        others_host_ns=0,
                        kernels=[
                            dict(
                                label="",
                                symbol=SYMBOL,
                                name="x" * 90000,
                                calls=1,
                                total_ns=100,
                            )
                        ],
                    )
                ]
            )
            samples[case] = [
                dict(
                    region=region,
                    label="",
                    symbol=SYMBOL,
                    metrics=dict.fromkeys({**METRICS, **CORE_METRICS}, 42),
                    rules=[],
                )
            ]
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / "performance/nsys").mkdir(parents=True)
            (output / "ncu").mkdir()
            (output / "performance/nsys/timings.json").write_text(
                json.dumps(dict(cases=profiles))
            )
            (output / "ncu/results.json").write_text(
                json.dumps(dict(detail="hybrid", cases=samples, errors=[]))
            )
            self.assertEqual(write_combined_report(output), [])
            self.assertGreater(
                (output / "nsight/report.md").stat().st_size, 1024 * 1024
            )
            summary = (output / "nsight/summary.md").read_text()
            self.assertLess(len(summary.encode()), 1024 * 1024)
            self.assertEqual(summary.count("\n## "), 14)
            self.assertIn("\n\n\n## ", summary)
            for case, _, _, _ in case_specs():
                self.assertIn(case.removeprefix("TaylorGreen/"), summary)

    def test_core_reader_requires_one_pass_and_complete_metrics(self):
        values = {name: 2.0 for name in CORE_METRICS.values()}
        values["profiler__replayer_passes"] = 1.0

        def metric(name):
            return (
                SimpleNamespace(has_value=lambda: True, as_double=lambda: values[name])
                if name in values
                else None
            )

        domain = SimpleNamespace(
            name=lambda: "<default domain>",
            push_pop_ranges=lambda: [CASE + "/iteration", CASE + "/iteration/solver"],
        )
        state = SimpleNamespace(domains=lambda: [0], domain_by_id=lambda _: domain)
        action = SimpleNamespace(
            name=lambda _: SYMBOL,
            nvtx_state=lambda: state,
            metric_by_name=metric,
            rule_results=lambda: [],
        )
        section = SimpleNamespace(num_actions=lambda: 1, action_by_idx=lambda _: action)
        report = SimpleNamespace(num_ranges=lambda: 1, range_by_idx=lambda _: section)
        api = SimpleNamespace(
            load_report=lambda _: report, IAction=SimpleNamespace(NameBase_MANGLED=1)
        )
        rows = read_compute_report(Path("report"), api, 1, detail="core")
        self.assertEqual(rows[0]["replay_passes"], 1)
        self.assertEqual(set(rows[0]["metrics"]), set(CORE_METRICS))
        values["profiler__replayer_passes"] = 8
        with self.assertRaisesRegex(ValueError, "requires one pass"):
            read_compute_report(Path("report"), api, 1, detail="core")
        values["profiler__replayer_passes"] = 1
        del values[CORE_METRICS["SM active (%)"]]
        with self.assertRaisesRegex(ValueError, "Missing core metric"):
            read_compute_report(Path("report"), api, 1, detail="core")

    def test_core_report_labels_activity_and_normalized_cache_bandwidth(
        self,
    ):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            profiles = capture_fixture(output / "capture.sqlite")
            samples = [
                dict(
                    region=region["name"],
                    label=kernel["label"],
                    symbol=kernel["symbol"],
                    metrics={name: 42 for name in CORE_METRICS},
                    rules=[],
                )
                for region in profiles[CASE]["regions"]
                for kernel in region["kernels"]
            ]
            (output / "performance/nsys").mkdir(parents=True)
            (output / "ncu").mkdir()
            (output / "performance/nsys/timings.json").write_text(
                json.dumps(dict(cases=profiles))
            )
            (output / "ncu/results.json").write_text(
                json.dumps(dict(detail="core", cases={CASE: samples}, errors=[]))
            )
            self.assertEqual(write_combined_report(output), [])
            report = (output / "nsight/report.md").read_text()
            self.assertIn("SM active (%)", report)
            self.assertIn("L1/TEX BW (%)", report)
            self.assertNotIn("| SM SOL (%) |", report)
            self.assertIn("| 42 | 42 | 42 | 42 |", report)
            with (output / "nsight/report.csv").open() as source:
                row = next(
                    row for row in csv.DictReader(source) if row["kind"] == "kernel"
                )
            self.assertEqual(float(row["L1/TEX bandwidth (%)"]), 42)

    def test_saturated_units_and_native_priority_metrics_are_highlighted(self):
        self.assertEqual(limiting_metrics({}), {})
        self.assertEqual(limiting_metrics({"SM (%)": 40, "DRAM (%)": 70}), {})
        self.assertEqual(
            limiting_metrics(
                {"SM (%)": 35, "L1/TEX (%)": 96, "L2 (%)": 85, "DRAM (%)": 93}
            ),
            {"L1/TEX (%)": 31, "L2 (%)": 31, "DRAM (%)": 31},
        )
        self.assertEqual(
            limiting_metrics({"SM (%)": 98, "DRAM (%)": 90}),
            {"SM (%)": 31, "DRAM (%)": 31},
        )
        self.assertEqual(
            limiting_metrics({"SM (%)": 90, "L2 (%)": 95, "DRAM (%)": None}),
            {"SM (%)": 31, "L2 (%)": 31},
        )
        self.assertEqual(
            limiting_metrics({"Achieved occupancy (%)": 20, "SM (%)": 84}), {}
        )
        self.assertEqual(limiting_metrics({"SM active (%)": 100}), {})
        self.assertEqual(
            limiting_metrics(
                {"L1/TEX bandwidth (%)": 90, "L2 bandwidth (%)": 95, "DRAM (%)": 50}
            ),
            {"L1/TEX bandwidth (%)": 31, "L2 bandwidth (%)": 31},
        )
        self.assertEqual(
            limiting_metrics(
                {"Registers/thread": 96},
                [
                    dict(
                        focus_metrics=[
                            dict(name="launch__registers_per_thread", severity=2)
                        ]
                    )
                ],
            ),
            {"Registers/thread": 31},
        )

    def test_near_peak_compute_sol_is_green(self):
        row = {"metrics": {"SM (%)": 90}, "rules": []}
        cells, ansi = metric_cells(row, {"Compute SOL (%)": "SM (%)"})
        self.assertEqual(cells, ["**90**"])
        self.assertEqual(ansi, ["\x1b[1;32m90\x1b[0m"])

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
            self.assertEqual(kernels[CASE + "/iteration/solver/A"]["calls"], 3)
            self.assertEqual(kernels[CASE + "/iteration/solver/A"]["mean_ns"], 10)
            self.assertEqual(kernels[CASE + "/iteration/solver/A"]["total_ns"], 30)
            self.assertEqual(kernels[CASE + "/iteration/solver/A"]["metrics"]["SM (%)"], 10)
            self.assertEqual(kernels[CASE + "/iteration/solver/B"]["metrics"]["SM (%)"], 90)
            self.assertIn(
                "Long diagnostic detail",
                kernels[CASE + "/iteration/solver/A"]["rules"][0]["message"],
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
            self.assertEqual(report.count("No solver data"), 13)
            self.assertIn("Not sampled", report)


class WorkflowPublication(unittest.TestCase):
    def test_compact_summary_is_published_instead_of_large_detail(self):
        workflow = (
            Path(__file__).resolve().parents[2]
            / ".github/workflows/benchmark_taylor_green_vortex.yml"
        ).read_text()
        start = workflow.index("        publish_nsight_report()")
        end = workflow.index("        trap publish_nsight_report EXIT", start)
        script = textwrap.dedent(workflow[start:end]) + "\npublish_nsight_report\n"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "results/nsight"
            directory.mkdir(parents=True)
            (directory / "report.md").write_text("x" * (2 * 1024 * 1024))
            (directory / "summary.md").write_text("All fourteen cases\n")
            (directory / "summary.ansi").write_text("Colored compact summary\n")
            summary = root / "job-summary.md"
            result = subprocess.run(
                ["bash", "-c", script],
                env=dict(
                    os.environ, CI_ROOT=str(root), GITHUB_STEP_SUMMARY=str(summary)
                ),
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertEqual(summary.read_text(), "All fourteen cases\n")
            self.assertEqual(result.stdout, "Colored compact summary\n")

    def test_cancellation_stops_the_compute_container(self):
        workflow = (
            Path(__file__).resolve().parents[2]
            / ".github/workflows/benchmark_taylor_green_vortex.yml"
        ).read_text()
        start = workflow.index("    - name: Collect first kernel invocations")
        start = workflow.index("        set -euo pipefail", start)
        end = workflow.index("    - name:", start)
        script = textwrap.dedent(workflow[start:end])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            docker = root / "docker"
            docker.write_text(
                "#!/bin/bash\n"
                'if [ "$1" = run ]; then\n'
                '  echo $$ > "${CI_ROOT}/started"\n'
                "  trap '' TERM\n"
                "  while :; do sleep 1; done\n"
                "else\n"
                '  echo "$*" > "${CI_ROOT}/cleanup"\n'
                '  kill -KILL "$(cat "${CI_ROOT}/started")"\n'
                "fi\n"
            )
            docker.chmod(0o755)
            child = subprocess.Popen(
                ["bash", "-c", script],
                env=dict(
                    os.environ,
                    PATH=str(root) + os.pathsep + os.environ["PATH"],
                    CI_ROOT=str(root),
                    BUILD_ROOT=str(root),
                    TARGET_SHA="test",
                    GITHUB_RUN_ID="123",
                    GITHUB_RUN_ATTEMPT="2",
                    FLUID="incompressible",
                    TRANSPORT="ideal",
                    COUPLING="hd",
                ),
                start_new_session=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            try:
                deadline = time.monotonic() + 5
                while not (root / "started").exists():
                    if child.poll() is not None or time.monotonic() > deadline:
                        self.fail("Mock Compute container did not start")
                    time.sleep(0.01)
                os.killpg(child.pid, signal.SIGTERM)
                child.communicate(timeout=5)
                self.assertEqual(child.returncode, 143)
                self.assertEqual(
                    (root / "cleanup").read_text().strip(),
                    "rm -f similie-tgv-123-2-incompressible-ideal-hd-compute",
                )
            finally:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.communicate(timeout=5)

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
