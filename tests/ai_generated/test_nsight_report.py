# SPDX-FileCopyrightText: 2026 Baptiste Legouix
# SPDX-License-Identifier: AGPL-3.0-or-later
# AI-GENERATED
"""Check symbol joins, complete case tables and missing profiling data."""

import csv
import json
import os
from pathlib import Path
import sys
import subprocess
import tempfile
import textwrap
import unittest

sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "benchmarks/taylor_green_vortex")
)
from nsight_report import METRICS, case_specs, write_combined_report
from nsys_report import summarize_regions


class CombinedReport(unittest.TestCase):
    def test_full_symbols_keep_specializations_separate_and_preserve_every_row(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            cases = list(case_specs())
            self.assertEqual(len(cases), 14)
            symbols = [f"_Zkernel_specialization_{index}" for index in range(25)]
            regions = []
            kernels = []
            samples = {}
            for case, _, _, _ in cases:
                regions.append(
                    dict(
                        case=case,
                        name=case + "/iteration",
                        calls=5,
                        host_ns=2000,
                        gpu_span_ns=1500,
                        kernel_ns=1000,
                        kernel_calls=25,
                    )
                )
                # Kokkos ranges do not necessarily start with TaylorGreen/.
                regions.append(
                    dict(
                        case=case,
                        name="Kokkos|<kernel>",
                        calls=25,
                        host_ns=1000,
                        gpu_span_ns=500,
                        kernel_ns=250,
                        kernel_calls=25,
                    )
                )
                kernels.extend(
                    dict(case=case, kernel=symbol, calls=10, total_ns=1000 + index)
                    for index, symbol in enumerate(symbols)
                )
                samples[case] = {
                    symbols[0]: dict(
                        display_name="kernel<int>",
                        metrics={label: 10 for label in METRICS},
                        rules=[
                            dict(
                                name="Bottleneck",
                                section="SpeedOfLight",
                                title="Small Grid",
                                message="Underutilized | <device>. See @section:LaunchStats:Launch Statistics@.",
                            )
                        ],
                    ),
                    symbols[1]: dict(
                        display_name="kernel<double>",
                        metrics={label: 90 for label in METRICS},
                        rules=[],
                    ),
                }
            (output / "performance/nsys").mkdir(parents=True)
            (output / "ncu").mkdir()
            (output / "performance/nsys/timings.json").write_text(
                json.dumps(dict(case_regions=regions, case_kernels=kernels))
            )
            (output / "ncu/results.json").write_text(
                json.dumps(dict(cases=samples, errors=[]))
            )
            self.assertEqual(write_combined_report(output), [])
            report = (output / "nsight/report.md").read_text()
            self.assertEqual(report.count("| Kind |"), 14)
            self.assertEqual(report.count("| Kernel |"), 14 * 25)
            self.assertEqual(report.count("| Region |"), 14 * 2)
            self.assertIn("Kokkos&#124;&lt;kernel&gt;", report)
            self.assertIn("Underutilized &#124; &lt;device&gt;", report)
            self.assertIn("See Launch Statistics.", report)
            self.assertNotIn("@section:", report)
            self.assertIn("Not sampled", report)
            records = json.loads((output / "nsight/report.json").read_text())["rows"]
            selected = [
                row
                for row in records
                if row["case"] == cases[0][0] and row["kind"] == "Kernel"
            ]
            by_symbol = {row["name"]: row for row in selected}
            self.assertEqual(by_symbol[symbols[0]]["SM (%)"], 10)
            self.assertEqual(by_symbol[symbols[1]]["SM (%)"], 90)
            self.assertEqual(by_symbol[symbols[0]]["calls"], 10)
            self.assertEqual(by_symbol[symbols[0]]["mean_us"], 0.1)
            self.assertNotIn("SM (%)", by_symbol[symbols[2]])
            with (output / "nsight/report.csv").open() as source:
                self.assertEqual(len(list(csv.DictReader(source))), 14 * 27)

    def test_missing_compute_data_still_reports_all_systems_cases(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / "performance/nsys").mkdir(parents=True)
            (output / "performance/nsys/timings.json").write_text(
                json.dumps(dict(case_regions=[], case_kernels=[]))
            )
            errors = write_combined_report(output)
            self.assertIn("Missing Nsight Compute diagnostics", errors)
            report = (output / "nsight/report.md").read_text()
            self.assertEqual(report.count("| Kind |"), 14)
            self.assertEqual(report.count("| Unavailable |"), 14)


class RegionAggregation(unittest.TestCase):
    def test_regions_include_all_threads_and_inclusive_kernel_counts(self):
        rows = {
            "nvtx_pushpop_sum": [
                {"Range": "region", "Instances": "2", "Total Time (ns)": "20"},
                {"Range": "region", "Instances": "3", "Total Time (ns)": "30"},
            ],
            "nvtx_gpu_proj_sum": [{"Range": "region", "Total Proj Time (ns)": "40"}],
            "nvtx_kern_sum:mangled": [
                {"NVTX Range": "region", "Total Time (ns)": "7", "Kern Inst": "4"},
                {"NVTX Range": "region", "Total Time (ns)": "8", "Kern Inst": "5"},
            ],
        }
        (region,) = summarize_regions(rows)
        self.assertEqual(region["calls"], 5)
        self.assertEqual(region["host_ns"], 50)
        self.assertEqual(region["gpu_span_ns"], 40)
        self.assertEqual(region["kernel_ns"], 15)
        self.assertEqual(region["kernel_calls"], 9)


class WorkflowPublication(unittest.TestCase):
    def test_exit_traps_publish_reports_without_hiding_step_failure(self):
        workflow = (
            Path(__file__).resolve().parents[2]
            / ".github/workflows/benchmark_taylor_green_vortex.yml"
        ).read_text()
        for name, phase in (
            ("publish_performance_report", "performance"),
            ("publish_nsight_report", "nsight"),
        ):
            start = workflow.index(f"        {name}()")
            end = workflow.index(f"        trap {name} EXIT", start) + len(
                f"        trap {name} EXIT"
            )
            script = "set -euo pipefail\n" + textwrap.dedent(workflow[start:end])
            for status, report_exists in ((0, True), (7, True), (7, False)):
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
                        self.assertEqual(result.stdout, "Collected report\n")
                    else:
                        self.assertFalse(summary.exists())


if __name__ == "__main__":
    unittest.main()
