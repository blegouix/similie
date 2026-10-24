# SPDX-FileCopyrightText: 2026 Baptiste Legouix
# SPDX-License-Identifier: AGPL-3.0-or-later
# AI-GENERATED

"""Check convergence of the constraint-preserving Taylor-Green initial field."""

import re
import subprocess
import sys


def initial_error(executable, nodes):
    result = subprocess.run(
        [
            executable,
            f"--N={nodes}",
            "--incompressible",
            "--ideal",
            "--hd",
            "--validate",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    match = re.search(r"initial_velocity_rms=([0-9.eE+-]+)", result.stdout)
    if match is None:
        raise AssertionError(f"Missing initial velocity diagnostic:\n{result.stdout}")
    return float(match.group(1))


coarse = initial_error(sys.argv[1], 8)
fine = initial_error(sys.argv[1], 16)
if not 0 < fine < 0.6 * coarse:
    raise AssertionError(f"Initial velocity does not converge: {coarse=}, {fine=}")
print(f"Constraint-preserving initial velocity converges: {coarse=}, {fine=}")
