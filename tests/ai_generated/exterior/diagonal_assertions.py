# SPDX-FileCopyrightText: 2026 Baptiste Legouix
# SPDX-License-Identifier: AGPL-3.0-or-later
# AI-GENERATED

"""Use separate CUDA processes because device assertion poisons its CUDA context."""

import subprocess
import sys

for operator, path in (
    (op, path)
    for op in ("reduction", "reconstruction", "weighted-hodge")
    for path in ("direct", "staged")
):
    for executable, geometry, should_assert in (
        (sys.argv[1], "aligned", False),
        (sys.argv[1], "skew", True),
        (sys.argv[2], "skew", False),
    ):
        result = subprocess.run(
            [executable, operator, geometry, path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=30,
        )
        if should_assert:
            assert result.returncode != 0, result.stdout
            assert "requires diagonal coefficients" in result.stdout, result.stdout
        else:
            assert result.returncode == 0, result.stdout

for operator in ("reduction", "reconstruction"):
    for path in ("direct", "staged"):
        for executable, geometry, should_assert in (
            (sys.argv[1], "isotropic", False),
            (sys.argv[1], "aligned", True),
            (sys.argv[2], "aligned", False),
        ):
            result = subprocess.run(
                [executable, operator, geometry, path, "scaling"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=30,
            )
            if should_assert:
                assert result.returncode != 0, result.stdout
                assert "requires equal diagonal coefficients" in result.stdout, (
                    result.stdout
                )
            else:
                assert result.returncode == 0, result.stdout
