# SPDX-FileCopyrightText: 2026 Baptiste Legouix
# SPDX-License-Identifier: AGPL-3.0-or-later
# AI-GENERATED

"""Use separate CUDA processes because device assertion poisons its CUDA context."""

import subprocess
import sys

for operator in ("reduction", "reconstruction"):
    for executable, geometry, should_assert in (
        (sys.argv[1], "aligned", False),
        (sys.argv[1], "skew", True),
        (sys.argv[2], "skew", False),
    ):
        result = subprocess.run(
            [executable, operator, geometry],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=30,
        )
        if should_assert:
            assert result.returncode != 0, result.stdout
            assert (
                "Diagonal staged operator requires diagonal coefficients"
                in result.stdout
            ), result.stdout
        else:
            assert result.returncode == 0, result.stdout
