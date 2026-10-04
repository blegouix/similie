# SPDX-FileCopyrightText: 2026 Baptiste Legouix
# SPDX-License-Identifier: AGPL-3.0-or-later
# // AI-GENERATED

"""Compare the circulation reported by SimiLie and the GetDP reference."""

import re
import sys
from pathlib import Path


similie_log = Path(sys.argv[1]).read_text()
match = re.search(
    r"SimiLie potential-flow diagnostics:.*circulation=([-+0-9.eE]+)", similie_log
)
if match is None:
    raise SystemExit("missing SimiLie circulation diagnostic")
similie = float(match.group(1))
getdp = float(Path(sys.argv[2]).read_text().splitlines()[0].split()[1])
error = abs(similie - getdp)
tolerance = max(0.05, 0.01 * abs(getdp))
print(f"SimiLie/GetDP circulation: {similie:.6g} / {getdp:.6g} m^2/s")
if error > tolerance:
    raise SystemExit(f"circulation difference {error:.6g} exceeds {tolerance:.6g}")
