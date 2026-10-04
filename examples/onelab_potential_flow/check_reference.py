# SPDX-FileCopyrightText: 2026 Baptiste Legouix
# SPDX-License-Identifier: AGPL-3.0-or-later
# // AI-GENERATED

"""Compare SimiLie potential-flow diagnostics with the GetDP reference."""

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
tolerance = max(0.05, 0.02 * abs(getdp))
print(f"SimiLie/GetDP circulation: {similie:.6g} / {getdp:.6g} m^2/s")
if error > tolerance:
    raise SystemExit(f"circulation difference {error:.6g} exceeds {tolerance:.6g}")

reference_lines = Path(sys.argv[2]).read_text().splitlines()
reference_mass_flow = float(reference_lines[1].split()[1])
if abs(reference_mass_flow) > 1e-12:
    mass_flow_match = re.search(
        r"SimiLie potential-flow diagnostics:.*mass flow rate=([-+0-9.eE]+)",
        similie_log,
    )
    if mass_flow_match is None:
        raise SystemExit("missing SimiLie mass-flow diagnostic")
    similie_mass_flow = float(mass_flow_match.group(1))
    mass_flow_error = abs(similie_mass_flow - reference_mass_flow)
    mass_flow_tolerance = max(0.05, 0.02 * abs(reference_mass_flow))
    print(
        "SimiLie/GetDP mass flow: "
        f"{similie_mass_flow:.6g} / {reference_mass_flow:.6g} kg/s"
    )
    if mass_flow_error > mass_flow_tolerance:
        raise SystemExit(
            f"mass-flow difference {mass_flow_error:.6g} exceeds "
            f"{mass_flow_tolerance:.6g}"
        )
