# SPDX-FileCopyrightText: 2026 Baptiste Legouix
# SPDX-License-Identifier: AGPL-3.0-or-later
# // AI-GENERATED

"""Compare SimiLie potential-flow diagnostics with the GetDP reference."""

import argparse
import math
import re
from pathlib import Path


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("similie_log", type=Path)
parser.add_argument("getdp_output", type=Path)
parser.add_argument("--imposed-circulation", type=int, choices=(0, 1), default=0)
args = parser.parse_args()
similie_log = args.similie_log.read_text()
match = re.search(
    r"SimiLie potential-flow diagnostics:.*circulation=([-+0-9.eE]+)", similie_log
)
if match is None:
    raise SystemExit("missing SimiLie circulation diagnostic")
similie = float(match.group(1))
getdp = float(args.getdp_output.read_text().splitlines()[0].split()[1])
if not math.isfinite(similie) or not math.isfinite(getdp):
    raise SystemExit("non-finite circulation diagnostic")
error = abs(similie - getdp)
# The two solvers use different discrete boundary closures on the
# mapped quad mesh. The absolute allowance matters when the cylinder's
# circulation is inferred from a small difference of much larger cut fluxes.
tolerance = max(0.6, 0.04 * abs(getdp))
print(f"SimiLie/GetDP circulation: {similie:.6g} / {getdp:.6g} m^2/s")
if error > tolerance:
    raise SystemExit(f"circulation difference {error:.6g} exceeds {tolerance:.6g}")

reference_lines = args.getdp_output.read_text().splitlines()
reference_mass_flow = float(reference_lines[1].split()[1])
if not math.isfinite(reference_mass_flow):
    raise SystemExit("non-finite GetDP mass-flow diagnostic")
mass_flow_match = re.search(
    r"SimiLie potential-flow diagnostics:.*mass flow rate=([-+0-9.eE]+)",
    similie_log,
)
if mass_flow_match is None:
    raise SystemExit("missing SimiLie mass-flow diagnostic")
similie_mass_flow = float(mass_flow_match.group(1))
if not math.isfinite(similie_mass_flow):
    raise SystemExit("non-finite SimiLie mass-flow diagnostic")
# GetDP omits the Dmdt equation when circulation is prescribed. Its exported
# associated quantity is then zero, rather than the physical cut mass flow.
# Still reject non-finite diagnostics above and validate the full fields.
if args.imposed_circulation:
    if reference_mass_flow != 0:
        raise SystemExit("expected unreported GetDP mass flow for imposed circulation")
    print("GetDP mass flow is unreported for imposed circulation; compare full fields")
    raise SystemExit(0)
mass_flow_error = abs(similie_mass_flow - reference_mass_flow)
mass_flow_tolerance = max(0.05, 0.04 * abs(reference_mass_flow))
print(
    f"SimiLie/GetDP mass flow: {similie_mass_flow:.6g} / {reference_mass_flow:.6g} kg/s"
)
if mass_flow_error > mass_flow_tolerance:
    raise SystemExit(
        f"mass-flow difference {mass_flow_error:.6g} exceeds {mass_flow_tolerance:.6g}"
    )
