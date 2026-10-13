#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Baptiste Legouix
# SPDX-License-Identifier: AGPL-3.0-or-later
# AI-GENERATED

"""Ideal-gas closure used only for the Taylor-Green reference campaign."""

import argparse
import importlib.util
import sys
from pathlib import Path

from sympy import symbols


def generate(output):
    physics = (
        Path(__file__).resolve().parents[2]
        / "src/similie/physics/fluid/navier_stokes_fourier.py"
    )
    spec = importlib.util.spec_from_file_location(
        "similie_navier_stokes_fourier", physics
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    rho, energy = symbols("rho e", positive=True)
    gamma, gas_constant = symbols("gamma gas_constant", positive=True)
    module.ThermodynamicRelation(
        density=rho,
        specific_internal_energy=energy,
        pressure=(gamma - 1) * rho * energy,
        temperature=(gamma - 1) * energy / gas_constant,
        parameters=(gamma, gas_constant),
    ).generate_cpp(output, "taylor_green_vortex", "IdealGas")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    generate(parser.parse_args().output)
