#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Baptiste Legouix
# SPDX-License-Identifier: AGPL-3.0-or-later
# AI-GENERATED

"""Full Maxwell Hamiltonian and charged-fluid coupling from the two field sectors."""

import importlib.util
import sys
from pathlib import Path

from sympy import Matrix, symbols


def _load(name, relative_path):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).resolve().parents[1] / relative_path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_electric = _load("similie_fmhd_electric", "electrodynamics/electrodynamics.py")
_magnetic = _load("similie_fmhd_magnetic", "magnetodynamics/magnetodynamics.py")


class Electromagnetism:
    def __init__(self):
        self.electric = _electric.Electrodynamics()
        self.magnetic = _magnetic.Magnetodynamics()
        self.displacement = self.electric.displacement
        self.induction = self.magnetic.induction
        self.velocity = self.electric.velocity
        self.charge_density = self.electric.charge_density
        self.permittivity = self.electric.permittivity
        self.permeability = self.magnetic.permeability
        self.electric_field = self.electric.electric_field
        self.magnetic_field = self.magnetic.magnetic_field
        self.hamiltonian = self.electric.hamiltonian + self.magnetic.hamiltonian
        self.conduction_current = Matrix.zeros(3, 1)

    @property
    def rest_frame_electric_field(self):
        return self.electric_field + self.velocity.cross(self.induction)

    @property
    def current(self):
        return self.charge_density * self.velocity + self.conduction_current

    @property
    def lorentz_force(self):
        return self.charge_density * self.electric_field + self.current.cross(
            self.induction
        )

    @property
    def fluid_energy_source(self):
        return self.current.dot(self.electric_field)

    @property
    def ohmic_heating(self):
        return self.conduction_current.dot(self.rest_frame_electric_field)

    @property
    def poynting_flux(self):
        return self.electric_field.cross(self.magnetic_field)


class ResistiveElectromagnetism(Electromagnetism):
    def __init__(self):
        super().__init__()
        self.resistivity = symbols("eta", positive=True)
        self.conduction_current = self.rest_frame_electric_field / self.resistivity
