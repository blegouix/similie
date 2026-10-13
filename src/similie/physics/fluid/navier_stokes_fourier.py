#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Baptiste Legouix
# SPDX-License-Identifier: AGPL-3.0-or-later
# AI-GENERATED

"""Explicit Navier-Stokes-Fourier evolution and symbolic thermodynamic closures.

Independent thermodynamic coordinates are density rho and specific internal
energy e. A caller supplies p(rho,e) and T(rho,e), not a Hamiltonian. SymPy
computes local thermodynamic derivatives; exterior supplies spatial derivatives.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from sympy import Matrix, Rational, Symbol, diff, simplify, symbols, sympify
from sympy.printing.codeprinter import cxxcode


def _cpp(expression, replacements=None):
    expression = sympify(expression)
    return cxxcode(expression.xreplace(replacements or {}), standard="c++17")


def _identifier(name):
    if not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", name):
        raise ValueError(f"Invalid C++ identifier: {name}")


def _write_header(output, text):
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text)
    formatter = shutil.which("clang-format")
    if formatter is not None:
        subprocess.run([formatter, "-i", str(output)], check=True)


@dataclass(frozen=True)
class ThermodynamicRelation:
    """Caller-supplied caloric closure, with no particular equation of state.

    cv=1/T_e. cp=(dh/dT)_p is evaluated as a ratio of Jacobians in (rho,e),
    with h=e+p/rho. c^2=p_rho+p*p_e/rho^2 follows from de=p/rho^2 drho
    along an isentrope. The closure must describe one smooth phase with an
    invertible local (p,T) map, positive heat capacities and sound speed.
    """

    density: Symbol
    specific_internal_energy: Symbol
    pressure: object
    temperature: object
    parameters: tuple[Symbol, ...] = ()

    def __post_init__(self):
        if self.density == self.specific_internal_energy:
            raise ValueError("Thermodynamic state variables must be distinct")
        declared = {self.density, self.specific_internal_energy, *self.parameters}
        if len(declared) != 2 + len(self.parameters):
            raise ValueError(
                "Thermodynamic parameters must be distinct from state variables"
            )
        for expression in (self.pressure, self.temperature):
            if sympify(expression).free_symbols - declared:
                raise ValueError("Undeclared symbols in thermodynamic relation")

    def derivatives(self):
        rho, energy = self.density, self.specific_internal_energy
        p, temperature = sympify(self.pressure), sympify(self.temperature)
        p_rho, p_e = diff(p, rho), diff(p, energy)
        t_rho, t_e = diff(temperature, rho), diff(temperature, energy)
        enthalpy = energy + p / rho
        jacobian = simplify(t_e * p_rho - t_rho * p_e)
        if simplify(t_e) == 0 or jacobian == 0:
            raise ValueError("Degenerate thermodynamic closure: cv or cp is undefined")
        return {
            "pressure_density_derivative": p_rho,
            "pressure_energy_derivative": p_e,
            "temperature_density_derivative": t_rho,
            "temperature_energy_derivative": t_e,
            "heat_capacity_at_constant_volume": simplify(1 / t_e),
            "heat_capacity_at_constant_pressure": simplify(
                (diff(enthalpy, energy) * p_rho - diff(enthalpy, rho) * p_e) / jacobian
            ),
            "sound_speed_squared": simplify(p_rho + p * p_e / rho**2),
        }

    def integrability_residual(self):
        """Exactness residual for ds=de/T-p drho/(T rho^2).

        Zero is required for thermodynamic consistency. It is exposed rather
        than forcing SymPy to decide identities for every user-defined function.
        """
        rho, energy = self.density, self.specific_internal_energy
        return simplify(
            diff(1 / self.temperature, rho)
            - diff(-self.pressure / (self.temperature * rho**2), energy)
        )

    def temperature_gradient(self, density_gradient, energy_gradient):
        derivatives = self.derivatives()
        return (
            Matrix(density_gradient) * derivatives["temperature_density_derivative"]
            + Matrix(energy_gradient) * derivatives["temperature_energy_derivative"]
        )

    def generate_cpp(self, output_path, namespace, class_name):
        """Generate a local, device-callable closure with analytic derivatives."""
        _identifier(class_name)
        for name in namespace.split("::"):
            _identifier(name)
        for parameter in self.parameters:
            _identifier(str(parameter))
        replacements = {
            self.density: Symbol("density"),
            self.specific_internal_energy: Symbol("specific_internal_energy"),
        }
        methods = {
            "pressure": self.pressure,
            "temperature": self.temperature,
            **self.derivatives(),
        }
        if any(
            str(parameter) in {"density", "specific_internal_energy", *methods}
            for parameter in self.parameters
        ):
            raise ValueError(
                "Thermodynamic parameter collides with a generated C++ method or argument"
            )
        members = "\n".join(f"    double const {p};" for p in self.parameters)
        arguments = ", ".join(f"double {p}_" for p in self.parameters)
        initializers = ", ".join(f"{p}({p}_)" for p in self.parameters)
        constructor = f"    KOKKOS_FUNCTION explicit {class_name}({arguments})"
        if initializers:
            constructor += f" : {initializers}"
        constructor += " {}"
        functions = "\n".join(
            f"    KOKKOS_FUNCTION double {name}(double density, double specific_internal_energy) const\n"
            f"    {{ return {_cpp(expression, replacements)}; }}"
            for name, expression in methods.items()
        )
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        _write_header(
            output,
            f"""// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED
// Generated from caller-supplied SymPy thermodynamics. Do not edit.
#pragma once
#include <cmath>
#include <Kokkos_Core.hpp>
namespace {namespace} {{
/** Thermodynamics in (rho,e), including analytic cv, cp, and isentropic c^2.
 * cp is a Jacobian ratio at fixed pressure; no ideal-gas identity is assumed.
 * \\important This operator and documentation is fully AI-generated.
 */
class {class_name} {{
public:
{members}
{constructor}
{functions}
}};
}} // namespace {namespace}
""",
        )


class NavierStokesFourier:
    """Symbolic mass, momentum, total-energy fluxes and explicit time rates.

    Newtonian stress uses Stokes' zero bulk-viscosity assumption in 3D;
    Fourier conductivity is mu*cp/Pr. No thermodynamic law is fixed here.
    """

    def __init__(self):
        self.density = Symbol("rho", positive=True)
        self.momentum = Matrix(symbols("m0:3"))
        self.total_energy = Symbol("rho_E")
        self.velocity = self.momentum / self.density
        self.specific_internal_energy = (
            self.total_energy - self.momentum.dot(self.momentum) / (2 * self.density)
        ) / self.density
        self.viscosity, self.prandtl, self.cp, self.pressure = symbols("mu Pr cp p")
        self.velocity_gradient = Matrix(3, 3, lambda a, b: Symbol(f"g{a}{b}"))
        self.temperature_gradient = Matrix(symbols("dT0:3"))
        self.stress = self.viscosity * (
            self.velocity_gradient
            + self.velocity_gradient.T
            - Rational(2, 3) * self.velocity_gradient.trace() * Matrix.eye(3)
        )
        self.heat_flux = (
            -self.viscosity * self.cp / self.prandtl * self.temperature_gradient
        )
        self.flux_divergence = Matrix(
            symbols("div_mass div_m0 div_m1 div_m2 div_energy")
        )
        self.evolution = -self.flux_divergence

    def fluxes(self, thermodynamics=None, stress=None, heat_flux=None):
        p, cp = self.pressure, self.cp
        if thermodynamics is not None:
            replacement = {
                thermodynamics.density: self.density,
                thermodynamics.specific_internal_energy: self.specific_internal_energy,
            }
            p = sympify(thermodynamics.pressure).xreplace(replacement)
            cp = thermodynamics.derivatives()[
                "heat_capacity_at_constant_pressure"
            ].xreplace(replacement)
        tau = self.stress if stress is None else stress
        q = self.heat_flux.subs(self.cp, cp) if heat_flux is None else heat_flux
        flux = Matrix.zeros(5, 3)
        for normal in range(3):
            flux[0, normal] = self.momentum[normal]
            for axis in range(3):
                flux[axis + 1, normal] = (
                    self.momentum[axis] * self.velocity[normal]
                    + (p if axis == normal else 0)
                    - tau[axis, normal]
                )
            flux[4, normal] = (
                (self.total_energy + p) * self.velocity[normal]
                - sum(tau[axis, normal] * self.velocity[axis] for axis in range(3))
                + q[normal]
            )
        return flux

    def generate_cpp(self, output_path):
        """Generate generic C++ physics, templated on caller thermodynamics."""
        replacements = {
            self.density: Symbol("state.density"),
            self.total_energy: Symbol("state.total_energy"),
            self.viscosity: Symbol("viscosity"),
            self.pressure: Symbol("primitive.pressure"),
            **{m: Symbol(f"state.momentum[{a}]") for a, m in enumerate(self.momentum)},
            **{
                self.velocity_gradient[a, b]: Symbol(f"gradient[{a}][{b}]")
                for a in range(3)
                for b in range(3)
            },
        }
        stress_code = "\n".join(
            f"        result[{a}][{b}] = {_cpp(self.stress[a, b], replacements)};"
            for a in range(3)
            for b in range(3)
        )
        tau = Matrix(3, 3, lambda a, b: Symbol(f"tau[{a}][{b}]"))
        q = Matrix([Symbol(f"q[{a}]") for a in range(3)])
        fluxes = self.fluxes(stress=tau, heat_flux=q)
        flux_code = []
        for normal in range(3):
            flux_code.append(f"        if (normal == {normal}) {{")
            flux_code.append(
                f"            result.mass = {_cpp(fluxes[0, normal], replacements)};"
            )
            for a in range(3):
                flux_code.append(
                    f"            result.momentum[{a}] = {_cpp(fluxes[a + 1, normal], replacements)};"
                )
            flux_code.append(
                f"            result.total_energy = {_cpp(fluxes[4, normal], replacements)};"
            )
            flux_code.append("        }")
        energy_code = _cpp(self.specific_internal_energy, replacements)
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        _write_header(
            output,
            _CPP_HEADER.format(
                energy=energy_code, stress=stress_code, flux="\n".join(flux_code)
            ),
        )


_CPP_HEADER = """// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED
// Generated by fluid/navier_stokes_fourier.py. Do not edit.
#pragma once
#include <array>
#include <cmath>
#include <Kokkos_Core.hpp>
namespace similie::physics::fluid {{
/** Explicit mass, momentum, and energy conservation with Newtonian/Fourier fluxes.
 * dt rho=-div(rho u), dt(rho u)=-div(rho u tensor u+p I-tau),
 * dt(rho E)=-div((rho E+p)u-tau.u+q), q=-mu cp/Pr grad T.
 * Thermodynamics supplies p(rho,e), T(rho,e), cv, cp and isentropic c^2.
 * Spatial derivatives are supplied by exterior; no optimizer or Hamiltonian.
 * \\important This operator and documentation is fully AI-generated.
 */
template<class Thermodynamics>
class NavierStokesFourier {{
public:
    struct ConservativeVariables {{ double density; std::array<double,3> momentum; double total_energy; }};
    struct PrimitiveVariables {{ double density; std::array<double,3> velocity; double pressure; double temperature; }};
    struct Flux {{ double mass; std::array<double,3> momentum; double total_energy; }};
    using VelocityGradient=std::array<std::array<double,3>,3>;
    Thermodynamics const thermodynamics;
    double const prandtl, viscosity;
    KOKKOS_FUNCTION explicit NavierStokesFourier(Thermodynamics closure, double prandtl_number,
            double dynamic_viscosity)
        : thermodynamics(closure), prandtl(prandtl_number), viscosity(dynamic_viscosity) {{}}
    KOKKOS_FUNCTION double specific_internal_energy(ConservativeVariables const& state) const
    {{ return {energy}; }}
    KOKKOS_FUNCTION double pressure(ConservativeVariables const& state) const
    {{ return thermodynamics.pressure(state.density,specific_internal_energy(state)); }}
    KOKKOS_FUNCTION PrimitiveVariables primitives(ConservativeVariables const& state) const
    {{
        double const e=specific_internal_energy(state);
        PrimitiveVariables result{{state.density,{{}},thermodynamics.pressure(state.density,e),thermodynamics.temperature(state.density,e)}};
        for(int a=0;a<3;++a) result.velocity[a]=state.momentum[a]/state.density;
        return result;
    }}
    KOKKOS_FUNCTION double heat_capacity_at_constant_volume(ConservativeVariables const& state) const
    {{ return thermodynamics.heat_capacity_at_constant_volume(state.density,specific_internal_energy(state)); }}
    KOKKOS_FUNCTION double heat_capacity_at_constant_pressure(ConservativeVariables const& state) const
    {{ return thermodynamics.heat_capacity_at_constant_pressure(state.density,specific_internal_energy(state)); }}
    KOKKOS_FUNCTION double sound_speed_squared(ConservativeVariables const& state) const
    {{ return thermodynamics.sound_speed_squared(state.density,specific_internal_energy(state)); }}
    KOKKOS_FUNCTION double conductivity(ConservativeVariables const& state) const
    {{ return viscosity*heat_capacity_at_constant_pressure(state)/prandtl; }}
    KOKKOS_FUNCTION std::array<double,3> temperature_gradient(ConservativeVariables const& state,
            std::array<double,3> const& density_gradient, std::array<double,3> const& energy_gradient) const
    {{
        double const e=specific_internal_energy(state);
        double const dr=thermodynamics.temperature_density_derivative(state.density,e);
        double const de=thermodynamics.temperature_energy_derivative(state.density,e);
        std::array<double,3> result{{}};
        for(int a=0;a<3;++a) result[a]=dr*density_gradient[a]+de*energy_gradient[a];
        return result;
    }}
    KOKKOS_FUNCTION VelocityGradient stress(VelocityGradient const& gradient) const
    {{
        VelocityGradient result{{}};
{stress}
        return result;
    }}
    KOKKOS_FUNCTION std::array<double,3> heat_flux(ConservativeVariables const& state,
            std::array<double,3> const& temperature_gradient) const
    {{
        double const kappa=conductivity(state);
        std::array<double,3> result{{}};
        for(int a=0;a<3;++a) result[a]=-kappa*temperature_gradient[a];
        return result;
    }}
    KOKKOS_FUNCTION Flux flux(ConservativeVariables const& state, PrimitiveVariables const& primitive,
            VelocityGradient const& velocity_gradient, std::array<double,3> const& temperature_gradient,
            int normal) const
    {{
        VelocityGradient const tau=stress(velocity_gradient);
        std::array<double,3> const q=heat_flux(state,temperature_gradient);
        Flux result{{}};
{flux}
        return result;
    }}
    KOKKOS_FUNCTION ConservativeVariables operator()(Flux const& divergence) const
    {{
        return {{-divergence.mass,{{-divergence.momentum[0],-divergence.momentum[1],-divergence.momentum[2]}},-divergence.total_energy}};
    }}
}};
}} // namespace similie::physics::fluid
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    NavierStokesFourier().generate_cpp(parser.parse_args().output)
