<!--
SPDX-FileCopyrightText: 2026 Baptiste Legouix
SPDX-License-Identifier: AGPL-3.0-or-later
AI-GENERATED
-->

# Electromagnetic Taylor–Green extensions

\important This operator and documentation is fully AI-generated.

These smooth periodic test problems extend the supplied hydrodynamic workshop
case. They are intended to compare solver performance and numerical invariants;
they do not have the workshop's reference energy histories. The original fluid
initialization, Re=1600, Pr=0.71, and Mach=0.1 for compressible flow remain fixed.
N is the only numerical resolution parameter exposed at runtime. Each Google
Benchmark iteration advances one adaptive time step; benchmark controls set
trajectory length. See [iteration semantics](benchmark_iterations.md).

## Selecting a model

Three independent groups of binary options select the equations:

| Group | Options | Default |
| --- | --- | --- |
| Fluid | `--incompressible`, `--compressible` | incompressible |
| Electromagnetic transport | `--ideal`, `--resistive` | ideal |
| Coupling | `--hd`, `--ehd`, `--mhd`, `--fmhd` | HD |

Conflicting choices within a group fail regardless of argument order. Repeating
the same flag is harmless. Omitted groups retain their defaults. Resistive HD
fails because HD contains no electrical transport. The legacy
`--physics=incompressible` and `--physics=compressible` spellings still work,
including conflict detection with the new fluid flags. Unknown options are
reported by Google Benchmark. Its reporting, filtering, and repetition flags
continue to work.

"Ideal" leaves the workshop fluid viscosity and Fourier conduction in place.
For MHD it means zero magnetic resistivity, E=-u×B. For EHD and FMHD it means
advected charge with zero conductivity, J=q u, as chosen for the library physics.
This FMHD limit still evolves full Maxwell waves; it does not impose E+u×B=0.

## Initial fields and material constants

The magnetic cases use

\[
B_0=0.25 e_z+0.15(\sin x\cos y\sin z,-\cos x\sin y\sin z,0).
\]

The perturbation is the curl of the periodic potential
A_z=0.15 sin(x) sin(y) sin(z). Its phase differs from the fluid velocity's
cos(z), so it produces magnetic stretching and Lorentz forces. The constant
guide field supplies a nonzero harmonic magnetic flux, which is evolved without
projecting it away. The continuum mean magnetic energy is 0.0340625, compared
with the fluid's initial mean kinetic energy 0.125.

The electric cases start from

\[
\phi_0=0.1\cos x\cos y\cos z,\qquad E_0=-\nabla\phi_0,\qquad D_0=\epsilon E_0.
\]

Charge is initialized by the **discrete** relation q=div D, rather than sampling
the continuum expression q=3 epsilon phi independently. Thus Gauss's law holds
on every grid and the total charge is zero. The continuum mean electric energy
is 0.001875 for epsilon=1. The initially gradient-like electric force alters the
incompressible pressure before charge advection changes its structure.

All sectors use constant epsilon=mu=1 in nondimensional units. Full Maxwell
waves therefore have speed 1; this is a numerical benchmark material, not a
vacuum/light-speed scaling for a tokamak. Resistive MHD uses eta=0.001 and
magnetic diffusivity eta/mu=0.001. Resistive EHD and FMHD use electrical
resistivity eta=10, hence conductivity 0.1 and charge relaxation time eta epsilon=10.
The two resistivities serve different model limits and are intentionally not
assigned the same value. They are fixed benchmark definitions, not CLI inputs.

EHD initializes and evolves no magnetic field. MHD has no independent electric
displacement or charge state; its electric field is derived from Ohm's law.
FMHD initializes both fields. Fluid pressure and temperature are initialized
exactly as in HD; no artificial magnetostatic balance is imposed. Incompressible
pressure is subsequently reconstructed from the full forced momentum equation.

## Compatible DEC evolution

`ElectromagneticSolver` in `electromagnetic_solver.hpp` uses the existing
`VorticityDEC` adapter to call native exterior reduction, reconstruction, d,
and metric codifferential delta operators. B is a primal two-cochain; D, E,
and J are primal one-cochains. Charge is a nodal scalar. On this uniform
periodic Euclidean grid, physical divergence of a one-cochain is -delta.

MHD advances B_t=-dE with J=delta B/mu. Its ideal motional EMF is reduced from
nodal -u×B. The resistive contribution eta J is added directly to the same
edge cochains, preserving the native diffusion operator.

EHD advances q_t=-div J and reconstructs D from
Delta_H phi=q/epsilon, D=-epsilon d phi. This uses matrix-free CG from the
solver module, with relative tolerance 1e-11, at most 8N iterations, and zero
mean potential. Zero net charge is checked before each periodic solve. The
electrostatic solve is distinct from a compressible fluid optimizer: all fluid
variables still follow evolution equations.

FMHD advances

\[
D_t=\delta B/\mu-J,\qquad B_t=-d(D/\epsilon),\qquad q_t=-\mathrm{div}J.
\]

The **same current cochain** enters Ampere and charge continuity. Consequently,
d B and div D-q remain unchanged by every spatial RHS evaluation. SSPRK3
preserves these linear constraints up to roundoff. Magnetic initialization
samples forward face centres and uses native two-form reduction; its staggered
divergence cancels exactly. The harmonic guide field survives time stepping.

Resistive electric currents add E/eta for EHD and (E+u×B)/eta for FMHD. The E
contribution uses the actual electric edge cochain, avoiding extra smoothing
of charge relaxation. Nonlinear products use arithmetic placement averages.
Library sector classes supply field constitutive relations and Lorentz force.

The fluid solver receives F=qE+J×B (with inactive sectors zero). Incompressible
evolution adds curl(F/rho) to the vorticity equation and includes F/rho in the
diagnostic pressure solve. The mean acceleration is integrated separately with
the same SSPRK3 stages, because a periodic curl cannot carry harmonic velocity.
Compressible momentum receives F, and **fluid-only** total energy receives J.E.
This differs in storage from the library MHD/FMHD combined-energy variables:
the benchmark adds field energy only when monitoring total energy, and never
adds Joule heat as a second energy source.

J.E is paired on the same edges as Ampere/Ohm and averaged back to nodes.
For uniform periodic geometry this gives the global spatial identity
d(field energy)/dt=-mean(J.E). Field energy is computed from reconstructed
edge/face values before nodal averaging, so it matches this pairing. An
independent RHS residual checks the identity using B_t and D_t; EHD instead
uses d(electric energy)/dt=mean(phi q_t). SSPRK3 introduces temporal energy
error, and nodal incompressible kinetic-energy monitors are not an exact
cochain energy invariant. Resolution and time-discretization convergence remain
necessary before interpreting long-run results.

Fluid, charge, and Maxwell states advance together at every SSPRK3 stage.
Automatic time-step bounds include fluid CFL/diffusion, a magnetic wave-speed
bound, Maxwell propagation, magnetic diffusion, and charge relaxation. There
are no user-supplied CFL, time-step, field-amplitude, or material parameters.

## Outputs and validation

Names of benchmark registrations, exported files, HDF5 metadata, and performance
JSON include all three selectors, for example `compressible_resistive_mhd`.
Active electromagnetic fields are exported alongside the fluid snapshots at
t=0, t=8 and t=10 when reached, and the final time. A scalar electric potential is exported only for electrostatic EHD;
it would not describe the evolving transverse electric field in FMHD.

`electromagnetic_monitors` stores time, magnetic energy, electric energy,
fluid electric work, Ohmic heating, magnetic divergence RMS, Gauss RMS, mean
charge, and the signed field/fluid energy-exchange residual. Inactive sector
energies/constraints are zero. The existing total-energy monitor includes
active field energy. For incompressible runs it excludes thermal energy and
therefore does not track Ohmic or viscous heating in a thermal reservoir.

`post_process.py` writes these histories to `.electromagnetic.dat`, exports
the active fields to VTK, and plots field energy, electric work/heating, and
constraints in `.electromagnetic.png`. Existing fluid spectra and plots remain
available. Field processing time and electrostatic Krylov iteration counts are
reported separately within solver timing; monitor/export work stays excluded
from Google Benchmark's manually timed solver measurement.

CMake registers short validation runs for all 14 supported configurations,
plus selector consistency tests. The coupled runs check initial/final Gauss
and magnetic constraints, charge and mass conservation, spatial energy exchange,
and short-run compressible combined-energy drift. Kinetic energy is allowed to
increase when fields drive the fluid. These C++ checks still need execution
inside the rebuilt CPU Docker image; symbolic and post-processing checks do not
replace a native-operator validation run.
