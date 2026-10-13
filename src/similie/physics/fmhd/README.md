<!--
SPDX-FileCopyrightText: 2026 Baptiste Legouix
SPDX-License-Identifier: AGPL-3.0-or-later
AI-GENERATED
-->

# Full Maxwell HydroDynamics (FMHD)

\important This operator and documentation is fully AI-generated.

FMHD couples hydrodynamics to both electric and magnetic fields with the full
Maxwell evolution, including displacement current. It is a separate model from
electric-only electrostatic EHD and magnetic-only quasistatic MHD.

`similie::physics::fmhd` exposes four `CombinePhysics`-based models:

| Type | Fluid | Field physics |
|---|---|---|
| `IdealIncompressibleFMHD` | `IncompressibleNavierStokes` | `Electromagnetism` |
| `IdealCompressibleFMHD<Thermodynamics>` | `NavierStokesFourier<Thermodynamics>` | `Electromagnetism` |
| `ResistiveIncompressibleFMHD` | `IncompressibleNavierStokes` | `ResistiveElectromagnetism` |
| `ResistiveCompressibleFMHD<Thermodynamics>` | `NavierStokesFourier<Thermodynamics>` | `ResistiveElectromagnetism` |

Construction takes the configured fluid and field objects. `get<0>()` is fluid,
`get<1>()` the full field sector. That field sector itself composes the electric
and magnetic constitutive physics with `CombinePhysics`. The compressible
thermodynamic relation remains caller-supplied.

Ideal uses the user's established advected-charge convention: zero conduction
current. Finite positive resistivity adds conduction and heat. A zero-resistivity
perfect conductor is a singular constrained limit, distinct from these explicit
FMHD models. Fluid viscosity and thermal conductivity remain independently
configured, as in the existing fluid physics.

## Hamiltonian and Maxwell equations

The electromagnetic Hamiltonian is the sum of the existing independently
SymPy-generated electric and magnetic Hamiltonians:

```
W(D,B) = |D|²/(2 epsilon) + |B|²/(2 mu),
E = δW/δD = D/epsilon,   H = δW/δB = B/mu.
```

`Electromagnetism(epsilon,mu)` obtains both constitutive fields from these
sectors. No duplicate field-energy generator or fluid Hamiltonian is introduced.
Material permittivity and permeability are uniform positive scalars here.
The full field and charge evolution is

```
D_t = curl H - J,
B_t = -curl E,
(rho_q)_t = -div J,
div D = rho_q,   div B = 0.
```

Rho_q is charge density, distinct from fluid mass density rho. Fields evolve
explicitly; there is no electrostatic potential solve in this model. In the
incompressible combination, velocity still uses the fluid's existing elliptic
Coulomb-gauge reconstruction from vorticity.

`operator()(div_J,curl_H,curl_E,J)` returns charge and field time derivatives.
Every spatial curl and divergence must come from native exterior operators.
In DEC, B and D are primal and dual two-cochains; E and H are primal and dual
one-cochains; current is a dual two-cochain and charge a dual three-cochain.
Then `D_t=dH-J`, `B_t=-dE`, and `rho_q,t=-dJ` preserve both Gauss constraints
through `d²=0`, provided **the same reduced current** is reused. The FMHD coupled
updates therefore take an explicit shared-current argument.

Initial fields must satisfy both Gauss constraints. Periodic Gauss's law
requires zero net charge, while magnetic harmonic flux may be nonzero.
`gauss_residual(div_D,rho_q)` exposes the electric residual; native div B is a
separate magnetic constraint monitor.

## Ideal and resistive current, force, and power

The current and full Lorentz force are

```
J = rho_q u + j,
f = rho_q E + J cross B.
```

For ideal FMHD, j=0. `ResistiveElectromagnetism(epsilon,mu,eta)` uses eta>0:

```
E_rest = E + u cross B,
j = E_rest/eta,
Q_Joule = j.E_rest = |E_rest|²/eta = eta |j|² >= 0.
```

Irreversible heat depends on conduction current, not the advective total
current. The lab-frame power identity is

```
J.E = u.f + Q_Joule.
```

`current`, `conduction_current`, `rest_frame_electric_field`, `lorentz_force`,
`fluid_energy_source`, and `ohmic_heating` provide these local expressions.
`wave_speed()` is `1/sqrt(epsilon mu)`. Finite-resistivity longitudinal charge
relaxation has rate `1/(eta epsilon)` in the zero-motion uniform-material case.
Explicit integration must account for fluid transport, Maxwell waves,
conduction relaxation and Lorentz-force source time scales.

## Incompressible FMHD

The evolved state is `(omega,rho_q,D,B)` and its vorticity balance is

```
omega_t = curl(u cross omega + f/rho) + nu Delta omega.
```

After fluid velocity reconstruction, `current(state,u)` computes the local
closure. The caller reduces it consistently. `transport(state,u,J)` supplies
the vector to be reduced and differentiated for vorticity transport. The coupled
`operator()` takes this transport curl, the physical negative-semidefinite
vorticity Laplacian, div J, curl H, curl E, and the shared current J.
`momentum_rate` supplies the same force for diagnostic pressure reconstruction.
Pressure and incompressibility forward to the fluid sector.
No thermal state is resolved; Joule and viscous heat enter an unresolved thermal
reservoir and are not claimed as conserved resolved energy.

## Compressible FMHD

The evolved state is `(rho,rho u,total_energy,rho_q,D,B)`, with

```
total_energy = rho e + rho |u|²/2 + W(D,B).
```

Both field energies are subtracted by `fluid_state` before the EOS evaluates
thermal variables. `primitives` and `pressure` delegate to the fluid with this
correct energy mapping.

Fluid momentum is rho u and receives the source f. It does not include field
momentum D cross B. Adding electromagnetic stress to its flux alone would omit
the field-momentum time derivative, so the full Lorentz force is retained as
an explicit source rather than borrowing the quasistatic MHD momentum flux.

The total-energy flux is

```
F_total = (fluid_energy+p)u - tau.u + q_heat + E cross H.
```

Maxwell's energy equation has source -J.E; the fluid energy equation has source
+J.E. Those internal exchanges cancel in the combined total-energy rate.
`flux(state,grad_u,grad_T,normal)` returns the fluid flux with Poynting energy
added, plus total current for numerical reduction. The coupled `operator()`
receives the combined flux divergence, div J, curl H, curl E, and that shared J.
It adds the Lorentz momentum source, updates all fields and charge, and keeps
the combined energy rate equal to the negative total-flux divergence.

`fluid_energy_source(state,J)` exposes J.E for formulations evolving fluid energy
separately; `ohmic_heating(state)` reports irreversible heating. Neither is
added again to combined total energy. Numerical conservation additionally
requires consistent cochain operators and matching force/power interpolation.

## Build and validation

The umbrella header exposes all FMHD types. Existing generated electric and
magnetic Hamiltonians are dependencies of the common physics generation target.
The Python `electromagnetism.py` composes their symbolic definitions for further
analytical work. All eight Docker-run `electromagnetism_symbolic_validation`
checks pass, covering combined energy derivatives, ideal/resistive current and
power, Gauss/Faraday compatibility, total-energy partition, and Maxwell wave
speed. C++ validation is registered as `unit_tests_fmhd`; compilation and
numerical simulations remain deferred, so these C++ checks have not been run.
