<!--
SPDX-FileCopyrightText: 2026 Baptiste Legouix
SPDX-License-Identifier: AGPL-3.0-or-later
AI-GENERATED
-->

# Electric-only electrohydrodynamics

\important This operator and documentation is fully AI-generated.

EHD couples fluid motion and charge transport to an electrostatic field.
There is no magnetic induction state or magnetic force in these models.
`CombinePhysics` composes the fluid and electric sectors:

| Type in `similie::physics::ehd` | Fluid | Electric sector |
|---|---|---|
| `IdealIncompressibleEHD` | `IncompressibleNavierStokes` | `Electrodynamics` |
| `IdealCompressibleEHD<Thermodynamics>` | `NavierStokesFourier<Thermodynamics>` | `Electrodynamics` |
| `ResistiveIncompressibleEHD` | `IncompressibleNavierStokes` | `ResistiveElectrodynamics` |
| `ResistiveCompressibleEHD<Thermodynamics>` | `NavierStokesFourier<Thermodynamics>` | `ResistiveElectrodynamics` |

Each combination takes its two configured sector objects. `get<0>()` is fluid
and `get<1>()` electric. Thermodynamics remains caller-supplied.

The user-selected ideal convention is advected charge with **zero conductivity**.
Finite positive electrical resistivity adds conduction and heating. Thus the
nonconducting ideal EHD limit is eta->infinity. The singular eta=0 perfect
conductor is a different constrained model. Ideal MHD uses a different Ohm-law
limit; the names do not imply identical resistivity limits in both formulations.
Fluid viscosity and heat conduction remain independently configured.

## Electric Hamiltonian and field reconstruction

`electrodynamics/electrodynamics.py` generates an independent electric-only
Hamiltonian with the project's existing SymPy generator:

```
W(D) = |D|²/(2 epsilon),
E = δW/δD = D/epsilon,
E = -grad phi,
div D = rho_q,
-div(epsilon grad phi) = rho_q.
```

Here rho_q is charge density, distinct from fluid mass density rho.
Permittivity is constant and scalar in this baseline implementation. The
potential phi enforces Gauss's constraint. Stationarity of electric energy
subject to that constraint gives the electrostatic gradient relation; the
Hamiltonian supplies the local constitutive derivative E(D).

Electric displacement is reconstructed rather than independently advanced
with an Ampere equation. The numerical solver must solve the elliptic potential
problem at each required state/stage using the native exterior operators, then
reconstruct the gradient and displacement with compatible Hodge mappings.
`field_from_potential_gradient` supplies the negative sign and `displacement`
multiplies the reconstructed electric field by epsilon. `gauss_residual` checks
native div D minus rho_q. These pointwise physics classes do not assemble their
own Cartesian stencils or implement a second elliptic solver.

On a periodic domain, use zero mean phi and require zero net charge. With a
uniform scalar permittivity, the potential operator is epsilon times the
positive native scalar Hodge Laplacian. Electrode problems instead require
appropriate Dirichlet potential values; insulated boundaries prescribe normal
displacement. Potential reconstruction belongs to the numerical solver, so
its boundary conditions and gauge are not hardcoded into the library physics.
The scalar potential-flow/elliptic machinery is relevant to this reconstruction.

A local update D_t=-J generally introduces a rotational electric field and
cannot replace the potential solve in electrostatic EHD. Any harmonic imposed
electric field or electrode forcing must be supplied consistently with the
chosen boundary problem.

## Charge, force, and irreversible heat

The electric sector provides

```
J = rho_q u + j,
f = rho_q E,
(rho_q)_t = -div J.
```

For ideal EHD, j=0. For finite resistivity,

```
j = E/eta,
Q_Joule = j.E = |E|²/eta = eta |j|² >= 0,
J.E = u.f + Q_Joule.
```

Only conduction current enters irreversible heating. Charge advection carries
electric power through mechanical force. `Electrodynamics(epsilon)` and
`ResistiveElectrodynamics(epsilon,eta)` provide the two local closures.
Their `current`, `conduction_current`, `electric_force`, `fluid_energy_source`,
and `ohmic_heating` methods implement these quantities. Applying `operator()`
to the native current divergence returns the charge time derivative.

For zero motion and constant epsilon and eta, Gauss's law gives charge
relaxation at `1/(eta epsilon)`. `charge_relaxation_rate()` exposes this scale;
fluid transport and electric-force coupling impose additional time-step limits.
The physical current must be reduced and differentiated through native DEC.

## Incompressible combination

The evolved state is `(omega, rho_q)`:

```
omega_t = curl(u cross omega + rho_q E/rho) + nu Delta omega,
(rho_q)_t = -div J.
```

The fluid reconstruction obtains u from omega with the existing Coulomb-gauge
vector potential. A separate scalar-potential solve obtains E from rho_q.
`transport(state,u,D)` returns the vector inside the vorticity curl;
`operator()(curl_transport,Delta_omega,div_J)` returns the vorticity and charge
rates. Delta_omega uses the negative-semidefinite physical diffusion sign.
`momentum_rate` includes electric force for diagnostic pressure reconstruction.
Pressure recovery and incompressibility forward to the fluid sector.
No thermal state is evolved; Joule and viscous heat enter an unresolved thermal
reservoir, while Joule heat remains available as a diagnostic.

## Compressible combination and energy partition

The evolved state is `(rho, rho u, total_energy, rho_q)`.
**Here total_energy is fluid energy only**:

```
total_energy = rho e + rho |u|²/2,
(total_energy)_t = -div F_fluid + J.E.
```

The fluid EOS receives this energy unchanged. The electrical source J.E already
contains electric-force work and Joule heating; adding another Q_Joule would
double-count the heat. `combined_energy(state,D)` returns fluid energy plus
W(D) for monitoring, without treating field energy as thermal energy.

`flux(state,D,grad_u,grad_T,normal)` returns the ordinary fluid conservative flux
and total electric current. `operator()(state,D,fluid_flux_divergence,div_J,J)`
returns all rates, adding rho_q E to momentum and J.E to fluid energy.
Its explicit J argument lets the solver reuse the same current used for charge
transport and power transfer.

Electrostatic energy is not transported by a magnetic Poynting flux. If a
formulation evolves combined fluid-plus-electric energy, differentiating Gauss
and charge conservation gives

```
div(J+D_t)=0,
W_t + div[phi (J+D_t)] = -J.E.
```

Thus its combined energy flux would require the potential and displacement
**time derivative**, which follows from the time-dependent elliptic
reconstruction. The present model evolves fluid energy with its correct source
and monitors reconstructed electric energy separately. On a periodic domain
with compatible field reconstruction, the volume electric-energy exchange
cancels the volume fluid power; electrode boundaries may supply electrical work.
Discrete conservation requires matching current, field reconstruction, and
force/power interpolation.

## Build and validation

CMake generates `electrodynamics_hamiltonian.hpp` through
`similie_physics_generated`. The public umbrella header exposes both electric
closures and all four EHD combinations. No magnetic physics is included by
these EHD headers.

All eight `electrodynamics_symbolic_validation` checks pass in Docker, including
electric Hamiltonian derivatives, ideal advection, force/Joule work identities,
the infinite-resistivity limit, Poisson sign, charge relaxation, electrostatic
energy flux, and generated headers. C++ checks of all four combinations are
registered in `unit_tests_ehd` but have not been compiled or run. No electric
initial condition or electrode configuration is invented for the hydrodynamic
Taylor–Green benchmark.
