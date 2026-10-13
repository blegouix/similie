<!--
SPDX-FileCopyrightText: 2026 Baptiste Legouix
SPDX-License-Identifier: AGPL-3.0-or-later
AI-GENERATED
-->

# Composing fluid and magnetic physics

\important This operator and documentation is fully AI-generated.

`CombinePhysics<Physics...>` stores several device-compatible physics objects.
Its recursive storage avoids a runtime dependency on `std::tuple`; `get<I>()`
returns the corresponding sector by reference, with const and mutable overloads.
Its `size` is the number of sectors. Coupled models inherit this composition
and explicitly provide shared state mappings, interaction fluxes, and sources.
Combining independent RHS values alone would not produce MHD: the fluid must
feel magnetic stress, and both sectors must exchange energy consistently.

`mhd.hpp` exposes four coupled physics types in `similie::physics::mhd`:

| Type | Fluid sector | Magnetic sector |
|---|---|---|
| `IdealIncompressibleMHD` | `IncompressibleNavierStokes` | `Magnetodynamics` |
| `IdealCompressibleMHD<Thermodynamics>` | `NavierStokesFourier<Thermodynamics>` | `Magnetodynamics` |
| `ResistiveIncompressibleMHD` | `IncompressibleNavierStokes` | `ResistiveMagnetodynamics` |
| `ResistiveCompressibleMHD<Thermodynamics>` | `NavierStokesFourier<Thermodynamics>` | `ResistiveMagnetodynamics` |

Each is a specialization of a coupled model derived from `CombinePhysics`.
Sector zero is the fluid; sector one is magnetic. Construction takes the two
configured sector objects. Ideal means zero **electrical resistivity** here;
fluid viscosity and Fourier conductivity retain their independently configured
values. Setting them to zero gives the nondissipative ideal-fluid limit.

These are reusable physics models rather than new Taylor–Green trajectories.
The hydrodynamic workshop reference specifies no magnetic initial condition;
its benchmark solver selector is therefore not automatically extended to an
invented MHD case. No electromagnetic optimizer or FFT reconstruction is added.

## The electromagnetic Hamiltonian

`magnetodynamics/magnetodynamics.py` uses the project's existing SymPy
Hamiltonian generator to produce `MagnetodynamicsHamiltonian` in namespace
`similie::physics::magnetodynamics`. For constant, scalar, positive electric
permittivity epsilon and magnetic permeability mu, its energy density is

```
H(D,B) = |D|²/(2 epsilon) + |B|²/(2 mu),
E = δH/δD = D/epsilon,
H_field = δH/δB = B/mu.
```

`Magnetodynamics` obtains the constitutive fields from these generated
derivatives, rather than duplicating their laws in a fluid Hamiltonian.
The Maxwell field Poisson operator contains spatial curls:

```
D_t = curl H_field - J,
B_t = -curl E.
```

Equivalently, in temporal gauge one may use canonical vector potential A and
momentum Pi=-D, with B=curl A and Hamiltonian
`|Pi|²/(2 epsilon)+|curl A|²/(2 mu)`. Hamilton's functional equations give
`A_t=-E` and `Pi_t=-curl H_field+J`. The generated pointwise energy derivatives
are consequently not meant to be fed into a local canonical time update
without the spatial curl operator.

With prescribed current, the power balance is

```
∂t H + div(E cross H_field) = -J.E.
```

`hamiltonian({D,B})`, `electric_field(D)`, `magnetic_field(B)`, and
`operator()(curl_H, curl_E, J)` expose this full electromagnetic sector.
CMake generates its Hamiltonian header through `similie_physics_generated`;
the generated header includes its AI annotation and operator documentation.

## The MHD limit and Ohm law

The MHD combinations use the quasineutral, nonrelativistic limit. They evolve
magnetic induction B, exclude displacement current, and do not include electric
energy in the conserved MHD energy. The current and ideal electric field are

```
J = curl H_field,
E = -u cross B,
B_t = -curl E = curl(u cross B),
div B = 0.
```

`ResistiveMagnetodynamics(mu, eta, epsilon=1)` adds electrical resistivity:

```
E = -u cross B + eta J,
B_t = curl(u cross B) - curl(eta J),
Q_J = eta |J|².
```

For constant eta and mu on a closed magnetic field, the resistive term is
`+(eta/mu) Delta B`. Thus magnetic diffusivity is eta/mu; it is not the input
resistivity itself. Resistive evolution is dissipative even though its stored
electromagnetic energy still comes from the same Hamiltonian.

The caller evaluates J and Faraday's curl through native exterior operators.
In DEC, B is a closed two-cochain, E is an electric one-cochain, and Faraday is
`B_t=-d E`. Native `d²=0` preserves the magnetic-divergence constraint without
an elliptic projection when the initial B is closed. Hodge stars map between
physical constitutive fields and primal/dual cochains. Interpolation needed for
u cross B and J cross B must respect the chosen numerical placement; these
local physics classes do not impose a mesh or replace the exterior operators.
Periodic harmonic magnetic flux may be nonzero and must be retained. A solver
must not remove the mean B merely because the fluid vector potential uses a
zero-mean Coulomb gauge.

## Incompressible coupling

For fixed fluid density rho, the coupled equations are

```
omega_t = curl(u cross omega + J cross B/rho) + nu Delta omega,
B_t = -curl E,
div u = 0,  div B = 0.
```

`transport(u, omega, B, J)` returns the vector inside the vorticity curl.
The caller reduces it and evaluates that curl with exterior, then passes
`(curl_transport, Delta_omega, curl_E)` to `operator()` to obtain the rates of
vorticity and magnetic induction. The Laplacian input has the physical,
negative-semidefinite diffusion sign; it is the negative of the positive
native Hodge Laplacian. The existing fluid reconstruction supplies u from
vorticity using its Coulomb-gauge vector potential.

`momentum_rate(...)` includes J cross B/rho for diagnostic pressure recovery.
`pressure(Pi,u)` returns `rho(Pi-|u|²/2)`; its Poisson RHS must include the full
magnetic force supplied by this coupled momentum balance. `incompressibility`
forwards the native velocity-divergence residual to the fluid sector.

There is no thermal state in this incompressible model. Viscous and Joule heat
therefore enter an unresolved thermal reservoir; `ohmic_heating(J)` reports the
magnetic contribution rather than silently claiming conservation of resolved
kinetic-plus-magnetic energy under resistive evolution.

## Compressible conservative coupling

The combined state is `(rho, rho u, total_energy, B)`, where

```
total_energy = rho e + rho |u|²/2 + |B|²/(2 mu).
```

`fluid_state(state)` subtracts magnetic energy before the Navier–Stokes–Fourier
sector evaluates thermodynamics. Otherwise an EOS would interpret magnetic
energy as thermal energy and give incorrect pressure and temperature.
No particular thermodynamic relation is introduced by the magnetic sector.

The normal momentum flux adds magnetic stress

```
M = (B.H_field/2) I - B tensor H_field,
F_m = rho u tensor u + p I - tau + M.
```

For closed B and constant mu, `-div M=J cross B`. The total-energy flux is

```
F_energy = (fluid_energy+p)u - tau.u + q + E cross H_field.
```

For ideal Ohm law this is the familiar
`(total_energy+p+|B|²/(2 mu))u - (u.B)B/mu - tau.u + q`.
Resistivity adds the transport flux `eta J cross B/mu`.
`flux(state, grad_u, grad_T, J, normal)` returns this conservative flux and E;
the numerical solver applies divergence to the complete conservative flux and
native Faraday to E. `operator()(total_flux_divergence, curl_E)` returns all
conserved time rates explicitly. `fast_magnetosonic_speed_squared(state,normal)`
supplies the ideal hyperbolic characteristic speed for a stability estimate;
resistive and thermal diffusion impose additional explicit-step limits.

If a numerical formulation instead evolves **fluid** energy separately, its
magnetic source is

```
J.E = u.(J cross B) + eta |J|².
```

The standalone helper `fluid_energy_source(magnetic,u,B,J)` exposes this power,
and `ohmic_heating(J)` exposes the irreversible part. Neither is added as an
extra source to the combined total-energy equation: doing so would double-count
internal energy exchange. Discrete conservation requires the numerical curl,
magnetic force, interpolation, and energy flux to implement the corresponding
power identity; the local continuum formulas alone do not guarantee it.

## Validation

The Docker-run symbolic test `magnetodynamics_symbolic_validation` verifies
Hamiltonian derivatives and positivity of their diagonal coefficients, ideal
Poynting flux, Lorentz work, Joule heating, the zero-resistivity limit, total
energy mapping, Maxwell power balance, and the resistive Fourier-mode decay
sign. All seven symbolic checks have passed, and the generated Hamiltonian API
has been inspected.

The C++ target `unit_tests_mhd` checks three-sector composition, Hamiltonian
constitutive fields, all four MHD combinations, magnetic pressure/energy flux,
Lorentz coupling, and absence of duplicated Joule sources. C++ compilation and
numerical simulations remain deferred; these C++ checks have not been run.
