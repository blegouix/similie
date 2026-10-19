<!--
SPDX-FileCopyrightText: 2026 Baptiste Legouix
SPDX-License-Identifier: AGPL-3.0-or-later
AI-GENERATED
-->

# Taylor–Green vortex {#taylor_green_vortex}

\important This documentation is fully AI-generated.

The Taylor–Green vortex starts from a smooth array of counter-rotating vortices
in a periodic cube, \f$[-\pi,\pi]^3\f$. Their interaction transfers kinetic
energy to smaller scales, where viscosity dissipates it. This makes the problem
useful both for checking a discretization and for measuring its cost as the
mesh is refined. The hydrodynamic case follows the
[Re=1600 workshop specification](../benchmarks/taylor_green_vortex/BS1_TaylorGreenVortexRe1600_0.pdf).
The electromagnetic cases are extensions for checking conservation and coupling;
the workshop's reference histories apply only to hydrodynamics.

## Fluid problem

With reference density and velocity equal to one, the initial velocity is
\f[
 u=(\sin x\cos y\cos z,-\cos x\sin y\cos z,0).
\f]
The Reynolds number is 1600. Compressible flow uses a perfect gas with
\f$\gamma=1.4\f$, Mach number 0.1 and Prandtl number 0.71. Its initial pressure is
\f[
 p=p_0+\frac{(\cos 2x+\cos 2y)(\cos 2z+2)}{16},
 \qquad p_0=\frac{1}{\gamma(0.1)^2}.
\f]
Temperature is initially uniform, so density follows from \f$\rho=p/p_0\f$.
Incompressible flow instead fixes \f$\rho=1\f$ and obtains pressure from the
momentum constraint. Both start with mean kinetic energy \f$1/8\f$.

## Electromagnetic extensions

Each field sector uses its Hamiltonian to obtain the constitutive fields:
\f$E=\partial H/\partial D\f$ and \f$H_B=\partial H/\partial B\f$.

| Selection | Field model and coupling |
| --- | --- |
| `--hd` | Navier–Stokes without electromagnetic fields. |
| `--ehd` | Electric quasistatic limit: solve Gauss's law for the potential, advect charge, and apply the electric force. Magnetic feedback is omitted. |
| `--mhd` | Magnetic quasistatic limit: advance magnetic flux with Faraday's law and an Ohm-law electric field; omit displacement current and apply the magnetic force. |
| `--fmhd` | Full Maxwell evolution of electric displacement and magnetic induction, with charge conservation and the Lorentz force. |

The benchmark uses \f$\epsilon=\mu=1\f$. Electric cases start from
\f$\phi=0.1\cos x\cos y\cos z\f$; charge is initialized from discrete Gauss's
law. Magnetic cases start from
\f$B=0.25e_z+\nabla\times(0,0,0.15\sin x\sin y\sin z)\f$.

`--ideal` means zero resistivity for MHD, and zero conductivity with advected
charge for EHD and FMHD. `--resistive` adds Ohmic conduction: electrical
resistivity is 0.001 for MHD and 10 for EHD and FMHD. The fluid viscosity and
heat conduction remain active in both selections. Full Maxwell propagation
remains active in ideal FMHD; its wave speed here is one.

## Discretization and use

`--N` selects the number of unique periodic nodes in each direction, with
spacing \f$2\pi/N\f$. The code stores vorticity and magnetic flux as two-cochains
and calls the exterior module for reduction, reconstruction, exterior derivative,
codifferential and Hodge Laplacian. Pointwise physics supplies material laws and
forces. The compressible solver currently adds a centered two-point face flux;
it is not positivity preserving. Replacing that flux requires a compatible
nonlinear transport operator, beyond the linear DEC operators.

Build with `SIMILIE_BUILD_BENCHMARKS=ON`, then run the executable from the build
directory, for example:

```sh
./benchmarks/taylor_green_vortex/taylor_green_vortex --N=64 --incompressible --ideal --hd
./benchmarks/taylor_green_vortex/taylor_green_vortex --N=64 --compressible --resistive --mhd
```

The default is incompressible ideal HD. Select one flag from each model group;
resistive HD is invalid. `--validate` performs the small-grid consistency checks.
Google Benchmark controls repetitions and timing. Each timed iteration advances
one adaptive step along a trajectory; inspect simulated time as well as timing
when comparing cases. Use energy histories, dissipation, Gauss residuals and
cochain closure errors to assess accuracy before drawing performance conclusions.
