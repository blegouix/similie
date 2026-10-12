<!--
SPDX-FileCopyrightText: 2026 Baptiste Legouix
SPDX-License-Identifier: AGPL-3.0-or-later
AI-GENERATED
-->

# Taylor–Green vortex at Re=1600

\important This operator and documentation is fully AI-generated.

This benchmark implements the structured-grid campaign in
[the workshop reference](BS1_TaylorGreenVortexRe1600_0.pdf), using native SimiLie
DEC incidence operators. It compares incompressible and compressible
Navier–Stokes on a periodic Cartesian grid. It uses no polynomial basis,
high-order reconstruction, or turbulence model. The implementation has not yet
been compiled or run: numerical results and performance figures must be validated
when the updated Docker image is available.

## Fixed problem and available choices

`--N` is the number of **unique periodic nodes in each direction**, with
`h=2π/N` and coordinates `-π+i h`, `i=0,...,N-1`. The upper endpoint is not
duplicated. N defaults to 64; the reference campaign uses 64, 128, and 256.
The implemented index range is 8≤N≤512. Storage and work scale as N³.

`--physics=incompressible` (default) or `--physics=compressible` chooses the
solver. `resistive_mhd` and `two_fluid_plasma_mhd` are reserved names that fail
explicitly until implemented. N and physics are the only simulation choices;
output location and Google Benchmark repetition/reporting flags do not change
the physical problem or the discretization.

Use nondimensional reference quantities ρ₀=V₀=L=R=1, μ=1/1600,
γ=1.4, Pr=0.71, M₀=0.1, and p₀=1/(γM₀²). The perfect-gas γ=1.4 is the
calorically perfect air assumption. The initial condition is

```
u = sin(x) cos(y) cos(z)
v = -cos(x) sin(y) cos(z)
w = 0
p = p₀ + [cos(2x)+cos(2y)] [cos(2z)+2] / 16.
```

For incompressible flow ρ=1. For compressible flow the reference prescribes
**uniform temperature**, so T=T₀=p₀/R and ρ=p/(R T₀); do not initialize a
uniform density together with the nonuniform pressure. Conservative variables
are `(ρ,ρu,ρv,ρw,ρE)`, with `ρE=p/(γ-1)+ρ|u|²/2`.
The final time is t/tc=10; a time step is shortened to land exactly on t=8
and t=10. The expected initial volume-averaged kinetic energy is 1/8 for both
initializations, including the density weighting.

## Physics and benchmark architecture

The library contains two pointwise, reusable physics classes:

- `similie::physics::fluid::IncompressibleNavierStokes(density, viscosity)` in
  `src/similie/physics/fluid/incompressible_navier_stokes.hpp`: rotational
  momentum balance, incompressibility residual, and physical pressure recovery.
- `similie::physics::fluid::NavierStokesFourier(gamma, prandtl, viscosity, R)` in
  `src/similie/physics/fluid/navier_stokes_fourier.hpp`: the mass, momentum, and
  total-energy equations, perfect-gas primitives, full physical flux, Newtonian
  stress, and Fourier/Prandtl heat-flux closure. Its conservation-law call takes
  the three physical flux divergences and returns the conservative time rates.

Neither class contains grid dimensions, a solver selector, workshop constants,
or a time-integration scheme. `Physics` is declared in the benchmark driver;
fixed workshop values are in benchmark `parameters.hpp`. `solvers.hpp` supplies
numerical flux construction and the pressure solve. `structured_grid.hpp`
provides benchmark storage, periodic sampling, and field placement while
calling the native exterior operators. There is no separate Cartesian DEC
operator implementation in the physics library.

The existing ONELAB potential-flow implementation wraps the free scalar-field
Hamiltonian and obtains velocity from a scalar-potential gradient. Such a field
is irrotational in the bulk (`d(d phi)=0`), whereas Taylor–Green has nonzero
vorticity and evolves viscous, vortical momentum. It therefore cannot replace
incompressible Navier–Stokes. Its scalar elliptic formulation is relevant to
pressure projection, which remains a constraint solve within the full fluid
system.

## Native exterior operators and field placement

Spatial operations delegate to the exterior module:

- `deriv` evaluates scalar gradients and the exterior derivative of a velocity
  one-cochain for vorticity.
- `Reduction` and `Reconstruction` map physical fields to integrated cochains
  and back using the position field.
- `StagedCodifferential` supplies flux divergence and the incompressibility
  constraint, including metric-dependent Hodge stages.
- `StagedLaplacian` supplies viscous diffusion using the native scalar Hodge
  Laplacian componentwise. Geometry and Hodge stars are assembled by exterior;
  the benchmark does not insert diagonal Hodge coefficients or implement its
  own Laplacian stencil.

The Euclidean metric and an **unwrapped coordinate halo** describe the periodic
box. Stored cochains use logical indices 1,...,N, allowing index zero in the
geometry halo. Field samples wrap across the seam; geometry continues linearly
through it. Consequently no native geometric edge acquires an erroneous length
`2*pi-h`. Scalar storage still corresponds to the N unique periodic points.

The module uses the positive Hodge Laplacian `Delta_H=d delta+delta d` and the
codifferential `delta=-div` on physical one-forms. The benchmark applies
`-Delta_H` for viscous diffusion and `-delta` for physical flux divergence.
On the uniform Euclidean mesh, the native operators reduce to the familiar
incidence/Hodge formulas; that simplification is used for independent checks,
not for implementation.

Arithmetic averages only place fields: scalar exterior derivatives are
reconstructed on edges, then incident edge gradients are averaged to nodes;
velocity is averaged to edge locations before reduction; exterior `d u_flat`
is reconstructed on faces, and four incident face values are averaged to a
nodal vorticity component. Vector face fluxes are reduced and passed to native
codifferential. The centered nodal gradient and divergence are adjoints in the
constant-volume inner product. The pressure operator is their composition.

For the initial Taylor–Green modes, this edge/face placement gives nodal
vorticity equal to the continuous initial vorticity times
`[sin(h)/h] cos²(h/2)`. Thus the initial discrete enstrophy is
`(3/8)[sin(h)/h]² cos⁴(h/2)` for incompressible flow, approaching 3/8 as N grows.
The curl is obtained from the native one-cochain exterior derivative, not from
an independently coded Cartesian curl formula.

## Incompressible equations and projection

Write the momentum equation in rotational form,

```
∂t u = u × ω - G Π + ν Δ_h u,
ω = reconstructed exterior curl of the velocity one-cochain,   Π = p + |u|²/2,   div_C u = 0.
```

Every RK stage evaluates `f=u×ω+νΔ_h u` and projects its acceleration:
solve `(-D G)Π=-D f`, then return `f-GΠ`. Here `G` is the native scalar
exterior derivative reconstructed and averaged to nodes; `D` is the native
codifferential with its physical divergence sign after velocity placement.
On this uniform grid each component reduces to
`C_a f(i)=[f(i+e_a)-f(i-e_a)]/(2h)`, so `G=(C_x,C_y,C_z)` and
`D u=Σ C_a u_a`. These formulas describe the resulting operators, rather
than additional implemented stencils. `Δ_h=-Delta_H` is the physical
diffusion Laplacian. The solve is matrix-free
conjugate gradient, starts at zero, and checks its residual; failure is reported,
never silently accepted. The dynamic pressure is Π−|u|²/2, shifted to zero
volume mean. Temperature is inactive, and exported heat flux is zero.

`IncompressibleSolver::compute_pressure(velocity, output, mean_pressure)`
reconstructs pressure independently of time integration, leaving the supplied
velocity unchanged. It evaluates the unprojected acceleration in a dedicated
workspace, solves the same elliptic Bernoulli-pressure problem, and recovers
physical pressure using the library physics class. All three directions have
periodic boundary conditions, consistent with the velocity and Taylor–Green
domain. The additive constant is fixed by the requested volume mean; exports
use `mean_pressure=p₀`. Zero-start CG selects zero unresolved null components
of the Bernoulli potential, including checkerboard modes on even grids.

Snapshot export explicitly invokes this reconstruction for the current state,
so it never relies on pressure left over from a previous Runge–Kutta stage.
The t=0 incompressible snapshot also uses reconstructed discrete pressure;
it approaches the prescribed analytic initial pressure as the grid is refined.
Compressible pressure is obtained directly from the equation of state. Pressure
reconstruction during snapshot export is counted in export time, separately
from the timed integration and its projection iterations.

Since D=−Gᵀ, this is an orthogonal projection. Starting with a projected velocity,
all RK stage derivatives remain in the divergence-free subspace. Pointwise
`u·(u×ω)=0`, so inviscid transport does no semi-discrete kinetic-energy work.
Viscosity removes kinetic energy through the negative of the native positive
Hodge Laplacian.

The centered projection has mean null modes and, for even N, checkerboard null
modes. The divergence RHS lies in the operator range and zero-start CG selects a
solution there. These modes are not removed by pressure projection; the compact
viscous Laplacian does damp velocity checkerboards. This is a collocated,
second-order Cartesian discretization, not a staggered pressure/velocity method.

## Compressible equations, stress, and heat-flux closure

The conservative equations are

```
∂t ρ       + div(ρu)                       = 0
∂t (ρu)    + div(ρu⊗u+pI-τ)                = 0
∂t (ρE)    + div[(ρE+p)u-τ·u+q]            = 0
p = (γ-1)(ρE-ρ|u|²/2),     T = p/(ρR)
τ = μ [grad u+(grad u)ᵀ-(2/3) div(u) I]
q = -κ grad T,    cp=γR/(γ-1),    κ=μ cp/Pr.
```

The last relation is the requested Prandtl/Fourier closure. Heat conduction is
part of the energy equation, not a post-processing approximation. Stokes' bulk
viscosity assumption is used. Each face gets one flux, used with opposite signs
in its two neighboring control volumes, so periodic mass and total energy are
conserved to roundoff, including viscous work and heat transfer.

For the inviscid flux, overbars denote arithmetic face means:

```
Fρ = ρ̄ ū_n
Fm = Fρ ū + p̄ n
FE = Fρ [ē + (u_L·u_R)/2] + p̄ ū_n,
e = p/[(γ-1)ρ].
```

The momentum flux is kinetic-energy preserving: the transport contribution to
the global kinetic-energy equation cancels because
`(u_R-u_L)·ū=(|u_R|²-|u_L|²)/2`. Pressure work remains, as it should.
This centered two-point flux is second order and has no reconstruction,
limiter, or acoustic artificial viscosity. Acoustic Rusanov dissipation would
be particularly intrusive at M₀=0.1. The centered scheme has no positivity or
entropy-stability guarantee; the driver detects invalid density/pressure and
nonfinite monitors and reports failure. Runtime robustness at the intended N
remains to be checked. A coarse run is not DNS merely because its physical
Reynolds number is 1600.

The face stress uses reconstructed native exterior gradients in the normal
direction and averaged node gradients tangentially. The energy flux includes
`-τ·ū-κ grad_n T`. No clipping, density repair, or hidden change of viscosity is
applied.

## Time discretization

Both solvers use SSPRK3:

```
q1 = qn + dt F(qn)
q2 = 3/4 qn + 1/4 [q1 + dt F(q1)]
qn+1 = 1/3 qn + 2/3 [q2 + dt F(q2)].
```

The step is chosen automatically from fixed constants:

```
dt = min(0.3 h / max_i Σ_a (|u_a|+c), 0.1 h² / max_i νeff)
c = sqrt(γp/ρ) for compressible flow, otherwise 0
νeff = max(4μ/3, κ/cv)/ρ,   cv=R/(γ-1).
```

For incompressible flow use ρ=1 and omit thermal diffusivity. These are fixed
stability bounds, not user parameters. Events at 8 and 10 only shorten the step.
SSPRK3 does not imply positivity for a centered spatial flux. Third-order time
integration does not make this a high-order spatial method.

## Monitors and files

Each computation has one base name `tgv_<physics>_N<N>_run<id>`. Existing HDF5
bases are skipped to preserve previous runs. All quantities are nondimensional.

At **every integration step**, including the initial condition, export

```
Ek          = mean(ρ |u|²/2) / ρ₀
𝓔           = mean(ρ |ω|²/2) / ρ₀
ε_enstrophy = 2 μ 𝓔 / ρ₀
ε_kinetic   = -dEk/dt.
```

The signed `dEk/dt` is calculated from the same projected/conservative RHS used
by SSPRK3. For incompressible flow it is `mean(u·F_u)`. For compressible flow it
is `mean(u·F_m-|u|² F_ρ/2)`. This is the scheme's instantaneous semi-discrete
energy derivative, not a difference of sparse snapshot energies. SSPRK3 has a
finite-step energy defect; the saved energy history can be used to assess it.
At finite resolution `ε_enstrophy` need not equal `ε_kinetic`: the former uses
the reconstructed and averaged exterior curl while viscosity uses the native
Hodge Laplacian without that nodal placement average.

The HDF5 history also includes `mean(p div u)`, `mean(τ:grad u)`, mean mass and
total energy, RMS divergence, minimum density/pressure, and each solver step's
wall time. The continuous identity is
`-dEk/dt = mean(τ:grad u)-mean(p div u)`; the plotted residual also reflects the
different face/node monitor operators and time discretization, so it should
not be interpreted solely as artificial viscosity.

Files:

- `.dat`: workshop header and `time Ek dEk/dt epsilon_enstrophy` at each step.
- `.h5`: `position[N,N,N,3]`, a `monitors[steps+1,14]` table whose
  `monitor_columns` attribute names each column, and `snapshot_0`,
  `snapshot_1`, `snapshot_2` at t=0,8,10. Each group contains time, velocity,
  vorticity, density, pressure, temperature, and heat flux. Vector fields are
  `[N,N,N,3]`, scalars `[N,N,N]`.
- `.xmf`: temporal XDMF view of the HDF5 fields, readable by ParaView, as in the
  scalar-field example. HDF5 is written directly through its C API here.
- `.performance.json`: step count, solver/monitor/export/wall seconds,
  time-stepping CG iteration count, and cell updates per solver second.
- Post-processing creates `.spectrum.dat`, `.vorticity_slice.dat`,
  `.monitors.dat`, PNG plots, and one legacy structured-grid `.vtk` per snapshot.

At exactly t=8, the script takes the face at x/L=−π (index zero) and computes
`|ω|L/V₀`. It also performs a full complex FFT of `sqrt(ρ/ρ₀) u`, using
normalization `1/N³`, so the all-mode energy equals Ek by Parseval's identity.
For constant density this is the usual velocity spectrum. For variable density
this is a density-weighted kinetic-energy spectrum, and the convention is
explicit in the file header. Modes are summed into integer spherical shells
with half-integer boundaries, up to floor(N/2). Omitted diagonal corner-mode
energy is reported separately. A periodic endpoint is added only to the plotted
slice, never to the FFT data.

## Performance measurement

One Google Benchmark iteration is a **complete t=0→10 computation**, with one
iteration by default. `UseManualTime` receives accumulated solver time and
`SetIterationTime` is called once per iteration, following the
[Google Benchmark manual timing API](https://github.com/google/benchmark/blob/main/docs/user_guide.md#manual-timing).
Kokkos is fenced before each solver interval ends. Timed work includes the
stability bound, three RHS evaluations, projection solves, and RK updates.
Allocations and initialization precede the wall-time interval; diagnostics,
HDF5/text output, and spectra/plots are excluded from solver timing. Monitor
and export times are reported separately. Wall time includes monitor and export
work but excludes later Python post-processing.

`cell_updates/s` counts completed steps times N³, not RHS stages. Acoustic
steps mean compressible flow can require many more steps than incompressible
flow; compare both the full physical trajectory cost and the per-cell-step
throughput. `step_seconds` in HDF5 records varying per-step solver cost. The
Python comparison plot shows solver/monitor/export wall costs and throughput;
Google Benchmark can report repetitions and JSON aggregates separately.
Google Benchmark’s manual-time column measures solver work; its CPU-time column
also includes work outside those manually measured intervals and should not be
used as solver-only CPU timing. Execution space and physical constants are added
to the benchmark report context. No timing numbers are claimed before building
and running.

## Build and run after rebuilding the CPU image

Follow AGENTS.md: build and run **inside Docker**, with the repository mounted
at the same absolute path, and never build with more than eight jobs. The current
Dockerfile supplies GCC 15; use it rather than assuming GCC 13 is installed.
The benchmark is opt-in (`SIMILIE_BUILD_BENCHMARKS=ON`) and requires Ubuntu's
`libbenchmark-dev` and `libhdf5-dev`. Post-processing requires NumPy, h5py, and
Matplotlib. If these last packages are absent, add/install `python3-numpy`,
`python3-h5py`, and `python3-matplotlib` in the image before post-processing.

Build the CPU image with both arguments:

```sh
docker build --build-arg BACKEND=cpu --build-arg BASE_IMAGE=ubuntu:26.04 \
  -t similie_env_cpu -f docker/similie_env/Dockerfile .
```

Inside the container, from `/home/blegouix/similie`:

```sh
cmake -DCMAKE_CXX_COMPILER=g++-15 -DCMAKE_BUILD_TYPE=Release \
  -DKokkos_ENABLE_OPENMP=ON -DSIMILIE_BUILD_BENCHMARKS=ON \
  -B agent_build_cpu -S .
cmake --build agent_build_cpu -j 8
ctest --test-dir agent_build_cpu --output-on-failure
cd agent_build_cpu
OMP_NUM_THREADS=8 OMP_PROC_BIND=spread OMP_PLACES=cores \
  benchmarks/taylor_green_vortex/taylor_green_vortex \
  --N=64 --physics=incompressible --benchmark_repetitions=3 \
  --benchmark_out=tgv_incompressible_N64.benchmark.json --benchmark_out_format=json
OMP_NUM_THREADS=8 OMP_PROC_BIND=spread OMP_PLACES=cores \
  benchmarks/taylor_green_vortex/taylor_green_vortex \
  --N=64 --physics=compressible --benchmark_repetitions=3 \
  --benchmark_out=tgv_compressible_N64.benchmark.json --benchmark_out_format=json
python3 ../benchmarks/taylor_green_vortex/post_process.py \
  taylor_green_vortex_results/*.h5
```

Repeat for N=128 and 256. Output directory defaults to
`taylor_green_vortex_results` relative to the working directory; override it
with `--output-dir=<directory>`. `--no-vtk` skips large ASCII VTK files, and
`--no-plots` skips Matplotlib entirely while preserving numerical tables.

## Pending validation

CTest registers small N=8 validations for both solvers. `--validate` also works
with a chosen N. These checks cover initial Ek, discrete initial enstrophy,
periodic Fourier gradient/Laplacian symbols in all three directions, the
Laplacian of a constant, projection of an exact gradient, uniform-gas steady
state, the sign and magnitude of Prandtl/Fourier diffusion, five RK steps, mass/total-energy conservation, divergence, and kinetic
energy decay. A Python CTest (when NumPy and h5py are available) checks independent analytic
FFT energies, density weighting, cutoff accounting, VTK coordinate ordering,
and the HDF5 post-processing/slice outputs. These checks are implemented but
**not executed**.

Before accepting results, run those tests in the rebuilt Docker image, then run
small complete t=10 trajectories and post-processing, inspect all three
snapshots, check FFT Parseval normalization and the VTK coordinate ordering,
and perform the reference 64³/128³/256³ convergence campaign. Verify conservation,
projection residuals, positivity, the discrepancy between the two dissipation
monitors, and sensitivity to fixed temporal resolution as part of numerical
validation. Performance comparisons require Release builds and a recorded CPU,
thread count, affinity, compiler, and backend.
