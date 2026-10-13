<!--
SPDX-FileCopyrightText: 2026 Baptiste Legouix
SPDX-License-Identifier: AGPL-3.0-or-later
AI-GENERATED
-->

# Taylor–Green continuous integration

\important This operator and documentation is fully AI-generated.

`.github/workflows/benchmark_taylor_green_vortex.yml` defines one matrix job
for each supported fluid/transport/coupling combination. The matrix contains
14 jobs: two ideal HD fluids, and four combinations each for EHD, MHD, and FMHD.
Resistive HD is excluded because it has no electrical transport equation.
Every job covers N=64,128,256, the three Cartesian resolutions in the supplied
[reference document](BS1_TaylorGreenVortexRe1600_0.pdf).

## Short performance campaign

Each configuration runs exactly five evolving SSPRK3 steps per repetition,
with three repetitions at each resolution. Google Benchmark receives
`--benchmark_min_time=5x --benchmark_repetitions=3`. A repetition starts from
the prescribed initial state; each iteration advances that state once.
Stability limits determine the physical time step. These short trajectories
are performance measurements and do not claim the complete workshop dynamics.

The job first runs the configuration's N=8 numerical validation. It then checks
each performance trajectory for iteration accounting, finite monitors,
positive density/pressure, initial kinetic-energy normalization, mass
conservation, and applicable DEC reconstruction/Gauss/magnetic constraints.
All three resolutions are attempted even if an individual mesh case fails.
An initial N=8 validation failure prevents measurements of that invalid solver.

`--no-snapshots` suppresses coordinate and field snapshots for this campaign.
HDF5 monitor histories, energy tables, raw Google Benchmark JSON, and separate
solver/monitor/export performance JSON are retained. This avoids large field
artifacts for every short repetition while keeping the same solver and monitor
definitions. I/O controls do not alter any numerical parameter.

## Complete HD reference campaign

Only the `incompressible_ideal_hd` and `compressible_ideal_hd` jobs also run full
trajectories, once at each of the same three resolutions. They use `--reference`
with the original workshop initial conditions, periodic box, Re=1600, and
compressible Mach=0.1 and Pr=0.71. The native DEC spatial discretization and
SSPRK3 time integration remain those requested for SimiLie.

Reference mode uses the same time-step operation and timing code as the Google
Benchmark loop, but continues until exactly t=10 rather than stopping at a
timing iteration budget. Steps land on t=8 and t=10. Its performance JSON records
the actual number of steps, total solver time, and mean step time. This mode is
restricted to HD; combining it with `--validate` is rejected.

Full runs retain HDF5/XDMF fields at t=0,8,10 and every monitor. CI requires the
complete interval and all three snapshots before post-processing. The required
t=8 density-weighted energy spectrum and x=-pi vorticity slice, monitor tables,
and plots are generated. ASCII VTK is omitted from CI artifacts because XDMF
already exposes the same HDF5 fields; it can be generated later with
`post_process.py` if needed.

For example, a full reference run is:

```sh
taylor_green_vortex --N=128 --compressible --ideal --hd --reference
```

## Build, scheduling, and performance reports

The preparation job compiles the benchmark once, in Release/OpenMP mode inside
the commit-tagged CPU image `similie_env_cpu:<sha>`. It uses at most eight
compiler jobs and runs selector, post-processing, CI-evidence, and native
benchmark-iteration tests. The executable and any build-local shared libraries
are transferred to the physics jobs as an artifact. Each job uses an isolated
directory and the matching commit image, so matrix jobs do not edit a shared
build tree.

Physics jobs use eight OpenMP threads with spread/core placement, and run one
at a time within the matrix to avoid simultaneous benchmark workloads.
Solver progress logging is disabled. Raw Google Benchmark reports retain
hardware and execution-space context. Timings include timestep selection,
elliptic reconstruction, and integration, while monitor/export costs are
recorded separately.

Each job publishes a GitHub step summary with resolution, steps, repetitions,
final physical time, mean milliseconds per solver step, standard deviation
across repetitions, and million cell updates per solver second. For complete
reference runs there is one trajectory, so no repeated-run variation is
estimated. Detailed timings and Krylov counts remain in the artifacts.
Failed cases include their logs and partial manifest in the artifact, and the
matrix uses `fail-fast: false` so other physics combinations still run.

The existing main/PR CI calls this workflow after its CPU build. It can also
run after the standalone `build` workflow, or be dispatched manually with a
full commit SHA whose CPU image is already present on the self-hosted runner.
No workflow is dispatched or published by adding these files locally.
