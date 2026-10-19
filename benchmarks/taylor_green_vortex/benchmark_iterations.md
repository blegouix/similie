<!--
SPDX-FileCopyrightText: 2026 Baptiste Legouix
SPDX-License-Identifier: AGPL-3.0-or-later
AI-GENERATED
-->

# Benchmark iterations and evolving simulation state

\important This operator and documentation is fully AI-generated.

One Google Benchmark iteration advances **one adaptive solver time step**.
The simulation is initialized before the iteration loop, and its fluid and
electromagnetic state carry over between iterations. Each step contains all
three SSPRK3 stages. The stability-based time step remains a function of N,
the selected physics, and the current solution; benchmark controls determine
the workload length rather than overriding the numerical stability limit.

For exactly 1000 time steps in a single trajectory:

```sh
taylor_green_vortex --N=64 --compressible --resistive --mhd \
    --benchmark_min_time=1000x --benchmark_repetitions=1
```

The `x` suffix requests an explicit iteration count. With a duration such as
`--benchmark_min_time=2s`, Google Benchmark instead chooses the number of steps
from the solver timing. Its calibration trials and separate repetitions invoke
the benchmark afresh, initializing separate trajectories from t=0 and producing
distinct output files. State persists within each invocation, not between them.
Use an explicit count when comparing identical physical trajectories across runs.
See the [Google Benchmark runtime guide](https://google.github.io/benchmark/user_guide.html#runtime-and-reporting-considerations).

`SetIterationTime` receives each step's solver duration. Timestep selection,
reconstruction, field evolution, fluid integration, and the completion fence
are included. Monitoring and export occur while Google Benchmark timing is
paused. The reported benchmark time is therefore the mean solver duration
per time step. The `steps` counter equals Google Benchmark's iteration count;
the driver checks this equality. `simulation_time` and `mean_dt` record the
resulting physical interval. Other duration/iteration counters are totals for
that trajectory. Performance JSON includes final time and mean solver step
duration; throughput still counts cell updates per solver second.

Physical time is no longer capped at t=10. The solver shortens a step to land
on t=8 and t=10 when it reaches them, exports those snapshots, then continues
while benchmark iterations remain. Every invocation exports its initial and
final states, without duplicating a final snapshot already taken at a target.
A fixed iteration count can end before either target or continue beyond both.

For the two complete HD reference campaigns, `--reference` instead runs the
same time-step operation until exactly t=10, with solver performance recorded
in JSON. This mode is restricted to HD. `--no-snapshots` suppresses field and
coordinate output for short performance runs while retaining HDF5 monitors.
See the [benchmark documentation](../../docs/taylor_green_vortex.md) for the physical models.

Post-processing accepts these shorter and longer trajectories. It checks that
the final snapshot matches the final monitored time. Spectra and the vorticity
slice use t=8 when reached; otherwise they use the final snapshot and label
its actual time. Short-run spectral/slice plots have suffix `.final.png`;
the workshop t=8 plots retain `.t8.png`. Comparison legends identify spectral
times, and performance plots normalize durations by step count.

The native `tgv_benchmark_iterations_validation` CTest runs one-step and
three-step HD trajectories for both fluid formulations. It checks iteration
accounting, state evolution, continuation after the first step, and reset
between separate repetitions. The test requires the newly compiled binary
and must run in Docker. Python tests separately cover partial trajectories,
t=8 selection in runs beyond t=10, and missing-final-snapshot rejection.
