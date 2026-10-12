# SPDX-FileCopyrightText: 2026 Baptiste Legouix
# SPDX-License-Identifier: AGPL-3.0-or-later
# AI-GENERATED
"""Workshop monitors, isotropic spectra, periodic-face slices, and VTK exports."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

COLUMNS = (
    "time",
    "kinetic_energy",
    "dEk_dt",
    "kinetic_dissipation",
    "enstrophy",
    "enstrophy_dissipation",
    "pressure_dilatation",
    "viscous_dissipation",
    "mass",
    "total_energy",
    "divergence_rms",
    "density_min",
    "pressure_min",
    "step_seconds",
)


def energy_spectrum(velocity, density):
    """Shell-integrated energy of sqrt(rho/rho0)*u, with rho0=1.

    Parseval gives sum(E_shell)=mean(rho*|u|^2/2). A full complex FFT
    avoids one-sided multiplicity ambiguities. All modes are binned before
    returning integer shells through floor(N/2), the workshop cutoff.
    """
    n = velocity.shape[0]
    if velocity.shape != (n, n, n, 3) or density.shape != (n, n, n):
        raise ValueError("Expected a cubic grid and three velocity components")
    if np.any(density <= 0) or not np.all(np.isfinite(velocity)):
        raise ValueError("Spectrum requires finite velocity and positive density")
    transform = np.fft.fftn(np.sqrt(density)[..., None] * velocity, axes=(0, 1, 2))
    mode_energy = 0.5 * np.sum(np.abs(transform) ** 2, axis=-1) / n**6
    k = np.fft.fftfreq(n) * n
    k2 = k * k
    radius = np.sqrt(k2[:, None, None] + k2[None, :, None] + k2[None, None, :])
    shells = np.floor(radius + 0.5).astype(int)
    full = np.bincount(shells.ravel(), weights=mode_energy.ravel())
    physical = np.mean(0.5 * density * np.sum(velocity**2, axis=-1))
    if not np.isclose(full.sum(), physical, rtol=1e-12, atol=1e-14):
        raise ValueError("FFT spectrum failed Parseval's identity")
    cutoff = n // 2
    return np.arange(cutoff + 1), full[: cutoff + 1], float(full[cutoff + 1 :].sum())


def workshop_header(path):
    lines = []
    with path.with_suffix(".dat").open() as source:
        for line in source:
            if not line.startswith("#"):
                break
            if ":" in line:
                lines.append(line.rstrip())
    return "\n".join(line.removeprefix("# ") for line in lines)


def write_vtk(path, position, fields, time):
    """Legacy structured grid; VTK expects x to vary fastest."""
    nx, ny, nz, _ = position.shape
    with path.open("w") as out:
        out.write(
            f"# vtk DataFile Version 3.0\nSimiLie Taylor-Green t={time:.17g}\n"
            f"ASCII\nDATASET STRUCTURED_GRID\nDIMENSIONS {nx} {ny} {nz}\n"
            f"POINTS {nx * ny * nz} double\n"
        )
        np.savetxt(out, position.transpose(2, 1, 0, 3).reshape(-1, 3), fmt="%.17g")
        out.write(f"POINT_DATA {nx * ny * nz}\n")
        out.write(f"FIELD FieldData {len(fields) + 1}\ntime 1 1 double\n{time:.17g}\n")
        for name, field in fields.items():
            components = 3 if field.ndim == 4 else 1
            axes = (2, 1, 0, 3) if components == 3 else (2, 1, 0)
            values = field.transpose(axes).reshape(-1, components)
            out.write(f"{name} {components} {nx * ny * nz} double\n")
            np.savetxt(out, values, fmt="%.17g")


def process(path, vtk=True, plots=True):
    if plots:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

    with h5py.File(path, "r") as data:
        columns = data.attrs["monitor_columns"]
        if isinstance(columns, bytes):
            columns = columns.decode()
        if tuple(columns.split(",")) != COLUMNS:
            raise ValueError(f"Unknown monitor schema in {path}")
        monitors = data["monitors"][:]
        if not np.all(np.isfinite(monitors)) or np.any(np.diff(monitors[:, 0]) <= 0):
            raise ValueError(f"Nonfinite monitors or invalid timestamps in {path}")
        if not np.isclose(monitors[0, 0], 0) or not np.isclose(monitors[-1, 0], 10):
            raise ValueError(f"Incomplete simulation in {path}")
        position = data["position"][:]
        snapshots = sorted(
            (name for name in data if name.startswith("snapshot_")),
            key=lambda name: float(data[name]["time"][0]),
        )
        t8 = next(
            (name for name in snapshots if np.isclose(data[name]["time"][0], 8)),
            None,
        )
        if t8 is None:
            raise ValueError(f"Missing t=8 snapshot in {path}")
        header = workshop_header(path)
        table_path = path.with_suffix(".monitors.dat")
        np.savetxt(table_path, monitors, header=header + "\n" + " ".join(COLUMNS))
        constraints = None
        if "reconstruction_monitors" in data:
            constraints = data["reconstruction_monitors"][:]
            if (
                constraints.shape != (len(monitors), 5)
                or not np.all(np.isfinite(constraints))
                or not np.allclose(constraints[:, 0], monitors[:, 0])
                or np.any(constraints[:, 1:] < 0)
            ):
                raise ValueError(f"Invalid reconstruction monitors in {path}")
            np.savetxt(
                path.with_suffix(".reconstruction.dat"),
                constraints,
                header=header
                + "\ntime coulomb_gauge_rms vorticity_divergence_rms"
                + " curl_reconstruction_rms vector_poisson_relative_residual",
            )
        snapshot = data[t8]
        wave_number, spectrum, outside = energy_spectrum(
            snapshot["velocity"][:], snapshot["density"][:]
        )
        np.savetxt(
            path.with_suffix(".spectrum.dat"),
            np.column_stack((wave_number, spectrum)),
            header=header
            + "\nt/tc: 8; half-integer shell boundaries; density-weighted velocity"
            + f"\nenergy outside reported cutoff: {outside:.17g}"
            + "\nwave_number spectral_energy",
        )
        norm = np.linalg.norm(snapshot["vorticity"][0], axis=-1)
        np.savetxt(
            path.with_suffix(".vorticity_slice.dat"),
            np.column_stack(
                (position[0, ..., 1].ravel(), position[0, ..., 2].ravel(), norm.ravel())
            ),
            header=header + "\nx/L=-pi; t/tc=8; y/L z/L vorticity_norm*L/V0",
        )
        if vtk:
            for name in snapshots:
                fields = {
                    key: data[name][key][:]
                    for key in (
                        "velocity",
                        "vorticity",
                        "density",
                        "pressure",
                        "temperature",
                        "heat_flux",
                    )
                }
                if "vector_potential" in data[name]:
                    fields["vector_potential"] = data[name]["vector_potential"][:]
                write_vtk(
                    path.with_suffix(f".{name}.vtk"),
                    position,
                    fields,
                    float(data[name]["time"][0]),
                )
        if plots:
            if constraints is not None:
                figure, axis = plt.subplots(figsize=(8, 4), constrained_layout=True)
                for index, label in enumerate(
                    (
                        "Coulomb gauge RMS",
                        "Vorticity divergence RMS",
                        "Velocity curl reconstruction RMS",
                        "Vector Poisson relative residual",
                    ),
                    start=1,
                ):
                    axis.semilogy(
                        constraints[:, 0],
                        np.maximum(constraints[:, index], 1e-18),
                        label=label,
                    )
                axis.set(xlabel="t/tc", ylabel="Reconstruction residual")
                axis.legend()
                figure.savefig(path.with_suffix(".reconstruction.png"), dpi=160)
                plt.close(figure)
            figure, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
            axes[0, 0].plot(monitors[:, 0], monitors[:, 1])
            axes[0, 0].set(xlabel="t/tc", ylabel="Ek/V0²")
            for index, label in (
                (3, "-dEk/dt"),
                (5, "2 nu enstrophy"),
                (7, "viscous stress work"),
            ):
                axes[0, 1].plot(monitors[:, 0], monitors[:, index], label=label)
            axes[0, 1].set(xlabel="t/tc", ylabel="Dissipation L/V0³")
            axes[0, 1].legend()
            axes[1, 0].plot(monitors[:, 0], monitors[:, 4])
            axes[1, 0].set(xlabel="t/tc", ylabel="Enstrophy L²/V0²")
            axes[1, 1].plot(monitors[:, 0], monitors[:, 6], label="pressure dilatation")
            axes[1, 1].plot(
                monitors[:, 0],
                monitors[:, 3] - monitors[:, 7] + monitors[:, 6],
                label="kinetic loss minus viscous/pressure work",
            )
            axes[1, 1].set(xlabel="t/tc", ylabel="L/V0³")
            axes[1, 1].legend()
            figure.savefig(path.with_suffix(".monitors.png"), dpi=160)
            plt.close(figure)
            figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
            positive = (wave_number > 0) & (spectrum > 0)
            axes[0].loglog(wave_number[positive], spectrum[positive])
            axes[0].set(
                xlabel="Wave number kL", ylabel="Shell energy/V0²", title="t/tc=8"
            )
            # Close the plotted periodic slice, without adding duplicate FFT samples.
            periodic = np.pad(norm, ((0, 1), (0, 1)), mode="wrap")
            coordinates = np.linspace(-np.pi, np.pi, norm.shape[0] + 1)
            image = axes[1].pcolormesh(
                coordinates, coordinates, periodic.T, shading="auto"
            )
            axes[1].set(
                xlabel="y/L",
                ylabel="z/L",
                title="|omega| L/V0; x/L=-pi; t/tc=8",
                aspect="equal",
            )
            figure.colorbar(image, ax=axes[1])
            figure.savefig(path.with_suffix(".t8.png"), dpi=160)
            plt.close(figure)
            figure, axes = plt.subplots(1, 3, figsize=(12, 3), constrained_layout=True)
            axes[0].plot(monitors[:, 0], monitors[:, 8] - monitors[0, 8])
            axes[0].set(xlabel="t/tc", ylabel="Mean density change")
            axes[1].plot(monitors[:, 0], monitors[:, 9] - monitors[0, 9])
            axes[1].set(xlabel="t/tc", ylabel="Total energy change")
            axes[2].semilogy(monitors[:, 0], np.maximum(monitors[:, 10], 1e-18))
            axes[2].set(xlabel="t/tc", ylabel="RMS divergence")
            figure.savefig(path.with_suffix(".conservation.png"), dpi=160)
            plt.close(figure)
        return monitors, wave_number, spectrum


def compare(paths, results):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for path, (monitors, k, energy) in zip(paths, results, strict=True):
        axes[0].plot(monitors[:, 0], monitors[:, 3], label=path.stem)
        axes[1].plot(monitors[:, 0], monitors[:, 5], label=path.stem)
        positive = (k > 0) & (energy > 0)
        axes[2].loglog(k[positive], energy[positive], label=path.stem)
    axes[0].set(xlabel="t/tc", ylabel="-dEk/dt L/V0³")
    axes[1].set(xlabel="t/tc", ylabel="2 nu enstrophy L/V0³")
    axes[2].set(xlabel="Wave number kL", ylabel="Shell energy/V0²", title="t/tc=8")
    for axis in axes:
        axis.legend(fontsize="small")
    figure.savefig(paths[0].parent / "tgv_comparison.png", dpi=160)
    plt.close(figure)
    records = []
    for path in paths:
        with path.with_suffix(".performance.json").open() as source:
            records.append(json.load(source))
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    x = np.arange(len(paths))
    bottom = np.zeros(len(paths))
    for key in ("solver_seconds", "monitor_seconds", "export_seconds"):
        values = np.array([record[key] for record in records])
        axes[0].bar(x, values, bottom=bottom, label=key.removesuffix("_seconds"))
        bottom += values
    axes[0].legend()
    axes[0].set(ylabel="Wall seconds")
    axes[1].bar(x, [record["cell_updates_per_second"] for record in records])
    axes[1].set(ylabel="Cell updates / solver second")
    for axis in axes:
        axis.set_xticks(
            x, [path.stem for path in paths], rotation=25, ha="right", fontsize="small"
        )
    figure.savefig(paths[0].parent / "tgv_performance.png", dpi=160)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", type=Path, nargs="+", help="Benchmark .h5 files")
    parser.add_argument(
        "--no-vtk", action="store_true", help="Skip large ASCII VTK exports"
    )
    parser.add_argument(
        "--no-plots", action="store_true", help="Only create numeric tables and VTK"
    )
    args = parser.parse_args()
    results = [
        process(path, vtk=not args.no_vtk, plots=not args.no_plots)
        for path in args.files
    ]
    if not args.no_plots:
        compare(args.files, results)


if __name__ == "__main__":
    main()
