# SPDX-FileCopyrightText: 2026 Baptiste Legouix
# SPDX-License-Identifier: AGPL-3.0-or-later
# // AI-GENERATED

"""Compare physical gradients of bilinear potentials on identical quad meshes."""

import argparse
import json
from pathlib import Path

import numpy as np


def read_field(path, kind="SQ", components=1, view=None):
    coordinates, values = [], []
    with Path(path).open() as stream:
        first = next(stream)
        if first.startswith("$MeshFormat"):
            nodes, elements = {}, {}
            for line in stream:
                if line.startswith("$Nodes"):
                    for _ in range(int(next(stream))):
                        row = next(stream).split()
                        nodes[int(row[0])] = [float(row[1]), float(row[2])]
                elif line.startswith("$Elements"):
                    for _ in range(int(next(stream))):
                        row = list(map(int, next(stream).split()))
                        if row[1] == 3:
                            elements[row[0]] = row[3 + row[2] :]
                elif line.startswith("$ElementNodeData"):
                    for _ in range(int(next(stream))):
                        next(stream)
                    for _ in range(int(next(stream))):
                        next(stream)
                    tags = [int(next(stream)) for _ in range(int(next(stream)))]
                    if tags[1] != components:
                        raise ValueError("unexpected field components")
                    for _ in range(tags[2]):
                        row = next(stream).split()
                        if int(row[0]) in elements:
                            coordinates.append(
                                [nodes[node] for node in elements[int(row[0])]]
                            )
                            values.append(list(map(float, row[2:])))
                    break
        else:
            active = view is None or view in first
            for line in stream:
                if line.startswith("View "):
                    active = view is None or view in line
                if line.startswith("};") and active:
                    break
                if active and line.startswith(kind + "("):
                    geometry, data = line.split("){")
                    coordinates.append(
                        np.fromstring(geometry[3:], sep=",").reshape(4, 3)[:, :2]
                    )
                    values.append(np.fromstring(data.split("}")[0], sep=","))
    coordinates, values = np.array(coordinates), np.array(values)
    if components != 1:
        values = values.reshape(-1, 4, components)
    if coordinates.shape != (len(values), 4, 2) or values.shape != (
        (len(values), 4) if components == 1 else (len(values), 4, components)
    ):
        raise ValueError("expected nonempty four-node quadrilateral potentials")
    if not np.isfinite(coordinates).all() or not np.isfinite(values).all():
        raise ValueError("non-finite geometry or potential")
    centers = np.round(coordinates.mean(axis=1), 9)
    order = np.lexsort((centers[:, 1], centers[:, 0]))
    return coordinates[order], values[order]


def jacobian(coordinates, x, y):
    a, b, c, d = (coordinates[:, i] for i in range(4))
    t = (1 - y) * (b - a) + y * (c - d)
    r = (1 - x) * (d - a) + x * (c - b)
    determinant = t[:, 0] * r[:, 1] - t[:, 1] * r[:, 0]
    if (determinant == 0).any():
        raise ValueError("degenerate quadrilateral")
    return t, r, determinant


def gradient(coordinates, potential, x=0.5, y=0.5):
    t, r, determinant = jacobian(coordinates, x, y)
    dt = (1 - y) * (potential[:, 1] - potential[:, 0]) + y * (
        potential[:, 2] - potential[:, 3]
    )
    dr = (1 - x) * (potential[:, 3] - potential[:, 0]) + x * (
        potential[:, 2] - potential[:, 1]
    )
    velocity = np.column_stack(
        (
            (dt * r[:, 1] - t[:, 1] * dr) / determinant,
            (t[:, 0] * dr - dt * r[:, 0]) / determinant,
        )
    )
    if not np.isfinite(velocity).all():
        raise ValueError("non-finite velocity")
    return velocity


def compare(similie, getdp):
    coordinates, potential = read_field(similie)
    reference_coordinates, reference = read_field(getdp)
    if coordinates.shape != reference_coordinates.shape or not np.allclose(
        coordinates, reference_coordinates, rtol=0, atol=1e-9
    ):
        raise ValueError("reference must use identical cells and vertex ordering")
    velocity_error, velocity_norm = 0.0, 0.0
    pressure_error, pressure_norm = 0.0, 0.0
    # Integrate at four Gauss points. Cell centers alone miss alternating nodal
    # errors whose bilinear gradient vanishes at the center.
    points = [0.5 - 0.5 / np.sqrt(3), 0.5 + 0.5 / np.sqrt(3)]
    for x in points:
        for y in points:
            velocity = gradient(coordinates, potential, x, y)
            reference_velocity = gradient(coordinates, reference, x, y)
            area = abs(jacobian(coordinates, x, y)[2]) / 4
            speed_squared = np.sum(velocity**2, axis=1)
            reference_speed_squared = np.sum(reference_velocity**2, axis=1)
            velocity_error += np.sum(
                area * np.sum((velocity - reference_velocity) ** 2, axis=1)
            )
            velocity_norm += np.sum(area * reference_speed_squared)
            # Bernoulli pressure is -rho/2 * |v|^2; its factor cancels.
            pressure_error += np.sum(
                area * (speed_squared - reference_speed_squared) ** 2
            )
            pressure_norm += np.sum(area * reference_speed_squared**2)

    def relative_l2(numerator, denominator):
        if denominator == 0:
            return 0.0 if numerator == 0 else float("inf")
        return float(np.sqrt(numerator / denominator))

    corners = [
        [x, y]
        for x in [coordinates[:, :, 0].min(), coordinates[:, :, 0].max()]
        for y in [coordinates[:, :, 1].min(), coordinates[:, :, 1].max()]
    ]
    corner_errors = []
    for corner in corners:
        distances = np.linalg.norm(coordinates.mean(axis=1) - corner, axis=1)
        near = distances < 0.25
        near[np.argmin(distances)] = True
        # Include vertices so a corner singularity cannot hide at cell centers.
        corner_errors.append(
            max(
                float(
                    np.linalg.norm(
                        gradient(coordinates[near], potential[near], x, y)
                        - gradient(coordinates[near], reference[near], x, y),
                        axis=1,
                    ).max()
                )
                for x, y in [(0, 0), (1, 0), (1, 1), (0, 1), (0.5, 0.5)]
            )
        )
    return {
        "cells": len(coordinates),
        "velocity_relative_area_l2": relative_l2(velocity_error, velocity_norm),
        "pressure_relative_area_l2": relative_l2(pressure_error, pressure_norm),
        "corner_velocity_error_max": corner_errors,
    }


def check_exports(path):
    coordinates, potential = read_field(path)
    velocity_coordinates, velocity = read_field(path, "VQ", 3, "SimiLie velocity")
    pressure_coordinates, pressure = read_field(
        path, view="SimiLie pressure difference"
    )
    for exported_coordinates in [velocity_coordinates, pressure_coordinates]:
        if coordinates.shape != exported_coordinates.shape or not np.allclose(
            coordinates, exported_coordinates, rtol=0, atol=1e-9
        ):
            raise ValueError("exported field mesh mismatch")
    expected = np.stack(
        [
            gradient(coordinates, potential, x, y)
            for x, y in [(0, 0), (1, 0), (1, 1), (0, 1)]
        ],
        axis=1,
    )
    if (
        abs(velocity[:, :, :2] - expected).max() > 1e-8
        or abs(velocity[:, :, 2]).max() > 1e-8
    ):
        raise ValueError("exported velocity differs from vertex potential gradients")
    if abs(pressure + 0.5 * 1.225 * np.sum(expected**2, axis=2)).max() > 1e-6:
        raise ValueError("exported pressure violates Bernoulli's law at vertices")


def check(metrics):
    # Reference closures differ locally. These bounds still reject the observed
    # GUI artifacts (40% velocity error and corner speeds above 170 m/s).
    limits = {
        "velocity_relative_area_l2": 0.01,
        "pressure_relative_area_l2": 0.02,
    }
    for name, limit in limits.items():
        if not np.isfinite(metrics[name]) or metrics[name] > limit:
            raise ValueError(f"{name}={metrics[name]:.6g} exceeds {limit}")
    if max(metrics["corner_velocity_error_max"]) > 0.5:
        raise ValueError("corner velocity disagreement exceeds 0.5 m/s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("similie")
    parser.add_argument("getdp")
    parser.add_argument("--check-exports", action="store_true")
    args = parser.parse_args()
    metrics = compare(args.similie, args.getdp)
    print(json.dumps(metrics, indent=2, allow_nan=False))
    check(metrics)
    if args.check_exports:
        check_exports(args.similie)
