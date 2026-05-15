#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Baptiste Legouix
# SPDX-License-Identifier: MIT

from __future__ import annotations

import argparse
from bisect import bisect_left
import re
from pathlib import Path

import h5py
import numpy as np


HEXAHEDRON_TYPE = 5
TOL = 1.0e-6

PERMEABILITY_VIEW = "SimiLie linear magnetostatics permeability"
CURRENT_DENSITY_VIEW = "SimiLie linear magnetostatics current density z"
MAGNETIC_VECTOR_POTENTIAL_VIEW = "SimiLie linear magnetostatics magnetic vector potential z"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--input-fields-pos", required=True)
    parser.add_argument("--h5-output", required=True)
    parser.add_argument("--xmf-output", required=True)
    return parser.parse_args()


def parse_msh2_hex_mesh(mesh_file: Path) -> tuple[list[tuple[int, float, float, float]], list[tuple[int, list[int]]]]:
    lines = mesh_file.read_text().splitlines()
    nodes: list[tuple[int, float, float, float]] = []
    cells: list[tuple[int, list[int]]] = []
    i = 0
    while i < len(lines):
        token = lines[i].strip()
        if token == "$Nodes":
            count = int(lines[i + 1].strip())
            for row in lines[i + 2 : i + 2 + count]:
                parts = row.split()
                nodes.append((int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])))
            i += count + 2
        elif token == "$Elements":
            count = int(lines[i + 1].strip())
            for row in lines[i + 2 : i + 2 + count]:
                parts = row.split()
                element_type = int(parts[1])
                num_tags = int(parts[2])
                tags = [int(x) for x in parts[3 : 3 + num_tags]]
                node_tags = [int(x) for x in parts[3 + num_tags :]]
                if element_type == HEXAHEDRON_TYPE:
                    cells.append((tags[0] if tags else 0, node_tags))
            i += count + 2
        else:
            i += 1
    if not nodes or not cells:
        raise RuntimeError("mesh does not contain nodes and first-order hexahedra")
    return nodes, cells


def unique_sorted(values: list[float]) -> list[float]:
    values = sorted(values)
    result: list[float] = []
    for value in values:
        if not result or abs(result[-1] - value) >= TOL:
            result.append(value)
    return result


def nearest_index(coordinates: list[float], value: float) -> int:
    idx = bisect_left(coordinates, value)
    if idx < len(coordinates) and abs(coordinates[idx] - value) < TOL:
        return idx
    if idx > 0 and abs(coordinates[idx - 1] - value) < TOL:
        return idx - 1
    raise RuntimeError("coordinate does not belong to the detected structured grid")


def parse_result_views(
    pos_file: Path,
) -> tuple[
    list[tuple[tuple[float, float, float], float]],
    list[tuple[tuple[float, float, float], float]],
    list[tuple[tuple[float, float, float], float]],
]:
    scalar_pattern = re.compile(r'SP\(([^,]+),([^,]+),([^)]+)\)\{([^}]+)\};')
    vector_pattern = re.compile(r'VP\(([^,]+),([^,]+),([^)]+)\)\{([^,]+),([^,]+),([^}]+)\};')
    view_pattern = re.compile(r'^View "([^"]+)" \{$')

    permeability: list[tuple[tuple[float, float, float], float]] = []
    current_density: list[tuple[tuple[float, float, float], float]] = []
    magnetic_vector_potential: list[tuple[tuple[float, float, float], float]] = []

    current_view: str | None = None
    for raw_line in pos_file.read_text().splitlines():
        line = raw_line.strip()
        view_match = view_pattern.match(line)
        if view_match:
            current_view = view_match.group(1)
            continue
        if line == "};":
            current_view = None
            continue
        if current_view is None:
            continue

        scalar_match = scalar_pattern.search(line)
        if scalar_match:
            xyz = (
                float(scalar_match.group(1)),
                float(scalar_match.group(2)),
                float(scalar_match.group(3)),
            )
            value = float(scalar_match.group(4))
            if current_view == PERMEABILITY_VIEW:
                permeability.append((xyz, value))
            elif current_view == CURRENT_DENSITY_VIEW:
                current_density.append((xyz, value))
            elif current_view == MAGNETIC_VECTOR_POTENTIAL_VIEW:
                magnetic_vector_potential.append((xyz, value))
            continue

        vector_match = vector_pattern.search(line)
        if not vector_match:
            continue

    return permeability, current_density, magnetic_vector_potential


def build_structured_arrays(
    nodes: list[tuple[int, float, float, float]],
    cells: list[tuple[int, list[int]]],
    permeability_values: list[tuple[tuple[float, float, float], float]],
    current_density_values: list[tuple[tuple[float, float, float], float]],
    magnetic_vector_potential_values: list[tuple[tuple[float, float, float], float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_coords = unique_sorted([node[1] for node in nodes])
    y_coords = unique_sorted([node[2] for node in nodes])
    z_coords = unique_sorted([node[3] for node in nodes])
    nx, ny, nz = len(x_coords), len(y_coords), len(z_coords)
    if nx * ny * nz != len(nodes):
        raise RuntimeError("the mesh nodes do not form a full structured grid")

    node_to_indices: dict[int, tuple[int, int, int]] = {}
    occupied_nodes: set[tuple[int, int, int]] = set()
    for tag, x, y, z in nodes:
        key = (nearest_index(x_coords, x), nearest_index(y_coords, y), nearest_index(z_coords, z))
        if key in occupied_nodes:
            raise RuntimeError("duplicated nodes on the structured grid")
        occupied_nodes.add(key)
        node_to_indices[tag] = key

    num_cells = (nx - 1) * (ny - 1) * (nz - 1)
    if len(cells) != num_cells:
        raise RuntimeError("the number of cells does not match the expected structured grid size")
    if len(permeability_values) != num_cells or len(current_density_values) != num_cells:
        raise RuntimeError("the input field views do not match the cell count")
    if len(magnetic_vector_potential_values) != len(nodes):
        raise RuntimeError("the magnetic vector potential view does not match the node count")
    magnetic_vector_potential = np.zeros((nx, ny, nz, 3), dtype=np.float64)
    for xyz, value in magnetic_vector_potential_values:
        ix = nearest_index(x_coords, xyz[0])
        iy = nearest_index(y_coords, xyz[1])
        iz = nearest_index(z_coords, xyz[2])
        magnetic_vector_potential[ix, iy, iz, 2] = value

    permeability = np.zeros((nx - 1, ny - 1, nz - 1), dtype=np.float64)
    current_density = np.zeros((nx - 1, ny - 1, nz - 1, 3), dtype=np.float64)
    occupied_cells: set[tuple[int, int, int]] = set()
    for cell_idx, (_, node_tags) in enumerate(cells):
        cell_indices = [node_to_indices[tag] for tag in node_tags]
        min_x = min(index[0] for index in cell_indices)
        min_y = min(index[1] for index in cell_indices)
        min_z = min(index[2] for index in cell_indices)
        max_x = max(index[0] for index in cell_indices)
        max_y = max(index[1] for index in cell_indices)
        max_z = max(index[2] for index in cell_indices)
        if (max_x, max_y, max_z) != (min_x + 1, min_y + 1, min_z + 1):
            raise RuntimeError("a hexahedron does not match a single structured grid cell")
        key = (min_x, min_y, min_z)
        if key in occupied_cells:
            raise RuntimeError("duplicated structured cell")
        occupied_cells.add(key)

    x_mid = [(x_coords[i] + x_coords[i + 1]) / 2.0 for i in range(nx - 1)]
    y_mid = [(y_coords[i] + y_coords[i + 1]) / 2.0 for i in range(ny - 1)]
    z_mid = [(z_coords[i] + z_coords[i + 1]) / 2.0 for i in range(nz - 1)]

    def cell_key_from_xyz(xyz: tuple[float, float, float]) -> tuple[int, int, int]:
        return (nearest_index(x_mid, xyz[0]), nearest_index(y_mid, xyz[1]), nearest_index(z_mid, xyz[2]))

    for xyz, value in permeability_values:
        ix, iy, iz = cell_key_from_xyz(xyz)
        permeability[ix, iy, iz] = value
    for xyz, value in current_density_values:
        ix, iy, iz = cell_key_from_xyz(xyz)
        current_density[ix, iy, iz, 2] = value
    return (
        np.asarray(x_coords),
        np.asarray(y_coords),
        np.asarray(z_coords),
        permeability,
        current_density,
        magnetic_vector_potential,
    )


def derive_cell_fields(
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    z_coords: np.ndarray,
    permeability: np.ndarray,
    magnetic_vector_potential: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    nx, ny, nz, _ = magnetic_vector_potential.shape
    magnetic_induction = np.zeros((nx - 1, ny - 1, nz - 1, 3), dtype=np.float64)
    magnetic_field = np.zeros((nx - 1, ny - 1, nz - 1, 3), dtype=np.float64)
    maxwell_stress = np.zeros((nx - 1, ny - 1, nz - 1, 3, 3), dtype=np.float64)
    force_density = np.zeros((nx - 1, ny - 1, nz - 1, 3), dtype=np.float64)

    az = magnetic_vector_potential[..., 2]
    for k in range(nz - 1):
        for j in range(ny - 1):
            for i in range(nx - 1):
                dy = y_coords[j + 1] - y_coords[j]
                dx = x_coords[i + 1] - x_coords[i]
                d_az_dy = (
                    (az[i, j + 1, k] + az[i + 1, j + 1, k] + az[i, j + 1, k + 1] + az[i + 1, j + 1, k + 1])
                    - (az[i, j, k] + az[i + 1, j, k] + az[i, j, k + 1] + az[i + 1, j, k + 1])
                ) / (4.0 * dy)
                d_az_dx = (
                    (az[i + 1, j, k] + az[i + 1, j + 1, k] + az[i + 1, j, k + 1] + az[i + 1, j + 1, k + 1])
                    - (az[i, j, k] + az[i, j + 1, k] + az[i, j, k + 1] + az[i, j + 1, k + 1])
                ) / (4.0 * dx)
                magnetic_induction[i, j, k, 0] = d_az_dy
                magnetic_induction[i, j, k, 1] = -d_az_dx

    magnetic_field = magnetic_induction / permeability[..., np.newaxis]

    bx = magnetic_induction[..., 0]
    by = magnetic_induction[..., 1]
    bz = magnetic_induction[..., 2]
    hx = magnetic_field[..., 0]
    hy = magnetic_field[..., 1]
    hz = magnetic_field[..., 2]
    half_trace = 0.5 * (bx * hx + by * hy + bz * hz)
    maxwell_stress[..., 0, 0] = bx * hx - half_trace
    maxwell_stress[..., 1, 1] = by * hy - half_trace
    maxwell_stress[..., 2, 2] = bz * hz - half_trace
    maxwell_stress[..., 0, 1] = bx * hy
    maxwell_stress[..., 1, 0] = maxwell_stress[..., 0, 1]
    maxwell_stress[..., 0, 2] = bx * hz
    maxwell_stress[..., 2, 0] = maxwell_stress[..., 0, 2]
    maxwell_stress[..., 1, 2] = by * hz
    maxwell_stress[..., 2, 1] = maxwell_stress[..., 1, 2]

    def centered_derivative(values: np.ndarray, coords: np.ndarray, axis: int) -> np.ndarray:
        result = np.zeros_like(values)
        if axis == 0:
            result[1:-1, :, :] = (values[2:, :, :] - values[:-2, :, :]) / (
                coords[2:, np.newaxis, np.newaxis] - coords[:-2, np.newaxis, np.newaxis]
            )
            result[0, :, :] = (values[1, :, :] - values[0, :, :]) / (coords[1] - coords[0])
            result[-1, :, :] = (values[-1, :, :] - values[-2, :, :]) / (coords[-1] - coords[-2])
        elif axis == 1:
            result[:, 1:-1, :] = (values[:, 2:, :] - values[:, :-2, :]) / (
                coords[np.newaxis, 2:, np.newaxis] - coords[np.newaxis, :-2, np.newaxis]
            )
            result[:, 0, :] = (values[:, 1, :] - values[:, 0, :]) / (coords[1] - coords[0])
            result[:, -1, :] = (values[:, -1, :] - values[:, -2, :]) / (coords[-1] - coords[-2])
        else:
            result[:, :, 1:-1] = (values[:, :, 2:] - values[:, :, :-2]) / (
                coords[np.newaxis, np.newaxis, 2:] - coords[np.newaxis, np.newaxis, :-2]
            )
            result[:, :, 0] = (values[:, :, 1] - values[:, :, 0]) / (coords[1] - coords[0])
            result[:, :, -1] = (values[:, :, -1] - values[:, :, -2]) / (coords[-1] - coords[-2])
        return result

    x_mid = 0.5 * (x_coords[:-1] + x_coords[1:])
    y_mid = 0.5 * (y_coords[:-1] + y_coords[1:])
    z_mid = 0.5 * (z_coords[:-1] + z_coords[1:])
    force_density[..., 0] = (
        centered_derivative(maxwell_stress[..., 0, 0], x_mid, 0)
        + centered_derivative(maxwell_stress[..., 0, 1], y_mid, 1)
        + centered_derivative(maxwell_stress[..., 0, 2], z_mid, 2)
    )
    force_density[..., 1] = (
        centered_derivative(maxwell_stress[..., 1, 0], x_mid, 0)
        + centered_derivative(maxwell_stress[..., 1, 1], y_mid, 1)
        + centered_derivative(maxwell_stress[..., 1, 2], z_mid, 2)
    )
    force_density[..., 2] = (
        centered_derivative(maxwell_stress[..., 2, 0], x_mid, 0)
        + centered_derivative(maxwell_stress[..., 2, 1], y_mid, 1)
        + centered_derivative(maxwell_stress[..., 2, 2], z_mid, 2)
    )
    return magnetic_induction, magnetic_field, maxwell_stress, force_density


def write_hdf5(
    h5_file: Path,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    z_coords: np.ndarray,
    permeability: np.ndarray,
    current_density: np.ndarray,
    magnetic_vector_potential: np.ndarray,
    magnetic_induction: np.ndarray,
    magnetic_field: np.ndarray,
    maxwell_stress: np.ndarray,
    force_density: np.ndarray,
) -> None:
    h5_file.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(h5_file, "w") as handle:
        handle.create_dataset("x_coordinates", data=x_coords)
        handle.create_dataset("y_coordinates", data=y_coords)
        handle.create_dataset("z_coordinates", data=z_coords)
        handle.create_dataset("magnetic_permeability", data=permeability)
        handle.create_dataset("current_density", data=current_density)
        handle.create_dataset("magnetic_vector_potential", data=magnetic_vector_potential)
        handle.create_dataset("magnetic_induction", data=magnetic_induction)
        handle.create_dataset("magnetic_field", data=magnetic_field)
        handle.create_dataset("maxwell_stress_xx", data=maxwell_stress[..., 0, 0])
        handle.create_dataset("maxwell_stress_yy", data=maxwell_stress[..., 1, 1])
        handle.create_dataset("maxwell_stress_zz", data=maxwell_stress[..., 2, 2])
        handle.create_dataset("maxwell_stress_xy", data=maxwell_stress[..., 0, 1])
        handle.create_dataset("maxwell_stress_xz", data=maxwell_stress[..., 0, 2])
        handle.create_dataset("maxwell_stress_yz", data=maxwell_stress[..., 1, 2])
        handle.create_dataset("force_density", data=force_density)


def write_xmf(
    xmf_file: Path,
    h5_file: Path,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    z_coords: np.ndarray,
    permeability: np.ndarray,
    current_density: np.ndarray,
    magnetic_vector_potential: np.ndarray,
    magnetic_induction: np.ndarray,
    magnetic_field: np.ndarray,
    maxwell_stress: np.ndarray,
    force_density: np.ndarray,
) -> None:
    nx = x_coords.size
    ny = y_coords.size
    nz = z_coords.size
    xmf_file.parent.mkdir(parents=True, exist_ok=True)
    xmf_file.write_text(
        f"""<?xml version="1.0" ?>
<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
<Xdmf Version="2.0">
 <Domain>
   <Grid Name="mesh1" GridType="Uniform">
     <Topology TopologyType="3DSMesh" NumberOfElements="{nx} {ny} {nz}"/>
     <Geometry GeometryType="VXVYVZ">
       <DataItem Dimensions="{nx}" NumberType="Float" Precision="8" Format="HDF">
        {h5_file.name}:/x_coordinates
       </DataItem>
       <DataItem Dimensions="{ny}" NumberType="Float" Precision="8" Format="HDF">
        {h5_file.name}:/y_coordinates
       </DataItem>
       <DataItem Dimensions="{nz}" NumberType="Float" Precision="8" Format="HDF">
        {h5_file.name}:/z_coordinates
       </DataItem>
     </Geometry>
     <Attribute Name="MagneticVectorPotential" AttributeType="Vector" Center="Node">
       <DataItem Dimensions="{nx} {ny} {nz} 3" NumberType="Float" Precision="8" Format="HDF">
        {h5_file.name}:/magnetic_vector_potential
       </DataItem>
     </Attribute>
     <Attribute Name="CurrentDensity" AttributeType="Vector" Center="Cell">
       <DataItem Dimensions="{nx - 1} {ny - 1} {nz - 1} 3" NumberType="Float" Precision="8" Format="HDF">
        {h5_file.name}:/current_density
       </DataItem>
     </Attribute>
     <Attribute Name="MagneticPermeability" AttributeType="Scalar" Center="Cell">
       <DataItem Dimensions="{nx - 1} {ny - 1} {nz - 1}" NumberType="Float" Precision="8" Format="HDF">
        {h5_file.name}:/magnetic_permeability
       </DataItem>
     </Attribute>
     <Attribute Name="MagneticInduction" AttributeType="Vector" Center="Cell">
       <DataItem Dimensions="{nx - 1} {ny - 1} {nz - 1} 3" NumberType="Float" Precision="8" Format="HDF">
        {h5_file.name}:/magnetic_induction
       </DataItem>
     </Attribute>
     <Attribute Name="MagneticField" AttributeType="Vector" Center="Cell">
       <DataItem Dimensions="{nx - 1} {ny - 1} {nz - 1} 3" NumberType="Float" Precision="8" Format="HDF">
        {h5_file.name}:/magnetic_field
       </DataItem>
     </Attribute>
     <Attribute Name="MaxwellStressXX" AttributeType="Scalar" Center="Cell">
       <DataItem Dimensions="{nx - 1} {ny - 1} {nz - 1}" NumberType="Float" Precision="8" Format="HDF">
        {h5_file.name}:/maxwell_stress_xx
       </DataItem>
     </Attribute>
     <Attribute Name="MaxwellStressYY" AttributeType="Scalar" Center="Cell">
       <DataItem Dimensions="{nx - 1} {ny - 1} {nz - 1}" NumberType="Float" Precision="8" Format="HDF">
        {h5_file.name}:/maxwell_stress_yy
       </DataItem>
     </Attribute>
     <Attribute Name="MaxwellStressZZ" AttributeType="Scalar" Center="Cell">
       <DataItem Dimensions="{nx - 1} {ny - 1} {nz - 1}" NumberType="Float" Precision="8" Format="HDF">
        {h5_file.name}:/maxwell_stress_zz
       </DataItem>
     </Attribute>
     <Attribute Name="MaxwellStressXY" AttributeType="Scalar" Center="Cell">
       <DataItem Dimensions="{nx - 1} {ny - 1} {nz - 1}" NumberType="Float" Precision="8" Format="HDF">
        {h5_file.name}:/maxwell_stress_xy
       </DataItem>
     </Attribute>
     <Attribute Name="MaxwellStressXZ" AttributeType="Scalar" Center="Cell">
       <DataItem Dimensions="{nx - 1} {ny - 1} {nz - 1}" NumberType="Float" Precision="8" Format="HDF">
        {h5_file.name}:/maxwell_stress_xz
       </DataItem>
     </Attribute>
     <Attribute Name="MaxwellStressYZ" AttributeType="Scalar" Center="Cell">
       <DataItem Dimensions="{nx - 1} {ny - 1} {nz - 1}" NumberType="Float" Precision="8" Format="HDF">
        {h5_file.name}:/maxwell_stress_yz
       </DataItem>
     </Attribute>
     <Attribute Name="ForceDensity" AttributeType="Vector" Center="Cell">
       <DataItem Dimensions="{nx - 1} {ny - 1} {nz - 1} 3" NumberType="Float" Precision="8" Format="HDF">
        {h5_file.name}:/force_density
       </DataItem>
     </Attribute>
   </Grid>
 </Domain>
</Xdmf>
"""
    )


def main() -> int:
    args = parse_args()
    mesh_file = Path(args.mesh)
    pos_file = Path(args.input_fields_pos)
    h5_file = Path(args.h5_output)
    xmf_file = Path(args.xmf_output)

    nodes, cells = parse_msh2_hex_mesh(mesh_file)
    permeability_values, current_density_values, magnetic_vector_potential_values = parse_result_views(pos_file)
    (
        x_coords,
        y_coords,
        z_coords,
        permeability,
        current_density,
        magnetic_vector_potential,
    ) = build_structured_arrays(
        nodes,
        cells,
        permeability_values,
        current_density_values,
        magnetic_vector_potential_values,
    )
    magnetic_induction, magnetic_field, maxwell_stress, force_density = derive_cell_fields(
        x_coords,
        y_coords,
        z_coords,
        permeability,
        magnetic_vector_potential,
    )
    write_hdf5(
        h5_file,
        x_coords,
        y_coords,
        z_coords,
        permeability,
        current_density,
        magnetic_vector_potential,
        magnetic_induction,
        magnetic_field,
        maxwell_stress,
        force_density,
    )
    write_xmf(
        xmf_file,
        h5_file,
        x_coords,
        y_coords,
        z_coords,
        permeability,
        current_density,
        magnetic_vector_potential,
        magnetic_induction,
        magnetic_field,
        maxwell_stress,
        force_density,
    )
    print(f"Paraview HDF5 result exported in {h5_file}.")
    print(f"Paraview XDMF model exported in {xmf_file}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
