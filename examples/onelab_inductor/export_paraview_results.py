#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Baptiste Legouix
# SPDX-License-Identifier: MIT

from __future__ import annotations

import argparse
import re
from pathlib import Path

import h5py
import numpy as np


HEXAHEDRON_TYPE = 5
TOL = 1.0e-12

PERMEABILITY_VIEW = "SimiLie linear magnetostatics permeability"
CURRENT_DENSITY_VIEW = "SimiLie linear magnetostatics current density"
MAGNETIC_VECTOR_POTENTIAL_VIEW = "SimiLie linear magnetostatics magnetic vector potential"
MAGNETIC_INDUCTION_VIEW = "SimiLie linear magnetostatics magnetic induction"
MAGNETIC_FIELD_VIEW = "SimiLie linear magnetostatics magnetic field"
FORCE_DENSITY_VIEW = "SimiLie linear magnetostatics force density"
STRESS_VIEW_NAMES = {
    "SimiLie linear magnetostatics Maxwell stress xx": (0, 0),
    "SimiLie linear magnetostatics Maxwell stress yy": (1, 1),
    "SimiLie linear magnetostatics Maxwell stress zz": (2, 2),
    "SimiLie linear magnetostatics Maxwell stress xy": (0, 1),
    "SimiLie linear magnetostatics Maxwell stress xz": (0, 2),
    "SimiLie linear magnetostatics Maxwell stress yz": (1, 2),
}


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
    for idx, candidate in enumerate(coordinates):
        if abs(candidate - value) < TOL:
            return idx
    raise RuntimeError("coordinate does not belong to the detected structured grid")


def parse_result_views(
    pos_file: Path,
) -> tuple[
    list[float],
    list[tuple[float, float, float]],
    list[tuple[float, float, float]],
    list[tuple[float, float, float]],
    list[tuple[float, float, float]],
    dict[str, list[float]],
    list[tuple[float, float, float]],
]:
    scalar_pattern = re.compile(r'SP\(([^,]+),([^,]+),([^)]+)\)\{([^}]+)\};')
    vector_pattern = re.compile(r'VP\(([^,]+),([^,]+),([^)]+)\)\{([^,]+),([^,]+),([^}]+)\};')
    view_pattern = re.compile(r'^View "([^"]+)" \{$')

    permeability: list[float] = []
    current_density: list[tuple[float, float, float]] = []
    magnetic_vector_potential: list[tuple[float, float, float]] = []
    magnetic_induction: list[tuple[float, float, float]] = []
    magnetic_field: list[tuple[float, float, float]] = []
    force_density: list[tuple[float, float, float]] = []
    stress_components = {name: [] for name in STRESS_VIEW_NAMES}

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
            value = float(scalar_match.group(4))
            if current_view == PERMEABILITY_VIEW:
                permeability.append(value)
            elif current_view in stress_components:
                stress_components[current_view].append(value)
            continue

        vector_match = vector_pattern.search(line)
        if not vector_match:
            continue
        value = (
            float(vector_match.group(4)),
            float(vector_match.group(5)),
            float(vector_match.group(6)),
        )
        if current_view == CURRENT_DENSITY_VIEW:
            current_density.append(value)
        elif current_view == MAGNETIC_VECTOR_POTENTIAL_VIEW:
            magnetic_vector_potential.append(value)
        elif current_view == MAGNETIC_INDUCTION_VIEW:
            magnetic_induction.append(value)
        elif current_view == MAGNETIC_FIELD_VIEW:
            magnetic_field.append(value)
        elif current_view == FORCE_DENSITY_VIEW:
            force_density.append(value)

    return (
        permeability,
        current_density,
        magnetic_vector_potential,
        magnetic_induction,
        magnetic_field,
        stress_components,
        force_density,
    )


def build_structured_arrays(
    nodes: list[tuple[int, float, float, float]],
    cells: list[tuple[int, list[int]]],
    permeability_values: list[float],
    current_density_values: list[tuple[float, float, float]],
    magnetic_vector_potential_values: list[tuple[float, float, float]],
    magnetic_induction_values: list[tuple[float, float, float]],
    magnetic_field_values: list[tuple[float, float, float]],
    stress_components: dict[str, list[float]],
    force_density_values: list[tuple[float, float, float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    if len(magnetic_induction_values) != num_cells or len(magnetic_field_values) != num_cells:
        raise RuntimeError("the magnetic induction and field views do not match the cell count")
    if len(force_density_values) != num_cells:
        raise RuntimeError("the force density view does not match the cell count")
    for name, values in stress_components.items():
        if len(values) != num_cells:
            raise RuntimeError(f"the stress view '{name}' does not match the cell count")

    position = np.zeros((nx, ny, nz, 3), dtype=np.float64)
    for ix, x in enumerate(x_coords):
        position[ix, :, :, 0] = x
    for iy, y in enumerate(y_coords):
        position[:, iy, :, 1] = y
    for iz, z in enumerate(z_coords):
        position[:, :, iz, 2] = z

    magnetic_vector_potential = np.zeros((nx, ny, nz, 3), dtype=np.float64)
    for value_idx, value in enumerate(magnetic_vector_potential_values):
        ix = value_idx % nx
        iy = (value_idx // nx) % ny
        iz = value_idx // (nx * ny)
        magnetic_vector_potential[ix, iy, iz, :] = value

    permeability = np.zeros((nx - 1, ny - 1, nz - 1), dtype=np.float64)
    current_density = np.zeros((nx - 1, ny - 1, nz - 1, 3), dtype=np.float64)
    magnetic_induction = np.zeros((nx - 1, ny - 1, nz - 1, 3), dtype=np.float64)
    magnetic_field = np.zeros((nx - 1, ny - 1, nz - 1, 3), dtype=np.float64)
    maxwell_stress = np.zeros((nx - 1, ny - 1, nz - 1, 3, 3), dtype=np.float64)
    force_density = np.zeros((nx - 1, ny - 1, nz - 1, 3), dtype=np.float64)

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

        permeability[min_x, min_y, min_z] = permeability_values[cell_idx]
        current_density[min_x, min_y, min_z, :] = current_density_values[cell_idx]
        magnetic_induction[min_x, min_y, min_z, :] = magnetic_induction_values[cell_idx]
        magnetic_field[min_x, min_y, min_z, :] = magnetic_field_values[cell_idx]
        force_density[min_x, min_y, min_z, :] = force_density_values[cell_idx]
        for name, (ii, jj) in STRESS_VIEW_NAMES.items():
            value = stress_components[name][cell_idx]
            maxwell_stress[min_x, min_y, min_z, ii, jj] = value
            maxwell_stress[min_x, min_y, min_z, jj, ii] = value

    return (
        position,
        permeability,
        current_density,
        magnetic_vector_potential,
        magnetic_induction,
        magnetic_field,
        maxwell_stress,
        force_density,
    )


def write_hdf5(
    h5_file: Path,
    position: np.ndarray,
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
        handle.create_dataset("position", data=position)
        handle.create_dataset("magnetic_permeability", data=permeability)
        handle.create_dataset("current_density", data=current_density)
        handle.create_dataset("magnetic_vector_potential", data=magnetic_vector_potential)
        handle.create_dataset("magnetic_induction", data=magnetic_induction)
        handle.create_dataset("magnetic_field", data=magnetic_field)
        handle.create_dataset("maxwell_stress_tensor", data=maxwell_stress)
        handle.create_dataset("force_density", data=force_density)


def write_xmf(
    xmf_file: Path,
    h5_file: Path,
    position: np.ndarray,
    permeability: np.ndarray,
    current_density: np.ndarray,
    magnetic_vector_potential: np.ndarray,
    magnetic_induction: np.ndarray,
    magnetic_field: np.ndarray,
    maxwell_stress: np.ndarray,
    force_density: np.ndarray,
) -> None:
    nx, ny, nz, _ = position.shape
    xmf_file.parent.mkdir(parents=True, exist_ok=True)
    xmf_file.write_text(
        f"""<?xml version="1.0" ?>
<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
<Xdmf Version="2.0">
 <Domain>
   <Grid Name="mesh1" GridType="Uniform">
     <Topology TopologyType="3DSMesh" NumberOfElements="{nx} {ny} {nz}"/>
     <Geometry GeometryType="XYZ">
       <DataItem Dimensions="{nx} {ny} {nz} 3" NumberType="Float" Precision="8" Format="HDF">
        {h5_file.name}:/position
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
     <Attribute Name="MaxwellStressTensor" AttributeType="Tensor" Center="Cell">
       <DataItem Dimensions="{nx - 1} {ny - 1} {nz - 1} 3 3" NumberType="Float" Precision="8" Format="HDF">
        {h5_file.name}:/maxwell_stress_tensor
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
    (
        permeability_values,
        current_density_values,
        magnetic_vector_potential_values,
        magnetic_induction_values,
        magnetic_field_values,
        stress_components,
        force_density_values,
    ) = parse_result_views(pos_file)
    (
        position,
        permeability,
        current_density,
        magnetic_vector_potential,
        magnetic_induction,
        magnetic_field,
        maxwell_stress,
        force_density,
    ) = build_structured_arrays(
        nodes,
        cells,
        permeability_values,
        current_density_values,
        magnetic_vector_potential_values,
        magnetic_induction_values,
        magnetic_field_values,
        stress_components,
        force_density_values,
    )
    write_hdf5(
        h5_file,
        position,
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
        position,
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
