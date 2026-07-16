#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2026 Baptiste Legouix
# SPDX-License-Identifier: MIT

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
geometry_file="${script_dir}/inductor.geo"
model_data_file="${script_dir}/inductor_data.geo"
problem_file="${SIMILIE_ONELAB_PROBLEM_FILE:-${script_dir}/inductor.silpro}"
output_dir="$(pwd)"
model_dimension=2
use_bool_test=0
physics_mode=""
use_matrix_free=""
open_cascade_model=0

gmsh_executable="${GMSH_EXECUTABLE:-gmsh}"
build_dir="${SIMILIE_ONELAB_BUILD_DIR:-${repo_root}/build}"
onelab_client="${SIMILIE_ONELAB_BINARY:-${build_dir}/onelab_interface/similie_onelab}"
mesh_file="${SIMILIE_ONELAB_MESH_FILE:-${output_dir}/inductor.msh}"
result_file="${SIMILIE_ONELAB_RESULT_FILE:-${output_dir}/similie_magnetostatics_inputs.pos}"
paraview_h5_file="${SIMILIE_PARAVIEW_H5_FILE:-${output_dir}/inductor.h5}"
paraview_xmf_file="${SIMILIE_PARAVIEW_XMF_FILE:-${output_dir}/inductor.xmf}"
paraview_export_script="${script_dir}/export_paraview_results.py"
mesh_output_dir="$(dirname "${mesh_file}")"
direct_h5_file="${mesh_output_dir}/similie_linear_magnetostatics.h5"
default_result_file="${mesh_output_dir}/similie_magnetostatics_inputs.pos"
legacy_result_file="${mesh_output_dir}/similie_linear_magnetostatics_inputs.pos"
analytical_l_rel_tolerance="${SIMILIE_ONELAB_ANALYTICAL_L_REL_TOLERANCE:-0.1}"

if [[ ! -f "${geometry_file}" ]]; then
    echo "missing inductor geometry file: ${geometry_file}" >&2
    exit 1
fi

if [[ ! -f "${model_data_file}" ]]; then
    echo "missing inductor model data file: ${model_data_file}" >&2
    exit 1
fi

if [[ ! -f "${problem_file}" ]]; then
    echo "missing SimiLie .silpro problem file: ${problem_file}" >&2
    exit 1
fi

if [[ ! -x "${onelab_client}" ]]; then
    echo "missing SimiLie ONELAB client executable: ${onelab_client}" >&2
    echo "build the project first, e.g. in ${build_dir}" >&2
    exit 1
fi

if [[ ! -f "${paraview_export_script}" ]]; then
    echo "missing Paraview export script: ${paraview_export_script}" >&2
    exit 1
fi

if ! command -v "${gmsh_executable}" >/dev/null 2>&1; then
    echo "gmsh executable not found: ${gmsh_executable}" >&2
    echo "set GMSH_EXECUTABLE if gmsh is installed under a different name or path" >&2
    exit 1
fi

gmsh_args=()
for arg in "$@"; do
    case "${arg}" in
        --dim=2)
            model_dimension=2
            ;;
        --dim=3)
            model_dimension=3
            ;;
        --bool)
            use_bool_test=1
            model_dimension=3
            open_cascade_model=1
            ;;
        --linear)
            physics_mode="LinearMagnetostatics"
            ;;
        --nonlinear)
            physics_mode="NonLinearMagnetostatics"
            ;;
        --matrix-free)
            use_matrix_free=1
            ;;
        --assembled-matrix)
            use_matrix_free=0
            ;;
        *)
            gmsh_args+=("${arg}")
            ;;
    esac
done

case "${model_dimension}" in
    2)
        gmsh_mesh_dimension=2
        fe_model_dimension=0
        ;;
    3)
        gmsh_mesh_dimension=3
        fe_model_dimension=1
        ;;
    *)
        echo "unsupported model dimension: ${model_dimension}" >&2
        echo "use --dim=2 or --dim=3" >&2
        exit 1
        ;;
esac
if [[ -z "${SIMILIE_ONELAB_ANALYTICAL_L_REL_TOLERANCE+x}" && "${model_dimension}" -eq 3 ]]; then
    # TODO restrict tolerancy
    analytical_l_rel_tolerance=0.3
fi

rm -f "${mesh_file}" "${result_file}" "${paraview_h5_file}" "${paraview_xmf_file}" "${direct_h5_file}"

control_file="$(mktemp "${script_dir}/.run_similie_onelab_XXXXXX.geo")"
log_file="$(mktemp "${TMPDIR:-/tmp}/run_similie_onelab_XXXXXX.log")"
effective_problem_file="${problem_file}"
patched_problem_file=""
if [[ -n "${physics_mode}" || -n "${use_matrix_free}" ]]; then
    patched_problem_file="$(mktemp "${script_dir}/.run_similie_onelab_XXXXXX.silpro")"
    sed_expressions=()
    if [[ -n "${physics_mode}" ]]; then
        sed_expressions+=(
            -e "s/^[[:space:]]*Physics[[:space:]].*;/  Physics ${physics_mode};/"
        )
    fi
    if [[ -n "${use_matrix_free}" ]]; then
        sed_expressions+=(
            -e "s/^[[:space:]]*UseMatrixFree[[:space:]].*;/  UseMatrixFree ${use_matrix_free};/"
        )
    fi
    sed \
        "${sed_expressions[@]}" \
        "${problem_file}" > "${patched_problem_file}"
    effective_problem_file="${patched_problem_file}"
fi
trap 'rm -f "${control_file}" "${patched_problem_file}" "${log_file}"' EXIT

cat > "${control_file}" <<EOF
Mesh ${gmsh_mesh_dimension};
OnelabRun("SimiLie", "${onelab_client}");
EOF

"${gmsh_executable}" \
    -parse_and_exit \
    -nopopup \
    -v 3 \
    -setnumber General.Terminal 1 \
    -setnumber Mesh.Binary 0 \
    -setnumber Mesh.MshFileVersion 2.2 \
    -setnumber "Input/00FE model" "${fe_model_dimension}" \
    -setnumber "Input/00OpenCASCADE model?" "${open_cascade_model}" \
    -setstring "0Modules/SimiLie/0Control/Problem file" "${effective_problem_file}" \
    -setstring "0Modules/SimiLie/0Control/Mesh file" "${mesh_file}" \
    "${geometry_file}" \
    "${control_file}" \
    "${gmsh_args[@]}" \
    2>&1 \
    | sed -u '/^Info[[:space:]]*: SimiLie -[[:space:]]*$/d' \
    | tee "${log_file}"

actual_result_file="${result_file}"
if [[ ! -f "${actual_result_file}" ]]; then
    if [[ -f "${default_result_file}" ]]; then
        actual_result_file="${default_result_file}"
    elif [[ -f "${legacy_result_file}" ]]; then
        actual_result_file="${legacy_result_file}"
    fi
fi

python3 "${paraview_export_script}" \
    --mesh "${mesh_file}" \
    --input-fields-pos "${actual_result_file}" \
    --h5-output "${paraview_h5_file}" \
    --xmf-output "${paraview_xmf_file}"

assert_example_results=0
if [[ -f "${build_dir}/CMakeCache.txt" ]] && grep -Eq \
    "^SIMILIE_ASSERT_EXAMPLE_RESULTS(_CORRECTNESS)?:BOOL=ON$" \
    "${build_dir}/CMakeCache.txt"; then
    assert_example_results=1
fi

if [[ "${assert_example_results}" -eq 0 ]]; then
    exit 0
fi

python3 - "${log_file}" "${mesh_file}" "${model_data_file}" "${analytical_l_rel_tolerance}" <<'PY'
import math
import re
import sys
from array import array
from pathlib import Path


def parse_airgap_tag(inductor_data_geo: Path) -> int:
    pattern = re.compile(r"^AIRGAP\s*=\s*(\d+);")
    for raw_line in inductor_data_geo.read_text().splitlines():
        match = pattern.match(raw_line.strip())
        if match:
            return int(match.group(1))
    raise RuntimeError("failed to locate AIRGAP tag in inductor_data.geo")


def parse_log(log_file: Path) -> tuple[float, float, float, float]:
    symmetry_pattern = re.compile(r"SymmetryFactor=([0-9.eE+-]+)")
    turns_pattern = re.compile(r"N=([0-9.eE+-]+)")
    length_z_pattern = re.compile(r"Lz=([0-9.eE+-]+)\s+m")
    inductance_pattern = re.compile(r"L=([0-9.eE+-]+)\s+H")
    symmetry_factor = None
    num_turns = None
    length_z = None
    numerical_l = None
    for raw_line in log_file.read_text().splitlines():
        if symmetry_factor is None:
            match = symmetry_pattern.search(raw_line)
            if match:
                symmetry_factor = float(match.group(1))
        if num_turns is None:
            match = turns_pattern.search(raw_line)
            if match:
                num_turns = float(match.group(1))
        if length_z is None:
            match = length_z_pattern.search(raw_line)
            if match:
                length_z = float(match.group(1))
        if numerical_l is None:
            match = inductance_pattern.search(raw_line)
            if match:
                numerical_l = float(match.group(1))
    if symmetry_factor is None:
        raise RuntimeError("failed to parse SymmetryFactor from ONELAB log")
    if num_turns is None:
        raise RuntimeError("failed to parse number of turns from ONELAB log")
    if length_z is None:
        raise RuntimeError("failed to parse length_z from ONELAB log")
    if numerical_l is None:
        raise RuntimeError("failed to parse numerical inductance from ONELAB log")
    return symmetry_factor, num_turns, length_z, numerical_l


def parse_mesh_airgap_geometry(
    mesh_file: Path, airgap_tag: int, logged_length_z: float
) -> tuple[float, float, float]:
    with mesh_file.open() as stream:
        for line in stream:
            if line.strip() == "$Nodes":
                break
        else:
            raise RuntimeError("missing $Nodes section in mesh file")
        node_count = int(next(stream).strip())
        xs = array("d", [0.0]) * (node_count + 1)
        ys = array("d", [0.0]) * (node_count + 1)
        zs = array("d", [0.0]) * (node_count + 1)
        for _ in range(node_count):
            tag_str, x_str, y_str, z_str = next(stream).split()
            tag = int(tag_str)
            xs[tag] = float(x_str)
            ys[tag] = float(y_str)
            zs[tag] = float(z_str)
        length_z = max(zs) - min(zs)
        for line in stream:
            if line.strip() == "$Elements":
                break
        else:
            raise RuntimeError("missing $Elements section in mesh file")
        element_count = int(next(stream).strip())
        airgap_cells: list[tuple[float, float, float, float, float, float]] = []
        mesh_dimension = None
        for _ in range(element_count):
            parts = next(stream).split()
            element_type = int(parts[1])
            if element_type not in (3, 5):
                continue
            num_tags = int(parts[2])
            physical_tag = int(parts[3]) if num_tags > 0 else 0
            if physical_tag != airgap_tag:
                continue
            node_count_for_element = 4 if element_type == 3 else 8
            node_tags = [int(value) for value in parts[3 + num_tags : 3 + num_tags + node_count_for_element]]
            x_values = [xs[tag] for tag in node_tags]
            y_values = [ys[tag] for tag in node_tags]
            z_values = [zs[tag] for tag in node_tags]
            airgap_cells.append(
                (
                    min(x_values),
                    max(x_values),
                    min(y_values),
                    max(y_values),
                    min(z_values),
                    max(z_values),
                )
            )
            mesh_dimension = 2 if element_type == 3 else 3
    if not airgap_cells:
        raise RuntimeError("failed to locate any AIRGAP cells in mesh file")
    top_y = max(cell[3] for cell in airgap_cells)
    gap_min_y = min(cell[2] for cell in airgap_cells)
    gap_length = top_y - gap_min_y
    tol = 1.0e-12
    if mesh_dimension == 2:
        length_z = logged_length_z
        upper_surface_measure = sum(
            (cell[1] - cell[0]) * length_z
            for cell in airgap_cells
            if abs(cell[3] - top_y) < tol
        )
    else:
        upper_surface_measure = sum(
            (cell[1] - cell[0]) * (cell[5] - cell[4])
            for cell in airgap_cells
            if abs(cell[3] - top_y) < tol
        )
    return upper_surface_measure, gap_length, length_z


log_file = Path(sys.argv[1])
mesh_file = Path(sys.argv[2])
inductor_data_geo = Path(sys.argv[3])
relative_tolerance = float(sys.argv[4])

symmetry_factor, num_turns, logged_length_z, numerical_l = parse_log(log_file)
airgap_tag = parse_airgap_tag(inductor_data_geo)
surface_measure, gap_length, length_z = parse_mesh_airgap_geometry(
    mesh_file, airgap_tag, logged_length_z
)
mu0 = 4.0e-7 * math.pi
# The structured example models the complete EI magnetic circuit. Its upper
# air-gap flux traverses four equivalent circuit sections. The inherited
# ONELAB SymmetryFactor describes the selected geometry reduction and is
# intentionally reported below, but it is not used by this full-grid estimate.
magnetic_circuit_factor = 4.0
analytical_l = mu0 * surface_measure / (
    magnetic_circuit_factor * gap_length * num_turns
)
relative_error = abs(numerical_l - analytical_l) / analytical_l

print(
    "Analytical air-gap estimate for logged L=Phi/(N*integral(J_z dS)):"
    f" L_ana={analytical_l:.9e} H"
    f" (mu0*S/(4*l*N), S={surface_measure:.9e} m^2,"
    f" l={gap_length:.9e} m, Lz={length_z:.9e} m, N={num_turns:.9e},"
    f" inherited SymmetryFactor={symmetry_factor:.9e}),"
    f" numerical L={numerical_l:.9e} H,"
    f" relative error={relative_error:.3%}"
)

if relative_error > relative_tolerance:
    raise SystemExit(
        f"Analytical logged-L check failed: relative error {relative_error:.3%} "
        f"exceeds tolerance {relative_tolerance:.3%}"
    )
PY
