#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2026 Baptiste Legouix
# SPDX-License-Identifier: AGPL-3.0-or-later

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
geometry_file="${script_dir}/wrench2D.geo"
problem_file="${SIMILIE_ONELAB_PROBLEM_FILE:-${script_dir}/elasticity.silpro}"
getdp_problem_file="${SIMILIE_GETDP_PROBLEM_FILE:-${script_dir}/getdp_ref/wrench2D.pro}"
output_dir="$(pwd)"
use_matrix_free=""
solver="similie"

gmsh_executable="${GMSH_EXECUTABLE:-gmsh}"
getdp_executable="${GETDP_EXECUTABLE:-getdp}"
build_dir="${SIMILIE_ONELAB_BUILD_DIR:-${repo_root}/build}"
onelab_client="${SIMILIE_ONELAB_BINARY:-${build_dir}/onelab_interface/similie_onelab}"
mesh_file="${SIMILIE_ONELAB_MESH_FILE:-${output_dir}/wrench2D.msh}"
result_file="${SIMILIE_ONELAB_RESULT_FILE:-${output_dir}/similie_elasticity_inputs.pos}"
getdp_deflection_rel_tolerance="${SIMILIE_ONELAB_GETDP_DEFLECTION_REL_TOLERANCE:-0.15}"

if [[ ! -f "${geometry_file}" ]]; then
    echo "missing elasticity geometry file: ${geometry_file}" >&2
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
        --solver=similie|--similie)
            solver="similie"
            ;;
        --solver=getdp|--solver=gmsh|--getdp|--gmsh)
            solver="getdp"
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

if [[ "${solver}" == "similie" ]]; then
    if [[ ! -f "${problem_file}" ]]; then
        echo "missing SimiLie .silpro problem file: ${problem_file}" >&2
        exit 1
    fi

    if [[ ! -x "${onelab_client}" ]]; then
        echo "missing SimiLie ONELAB client executable: ${onelab_client}" >&2
        echo "build the project first, e.g. in ${build_dir}" >&2
        exit 1
    fi
else
    if [[ ! -f "${getdp_problem_file}" ]]; then
        echo "missing GetDP reference problem file: ${getdp_problem_file}" >&2
        exit 1
    fi

    if ! command -v "${getdp_executable}" >/dev/null 2>&1; then
        echo "getdp executable not found: ${getdp_executable}" >&2
        echo "set GETDP_EXECUTABLE if getdp is installed under a different name or path" >&2
        exit 1
    fi
fi

# GetDP has its own parameter database. Convert the geometry UI units once
# and pass identical physical data to both solvers.
getdp_args=()
for ((i=0; i<${#gmsh_args[@]}; ++i)); do
    if [[ "${gmsh_args[i]}" != "-setnumber" ]]; then continue; fi
    name="${gmsh_args[i+1]}"
    value="${gmsh_args[i+2]}"
    case "${name}" in
        "Material/Young modulus [GPa]") variable=Young; scale=1e9 ;;
        "Material/Poisson coefficient []") variable=Poisson; scale=1 ;;
        "Material/Applied force [N]") variable=AppliedForce; scale=1 ;;
        "Geometry/4Thickness (mm)") variable=WrenchThickness; scale=1e-3 ;;
        "Geometry/5Arm Width (mm)") variable=LoadWidth; scale=1e-3 ;;
        *) continue ;;
    esac
    converted="$(python3 -c 'import sys; print(float(sys.argv[1])*float(sys.argv[2]))' "${value}" "${scale}")"
    getdp_args+=(-setnumber "${variable}" "${converted}")
done

rm -f "${mesh_file}" "${result_file}"

if [[ "${solver}" == "getdp" ]]; then
    "${gmsh_executable}" \
        -2 \
        -nopopup \
        -v 3 \
        -format msh2 \
        -o "${mesh_file}" \
        "${geometry_file}" \
        "${gmsh_args[@]}"

    "${getdp_executable}" \
        "${getdp_problem_file}" \
        -msh "${mesh_file}" \
        -name "${output_dir}/wrench2D" \
        -setstring "GetDPOutputDir" "${output_dir}/res_elasticity" \
        "${getdp_args[@]}" \
        -solve Elast_u \
        -pos Get_LocalFields \
        -v2
    exit 0
fi

control_file="$(mktemp "${script_dir}/.run_similie_onelab_XXXXXX.geo")"
log_file="$(mktemp "${TMPDIR:-/tmp}/run_similie_onelab_elasticity_XXXXXX.log")"
effective_problem_file="${problem_file}"
patched_problem_file=""
if [[ -n "${use_matrix_free}" ]]; then
    patched_problem_file="$(mktemp "${script_dir}/.run_similie_onelab_XXXXXX.silpro")"
    sed \
        -e "s/^[[:space:]]*UseMatrixFree[[:space:]].*;/  UseMatrixFree ${use_matrix_free};/" \
        "${problem_file}" > "${patched_problem_file}"
    effective_problem_file="${patched_problem_file}"
fi
cleanup() {
    rm -f "${control_file}" "${patched_problem_file}" "${log_file}"
}
trap cleanup EXIT

cat > "${control_file}" <<EOF
Mesh 2;
OnelabRun("SimiLie", "${onelab_client}");
EOF

"${gmsh_executable}" \
    -parse_and_exit \
    -nopopup \
    -v 3 \
    -setnumber General.Terminal 1 \
    -setnumber Mesh.Binary 0 \
    -setnumber Mesh.MshFileVersion 2.2 \
    -setstring "0Modules/SimiLie/0Control/Problem file" "${effective_problem_file}" \
    -setstring "0Modules/SimiLie/0Control/Mesh file" "${mesh_file}" \
    "${geometry_file}" \
    "${control_file}" \
    "${gmsh_args[@]}" \
    2>&1 \
    | sed -u '/^Info[[:space:]]*: SimiLie -[[:space:]]*$/d' \
    | tee "${log_file}"

# Gmsh can return success even when its ONELAB client failed.
if ! grep -q "SimiLie elasticity diagnostics:" "${log_file}"; then
    echo "SimiLie elasticity solve did not complete" >&2
    exit 1
fi
actual_result="$(dirname "${mesh_file}")/similie_elasticity_inputs.pos"
if [[ "${actual_result}" != "${result_file}" ]]; then
    cp "${actual_result}" "${result_file}"
fi

if [[ ! -f "${build_dir}/CMakeCache.txt" ]] || ! grep -Fqx \
    "SIMILIE_ASSERT_EXAMPLE_RESULTS_CORRECTNESS:BOOL=ON" \
    "${build_dir}/CMakeCache.txt"; then
    exit 0
fi

if [[ ! -f "${getdp_problem_file}" ]]; then
    echo "missing GetDP reference problem file: ${getdp_problem_file}" >&2
    exit 1
fi
if ! command -v "${getdp_executable}" >/dev/null 2>&1; then
    echo "getdp executable not found: ${getdp_executable}" >&2
    exit 1
fi

getdp_output_dir="$(dirname "${mesh_file}")/getdp_reference"
mkdir -p "${getdp_output_dir}"
set +e
"${getdp_executable}" \
        "${getdp_problem_file}" \
        -msh "${mesh_file}" \
        -name "${getdp_output_dir}/wrench2D" \
        -solver "${script_dir}/getdp_ref/solver.par" \
        -Scaling 1 \
        -Algorithm 8 \
        -Krylov_Size 200 \
        -Nb_Iter_Max 100000 \
        -Stopping_Test 1e-10 \
        -setstring "GetDPOutputDir" "${getdp_output_dir}" \
        "${getdp_args[@]}" \
        -solve Elast_u \
        -pos Get_Probe_Displacement
getdp_status=$?
set -e
if [[ "${getdp_status}" -ne 0 ]]; then
    if [[ "${getdp_status}" -ne 134 || ! -s "${getdp_output_dir}/u_probe.txt" ]]; then
        echo "GetDP reference solve failed with exit status ${getdp_status}" >&2
        exit "${getdp_status}"
    fi
    echo "warning: GetDP aborted during final cleanup after writing u_probe.txt" >&2
fi

python3 - "${log_file}" "${getdp_output_dir}/u_probe.txt" "${getdp_deflection_rel_tolerance}" <<'PY'
import re
import sys
from pathlib import Path


def parse_similie_probe_displacement(log_file: Path) -> float:
    pattern = re.compile(r"SimiLie elasticity diagnostics:.*uy_probe=([0-9.eE+-]+)\s+m")
    for line in log_file.read_text().splitlines():
        if match := pattern.search(line):
            return float(match.group(1))
    raise RuntimeError("failed to parse uy_probe from SimiLie elasticity diagnostics")


def parse_getdp_probe_displacement(displacement_file: Path) -> float:
    rows = [line.split() for line in displacement_file.read_text().splitlines() if line.strip()]
    if len(rows) != 1 or len(rows[0]) < 10:
        raise RuntimeError(f"invalid GetDP point-probe data in {displacement_file}")
    return float(rows[0][9])


similie_displacement = parse_similie_probe_displacement(Path(sys.argv[1]))
getdp_displacement = parse_getdp_probe_displacement(Path(sys.argv[2]))
relative_tolerance = float(sys.argv[3])
if getdp_displacement == 0.0:
    raise RuntimeError("GetDP returned a zero probe displacement")
relative_error = abs(similie_displacement - getdp_displacement) / abs(getdp_displacement)

print(
    "SimiLie/GetDP probe-deflection consistency:"
    f" SimiLie={similie_displacement:.9e} m,"
    f" GetDP={getdp_displacement:.9e} m,"
    f" relative error={relative_error:.3%}"
)
if relative_error > relative_tolerance:
    raise SystemExit(
        "SimiLie/GetDP probe-deflection consistency check failed: "
        f"relative error {relative_error:.3%} exceeds tolerance {relative_tolerance:.3%}"
    )
PY
