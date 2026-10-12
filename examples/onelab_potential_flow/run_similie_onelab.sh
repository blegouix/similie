#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2026 Baptiste Legouix
# SPDX-License-Identifier: AGPL-3.0-or-later
# // AI-GENERATED

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
problem_file="${SIMILIE_ONELAB_PROBLEM_FILE:-${script_dir}/potential_flow.silpro}"
getdp_problem_file="${SIMILIE_GETDP_PROBLEM_FILE:-${script_dir}/ref/magnus.pro}"
build_dir="${SIMILIE_ONELAB_BUILD_DIR:-${repo_root}/build}"
onelab_client="${SIMILIE_ONELAB_BINARY:-${build_dir}/onelab_interface/similie_onelab}"
mesh_file="${SIMILIE_ONELAB_MESH_FILE:-${PWD}/magnus.msh}"
result_file="${SIMILIE_ONELAB_RESULT_FILE:-$(dirname "${mesh_file}")/similie_potential_flow.pos}"
geometry_dir="$(mktemp -d "$(dirname "${mesh_file}")/.run_similie_onelab_geometry_XXXXXX")"
trap 'rm -rf "${geometry_dir}"' EXIT
cp "${script_dir}/magnus.geo" "${script_dir}/magnus_common.pro" \
    "${script_dir}/nacaAirfoil.geo" "${geometry_dir}/"
geometry_file="${geometry_dir}/magnus.geo"
gmsh_executable="${GMSH_EXECUTABLE:-gmsh}"
getdp_executable="${GETDP_EXECUTABLE:-getdp}"
solver=similie
matrix_free=""
gmsh_args=()
for arg in "$@"; do
    case "${arg}" in
        --solver=similie|--similie) solver=similie ;;
        --solver=getdp|--solver=gmsh|--getdp|--gmsh) solver=getdp ;;
        --matrix-free) matrix_free=1 ;;
        --assembled-matrix) matrix_free=0 ;;
        *) gmsh_args+=("${arg}") ;;
    esac
done

if ! command -v "${gmsh_executable}" >/dev/null 2>&1; then
    echo "gmsh executable not found: ${gmsh_executable}" >&2
    exit 1
fi
if [[ "${solver}" == getdp ]]; then
    if ! command -v "${getdp_executable}" >/dev/null 2>&1; then
        echo "getdp executable not found: ${getdp_executable}" >&2
        exit 1
    fi
    if [[ "${SIMILIE_GETDP_REUSE_MESH:-0}" != 1 ]]; then
        "${gmsh_executable}" -2 -nopopup -v 3 -format msh2 -o "${mesh_file}" \
            "${geometry_file}" "${gmsh_args[@]}"
    fi
    getdp_output_name="$(dirname "${mesh_file}")/magnus"
    # Keep the default problem and its output prefix together inside this case.
    if [[ -z "${SIMILIE_GETDP_PROBLEM_FILE:-}" ]]; then
        reference_dir="$(dirname "${mesh_file}")/getdp_reference"
        mkdir -p "${reference_dir}/ref"
        cp "${script_dir}/magnus_common.pro" "${reference_dir}/"
        cp "${script_dir}/ref/magnus.pro" "${reference_dir}/ref/"
        getdp_problem_file="${reference_dir}/ref/magnus.pro"
        getdp_output_name="${reference_dir}/ref/magnus"
        rm -f "${reference_dir}/ref/"{output.txt,KJ.pos,phi.pos,pressure.pos,velocity.pos}
    fi
    getdp_args=()
    for ((i=0; i<${#gmsh_args[@]}; ++i)); do
        if [[ "${gmsh_args[i]}" != -setnumber ]]; then continue; fi
        name="${gmsh_args[i+1]}"
        value="${gmsh_args[i+2]}"
        case "${name}" in
            "Model/Object") variable=RefObject; scale=1 ;;
            "Model/Impose circulation") variable=RefImposeCirculation; scale=1 ;;
            "Model/Air box size [m]") variable=RefBoxSize; scale=1 ;;
            "Model/V") variable=RefVelocity; scale=0.2777777777777778 ;;
            "Model/Angle of attack [deg]") variable=RefIncidence; scale=-0.017453292519943295 ;;
            "Model/Circ") variable=RefCirculation; scale=-1 ;;
            "Model/Dmdt") variable=RefMassFlowRate; scale=1 ;;
            *) continue ;;
        esac
        converted="$(python3 -c 'import sys; print(float(sys.argv[1])*float(sys.argv[2]))' "${value}" "${scale}")"
        getdp_args+=(-setnumber "${variable}" "${converted}")
    done
    "${getdp_executable}" "${getdp_problem_file}" -msh "${mesh_file}" \
        -name "${getdp_output_name}" -solve PotentialFlow -pos PotentialFlow \
        "${getdp_args[@]}" -v2
    if [[ -z "${SIMILIE_GETDP_PROBLEM_FILE:-}" ]]; then
        cp "${reference_dir}/ref/"{output.txt,pressure.pos,velocity.pos} \
            "$(dirname "${mesh_file}")/"
        reference_potential="${reference_dir}/ref/KJ.pos"
        if [[ ! -f "${reference_potential}" ]]; then
            reference_potential="${reference_dir}/ref/phi.pos"
        fi
        cp "${reference_potential}" "$(dirname "${mesh_file}")/getdp_potential.pos"
    fi
    exit 0
fi
if [[ ! -x "${onelab_client}" ]]; then
    echo "SimiLie ONELAB client not found: ${onelab_client}" >&2
    exit 1
fi
if [[ ! -f "${problem_file}" ]]; then
    echo "SimiLie problem file not found: ${problem_file}" >&2
    exit 1
fi
control_file="$(mktemp "${geometry_dir}/control_XXXXXX.geo")"
log_file="$(mktemp "${TMPDIR:-/tmp}/run_similie_onelab_potential_flow_XXXXXX.log")"
effective_problem_file="${problem_file}"
patched_problem_file=""
cleanup() {
    rm -f "${control_file}" "${log_file}" "${patched_problem_file}"
    rm -rf "${geometry_dir}"
}
trap cleanup EXIT
if [[ -n "${matrix_free}" ]]; then
    patched_problem_file="$(mktemp "${geometry_dir}/problem_XXXXXX.silpro")"
    sed -e "s/^[[:space:]]*UseMatrixFree[[:space:]].*;/  UseMatrixFree ${matrix_free};/" \
        "${problem_file}" > "${patched_problem_file}"
    effective_problem_file="${patched_problem_file}"
fi
cat > "${control_file}" <<CONTROL
Mesh 2;
OnelabRun("SimiLie", "${onelab_client}");
CONTROL
rm -f "${mesh_file}" "${result_file}"
"${gmsh_executable}" -parse_and_exit -nopopup -v 3 \
    -setnumber General.Terminal 1 \
    -setnumber Mesh.Binary 0 \
    -setnumber Mesh.MshFileVersion 2.2 \
    -setstring "0Modules/SimiLie/0Control/Problem file" "${effective_problem_file}" \
    -setstring "0Modules/SimiLie/0Control/Mesh file" "${mesh_file}" \
    "${geometry_file}" "${control_file}" "${gmsh_args[@]}" \
    2>&1 | tee "${log_file}"
if ! grep -q 'SimiLie potential-flow diagnostics:' "${log_file}"; then
    echo "SimiLie potential-flow solve did not complete" >&2
    exit 1
fi
actual_result="$(dirname "${mesh_file}")/similie_potential_flow.pos"
if [[ "${actual_result}" != "${result_file}" ]]; then
    cp "${actual_result}" "${result_file}"
fi
