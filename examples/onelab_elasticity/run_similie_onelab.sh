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
        -solver "${script_dir}/getdp_ref/solver.par" \
        -setstring "GetDPOutputDir" "${output_dir}/res_elasticity" \
        -solve Elast_u \
        -pos Get_LocalFields \
        -v2
    exit 0
fi

control_file="$(mktemp "${script_dir}/.run_similie_onelab_XXXXXX.geo")"
effective_problem_file="${problem_file}"
patched_problem_file=""
if [[ -n "${use_matrix_free}" ]]; then
    patched_problem_file="$(mktemp "${script_dir}/.run_similie_onelab_XXXXXX.silpro")"
    sed \
        -e "s/^[[:space:]]*UseMatrixFree[[:space:]].*;/  UseMatrixFree ${use_matrix_free};/" \
        "${problem_file}" > "${patched_problem_file}"
    effective_problem_file="${patched_problem_file}"
fi
trap 'rm -f "${control_file}" "${patched_problem_file}"' EXIT

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
    | sed -u '/^Info[[:space:]]*: SimiLie -[[:space:]]*$/d'
