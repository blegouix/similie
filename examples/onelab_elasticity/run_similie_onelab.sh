#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2026 Baptiste Legouix
# SPDX-License-Identifier: AGPL-3.0-or-later

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
geometry_file="${script_dir}/wrench_structured.geo"
problem_file="${SIMILIE_ONELAB_PROBLEM_FILE:-${script_dir}/elasticity.silpro}"
output_dir="$(pwd)"

gmsh_executable="${GMSH_EXECUTABLE:-gmsh}"
build_dir="${SIMILIE_ONELAB_BUILD_DIR:-${repo_root}/build}"
onelab_client="${SIMILIE_ONELAB_BINARY:-${build_dir}/onelab_interface/similie_onelab}"
mesh_file="${SIMILIE_ONELAB_MESH_FILE:-${output_dir}/wrench_structured.msh}"
result_file="${SIMILIE_ONELAB_RESULT_FILE:-${output_dir}/similie_elasticity_inputs.pos}"

if [[ ! -f "${geometry_file}" ]]; then
    echo "missing elasticity geometry file: ${geometry_file}" >&2
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

if ! command -v "${gmsh_executable}" >/dev/null 2>&1; then
    echo "gmsh executable not found: ${gmsh_executable}" >&2
    echo "set GMSH_EXECUTABLE if gmsh is installed under a different name or path" >&2
    exit 1
fi

rm -f "${mesh_file}" "${result_file}"

control_file="$(mktemp "${script_dir}/.run_similie_onelab_XXXXXX.geo")"
trap 'rm -f "${control_file}"' EXIT

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
    -setstring "0Modules/SimiLie/0Control/Problem file" "${problem_file}" \
    -setstring "0Modules/SimiLie/0Control/Mesh file" "${mesh_file}" \
    "${geometry_file}" \
    "${control_file}" \
    "$@" \
    2>&1 \
    | sed -u '/^Info[[:space:]]*: SimiLie -[[:space:]]*$/d'
