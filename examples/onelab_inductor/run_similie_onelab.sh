#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
inductor_dir="${script_dir}/Inductor"
geometry_file="${script_dir}/similie_rectilinear_inductor.geo"

gmsh_executable="${GMSH_EXECUTABLE:-gmsh}"
build_dir="${SIMILIE_ONELAB_BUILD_DIR:-${repo_root}/build}"
onelab_client="${SIMILIE_ONELAB_BINARY:-${build_dir}/onelab_interface/similie_onelab}"
mesh_file="${SIMILIE_ONELAB_MESH_FILE:-${script_dir}/similie_inductor_rectilinear.msh}"
result_file="${SIMILIE_ONELAB_RESULT_FILE:-${script_dir}/similie_inductor_node_positions.pos}"

if [[ ! -d "${inductor_dir}" ]]; then
    echo "missing Inductor example directory: ${inductor_dir}" >&2
    exit 1
fi

if [[ ! -f "${geometry_file}" ]]; then
    echo "missing Inductor geometry file: ${geometry_file}" >&2
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

control_file="$(mktemp "${script_dir}/.run_similie_onelab_XXXXXX.geo")"
trap 'rm -f "${control_file}"' EXIT

cat > "${control_file}" <<EOF
Mesh 3;
Save "${mesh_file}";
OnelabRun("SimiLie", "${onelab_client}");
EOF

"${gmsh_executable}" \
    -parse_and_exit \
    -nopopup \
    -v 3 \
    -setnumber General.Terminal 1 \
    -setnumber Mesh.Binary 0 \
    -setnumber Mesh.MshFileVersion 2.2 \
    -setstring "0Modules/SimiLie/0Control/Output view file" "${result_file}" \
    -setnumber "0Modules/SimiLie/0Control/Export fake view" 1 \
    -setstring "0Modules/SimiLie/0Control/Input mesh file" "${mesh_file}" \
    "${geometry_file}" \
    "${control_file}" \
    "$@"
