#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
inductor_dir="${script_dir}/Inductor"
geometry_file="${script_dir}/inductor.geo"
problem_file="${SIMILIE_ONELAB_PROBLEM_FILE:-${script_dir}/inductor.silpro}"
output_dir="$(pwd)"
model_dimension=2
use_bool_test=0
physics_mode=""
open_cascade_model=0

gmsh_executable="${GMSH_EXECUTABLE:-gmsh}"
build_dir="${SIMILIE_ONELAB_BUILD_DIR:-${repo_root}/build}"
onelab_client="${SIMILIE_ONELAB_BINARY:-${build_dir}/onelab_interface/similie_onelab}"
mesh_file="${SIMILIE_ONELAB_MESH_FILE:-${output_dir}/inductor.msh}"
result_file="${SIMILIE_ONELAB_RESULT_FILE:-${output_dir}/similie_magnetostatics_inputs.pos}"
paraview_h5_file="${SIMILIE_PARAVIEW_H5_FILE:-${output_dir}/inductor.h5}"
paraview_xmf_file="${SIMILIE_PARAVIEW_XMF_FILE:-${output_dir}/inductor.xmf}"
paraview_export_script="${script_dir}/export_paraview_results.py"
direct_h5_file="${output_dir}/similie_linear_magnetostatics.h5"

if [[ ! -d "${inductor_dir}" ]]; then
    echo "missing Inductor example directory: ${inductor_dir}" >&2
    exit 1
fi

if [[ ! -f "${geometry_file}" ]]; then
    echo "missing Inductor geometry file: ${geometry_file}" >&2
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

rm -f "${mesh_file}" "${result_file}" "${paraview_h5_file}" "${paraview_xmf_file}" "${direct_h5_file}"

control_file="$(mktemp "${script_dir}/.run_similie_onelab_XXXXXX.geo")"
effective_problem_file="${problem_file}"
patched_problem_file=""
if [[ -n "${physics_mode}" ]]; then
    patched_problem_file="$(mktemp "${script_dir}/.run_similie_onelab_XXXXXX.silpro")"
    sed \
        "s/^[[:space:]]*Physics[[:space:]].*;/  Physics ${physics_mode};/" \
        "${problem_file}" > "${patched_problem_file}"
    effective_problem_file="${patched_problem_file}"
fi
trap 'rm -f "${control_file}" "${patched_problem_file}"' EXIT

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
    "${gmsh_args[@]}"

python3 "${paraview_export_script}" \
    --mesh "${mesh_file}" \
    --input-fields-pos "${result_file}" \
    --h5-output "${paraview_h5_file}" \
    --xmf-output "${paraview_xmf_file}"
