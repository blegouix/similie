#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

image_name="${SIMILIE_ONELAB_IMAGE:-similie_env:latest}"
build_dir="${SIMILIE_ONELAB_BUILD_DIR:-${repo_root}/agent_build_cpu}"
binary_path="${SIMILIE_ONELAB_BINARY:-${build_dir}/onelab_interface/similie_onelab}"

if [[ ! -x "${binary_path}" ]]; then
    echo "missing ONELAB client executable: ${binary_path}" >&2
    echo "build the project first, e.g. in ${build_dir}" >&2
    exit 1
fi

exec docker run --rm \
    -v "${repo_root}:${repo_root}" \
    -w "${repo_root}" \
    "${image_name}" \
    "${binary_path}" \
    "$@"
