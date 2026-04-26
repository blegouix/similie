<!--
SPDX-FileCopyrightText: 2026 Baptiste Legouix
SPDX-License-Identifier: MIT
-->

# AGENTS.md for SimiLie

<!-- Example usage: "run tests" -->

# Clone
- Clone with `git clone --recurse-submodules git@github.com:blegouix/similie.git` or any similar commande which suits the network configuration. Be sure to pull the submodules recursively.

# Build environment
- **Default build** By default (no particular contradictory instructions), you will check if a GPU is available with `nvidia-smi`. If so, you will try to compile for it.
- **Requirement:** By default (no particular contradictory instructions), build and run in the `docker/similie_env/Dockerfile` image that you call `similie_env:latest`. You'll need to mount the `similie/` folder before compiling it. Compile directly in the host environment only if specifically asked for (if so, check if all the necessary dependencies listed in `docker/similie_env/Dockerfile` are available in the current environment, in particular `nvcc` and `openmpi`).
- **Tests and simulations:** Always run tests, examples, and simulations inside Docker unless the user explicitly asks for a host run. Do not use host-side build directories or host-side executables for validation by default.
- **Mount path in Docker:** If you reuse an existing `build/` or `build_cpu/` directory from the host, mount the repository in Docker at the same absolute path as on the host. Otherwise CMake may fail because the cached source/build paths no longer match.
- **CUDA in Docker:** When building or running the CUDA backend inside Docker, start the container with `--gpus all` so the NVIDIA driver and GPU devices are exposed in the container.
- **Git safe.directory in Docker:** Some vendored submodules query Git metadata during CMake. If Docker reports "detected dubious ownership" on the mounted repository or its submodules, add temporary `git config --global --add safe.directory <path>` entries inside the container for the repository and the relevant submodules before building.
- **Configure CMake:** For CPU with `cmake -DCMAKE_CXX_COMPILER=g++-13 -DCMAKE_BUILD_TYPE=Debug -DKokkos_ENABLE_OPENMP=ON -B build -S .` or for CUDA with `cmake -DCMAKE_CXX_COMPILER=$PWD/vendor/ddc/vendor/kokkos/bin/nvcc_wrapper -DCMAKE_BUILD_TYPE=Debug -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_<YOUR_ARCH_NAMECODE>=ON -B build -S .`. Of course replace `<YOUR_ARCH_NAMECODE>` with the arch namecode which suits the GPU.
- **Compile:** As usual once configured (`make -jN_PROC` from the `build/` folder). Never exceed N_PROC=8. Be sure to compile in the `build/` for CUDA backend and `build_cpu/` for CPU backend.

## Run tests
- Run tests from inside Docker only. From `build`, first recompile then run `ctest`.
