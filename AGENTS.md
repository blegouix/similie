<!--
SPDX-FileCopyrightText: 2026 Baptiste Legouix
SPDX-License-Identifier: MIT
-->

# AGENTS.md for SimiLie

<!-- Example usage: "run tests" -->

# Clone
- Clone with `git clone --recurse-submodules git@github.com:blegouix/similie.git` or any similar commande which suits the network configuration. Be sure to pull the submodules recursively.

# Build environment
- **Requirement:** By default (no particular contradictory instructions), build and run in the `docker/similie_env/Dockerfile` image that you call `similie_env:latest`. You'll need to mount the `similie/` folder before compiling it. Compile directly in the host environment only if specifically asked for (if so, check if all the necessary dependencies listed in `docker/similie_env/Dockerfile` are available in the current environment, in particular `nvcc` and `openmpi`).
- **CUDA in Docker:** When building or running the CUDA backend inside Docker, start the container with `--gpus all` so the NVIDIA driver and GPU devices are exposed in the container.
- **Configure CMake:** For CPU with `cmake -DCMAKE_CXX_COMPILER=g++-13 -DCMAKE_BUILD_TYPE=Debug -DKokkos_ENABLE_OPENMP=ON -B build -S .` or for CUDA with `cmake -DCMAKE_CXX_COMPILER=$PWD/vendor/ddc/vendor/kokkos/bin/nvcc_wrapper -DCMAKE_BUILD_TYPE=Debug -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_<YOUR_ARCH_NAMECODE>=ON -B build -S .`. Of course replace `<YOUR_ARCH_NAMECODE>` with the arch namecode which suits the GPU.
- **Compile:** As usual once configured (`make -jN_PROC` from the `build/` folder). Never exceed N_PROC=8. Be sure to compile in the `build/` for CUDA backend and `build_cpu/` for CPU backend.

## Run tests
- From `build`, first recompile then run `ctest`.
