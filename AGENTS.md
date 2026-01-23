<!--
SPDX-FileCopyrightText: 2026 Baptiste Legouix
SPDX-License-Identifier: MIT
-->

# AGENTS.md for SimiLie

# Clone
- Clone with `git clone --recurse-submodules git@github.com:blegouix/similie.git` or any similar commande which suits the network configuration. Be sure to pull the submodules recursively.

# Build environment
- **Requirement:** Build and use the `docker/similie/Dockerfile` image before compiling or running tests. Do not build on the host directly. If `docker` is not installed, install it first (refer to the online documentation).
- Configure `CMake` for CPU with `cmake -DCMAKE_CXX_COMPILER=g++-13 -DCMAKE_BUILD_TYPE=Debug -DKokkos_ENABLE_OPENMP=ON -B build -S .` or for CUDA with `cmake -DCMAKE_CXX_COMPILER=$PWD/vendor/ddc/vendor/kokkos/bin/nvcc_wrapper -DCMAKE_BUILD_TYPE=Debug -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_<YOUR_ARCH_NAMECODE>=ON -B build -S .`. Of course replace `<YOUR_ARCH_NAMECODE>` with the arch namecode which suits the GPU.
- Compile as usual once configured (`make -jN_PROC` from `build`).

## Run tests
- From `build`, run `ctest`
