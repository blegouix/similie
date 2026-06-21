<!--
SPDX-FileCopyrightText: 2026 Baptiste Legouix
SPDX-License-Identifier: MIT
-->

# AGENTS.md for SimiLie

# Clone
- Clone with `git clone --recurse-submodules git@github.com:blegouix/similie.git` or any similar commande which suits the network configuration. Be sure to pull the submodules recursively.

# Build environment
- **Requirement:** By default (no particular contradictory instructions), build and run in the `docker/similie_env/Dockerfile` image that you call `similie_env:latest`. You'll need to mount the `similie/` folder before compiling it. Compile directly in the host environment only if specifically asked for (if so, check if all the necessary dependencies listed in `docker/similie_env/Dockerfile` are available in the current environment, in particular `nvcc` and `openmpi`).
- **Choose build** Except if already specified by user, before building you will ask if you have to build for CPU or for CUDA. The CPU backend must be selected explicitly with `--build-arg BACKEND=cpu`.
- **Adapt Docker base image** By default (no particular contradictory instructions), the CPU backend always uses the official Ubuntu 24 base image. In practice, when building the CPU image, always pass both `--build-arg BACKEND=cpu` and `--build-arg BASE_IMAGE=ubuntu:24.04` to `docker build` instead of using the NVIDIA CUDA image.
- **Tests and simulations:** Always run tests, examples, and simulations inside Docker unless the user explicitly asks for a host run. Do not use host-side build directories or host-side executables for validation by default.
- **Mount path in Docker:** By default (no particular contradictory instructions), you will compile into a `agent_build/` or `agent_build_cpu/` repository mounted in Docker at the same absolute path as on the host.
- **CUDA in Docker:** When building or running the CUDA backend inside Docker, start the container with `--gpus all` so the NVIDIA driver and GPU devices are exposed in the container. Check `nvidia-smi` output once inside the docker image.
- **Git safe.directory in Docker:** Some vendored submodules query Git metadata during CMake. If Docker reports "detected dubious ownership" on the mounted repository or its submodules, add temporary `git config --global --add safe.directory <path>` entries inside the container for the repository and the relevant submodules before building.
- **Configure CMake:** For CPU with `cmake -DCMAKE_CXX_COMPILER=g++-13 -DCMAKE_BUILD_TYPE=Debug -DKokkos_ENABLE_OPENMP=ON -B agent_build_cpu -S .` or for CUDA with `cmake -DCMAKE_CXX_COMPILER=$PWD/vendor/ddc/vendor/kokkos/bin/nvcc_wrapper -DCMAKE_BUILD_TYPE=Debug -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_<YOUR_ARCH_NAMECODE>=ON -B agent_build -S .`. Of course replace `<YOUR_ARCH_NAMECODE>` with the arch namecode which suits the GPU.
- **Compile:** As usual once configured (`make -jN_PROC` from the `agent_build/` or `agent_build_cpu/` folder). Never exceed N_PROC=8.

## Run tests
- From `agents_build` or `agents_build_cpu`, first recompile then run `ctest`.

## Run examples
- Running `free_scalar_field` example in Debug mode on CPU can be very long (on GPU this can be ok though). By default (no particular contradictory instruction), if you have to run it on CPU, first recompile in Release mode.

## Coding
- `clang-format 20` is used to format C++, `ruff format` is used to format python.
- Avoid introducing unecessary constexpr aliases (like, using the `using` keyword for type aliases of `constexpr` for variables aliases). It can be used but not just for readability purpose, except is explicity asked.

## ONELAB interface
- `onelab_interface/GmshSocket.h` and `onelab_interface/onelab.h` are forks from existing repos, you will never modify them.
- Always run `run_onelab_inductor.sh` from the `agent_build/` or `agent_build_cpu/` folder such that the exported files won't polluate the repo.
- When running `run_onelab_inductor.sh`, you will set SIMILIE_ONELAB_BUILD_DIR to the `agent_build/` or `agent_build_cpu/` path, but dont try to overwrite it's default value in the file.
