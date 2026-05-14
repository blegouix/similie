# ONELAB Interface

This directory contains a first ONELAB client for SimiLie.

Current behavior:
- Gmsh can launch `similie_onelab` as an external ONELAB client.
- The client exposes a few control parameters in the ONELAB tree.
- Each run performs a minimal fake SimiLie task by allocating and filling a `sil::tensor::Tensor`.
- The client publishes scalar outputs and can generate a trivial `.pos` file that Gmsh can merge for visualization.

Build notes:
- The CMake target expects gmsh's `onelab.h` and `GmshSocket.h` to be available through `SIMILIE_ONELAB_SOURCE_DIR`.
- In the default Docker image from this repository, the expected path is `/gmsh/utils/solvers/c++`.

Typical workflow:
1. Build SimiLie in Docker with `SIMILIE_BUILD_ONELAB_INTERFACE=ON`.
2. Register either the built executable or `onelab_interface/run_in_docker.sh` as a new ONELAB client in Gmsh.
3. Let Gmsh launch the client with its usual `-onelab <name> <socket>` arguments.
