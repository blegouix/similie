# ONELAB Interface

This directory contains a first ONELAB client for SimiLie.

Current behavior:
- Gmsh can launch `similie_onelab` as an external ONELAB client.
- The code is now structured around a reusable `BaseOnelabInterface` class.
- The current executable entry point lives in `magnetostatics/` and instantiates `LinearMagnetostaticsOnelabInterface`.
- The linear magnetostatics interface currently prepares the ONELAB session and exports the mesh from Gmsh, but the actual solver implementation is still pending.

Build notes:
- The ONELAB headers `onelab.h` and `GmshSocket.h` are vendored directly in this directory.

Typical workflow:
1. Build SimiLie in Docker with `SIMILIE_BUILD_ONELAB_INTERFACE=ON`.
2. Register the built executable as a new ONELAB client in Gmsh.
3. Let Gmsh launch the client with its usual `-onelab <name> <socket>` arguments.
