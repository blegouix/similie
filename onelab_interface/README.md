# ONELAB Interface

This directory contains the self-contained SimiLie ONELAB client.

Current structure:
- `onelab_interface.hpp` contains the ONELAB driver, the `.silpro` parser, and the current problem dispatch.
- `main.cpp` instantiates `similie::onelab_interface::OnelabInterface`.
- the vendored `onelab.h` and `GmshSocket.h` stay local to keep the build independent from a system Gmsh SDK.

Current runtime support:
- `.silpro` parsing for `ScalarFieldWithPowerCoupling` and `Magnetostatics`
- `MinimizeStrongFormulationResidual` solver settings assembly
- structured rectilinear magnetostatics execution from Gmsh/ONELAB

The `.silpro` syntax is documented in:
- [docs/onelab_interface.md](/home/cart3sianbear/similie/docs/onelab_interface.md)
