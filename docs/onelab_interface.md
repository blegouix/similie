# The ONELAB interface {#onelab_interface}
<!--
SPDX-FileCopyrightText: 2026 Baptiste Legouix
SPDX-License-Identifier: AGPL-3.0-or-later
-->

[ONELAB](https://onelab.info/) is a standard interface specification that allows [Gmsh](https://gmsh.info/) to control external PDE solvers. SimiLie provides an ONELAB interface for several standard physics problems, including scalar field, magnetostatics and potential flow.

\important The ONELAB interface in SimiLie is mostly AI-generated and has received very limited testing. Unfortunately, I do not have the resources to do better. Please keep in mind that, without AI assistance, this interface would not exist at all. Contributions would of course be greatly appreciated.

The geometry is defined in a `.geo` file, which is interpreted entirely by Gmsh. Gmsh then communicates with SimiLie through the ONELAB interface. The physical problem definition and numerical solver configuration are specified in a `.silpro` file, which can be viewed as the SimiLie counterpart of [GetDP](https://getdp.info/) `.pro` files. Of course, `.silpro` files do not follow the same specification as `.pro` files, since SimiLie is far for being able to claim having capabilities comparable to GetDP.

\important Because the ONELAB interface is evolving rapidly, the `.silpro` format is currently undocumented. For a basic introduction, please refer to the [onelab_inductor](https://github.com/blegouix/similie/tree/main/examples/onelab_inductor) example.

\important This documentation is fully AI-generated.

The mesh reader accepts supported MSH2 quadrilateral and hexahedral meshes.
Each physics adapter imposes further structured-grid requirements. In
particular, potential flow uses a mapped annular quadrilateral mesh rather
than a rectilinear grid. Its physical region and boundary tags define a runtime
graph of connected patches, including a circulation cut. See
\ref multidomains "multidomains and potential flow" for the model, `.silpro`
controls, example commands and GetDP comparison workflow.
