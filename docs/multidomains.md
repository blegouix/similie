# Multidomains and potential flow {#multidomains}
<!--
SPDX-FileCopyrightText: 2026 Baptiste Legouix
SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED
-->

\important This documentation is fully AI-generated.

## Domain graph and field ownership

`sil::multidomains::Topology<N>` describes the adjacency of aligned structured
patches at runtime. Each physical patch owns its nodes and has one connection
per lower or upper face of each logical axis. A connection points to another
patch face or to a boundary node. `connect()` records the inverse relation on
the other patch automatically. `validate()` requires every face to be connected.
The graph supports any runtime number of patches, including periodic links to
the same patch. The dimension tags and ordering must agree across patches;
interfaces must have matching tangential extents. The graph does not remesh or
interpolate nonconforming faces.

An interface can map coincident boundary traces (`shared_boundary=true`),
or put the donor after the local last node (`shared_boundary=false`). Its affine
trace relation is

\f[ u_A = s u_B + j, \f]

where `orientation` supplies \f$s\f$ and `jump` supplies \f$j\f$. The reverse
connection has coefficients \f$1/s\f$ and \f$-j/s\f$. These numbers are
sampling rules; they do not merge storage or assign global unknowns. When both
patches store the shared trace, their global numbering must express the intended
ownership or identification. The potential flow example uses disjoint ownership.
`IndexedScalarField` maps local nodes to those global numbers. The caller keeps
its index allocation alive while kernels or assembly use it.

`Multidomain` couples the topology to each patch's primal field, geometry,
physics and optional flux field. `FieldRole` selects which field a boundary rule
acts on. Physical boundaries can prescribe a scalar value, use natural or
constant continuation, clamp, or set an exterior value. The default flux
boundary is zero. `extrapolation_rules(id)` returns the ordinary tensor
extrapolation API, so a differential operator samples across interfaces without
changing its local formula. The graph stores metadata; donor views borrow their
source allocations. Keep both the source and transformed bindings alive until
their kernels finish. `DomainExecution` partitions the Kokkos execution space by
patch and fences each partition at the end of a phase.

## Differential operators and assembly

An exterior operator evaluates a local stencil with `value()` and a field value
with `operator()`. `make_multidomain_operator(domains, operation)` copies a local
operator to each physical patch; `MultidomainOperator<Operator>` forwards
`value(id, ...)` and `operator()(id, ...)` to that patch's local operator and can
run its bulk `operator()` over partitioned execution spaces. The caller supplies
the local sampling rules and any geometry arguments required by the operator.
Copy `operators[id]` into a device kernel for local pointwise calls. The
assembly path in
`multidomains/differential_operator.hpp` accepts a stencil factory that runs in
device kernels and supplies the operator's `value()` arguments. It composes
each local stencil coefficient with the input field's extrapolation basis,
converting donor samples to global columns and prescribed values to a constant
term. Output rows and input columns may have different component indices, as
with a coboundary from scalar nodes to edges. This lets the same assembly path
serve a coboundary, codifferential, or a composite operator when it provides a
stencil-valued `value()` call. `StencilCoordinates::Absolute` uses entries as
physical sample indices, as returned by `Coboundary::value()`. Select
`StencilCoordinates::Relative` for stencils whose indices represent offsets
plus one, with the center at one, as returned by the scalar dual-field
`Codifferential::value()` overload. `FieldRole` selects the input field and its
corresponding rules; primal is the default.

The caller chooses a capacity large enough for the global columns after donor
and lazy-cochain composition, and supplies globally numbered output and input
domains. The matrix may be rectangular. Every output row needs a unique owner,
and each output component needs its own
global row number; `IndexedScalarField` supplies numbering only for scalars.
For a field \f$u\f$, assembly gives \f$F(u)=Au+c\f$ and returns the load
\f$b=-c\f$, so homogeneous differential equations read \f$Au=b\f$. Empty
rows are retained, with unit scaling when normalization is requested. The
public generic assembler does
not impose physical Dirichlet rows; operators needing those constraints add
them before finalization.

For example, after binding numbered scalar input and one-form output patches,
a two-dimensional coboundary can be assembled as follows. `Direction` is the
covariant natural index of the two logical axes, `Scalar` is the scalar index,
and `Grid` is their spatial DDC domain:

```cpp
auto const operators = sil::multidomains::make_multidomain_operator(
        input_domains, sil::exterior::Coboundary<Direction, Scalar> {});
auto const system = sil::multidomains::assemble_multidomain_differential_operator<2>(
        exec, output_domains, input_domains, operators,
        KOKKOS_LAMBDA(auto const& op, auto const&, auto const&, auto elem, auto component) {
            auto const chain = sil::exterior::tangent_basis<
                    1, Grid, Kokkos::DefaultExecutionSpace::memory_space>(
                    typename Grid::discrete_element_type {});
            auto const lower = sil::exterior::tangent_basis<
                    0, Grid, Kokkos::DefaultExecutionSpace::memory_space>(
                    typename Grid::discrete_element_type {});
            return op.value(chain, lower, elem, component);
        },
        output_rows, input_columns);
```

The scalar Laplacian adapter constructs a lazy dual flux \f$\star d\phi\f$ on
each patch, with the material diffusivity in the constitutive Hodge map, then
applies transposed incidence and inverse dual volume to assemble the scalar
Laplacian. The local Hodge pairing uses the full cell Jacobian, so skew cells
can couple derivatives along different logical axes. A complementary
stabilization term vanishes on constant physical forms. Flux samples across
an interface use the donor patch's geometry and material. The matrix stores
the linear response to unknowns, while interface jumps and prescribed boundary
values contribute to the right-hand side. Dirichlet nodes become identity
rows. `assemble_multidomain_laplacian_rhs()` recomputes a load with the saved
row scales and matrix; it requires unchanged geometry, material, ownership and
boundary kinds. Matrix normalization scales each row by its largest absolute
coefficient.

## ONELAB potential flow example

The [potential flow example](https://github.com/blegouix/similie/tree/main/examples/onelab_potential_flow)
uses mapped quadrilateral patches around a cylinder or NACA airfoil. Gmsh
supplies a complete annular quad mesh with physical tags for the body, upstream
and downstream outer boundaries, and wake cut. The ONELAB client checks mesh
connectivity and groups ring columns into patches whenever the material region,
outer boundary law or wake relation changes. A nonconforming mesh or missing
physical boundary is rejected.

For steady, incompressible, irrotational flow the velocity derives from a
potential, and continuity gives

\f[ \mathbf v = \nabla\phi, \qquad \nabla\cdot(\rho\nabla\phi)=0. \f]

The code obtains velocity components through the free scalar field Hamiltonian
and reconstructs \f$d\phi\f$ with the exterior derivative. The body has zero
normal flux. Upstream potential is zero; downstream potential is prescribed
from the incoming velocity and air-box size. The wake cut carries a potential
jump corresponding to circulation. The mapped quad geometry enters the Hodge
flux and the conversion from logical differences to Cartesian velocity.

The solver assembles one matrix and two right-hand sides: a base solve with no
cut jump and a unit-jump response. Linearity then gives

\f[ \phi = \phi_0 + \Gamma\phi_1. \f]

The oriented wake relation is \f$\phi_A-\phi_B=-\Gamma\f$ for the connection
from the preceding patch's upper face to the cut patch's lower face. For an
imposed circulation, \f$\Gamma\f$ is the signed input. For the cylinder it is
chosen to match the requested mass flow through the cut. For the airfoil it is
chosen so the reconstructed velocity satisfies a Kutta condition near the
trailing edge. The example exports potential, velocity, and the Bernoulli
pressure difference from stagnation pressure, \f$-\rho|\mathbf v|^2/2\f$,
as Gmsh `.pos` views. Its
diagnostics include circulation, mass flow, lift from the Kutta–Joukowski
relation \f$L=-\rho V\Gamma\f$, and maximum speed. The example uses the
default constant density of 1.225 kg/m³.

The example's `potential_flow.silpro` names the ONELAB model inputs and solver
settings. The Gmsh model exposes these controls:

| Parameter | Meaning |
| --- | --- |
| `Model/Object` | `0` for cylinder, `1` for airfoil |
| `Model/V` | Incoming speed in km/h |
| `Model/Air box size [m]` | Length of the surrounding flow box in meters |
| `Model/Angle of attack [deg]` | Airfoil incidence in degrees |
| `Model/Impose circulation` | `1` to use `Model/Circ`; `0` to infer circulation |
| `Model/Circ` | Circulation control in m²/s; the client sets \f$\Gamma=-\mathrm{Model/Circ}\f$ |
| `Model/Dmdt` | Requested cylinder cut mass flow in kg/s when circulation is inferred |

From a configured build directory, set `SIMILIE_ONELAB_BUILD_DIR` to
that directory and invoke `examples/onelab_potential_flow/run_similie_onelab.sh`.
The potential flow adapter uses an assembled matrix. The script accepts Gmsh
`-setnumber` arguments, `--assembled-matrix`, and `--solver=getdp` for the reference solve. Put
`SIMILIE_ONELAB_MESH_FILE` in a writable output directory; the script writes
`similie_potential_flow.pos` beside it. The checks in `check_reference.py` and
`check_fields.py` compare diagnostics and exported fields with GetDP.

For a cylinder with imposed circulation, the following commands keep the
mesh, solver logs and views in the build directory. Run them inside the chosen
SimiLie Docker environment, after configuration and compilation:

```sh
cd /path/to/similie/agent_build_cpu
SIMILIE_ONELAB_BUILD_DIR="$PWD" \
  ../examples/onelab_potential_flow/run_similie_onelab.sh \
  --assembled-matrix -setnumber "Model/Object" 0 \
  -setnumber "Model/Impose circulation" 1
```
