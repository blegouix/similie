# The exterior module {#exterior_module}
<!--
SPDX-FileCopyrightText: 2024 Baptiste Legouix
SPDX-License-Identifier: AGPL-3.0-or-later
-->

## Why integro-differential calculus is important for physics ?

Physics is governed by [Partial differential algebraic equations](https://en.wikipedia.org/wiki/Partial_differential_algebraic_equation) which involve derivatives of fields. The ability to differentiate and integrate tensor fields is thus a key feature in numerical codes. In general relativity or in non-cartesian coordinate systems, differentiation and integration takes place on manifolds with non-necessarily-Lorentzian metric.

Examples of tensor fields which are linked one-to-the-other by differentiations are:
- Cauchy momentum equation: \f$f = -\underline{\underline{\nabla}}\cdot \sigma\f$ in steady structural mechanics.
- \f$\vec{q} = -k\vec{\nabla}T\f$ in heat transfert.
- \f$F = dA\f$ in electromagnetism.
- Christoffel symbols of the first kinds \f$\Gamma^c_{ab} = g_{ca,b} + g_{cb,a} - g_{ab,c}\f$ in Riemannian geometry.

Integration is used to compute global quantities from local ones, ie. the current flowing through a wire, the energy in a physical system, etc. The generalized Stokes theorem (discussed below) is of particular importance when computing the integral of a derivative:

\f\[
\int_{\delta\Omega} \omega = \int_\Omega d\omega
\f\]

## Discrete exterior calculus

Please refer to the reference document [Discrete Differential Forms for Computational Modeling](http://www.geometry.caltech.edu/pubs/DKT05.pdf) for a way more detailled description of the theory presented in this section.

A whole zoo of differential operators has been developed over the centuries : gradient, rotational, divergency, 4-gradient, 4-divergency, directional derivative, material derivative, covariant derivative, Lie derivative, etc. During the 20th century, mathematicians proposed a refactoring of it known as Exterior calculus, which proved highly suitable for discretization (now known as Discrete exterior calculus). Only two fundamental operators are defined and can be used to build any possible differential operator on any-dimension manifold:

- The <em>exterior derivative</em> \f$d\f$.
- The <em>Hodge star</em> \f$\star\f$.

Those operators must be applied on <em>differential forms</em>, which can be assimilated to antisymmetric tensor fields. The exterior derivative transforms a \f$k-\f$form into a \f$k+1-\f$form. The Hodge star transforms a \f$k-\f$form into a \f$n-k-\f$form where \f$n\f$ is the dimension of the manifold. The <em>codifferential operator</em> \f$\delta = (-1)^{n(k+1)+1}\star d^\top \star\f$ transforms a \f$k-\f$form into a \f$k-1-\f$form.

\note Partially-antisymmetric tensor fields may be assimilated to differential form-valued lower rank tensor fields (ie. the Cauchy stress tensor or the Christoffel symbols).

The generic Laplacian is defined as \f$\delta d + d \delta\f$.

An important relation is the Poincarré Lemma \f$dd = 0\f$. It leads to the very deep physical concept of gauge invariance, which tells us potentials are defined up to a derivative (because \f$d(A+d\alpha) = dA + dd\alpha = dA\f$).

\important A key paradigm of SimiLie is the exclusive support of structured meshes to avoid sparse linear algebra and produce mostly-embarrassingly parallel code. It implies that the main difference between the theory described in [Discrete Differential Forms for Computational Modeling](http://www.geometry.caltech.edu/pubs/DKT05.pdf) and the implementation in SimiLie is that the discrete exterior derivative is not built upon a sparse adjacency matrix, but directly computed locally for each node of the mesh (matrix-free approach). Otherwise, SimiLie follows quite closely the construction presented in the document.

### Simplex

A \f$k-\f$simplex is an oriented discrete element of dimension \f$k\f$ belonging to the discrete manifold. Ie. a \f$0\f$-simplex is a node, a \f$1\f$-simplex is an edge, a \f$2\f$-simplex is a face and a \f$3\f$-simplex is a cell. It can be defined for any dimension \f$n\f$ of the discrete manifold.

In SimiLie, a simplex derives from `ddc::DiscreteElement`, associated to a shift vector with \f$k\f$ `1` and \f$n-k\f$ `0`, and a boolean flag `negative` giving the orientation.

### Chain

A \f$k-\f$chain - at least in the sense it is used in SimiLie - is basically a set of \f$k-\f$simplices.

\note In [Discrete Differential Forms for Computational Modeling](http://www.geometry.caltech.edu/pubs/DKT05.pdf) chains are defined as linear combinations of simplices, but in SimiLie we assume all the coefficients to be `1`, every simplex belonging to the chain thus has to be oriented accordingly to the chain purpose.

\attention In SimiLie we also have the non-standard LocalChain class which allows to define a chain in which all the simplices share a node in common.

### Boundary operator

The boundary operator associates to a \f$k-\f$simplex a \f$k-1-\f$chain containing the simplices which form the boundary of the simplex. It can also associate to a \f$k-\f$chain a \f$k-1-\f$chain forming the boundary of the chain.

### Cosimplex

A \f$k-\f$cosimplex is the association of a simplex and of a real (or complex) number.

### Cochain

A \f$k-\f$cochain is basically a set of cosimplices. This is a discrete differential \f$k-\f$form. It can be represented as a rank-k antisymmetric tensor field (there is a Cochain constructor which takes an antisymmetric tensor as argument).

### Coboundary operator

The coboundary operator associates to a \f$k-\f$cochain a \f$k+1-\f$cosimplex or a \f$k+1-\f$cochain containing the cosimplices which form the coboundary of the cochain. Please refer to [Discrete Differential Forms for Computational Modeling](http://www.geometry.caltech.edu/pubs/DKT05.pdf) for more details. This is the discrete exterior derivative \f$d\f$.

At the implementation level, `Coboundary` evaluates the integral of the input cochain on the boundary of each local \f$(k+1)\f$-simplex. For a primal basis simplex \f$\sigma^{k+1}\f$, the discrete exterior derivative is therefore:

\f\[
(d\omega)(\sigma^{k+1}) = \omega(\partial \sigma^{k+1}).
\f\]

Equivalently, if \f$\omega\f$ is a discrete \f$k\f$-cochain, then:

\f\[
\left(d\omega\right)(\sigma^{k+1})
= \sum_{\tau^k \subset \partial \sigma^{k+1}}
\mathrm{sgn}(\tau^k,\sigma^{k+1})\,\omega(\tau^k),
\f\]

where the sum runs over the oriented boundary \f$k\f$-simplices of \f$\sigma^{k+1}\f$ and \f$\mathrm{sgn}(\tau^k,\sigma^{k+1})\f$ is the induced orientation sign.

### Transposed coboundary operator

The transposed coboundary operator accumulates the contributions of incident higher-dimensional simplices onto the current simplex. It is the transpose of the incidence operator underlying the coboundary. Explicitly, if \f$\eta\f$ is a discrete \f$(k+1)\f$-cochain and \f$\tau^k\f$ is a \f$k\f$-simplex, then:

\f\[
\left(d^\top \eta\right)(\tau^k)
= \sum_{\sigma^{k+1} \supset \tau^k}
\mathrm{sgn}(\tau^k,\sigma^{k+1})\,\eta(\sigma^{k+1}),
\f\]

However this is **not** the adjoint of the coboundary operator, because the latter has to involve the metric in some way. The adjoint operator on forms is the codifferential, presented below.

\important Boundary handling is currently explicit but not yet user-configurable.
- `Coboundary` (and its alias `deriv(...)`) use a clamped out-of-domain sampling rule: if a forward sample exits the mesh, the nearest in-domain node is reused. On a scalar field this behaves like a one-sided "free"/zero-normal-variation closure for the first derivative.
- `TransposedCoboundary` does **not** clamp. Missing incident simplices on the far side of the boundary are simply omitted. This is what restores the weighted adjointness needed by the codifferential.
- Periodic, Dirichlet and generic Neumann boundary conditions are not yet exposed as configurable policies in this module.

### Primal and dual cell complexes

In DEC, cochains are not only characterized by their degree, but also by the cell complex on which they live. There is the primal complex, made of the mesh simplices themselves, but the operators presented below make use of a dual cell complex which allows to introduce metric information. Indeed, the dual cell complex provides the complementary cells on which the metric-dependent "orthogonal counterparts" live. This is why the discrete Hodge star, which is the operator encoding the metric in DEC, naturally maps primal cochains to dual cochains.

However, a dual mesh has to be chosen according to some "dualization strategy". In SimiLie, this choice is represented explicitly by the `CellComplex` tag:
- `CellComplex::Primal` (no dualization)
- `CellComplex::CircumcentricDual`
- `CellComplex::BarycentricDual`

### Volume operator

To build a discrete operator able to encode the metric (the discrete Hodge star), we need a volume operator (in the sense of \f$k-\f$volume, ie. length, area...) to provide the ability to measure the "size" of a simplex. Given a \f$k\f$-simplex spanned by edge vectors \f$e_1,\dots,e_k\f$, SimiLie builds the local Gram matrix:

\f\[
G_{\mu\nu} = g_{\rho\sigma} e_\mu^\rho e_\nu^\sigma
\f\]

where \f$g\f$ is the metric evaluated at the current mesh node. The \f$k\f$-volume is then obtained from

\f\[
|\sigma^k| = \sqrt{|\det G|}
\f\]

This is exactly what `SimplexVolume` computes: it first extracts the local edge vectors from the position field, assembles the Gram matrix, and finally takes the square root of its determinant. `DualSimplexVolume` applies the same formula to the complementary \f$(n-k)\f$ directions. The associated dual cell complex is selected through the `CellComplex` tag, typically circumcentric or barycentric.

### Reduction and reconstruction

Reduction and reconstruction are the bridges between pointwise differential-form coefficients and geometric cochains. Schematically, reduction is the same as integrating a density field along simplices, providing a natural way to encode a differential form on a mesh so that the geometric measures are already absorbed into the cochain values. Reconstruction is the inverse operation. They always act on a chosen cell complex, selected by the `CellComplex` tag. The reduction maps a coefficient \f$k\f$-form to the cochain obtained by integrating it on the corresponding \f$k\f$-cell:

\f\[
R_k(\omega, \sigma^k) = \int_{\sigma^k} \omega
\f\]

Depending on the chosen `CellComplex`, \f$\sigma^k\f$ is interpreted as:
- a primal \f$k\f$-simplex for `CellComplex::Primal`,
- a circumcentric dual \f$k\f$-cell for `CellComplex::CircumcentricDual`,
- a barycentric dual \f$k\f$-cell for `CellComplex::BarycentricDual`.

The reconstruction is the inverse map assuming a piecewise-constant ansatz:

\f\[
Q_k(c, \sigma^k) = \frac{c(\sigma^k)}{|\sigma^k|}
\f\]

Here again, the meaning of \f$\sigma^k\f$ depends on the selected cell complex. For `CellComplex::Primal`, reduction and reconstruction only need the position field because the required measure is purely primal. No metric needs to be passed. For dual cell complexes, the dual volumes depend on the metric, so dual reduction needs both the position and the metric.

In the current implementation:
- primal reduction/reconstruction are diagonal local operators based on primal simplex volumes;
- circumcentric-dual reduction is the metric-dependent dual geometric map used by the discrete Hodge star;
- reconstruction is currently only implemented where the piecewise-constant inverse is well defined locally. Unsupported `CellComplex` choices are rejected with `static_assert`.

### Hodge star operator

The Hodge star operator is implemented both in continuous and discrete form.

The continuous Hodge star acts pointwise on coefficient forms. It is the exterior-calculus counterpart of the metric. In an \f$n\f$-dimensional manifold, for a \f$k\f$-form \f$\omega\f$ with components \f$\omega_{i_1 \dots i_k}\f$, it is defined as (\f$\varepsilon\f$ is the Levi-Civita symbol):

\f\[
(\star \omega)_{j_1 \dots j_{n-k}}
= \frac{\sqrt{|g|}}{k!}
\omega_{i_1 \dots i_k}
g^{i_1 \ell_1} \dots g^{i_k \ell_k}
\varepsilon_{\ell_1 \dots \ell_k j_1 \dots j_{n-k}}.
\f\]

The discrete Hodge star acts on cochains. For a \f$k\f$-simplex \f$\sigma\f$ in an \f$n\f$-dimensional mesh, it is the composition:
- Reconstruction of the primal cochain into a pointwise coefficient \f$k\f$-form,
- Application of the continuous Hodge star,
- Reduction of the resulting \f$(n-k)\f$-form onto the chosen dual cell complex.

This composition reproduces exactly the usual diagonal DEC factor:

\f\[
\star_\sigma = \frac{|\star \sigma|}{|\sigma|}
\f\]

### Codifferential

This is the adjoint of the coboundary operator in the metric sense. It is built from the discrete Hodge star and the transposed coboundary operator:

\f\[
\delta = (-1)^{n(k+1)+1}\,\star^{-1}\, d^\top_\mathrm{dual}\, \star,
\f\]

where the "right" Hodge star dualizes the primal \f$k\f$-cochain and the "left" one brings the result back to the primal side.

The reason this is the adjoint is that the Hodge star defines the inner product \f$\langle \alpha,\beta\rangle = \int \alpha \wedge \star \beta\f$ on differential forms (with \f$\wedge\f$ the exterior product). The codifferential is then characterized by:

\f\[
\langle d\omega,\eta\rangle = \int d(\omega \wedge \star \eta) - (-1)^k \int \omega \wedge d(\star \eta) = \int \omega \wedge \star\left( (-1)^{n(k+1)+1}\star^{-1} d \star \eta \right) = \langle \omega,\delta\eta\rangle.
\f\]

up to the boundary term and the usual sign convention.

### Laplacian

The generic DEC Laplacian combines the exterior derivative and the codifferential:

\f\[
\Delta = \delta d + d \delta
\f\]
