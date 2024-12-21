# Exterior module {#exterior_module}
<!--
SPDX-FileCopyrightText: 2024 Baptiste Legouix
SPDX-License-Identifier: GPL-3.0-or-later
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

A whole zoo of differential operators has been developed over the centuries : gradient, rotational, divergency, 4-gradient, 4-divergency, directional derivative, material derivative, covariant derivative, Lie derivative, etc. During the 20th century, mathematicians proposed a refactoring of it known as Exterior calculus, which revealed highly suitable for discretization (now known as Discrete exterior calculus). Only two fundamental operators are defined and can be used to build any possible differential operator on any-dimension manifold:

- The <em>exterior derivative</em> \f$d\f$.
- The <em>Hodge star</em> \f$\star\f$.

Those operators must be applied on <em>differential forms</em>, which can be assimilated to antisymmetric tensor fields. The exterior derivative transforms a \f$k-\f$form into a \f$k+1-\f$form. The Hodge star transforms a \f$k-\f$form into a \f$n-k-\f$form where \f$n\f$ is the dimension of the manifold. The <em>codifferential operator</em> \f$\delta = (-1)^{n(k+1)+1}\star d \star\f$ transforms a \f$k-\f$form into a \f$k-1-\f$form.

\note Partially-antisymmetric tensor fields may be assimilated to differential form-valued lower rank tensor fields (ie. the Cauchy stress tensor or the Christoffel symbols).

The generic Laplacian is defined as \f$\delta d + d \delta\f$.

An important relation is the Poincarré Lemma \f$dd = 0\f$. It leads to the very deep physical concept of gauge invariance, which tells us potentials are defined up to a derivative (because \f$d(A+d\alpha) = dA + dd\alpha = dA\f$).

\important A key paradigm of SimiLie is the exclusive support of structured meshes to avoid sparse linear algebra and produce mostly-embarrassing parallel code. It implies that the main difference between the theory described in [Discrete Differential Forms for Computational Modeling](http://www.geometry.caltech.edu/pubs/DKT05.pdf) and the implementation in SimiLie is that discrete exterior derivative is not build upon a sparse adjacency matrix, but directly computed locally for each node of the mesh. Otherwise, SimiLie follows closely he constrution described in the document.

### Simplex

A \f$k-\f$simplex is an oriented discrete element of dimension \f$k\f$ belonging to the discrete manifold. Ie. a \f$0\f$-simplex is a node, a \f$1\f$-simplex is an edge, a \f$2\f$-simplex is a face and a \f$3\f$-simplex is a cell. It can be defined for any dimension \f$n\f$ of the discrete manifold.

In SimiLie, a simplex derives from `ddc::DiscreteElement`, associating to it a shift vector with \f$k\f$ `1` and \f$n-k\f$ `0`, and a boolean flag `negative` giving the orientation.

### Chain

A \f$k-\f$chain - at least in the sense it is used in SimiLie - is basically a set of \f$k-\f$simplices.

\note In [Discrete Differential Forms for Computational Modeling](http://www.geometry.caltech.edu/pubs/DKT05.pdf) chains are defined as linear combination of simplices, but in SimiLie we assume all the coefficients to be `1`, every simplex belonging to the chain is thus already correctly oriented.

\attention In SimiLie we also have the non-standard LocalChain class which allows to define a chain whose every simplex shares a node in common.

### Boundary operator

The boundary operator associates to a \f$k-\f$simplex a \f$k-1-\f$chain containing the simplices which form the boundary of the simplex. It can also associate to a \f$k-\f$chain a \f$k-1-\f$chain forming the boundary of the chain.

### Cosimplex

A \f$k-\f$cosimplex is the association of a simplex and of a real (or complex) number.

### Cochain

A \f$k-\f$cochain is basically a set of cosimplices. This is a discrete differential \f$k-\f$form.

### Coboundary operator

The coboundary operator associates to a \f$k-\f$cochain a \f$k+1-\f$cosimplex or a \f$k+1-\f$cochain containing the cosimplices which form the coboundary of the cochain. Please refer to [Discrete Differential Forms for Computational Modeling](http://www.geometry.caltech.edu/pubs/DKT05.pdf) for more details. This is the discrete exterior derivative \f$d\f$.

### Hodge star operator

The Hodge star operator implements the formula:

\f\[
\star^{i_1, \dots, i_k}_{j_{k+1}, \dots, j_n} = \frac{\sqrt{\left|\det [g_{ab}]\right|}}{k!} g^{i_1 j_1}\cdots g^{i_k j_k} \,\varepsilon_{j_1, \dots, j_n}
\f\]

With \f$g\f$ the metric and \f$\varepsilon\f$ the Levi-Civita tensor. It is expected to be applied on a differential form \f$\omega_{i_1, \dots, i_k}\f$.
