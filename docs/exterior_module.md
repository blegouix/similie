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

A whole zoo of differential operators has been developed over the centuries : gradient, rotational, divergency, 4-gradient, 4-divergency, directional derivative, material derivative, covariant derivative, Lie derivative, etc. During the 20th century, mathematicians proposed a refactoring of it known as Exterior calculus, which revealed highly suitable for discretization (now known as Discrete exterior calculus). Only two fundamental operators are defined and can be used to build every possible differential operators:  

- The <em>exterior derivative</em> \f$d\f$.
- The <em>Hodge star</em> \f$\star\f$.

Those operators must be applied on <em>differential forms</em>, which can be assimilated to antisymmetric tensor fields. Note that partially-antisymmetric tensor fields may be assimilated to differential form-valued lower rank tensor fields (ie. the Cauchy stress tensor or the Christoffel symbols).
