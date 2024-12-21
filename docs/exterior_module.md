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
