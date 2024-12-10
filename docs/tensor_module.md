# Tensor module {#tensor_module}
<!--
SPDX-FileCopyrightText: 2024 Baptiste Legouix
SPDX-License-Identifier: GPL-3.0-or-later
-->

# Why tensors are important in physics ?

[Tensors](https://en.wikipedia.org/wiki/Tensor) are the numerical counterparts of algebraic structures which form the base of fundamental physics as Lie algebra (represent gauge fields) or Clifford algebra (represent particles with spin). They also appear in continuum mechanics, with ie. the [Cauchy stress tensor](https://en.wikipedia.org/wiki/Cauchy_stress_tensor) which describes internal forces in a fluid or solid. The [Metric tensor](https://en.wikipedia.org/wiki/Metric_tensor) is of particular importance because it allows to evaluate distances on a manifold (like space-time in general relativity or more simply non-cartesian coordinate system).

A special class of tensors are the [antisymmetric tensors](https://en.wikipedia.org/wiki/Antisymmetric_tensor) which correspond to differential forms, implemented in the Exterior module of Similie. Even more specials are the [vectors](https://en.wikipedia.org/wiki/Vector_(mathematics_and_physics)) which correspond to 1-forms and constitutes one of the most elementary concepts of physics.

The [Scalar](https://en.wikipedia.org/wiki/Scalar) is a degenerated type of tensors.

# Do Kokkos and DDC support tensor fields ?
