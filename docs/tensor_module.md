# Tensor module {#tensor_module}
<!--
SPDX-FileCopyrightText: 2024 Baptiste Legouix
SPDX-License-Identifier: GPL-3.0-or-later
-->

# Why tensors are importants for physics ?

[Tensors](https://en.wikipedia.org/wiki/Tensor) are the numerical counterparts of algebraic structures which form the base of fundamental physics as Lie algebra (represent gauge fields) or Clifford algebra (represent fermionic fields). They can thus constitute the building blocks of quantum physics solvers (long term roadmap of SimiLie).

They also appear in classical physics, with ie. the [Cauchy stress tensor](https://en.wikipedia.org/wiki/Cauchy_stress_tensor) which describes internal forces in a fluid or solid or the [Maxwell stress tensor](https://en.wikipedia.org/wiki/Maxwell_stress_tensor) if those forces are due to the interaction of the medium with the electromagnetic field. The [Metric tensor](https://en.wikipedia.org/wiki/Metric_tensor) is of particular importance because it allows to evaluate distances on a manifold (like space-time in general relativity or more simply non-cartesian coordinate system).

In general relativity, the [Christoffel symbols](https://en.wikipedia.org/wiki/Christoffel_symbols) are not technically tensors (because they do not transform as such under diffeomorphism) but are at least multidimensional arrays which allow to compute geodesics. The [Riemann tensor](https://en.wikipedia.org/wiki/Riemann_curvature_tensor) and the [Stress-energy tensor](https://en.wikipedia.org/wiki/Stress%E2%80%93energy_tensor) are also at the hearth of the theory, as they respectively describe the curvature of space-time and its content in stress, momentum and energy (while being [invariant under change of basis](https://en.wikipedia.org/wiki/Covariant_transformation)).

A special class of tensors are the [antisymmetric tensors](https://en.wikipedia.org/wiki/Antisymmetric_tensor) which correspond to differential forms, implemented in the \subpage exterior_module "Exterior module" of SimiLie. Even more specials are the [vectors](https://en.wikipedia.org/wiki/Vector_(mathematics_and_physics)) which correspond to 1-forms and constitutes one of the most elementary concepts of physics.

The [Scalar](https://en.wikipedia.org/wiki/Scalar) is a degenerated type of tensors.

# Do Kokkos and DDC support tensor fields ? What more can SimiLie do ?

Kokkos aims to handle [Multidimensional arrays](https://en.wikipedia.org/wiki/Array_\(data_structure\)#Multidimensional_arrays). A `N-rank` tensor field defined over a `M-dimension` discrete manifold can be stored as a `N*M-dimensions` array. But this is not very conveniant, because Kokkos does not allow to labelize the dimensions of the array, so the user has to keep track of the ordering of physical dimensions and tensor indices, which may be very cumbersome and can easily lead to silent bugs.

DDC offers the possibility to labelize the physical dimensions, but its support for tensor fields is limited. Scalar fields are described using the `ddc::Chunk` and `ddc::ChunkSpan` classes, which internally relie on `mdspan` (the C++ standardization of `Kokkos::View`). They take as a template argument a `ddc::DiscreteDomain`, which is itself template by the physical dimensions of the problem:

```
ddc::DiscreteDomain<DDimX, DDimY, DDimZ>
```

A simple way to support a tensor field would be adding non-physical dimensions representing tensor indices, ie.:

```
ddc::DiscreteDomain<DDimX, DDimY, DDimZ, Mu, Nu>
```

This is precisely the starting point of SimiLie, which extents this natural support of tensor fields in DDC to tensor fields with internal symmetries (diagonal tensor, symmetric tensor, antisymmetric tensor, etc... or tensor product of lower-rank tensors with their own internal symmetries). Under the hood there are still `ddc::Chunk` and `mdspan` but conveniant interfaces are provided to manipulate tensors with internal symmetries.

SimiLie also provides a dictionnary of functions to perform tensor products, with the ability to run on GPU and benefiting from the structure of the involved tensors.

\important The `Tensor` terminology in SimiLie does not guarantee the invariance under diffeomorphism, as it is the case in maths and physics. Tensor fields in Similie are just multidimensional arrays mapped over discrete manifolds and with conveniant accessors to benefit from internal symmetries. 

# How does it work ?

TODO
