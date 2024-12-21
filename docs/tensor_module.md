# Tensor module {#tensor_module}
<!--
SPDX-FileCopyrightText: 2024 Baptiste Legouix
SPDX-License-Identifier: GPL-3.0-or-later
-->

## Why tensors are importants for physics ?

[Tensors](https://en.wikipedia.org/wiki/Tensor) are the numerical counterparts of algebraic structures which form the base of fundamental physics as Lie algebra (represent gauge fields) or Clifford algebra (represent fermionic fields). They can thus constitute the building blocks of quantum physics solvers (long term roadmap of SimiLie).

They also appear in classical physics, with ie. the [Cauchy stress tensor](https://en.wikipedia.org/wiki/Cauchy_stress_tensor) which describes internal forces in a fluid or solid or the [Maxwell stress tensor](https://en.wikipedia.org/wiki/Maxwell_stress_tensor) if those forces are due to the interaction of the medium with the electromagnetic field. The [Metric tensor](https://en.wikipedia.org/wiki/Metric_tensor) is of particular importance because it allows to evaluate distances on a manifold (like space-time in general relativity or more simply non-cartesian coordinate system).

In general relativity, the [Christoffel symbols](https://en.wikipedia.org/wiki/Christoffel_symbols) are not technically tensors (because they do not transform as such under diffeomorphism) but are at least multidimensional arrays which allow to compute geodesics. The [Riemann tensor](https://en.wikipedia.org/wiki/Riemann_curvature_tensor) and the [Stress-energy tensor](https://en.wikipedia.org/wiki/Stress%E2%80%93energy_tensor) are also at the heart of the theory, as they respectively describe the curvature of space-time and its content in stress, momentum and energy (while being [invariant under change of basis](https://en.wikipedia.org/wiki/Covariant_transformation)).

A special class of tensors are the [antisymmetric tensors](https://en.wikipedia.org/wiki/Antisymmetric_tensor) which correspond to differential forms (ie. the [Faraday tensor](https://en.wikipedia.org/wiki/Electromagnetic_tensor) of electromagnetism), implemented in the \ref exterior_module "Exterior module" of SimiLie. Even more specials are the [vectors](https://en.wikipedia.org/wiki/Vector_(mathematics_and_physics)) which correspond to 1-forms.

The [Scalar](https://en.wikipedia.org/wiki/Scalar) corresponds to a 0-form and is a degenerated (rank-0) type of tensors.

## Do Kokkos and DDC support tensor fields ? What can SimiLie do in addition ?

Kokkos aims to handle [Multidimensional arrays](https://en.wikipedia.org/wiki/Array_\(data_structure\)#Multidimensional_arrays). A rank-N tensor field defined over a M-dimension discrete manifold can be stored as a N\*M-dimensions array. But this is not very convenient, because Kokkos does not allow to labelize the dimensions of the array, so the user has to keep track of the ordering of physical dimensions and tensor indices, which may be very cumbersome and can easily lead to silent bugs.

DDC offers the possibility to labelize the physical dimensions, but its support for tensor fields is limited. Scalar fields are described using the `ddc::Chunk` and `ddc::ChunkSpan` classes, which internally relie on `mdspan` (the C++ standardization of `Kokkos::View`). They take as a template argument a `ddc::DiscreteDomain`, which is itself template by the physical dimensions of the problem:

```
ddc::DiscreteDomain<DDimX, DDimY, DDimZ>
```

A simple way to support a tensor field would be adding non-physical dimensions representing tensor indices, ie.:

```
ddc::DiscreteDomain<DDimX, DDimY, DDimZ, Mu, Nu>
```

This is precisely the starting point of SimiLie, which extents this natural support of tensor fields in DDC to tensor fields with internal symmetries (diagonal tensor, symmetric tensor, antisymmetric tensor, etc... or tensor product of lower-rank tensors with their own internal symmetries). Under the hood there are still `ddc::Chunk` and `mdspan` but convenient interfaces are provided to manipulate tensors with internal symmetries.

\important The `Tensor` terminology in SimiLie does not guarantee the invariance under diffeomorphism, as it is the case in maths and physics. Tensor fields in SimiLie are just multidimensional arrays mapped over discrete manifolds and with convenient accessors to benefit from internal symmetries.

## Tensor indices

First, the class `TensorNaturalIndex` allows to build natural indices like `Mu` or `Nu` while giving the different labels it can take (ie. `T`, `X`, `Y` and `Z`). When the following DDC code declares a new dimension:

```
struct X;
struct DDimX : ddc::UniformPointSampling<X> {};
```

The following SimiLie code declares a new natural index:

```
struct T;
struct X;
struct Y;
struct Z;
struct Mu : sil::tensor::TensorNaturalIndex<T, X, Y, Z> {};
```

But several indices like `Mu, Nu` in the template arguments of a `ddc::DiscreteDomain` do not have any particular symmetry (that's why they are so-said <em>natural</em>). A `TensorNaturalIndex` is necessarily rank-1. Then, several classes sharing the same API are provided to describe a tensor index (which may be of any rank, which means it may correspond to any arbitrary number of natural indices):

- `TensorIdentityIndex`: identity tensor index, `1` on the diagonal, `0` out of it. No runtime allocation required.
- `TensorLorentzianSignIndex`: Lorentzian signature tensor index, diagonal with `p` `-1` and `q` `1`, ie. the Lorentz metric of special relativity. No runtime allocation required.
- `TensorLeviCivitaIndex`: [Levi-Civita tensor](https://en.wikipedia.org/wiki/Levi-Civita_symbol) index. Must be squared and have a number of elements in a dimension equal to its rank. No runtime allocation required.
- `TensorDiagonalIndex`: Diagonal tensor index. Only diagonal is stored at runtime.
- `TensorSymmetricIndex`: Symmetric tensor index. Only upper-left triangle (or higher-rank equivalent) is stored at runtime.
- `TensorAntisymmetricIndex`: Skew-symmetric tensor index. Only upper-left triangle excluding diagonal (or higher-rank equivalent) is stored at runtime.
- `TensorFullIndex`: Full tensor product index. Every elements are stored at runtime.
- `TensorYoungTableauIndex`: This one is very special and not yet documented. Ie. the Riemann tensor.

All those classes are templated by any number of `TensorNaturalIndex`. Keep in mind that dimensions, natural indices and non-natural indices can coexist in any order in the template arguments of a `ddc::DiscreteDomain` type. Ie.:

```
struct DDimX : ddc::UniformPointSampling<X> {};
struct DDimY : ddc::UniformPointSampling<Y> {};
struct Mu : sil::tensor::TensorNaturalIndex<X, Y> {};
struct Nu : sil::tensor::TensorNaturalIndex<X, Y> {};
struct Rho : sil::tensor::TensorNaturalIndex<X, Y> {};
using DDom = ddc::DiscreteDomain<DDimX, Rho, DDimY, sil::tensor::TensorSymmetricIndex<Mu, Nu>>;
```

is perfectly valid and `DDom` represents the type of the support of a rank-3 tensor field on a `XY` manifold, with the pair `Mu, Nu` being symmetric.

\attention The contiguity-coalescence rule of DDC sill stands in SimiLie: the leftest template argument of the `ddc::DiscreteDomain` declaration will be the most coalescent in the underlying multidimensional array while the rightest will be the most contiguous.

\note `ddc::DiscreteDomain<Mu, Nu>` and `ddc::DiscreteDomain<sil::tensor::TensorFullIndex<Mu, Nu>>` are mostly the same thing.

\important Nested non-natural tensor indices are forbidden, ie. `sil::tensor::TensorSymmetricIndex<sil::tensor::TensorAntisymmetricIndex<Mu, Nu>, Rho>` is not valid.

## Internal mechanism

The most important concept to understand is that elements of tensors are indexed (in the sense of <em>associated to integers which identify their positions</em>) using three different kind of `id`s:

- The `natural_ids` is the most intuitive, ie. `(1, 2)` identifies the tensor element at index `1` along the rows and at index `2` along the columns, regardless of the eventual symmetries of the tensor.
- The `access_id` gives a way to benefit from internal symmetries when trying to access to an element of a tensor. Ie. all the non-diagonal elements of a diagonal tensor share the same value `0` and are thus accessed using the same `access_id` (which is also `0`) while diagonal terms are accessed starting from `1`.
- The `mem_id` corresponds to the location in the memory, ie. the non-diagonal elements of a diagonal tensor are known to be `0` so they do not need to be stored, thus only the diagonal is stored starting from `0`.

Most of the logic implemented in the `tensor_impl.hpp` file aims to provide kind-of-converters between those three kinds of `id`s.

## Tensor accessor

Once the different tensor indices involved in your physical problem are declared, you need to declare a `TensorAccessor` templated by those indices. Ie.:

```
struct Mu : sil::tensor::TensorNaturalIndex<X, Y, Z> {};
struct Nu : sil::tensor::TensorNaturalIndex<X, Y, Z> {};
struct Rho : sil::tensor::TensorNaturalIndex<X, Y, Z> {};
using Sigma = sil::tensor::TensorAntisymmetricIndex<Mu, Nu>;
[[maybe_unused]] sil::tensor::TensorAccessor<Rho, Sigma> tensor_accessor;
```

\note The `[[maybe_unused]]` is convenient because all the member functions of `TensorAccessor` are `static`, so it avoids a compilation warning.

\important The `TensorAccessor` class only handles the indicial part of a tensor (ie. not the dimensions of the manifold supporting a tensor field). You must not pass dimensions to its template arguments!

Then using the accessor you can get the required domain supporting the allocation of such a tensor with:

```
ddc::DiscreteDomain<Rho, Sigma> single_tensor_dom = tensor_accessor.mem_domain();
```

But in practice it is more convenient to directly compute the cartesian product with the mesh:


```
ddc::DiscreteDomain<DDimX, DDimY> mesh_xy(ddc::DiscreteElement(0, 0), ddc::DiscreteVector(1000, 1000));
ddc::DiscreteDomain<DDimX, DDimY, Rho, Sigma> tensor_field_dom(mesh_xy, tensor_accessor.mem_domain());
```

The `mem_domain()` functions produces the domain in which the `mem_id`s stand. Ie. for an identity tensor, `mem_domain()` will produce a domain of size `0`.

\note `TensorAccessor` also provides several member functions to perform kind-of-conversions between the three kinds of indices. Please check its API, examples and tests for common usage.

## Tensor class

In DDC an allocation is made using the `ddc::Chunk` class. We do the same in SimiLie (ie. on CPU) :

```
ddc::Chunk tensor_field_alloc(tensor_field_dom, ddc::HostAllocator<double>());
```

In DDC we would then build a `ddc::ChunkSpan` to manipulate this data. In SimiLie we have a class `Tensor` which derives from `ChunkSpan` and which is more convenient to represent tensor fields:

```
sil::tensor::Tensor tensor_field(tensor_field_alloc);
```

\note `Tensor` provides a lot of member functions to perform usefull operations like accessing an element or slicing using natural indices, etc. 

\attention Remember that `TensorAccessor` only handles the indicial part of the tensor, while `Tensor` handles the whole field involving indicial and non-indicial parts of its support. Please check their respective APIs, the tests and examples for common usages.

## Metric

A metric is a squared symmetric rank-2 tensor field \f$g\f$ with number of elements along one of the tensor dimensions equal to the dimension of the manifold. It encodes the local information required to compute a distance \f$s\f$ on a Riemannian manifold:

\f\[
ds^2 = g_{\mu\nu}\;dx^\mu\; dx^\nu
\f\]

The simplest metric is the Euclidean metric which is the identity tensor. A Lorentzian metric is the equivalent in special relativistic frame. More complicated metrics are used in general relativity or non-cartesian coordinate systems.

In SimiLie a metric is defined using the two natural tensor indices `MetricIndex1` and `MetricIndex2`. A metric index is defined as ie.:

```
using MetricIndex = sil::tensor::TensorSymmetricIndex<sil::tensor::MetricIndex1<X, Y, Z>, sil::tensor::MetricIndex2<X, Y, Z>>;
```

And the metric tensor field is build upon it as described previously.

The most usefull usage of metric is <em>lowering</em> or <em>raising</em> indices. Indeed, on Riemannian manifolds the natural indices are attributed with a <em>covariant</em> or <em>contravariant character</em>. It is very convenient to produce a tensor product of metrics (like \f$g_{\mu\mu'}\;g_{\nu\nu'}\;g_{\rho\rho'}\f$), which is permitted by the `fill_metric_prod` function.

Lowering indices is thus made by performing a tensor product with a contravariant tensor, ie.:

\f\[
T_{\mu\nu\rho} = g_{\mu\mu'}\;g_{\nu\nu'}\;g_{\rho\rho'}\;T^{\mu'\nu'\rho'}
\f\]

using the `tensor_prod` function described in next section, or the `inplace_apply_metric` function if you want to reuse the allocation of \f$T^{\mu'\nu'\rho'}\f$ to store \f$T_{\mu\nu\rho}\f$ (which internally relies on `tensor_prod`).

\note The `fill_inverse_metric` function relies on `kokkos-kernels` `KokkosBatched::SerialInverseLU` function to compute the inverse metric (contravariant indexing of the metric, used to raise indices afterward).

## Tensor product

The generic `tensor_prod` function supports input and output tensors with any kind of internal symmetries. The Einstein convention stands: indices appearing in the two input tensors (one covariant and the other contravariant) are contracted while other indices are spectators (outer product). Any memory layouts are supported but bad choice of layout may highly deteriorate the performance. 

\important `tensor_prod` does not support tensor fields. You need to iterate over the nodes of the mesh, extract tensors (using the `Tensor::operator[]` function) and call `tensor_prod` at each node.
