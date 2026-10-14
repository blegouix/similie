<!--
SPDX-FileCopyrightText: 2026 Baptiste Legouix
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# API.md for SimiLie

Follow these interfaces for consistency and interoperability. Avoid additional
public members or arguments unless required by the operation. These rules apply
to the listed class families, including specializations and derived index tags.

## Tensor module

### Tensor indices

Static members: `is_tensor_index`, `is_explicitely_stored_tensor`, and
`subindices_domain_t`. Natural indices additionally expose
`is_tensor_natural_index` and `type_seq_dimensions`.

```cpp
KOKKOS_FUNCTION static constexpr subindices_domain_t subindices_domain();
static constexpr std::size_t rank();
static constexpr std::size_t size();
static constexpr std::size_t mem_size();
static constexpr std::size_t access_size();
```

Respectively: subindex domain, tensor rank, natural component count, stored
component count, and access-domain size. `size()` includes reconstructed
components; `access_size()` covers every identifier returned by `access_id()`.
Allocate storage with `mem_size()`.

Natural-index mappings:

```cpp
template <class ODim>
KOKKOS_FUNCTION static constexpr std::size_t mem_id();
KOKKOS_FUNCTION static constexpr std::size_t mem_id(std::size_t const natural_id);
KOKKOS_FUNCTION static constexpr std::size_t access_id(std::size_t const natural_id);
```

Structured-index mappings (`TensorIndex...` are the constituent natural indices):

```cpp
KOKKOS_FUNCTION static constexpr std::size_t mem_id(std::array<std::size_t, sizeof...(TensorIndex)> const natural_ids);
KOKKOS_FUNCTION static constexpr std::size_t access_id(std::array<std::size_t, sizeof...(TensorIndex)> const natural_ids);
```

`mem_id()` selects storage; `access_id()` encodes component reconstruction.
The array extent matches the constituent natural-index count. `mem_id()`
requires a stored component; `access_id()` also accepts reconstructed components.
A storage-free component may use `std::numeric_limits<std::size_t>::max()` as
its storage identifier.

Shared direct-access interface:

```cpp
KOKKOS_FUNCTION static constexpr std::size_t access_id_to_mem_id(std::size_t access_id);

template <class Tensor, class Elem, class Id, class FunctorType>
KOKKOS_FUNCTION static constexpr Tensor::element_type process_access(FunctorType const& access, Tensor tensor, Elem elem);

KOKKOS_FUNCTION static constexpr std::array<std::size_t, rank()> mem_id_to_canonical_natural_ids(std::size_t mem_id);
```

`process_access()` reconstructs the component. Reverse mappings require an
identifier with corresponding storage.

Indices with `is_explicitely_stored_tensor = false` use linear-combination
mappings instead of single storage identifiers:

```cpp
static constexpr std::pair<std::vector<double>, std::vector<std::size_t>> mem_lin_comb(std::array<std::size_t, sizeof...(TensorIndex)> const natural_ids);
static constexpr std::pair<std::vector<double>, std::vector<std::size_t>> access_id_to_mem_lin_comb(std::size_t access_id);
```

They retain `access_id()` and `process_access()`; a canonical reverse mapping
is only required when each stored component has a natural representative.

### Extrapolation rules

```cpp
template <class Domain, class Element>
KOKKOS_FUNCTION auto value(Domain domain, Element element) const;

template <class Field, class Element>
KOKKOS_FUNCTION auto operator()(Field const& field, Element element) const;
```

- `value()`: spatial basis coefficients, independent of field values; `domain` is the non-index domain.
- `operator()`: pointwise tensor sample, applying extrapolation outside the stored domain.

Collections of boundary policies expose the same interface, plus:

```cpp
template <std::size_t Axis, bool Upper>
KOKKOS_FUNCTION auto const& boundary_rule() const;
```

## Exterior module

### Incidence operators

Tensor-stencil interface:

```cpp
template <class ChainType, class LowerChainType, class Elem, class NaturalElem>
KOKKOS_FUNCTION static auto value(ChainType chain, LowerChainType lower_chain, Elem elem, NaturalElem natural_elem);

template <class CoboundaryTensorType, class Evaluator, class ChainType, class LowerChainType, class Elem>
KOKKOS_FUNCTION static void operator()(CoboundaryTensorType coboundary_tensor, Evaluator evaluator, ChainType chain, LowerChainType lower_chain, Elem elem);
```

- `value()`: stencil for the selected natural output component.
- `operator()`: evaluate through `evaluator` and write the output at `elem`.

### Hodge maps

For geometry-dependent maps (`MetricType`, `PositionType`, and `BatchElem`
are class template parameters):

```cpp
KOKKOS_FUNCTION static double value(MetricType metric, PositionType position, BatchElem elem, auto natural_elem);

template <misc::Specialization<tensor::Tensor> HodgeTensorType, misc::Specialization<tensor::Tensor> FormTensorType>
KOKKOS_FUNCTION static void operator()(HodgeTensorType hodge_tensor, FormTensorType form_tensor, MetricType metric, PositionType position, BatchElem elem);
```

- `value()`: coefficient selected by source and target natural indices.
- `operator()`: apply the map and write `hodge_tensor` at `elem`.

Metric-only maps omit the position argument. More generally, exterior
operators expose `value()` for local coefficients and `operator()` for
application, with geometry and boundary arguments as required. Staged
operators need only expose `operator()`.

## Physics module

### Hamiltonians

Static members: `std::size_t N` (moment count) and `bool IS_LINEAR`.
For a scalar potential and `N > 1` moments:

```cpp
KOKKOS_FUNCTION constexpr double hamiltonian(double phi, std::span<double const, N> pi) const;

template <class Index>
KOKKOS_FUNCTION constexpr double dhamiltonian_dmoments(double moments) const;

KOKKOS_FUNCTION constexpr double dhamiltonian_dpotential(double phi) const;
```

Respectively: energy density, selected moment derivative, and potential derivative.
Generated argument names and types follow the model; one-component states use
`double`, and spatially varying parameters add a templated element argument.

Quantity-based models accept `Moments moments, Elem elem` in
`dhamiltonian_dmoments<Index>()`, preserving cross-component coupling. They may
also expose `jacobian<RowIndex, ColumnIndex>(moments, elem)` for nonlinear solvers.

Nonlocal models can additionally expose:

```cpp
template <class Index, class Elem>
KOKKOS_FUNCTION constexpr auto dhamiltonian_dmoments_value(Elem elem) const;
```

Returns moment-derivative coefficients. An invertible moment relation may
also expose `moments<Index>(double dpotential_dx)`, with a trailing element
argument for spatially varying parameters.

### Equation wrappers

Equation wrappers expose `static constexpr bool IS_LINEAR = Hamiltonian::IS_LINEAR`.
All following members return `double` and are
`[[nodiscard]] KOKKOS_FUNCTION constexpr ... const`:

| Equation form | Member signatures |
| --- | --- |
| Time evolution | `template <class Index> dpotential_dt(double spatial_moments_component)`; `template <class Index> dpotential_dt(std::span<double const, Hamiltonian::N> spatial_moments)` |
| Time evolution | `dmoments_dt(double potential)`; `template <class Index> dmoments_dt(std::span<double const, 1> potential)` |
| Covariant | `template <class Index> potential_grad(double moments_component)`; `template <class Index> potential_grad(std::span<double const, Hamiltonian::N> moments)` |
| Covariant | `moments_div(double potential)`; `template <class Index> moments_div(std::span<double const, 1> potential)` |

Moment span overloads require a Hamiltonian accepting the span, or `N == 1`.
Otherwise, supply the selected component through the scalar overload.

Element-dependent wrappers add a trailing `Elem elem` with
`template <class Index, class Elem>`. Quantity-object and coefficient interfaces
follow the same naming pattern:

```cpp
template <class Index, class Moments, class Elem>
[[nodiscard]] KOKKOS_FUNCTION constexpr double dpotential_dt(Moments moments, Elem elem) const;

template <class Index, class Elem>
[[nodiscard]] KOKKOS_FUNCTION constexpr auto dpotential_dt_value(Elem elem) const;

template <class Index, class Elem>
[[nodiscard]] KOKKOS_FUNCTION constexpr auto dmoments_dt_value(Elem elem) const;
```

The quantity overload requires `get<Index>()` or a matching Hamiltonian
derivative. Coefficient overloads require the corresponding Hamiltonian
`dhamiltonian_*_value()` member. Constraints are omitted above.

### Constitutive laws

For a generated law with one state argument (argument names follow the definition):

```cpp
KOKKOS_FUNCTION constexpr double operator()(double x) const;
KOKKOS_FUNCTION constexpr double value(double x) const;
KOKKOS_FUNCTION constexpr double jacobian(double x) const;
KOKKOS_FUNCTION constexpr double inverse(double y) const;
KOKKOS_FUNCTION constexpr double inverse_value(double y) const;
```

Respectively: response, state derivative, synonym for `value()`, inverse
response, and inverse derivative. Additional generated `double` arguments
precede the state argument; inverse methods replace that argument with the output.

Change generated APIs through the Python definitions or generators in
`src/similie/physics/`.
