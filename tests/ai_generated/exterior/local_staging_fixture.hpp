// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#pragma once

#include <similie/exterior/exterior.hpp>

namespace local_staging_test {
struct X
{
};
struct Y
{
};
struct Z
{
};
struct GridX
{
    using continuous_dimension_type = X;
};
struct GridY
{
    using continuous_dimension_type = Y;
};
struct GridZ
{
    using continuous_dimension_type = Z;
};
struct Direction : sil::tensor::TensorNaturalIndex<X, Y, Z>
{
};
struct Other : sil::tensor::TensorNaturalIndex<X, Y, Z>
{
};
struct Third : sil::tensor::TensorNaturalIndex<X, Y, Z>
{
};
using Scalar = sil::tensor::Covariant<sil::tensor::ScalarIndex>;
using One = sil::tensor::Covariant<Direction>;
using Two = sil::tensor::TensorAntisymmetricIndex<One, sil::tensor::Covariant<Other>>;
using Three = sil::tensor::
        TensorAntisymmetricIndex<One, sil::tensor::Covariant<Other>, sil::tensor::Covariant<Third>>;
using PositionIndex = sil::tensor::Contravariant<Direction>;
using MetricIndex = sil::tensor::TensorIdentityIndex<
        sil::tensor::Covariant<sil::tensor::MetricIndex1<X, Y, Z>>,
        sil::tensor::Covariant<sil::tensor::MetricIndex2<X, Y, Z>>>;
using Domain = ddc::DiscreteDomain<GridX, GridY, GridZ>;
using Exec = Kokkos::DefaultExecutionSpace;
using Memory = Exec::memory_space;
template <class Index>
using FullDomain = ddc::DiscreteDomain<GridX, GridY, GridZ, Index>;
template <class Index>
using Field = sil::tensor::Tensor<double, FullDomain<Index>, Kokkos::layout_right, Memory>;
template <class Index>
using Allocation = ddc::Chunk<double, FullDomain<Index>, ddc::KokkosAllocator<double, Memory>>;

template <class Index>
Allocation<Index> allocate(Domain const& domain)
{
    return Allocation<Index>(
            FullDomain<Index>(domain, sil::tensor::TensorAccessor<Index>::domain()),
            ddc::KokkosAllocator<double, Memory>());
}
inline Domain domain()
{
    return Domain(
            ddc::DiscreteElement<GridX, GridY, GridZ>(1, 1, 1),
            ddc::DiscreteVector<GridX, GridY, GridZ>(2, 2, 2));
}
inline Domain halo()
{
    return Domain(
            ddc::DiscreteElement<GridX, GridY, GridZ>(0, 0, 0),
            ddc::DiscreteVector<GridX, GridY, GridZ>(4, 4, 4));
}
inline void fill_geometry(Field<PositionIndex> const& position, bool skew, double scale = 1.0)
{
    ddc::parallel_for_each(
            Exec(),
            position.non_indices_domain(),
            KOKKOS_LAMBDA(Domain::discrete_element_type element) {
                double const x = element.template uid<GridX>();
                double const y = element.template uid<GridY>();
                double const z = element.template uid<GridZ>();
                position.mem(element, ddc::DiscreteElement<PositionIndex>(0))
                        = scale * (0.7 * x + (skew ? 0.12 * y + 0.01 * y * y : 0.0));
                position.mem(element, ddc::DiscreteElement<PositionIndex>(1))
                        = scale * (0.9 * y + (skew ? 0.04 * z * z : 0.0));
                position.mem(element, ddc::DiscreteElement<PositionIndex>(2)) = scale * 1.1 * z;
            });
}
template <class Form>
void fill_form(Field<Form> const& input, double offset = 0.0)
{
    ddc::parallel_for_each(
            Exec(),
            input.domain(),
            KOKKOS_LAMBDA(typename FullDomain<Form>::discrete_element_type element) {
                input.mem(element) = offset + 0.4 + 0.3 * element.template uid<Form>()
                                     + 0.02 * element.template uid<GridX>()
                                     - 0.04 * element.template uid<GridY>();
            });
}
} // namespace local_staging_test
