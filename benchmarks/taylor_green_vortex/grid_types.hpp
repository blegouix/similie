// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#pragma once

#include <ddc/ddc.hpp>

#include <similie/tensor/metric.hpp>
#include <similie/tensor/tensor_indices/antisymmetric_tensor.hpp>
#include <similie/tensor/tensor_indices/character.hpp>
#include <similie/tensor/tensor_indices/identity_tensor.hpp>

namespace taylor_green_vortex {
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
using Field = Kokkos::View<double**, Kokkos::LayoutRight>;
using SpatialDomain = ddc::DiscreteDomain<GridX, GridY, GridZ>;
using Scalar = sil::tensor::Covariant<sil::tensor::ScalarIndex>;
struct Direction : sil::tensor::TensorNaturalIndex<X, Y, Z>
{
};
struct CurlDirection : sil::tensor::TensorNaturalIndex<X, Y, Z>
{
};
using Vector = sil::tensor::Covariant<Direction>;
using TwoForm
        = sil::tensor::TensorAntisymmetricIndex<sil::tensor::Covariant<CurlDirection>, Vector>;
using PositionIndex = sil::tensor::Contravariant<Direction>;
using MetricIndex = sil::tensor::TensorIdentityIndex<
        sil::tensor::Covariant<sil::tensor::MetricIndex1<X, Y, Z>>,
        sil::tensor::Covariant<sil::tensor::MetricIndex2<X, Y, Z>>>;
using ScalarTensor = sil::tensor::Tensor<
        double,
        ddc::DiscreteDomain<GridX, GridY, GridZ, Scalar>,
        Kokkos::layout_right,
        Field::memory_space>;
using VectorTensor = sil::tensor::Tensor<
        double,
        ddc::DiscreteDomain<GridX, GridY, GridZ, Vector>,
        Kokkos::layout_right,
        Field::memory_space>;
using TwoFormTensor = sil::tensor::Tensor<
        double,
        ddc::DiscreteDomain<GridX, GridY, GridZ, TwoForm>,
        Kokkos::layout_right,
        Field::memory_space>;
using PositionTensor = sil::tensor::Tensor<
        double,
        ddc::DiscreteDomain<GridX, GridY, GridZ, PositionIndex>,
        Kokkos::layout_right,
        Field::memory_space>;
using MetricTensor = sil::tensor::Tensor<
        double,
        ddc::DiscreteDomain<GridX, GridY, GridZ, MetricIndex>,
        Kokkos::layout_right,
        Field::memory_space>;

} // namespace taylor_green_vortex
