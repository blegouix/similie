// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#pragma once

#include <similie/tensor/tensor.hpp>

namespace similie::physics::magnetostatics {

template <class... SpatialIndex>
using ForceDensityIndex = sil::tensor::Covariant<sil::tensor::TensorNaturalIndex<SpatialIndex...>>;

} // namespace similie::physics::magnetostatics
