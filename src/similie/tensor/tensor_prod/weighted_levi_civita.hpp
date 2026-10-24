// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#pragma once

#include <similie/tensor/tensor_indices/weighted_levi_civita_tensor.hpp>

#include "levi_civita.hpp"

namespace sil::tensor::detail {
// The weights change coefficients, but preserve Levi-Civita's structural zeros.
// The shared structured contraction prunes repeated IDs on either operand.
template <class Index>
struct WeightedLeviCivitaTensorProductMatcher : LeviCivitaTensorProductMatcher<Index>
{
};
} // namespace sil::tensor::detail
