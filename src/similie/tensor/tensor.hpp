// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

#include <ddc/ddc.hpp>

#include <similie/misc/macros.hpp>
#include <similie/misc/small_matrix.hpp>
#include <similie/misc/specialization.hpp>
#include <similie/tensor/tensor_impl.hpp>
#include <similie/tensor/tensor_indices/antisymmetric_tensor.hpp>
#include <similie/tensor/tensor_indices/character.hpp>
#include <similie/tensor/tensor_indices/diagonal_tensor.hpp>
#include <similie/tensor/tensor_indices/full_tensor.hpp>
#include <similie/tensor/tensor_indices/identity_tensor.hpp>
#include <similie/tensor/tensor_indices/levi_civita_tensor.hpp>
#include <similie/tensor/tensor_indices/weighted_levi_civita_tensor.hpp>
#include <similie/tensor/tensor_indices/lorentzian_sign_tensor.hpp>
#include <similie/tensor/tensor_indices/prime.hpp>
#include <similie/tensor/tensor_indices/relabelization.hpp>
#include <similie/tensor/tensor_indices/symmetric_tensor.hpp>
#include <similie/tensor/tensor_prod/tensor_prod.hpp>

#include "determinant.hpp"
#include "external_domain_extrapolation_rule.hpp"
#include "metric.hpp"
#include "scalar_extrapolation_rules.hpp"
#include "tensor_with_extrapolation_rules.hpp"
#if defined BUILD_YOUNG_TABLEAU
#include <similie/tensor/tensor_indices/young_tableau_tensor.hpp>
#endif
