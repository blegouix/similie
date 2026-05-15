// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: MIT

#pragma once

//! @brief The top-level namespace of SimiLie.
//! All SimiLie symbols are defined either in this namespace or in a nested namespace.
namespace sil {
}

#include "csr/csr.hpp"
#include "exterior/exterior.hpp"
#include "mesher/mesher.hpp"
#include <similie/physics/dedonder_weyl.hpp>
#include <similie/physics/magnetostatics/magnetostatics_indices.hpp>
#include <similie/physics/magnetostatics/magnetostatics_quantities.hpp>
#include <similie/physics/magnetostatics/structured_linear_magnetostatics.hpp>
#include <similie/physics/scalar_field/scalar_field_with_power_coupling.hpp>
#include <similie/solvers/minimize_strong_formulation_residual.hpp>
#include "tensor/tensor.hpp"
#include "young_tableau/young_tableau.hpp"
