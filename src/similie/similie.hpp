// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

//! @brief The top-level namespace of SimiLie.
//! All SimiLie symbols are defined either in this namespace or in a nested namespace.
namespace sil {
}

#include <similie/physics/dedonder_weyl_equations.hpp>
#include <similie/physics/hamilton_equations.hpp>
#include <similie/physics/magnetostatics/magnetostatics_quantities.hpp>
#include <similie/physics/scalar_field/scalar_field_with_power_coupling.hpp>
#include <similie/solvers/minimize_strong_formulation_residual.hpp>

#include "csr/csr.hpp"
#include "exterior/exterior.hpp"
#include "mesher/mesher.hpp"
#include "tensor/tensor.hpp"
#include "young_tableau/young_tableau.hpp"
