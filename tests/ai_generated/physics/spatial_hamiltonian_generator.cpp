// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#include <cstddef>
#include <type_traits>

#include "spatial_hamiltonian.hpp"

using Index = std::integral_constant<std::size_t, 0>;
constexpr test::SpatialScalarHamiltonian model(2.0);
constexpr test::SpatialScalarHamiltonian::State state {{3.0}, {3.0}};
constexpr test::SpatialScalarHamiltonian::State inverse_state {{0.0}, {30.0}};
static_assert(model.N == 1);
static_assert(model.IS_LINEAR);
static_assert(model.dhamiltonian_dmoments<Index>(state, 4) == 30.0);
static_assert(model.dhamiltonian_dpotential<Index>(state, 4) == 30.0);
static_assert(model.moments<Index>(inverse_state, 4) == 3.0);

int main()
{
    // std::pow is not constexpr on every supported compiler.
    return model.hamiltonian(state, 4) == 90.0 ? 0 : 1;
}
