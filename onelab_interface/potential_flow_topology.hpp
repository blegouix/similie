// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#pragma once

#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

#include <similie/multidomains/multidomains.hpp>
#include <similie/physics/hamilton_equations.hpp>
#include <similie/physics/scalar_field/scalar_field_with_power_coupling.hpp>
#include <similie/tensor/tensor_indices/full_tensor.hpp>

namespace similie::onelab_interface::potential_flow_onelab {
struct X
{
};
struct Y
{
};
struct T
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

using PotentialIndex = sil::tensor::Covariant<sil::tensor::ScalarIndex>;
using GradientIndex = sil::tensor::Covariant<sil::tensor::TensorNaturalIndex<X, Y>>;
using PositionIndex = sil::tensor::Contravariant<sil::tensor::TensorNaturalIndex<X, Y>>;

using GridDomain = ddc::DiscreteDomain<GridX, GridY>;
using PotentialDomain = ddc::DiscreteDomain<GridX, GridY, PotentialIndex>;
using PositionDomain = ddc::DiscreteDomain<GridX, GridY, PositionIndex>;

using FreeScalarFieldHamiltonian
        = physics::scalar_field::ScalarFieldWithPowerCouplingHamiltonian<T, X, Y>;
using PotentialFlowPhysics = physics::HamiltonEquations<FreeScalarFieldHamiltonian>;


/** Bind scalar physics to the runtime graph inferred from the ONELAB mesh.
 * \important This operator and documentation is fully AI-generated.
 */
template <class Fields, class Positions>
auto bind_potential_flow_domains(
        sil::multidomains::Topology<2> topology,
        Fields const& fields,
        Positions const& positions,
        PotentialFlowPhysics physics)
{
    if (fields.size() != positions.size() || fields.size() != topology.domain_count())
        throw std::invalid_argument("potential-flow domain data does not match topology");
    std::vector<
            decltype(sil::multidomains::domain_data(fields.front(), positions.front(), physics))>
            data;
    data.reserve(fields.size());
    for (std::size_t id = 0; id < fields.size(); ++id)
        data.push_back(sil::multidomains::domain_data(fields[id], positions[id], physics));
    return sil::multidomains::Multidomain(std::move(topology), std::move(data));
}
} // namespace similie::onelab_interface::potential_flow_onelab
