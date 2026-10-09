// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#pragma once

#include <cstddef>
#include <utility>

#include <similie/multidomains/multidomains.hpp>
#include <similie/physics/hamilton_equations.hpp>
#include <similie/physics/scalar_field/scalar_field_with_power_coupling.hpp>
#include <similie/tensor/full_tensor.hpp>

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

template <std::size_t Index>
struct Patch
{
    static constexpr std::size_t INDEX = Index;
};
struct Wall
{
};
struct Upstream
{
};
struct Downstream
{
};

template <std::size_t Index, class Dimension, sil::exterior::BoundarySide Side>
using PatchFace = sil::multidomains::Face<Patch<Index>, Dimension, Side>;

/** Six physical scalar-potential domains, with explicit exterior boundary nodes.
 * \important This documentation is fully AI-generated.
 * Each angular interface is declared once. The 2->3 connection carries the
 * circulation cut; the graph derives its reverse and generates every rule.
 * Radial faces meet wall, upstream, or downstream nodes with no allocated mesh.
 */
using PotentialFlowTopology = sil::multidomains::Topology<
        ddc::TypeSeq<
                sil::multidomains::Domain<Patch<0>, PotentialFlowPhysics, GridX, GridY>,
                sil::multidomains::Domain<Patch<1>, PotentialFlowPhysics, GridX, GridY>,
                sil::multidomains::Domain<Patch<2>, PotentialFlowPhysics, GridX, GridY>,
                sil::multidomains::Domain<Patch<3>, PotentialFlowPhysics, GridX, GridY>,
                sil::multidomains::Domain<Patch<4>, PotentialFlowPhysics, GridX, GridY>,
                sil::multidomains::Domain<Patch<5>, PotentialFlowPhysics, GridX, GridY>,
                sil::multidomains::
                        BoundaryDomain<Wall, sil::exterior::NaturalScalarExtrapolationRule>,
                sil::multidomains::
                        BoundaryDomain<Upstream, sil::exterior::PrescribedScalarExtrapolationRule>,
                sil::multidomains::BoundaryDomain<
                        Downstream,
                        sil::exterior::PrescribedScalarExtrapolationRule>>,
        sil::multidomains::Connection<
                PatchFace<0, GridX, sil::exterior::BoundarySide::Upper>,
                PatchFace<1, GridX, sil::exterior::BoundarySide::Lower>,
                false>,
        sil::multidomains::Connection<
                PatchFace<1, GridX, sil::exterior::BoundarySide::Upper>,
                PatchFace<2, GridX, sil::exterior::BoundarySide::Lower>,
                false>,
        sil::multidomains::Connection<
                PatchFace<2, GridX, sil::exterior::BoundarySide::Upper>,
                PatchFace<3, GridX, sil::exterior::BoundarySide::Lower>,
                false,
                1.0,
                -1.0>,
        sil::multidomains::Connection<
                PatchFace<3, GridX, sil::exterior::BoundarySide::Upper>,
                PatchFace<4, GridX, sil::exterior::BoundarySide::Lower>,
                false>,
        sil::multidomains::Connection<
                PatchFace<4, GridX, sil::exterior::BoundarySide::Upper>,
                PatchFace<5, GridX, sil::exterior::BoundarySide::Lower>,
                false>,
        sil::multidomains::Connection<
                PatchFace<5, GridX, sil::exterior::BoundarySide::Upper>,
                PatchFace<0, GridX, sil::exterior::BoundarySide::Lower>,
                false>,
        sil::multidomains::
                BoundaryConnection<PatchFace<0, GridY, sil::exterior::BoundarySide::Lower>, Wall>,
        sil::multidomains::
                BoundaryConnection<PatchFace<1, GridY, sil::exterior::BoundarySide::Lower>, Wall>,
        sil::multidomains::
                BoundaryConnection<PatchFace<2, GridY, sil::exterior::BoundarySide::Lower>, Wall>,
        sil::multidomains::
                BoundaryConnection<PatchFace<3, GridY, sil::exterior::BoundarySide::Lower>, Wall>,
        sil::multidomains::
                BoundaryConnection<PatchFace<4, GridY, sil::exterior::BoundarySide::Lower>, Wall>,
        sil::multidomains::
                BoundaryConnection<PatchFace<5, GridY, sil::exterior::BoundarySide::Lower>, Wall>,
        sil::multidomains::
                BoundaryConnection<PatchFace<0, GridY, sil::exterior::BoundarySide::Upper>, Wall>,
        sil::multidomains::BoundaryConnection<
                PatchFace<1, GridY, sil::exterior::BoundarySide::Upper>,
                Upstream>,
        sil::multidomains::
                BoundaryConnection<PatchFace<2, GridY, sil::exterior::BoundarySide::Upper>, Wall>,
        sil::multidomains::
                BoundaryConnection<PatchFace<3, GridY, sil::exterior::BoundarySide::Upper>, Wall>,
        sil::multidomains::BoundaryConnection<
                PatchFace<4, GridY, sil::exterior::BoundarySide::Upper>,
                Downstream>,
        sil::multidomains::
                BoundaryConnection<PatchFace<5, GridY, sil::exterior::BoundarySide::Upper>, Wall>>;

template <class Fields, class Positions, std::size_t... Indices>
auto bind_potential_flow_domains(
        Fields const& fields,
        Positions const& positions,
        PotentialFlowPhysics physics,
        double upstream,
        double downstream,
        std::index_sequence<Indices...>)
{
    return sil::multidomains::Multidomain(
            PotentialFlowTopology {},
            sil::multidomains::domain_data<
                    Patch<Indices>>(fields[Indices], positions[Indices], physics)...,
            sil::multidomains::boundary_data<Wall>(
                    sil::exterior::NaturalScalarExtrapolationRule {}),
            sil::multidomains::boundary_data<Upstream>(
                    sil::exterior::PrescribedScalarExtrapolationRule {upstream}),
            sil::multidomains::boundary_data<Downstream>(
                    sil::exterior::PrescribedScalarExtrapolationRule {downstream}));
}

template <class Fields, class Positions>
auto bind_potential_flow_domains(
        Fields const& fields,
        Positions const& positions,
        PotentialFlowPhysics physics,
        double upstream,
        double downstream)
{
    return bind_potential_flow_domains(
            fields,
            positions,
            physics,
            upstream,
            downstream,
            std::make_index_sequence<6> {});
}
} // namespace similie::onelab_interface::potential_flow_onelab
