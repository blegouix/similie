// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

// Check the generated spatial model without relying on exterior transitive includes.
#include <similie/physics/magnetostatics/linear_magnetostatics.hpp>

// Include each standardized family so its inline assertions run in every test build.
#include <similie/exterior/codifferential/codifferential.hpp>
#include <similie/exterior/continuous_hodge_star/continuous_hodge_star.hpp>
#include <similie/exterior/laplacian/laplacian.hpp>
#include <similie/physics/dedonder_weyl_equations.hpp>
#include <similie/physics/elasticity/linear_elasticity.hpp>
#include <similie/physics/elasticity/linear_elasticity_constitutive_law.hpp>
#include <similie/physics/hamilton_equations.hpp>
#include <similie/physics/magnetostatics/linear_magnetostatics_constitutive_law.hpp>
#include <similie/physics/magnetostatics/nonlinear_magnetostatics.hpp>
#include <similie/physics/scalar_field/scalar_field_with_power_coupling.hpp>
#include <similie/tensor/tensor.hpp>

namespace concept_test {

struct MissingInterface
{
};

struct NonHamiltonianPhysics
{
    static constexpr std::size_t N = 2;
    static constexpr bool IS_LINEAR = false;
    struct State
    {
        std::array<double, N> value {};
    };
};

struct WrongPhysicsMetadata : NonHamiltonianPhysics
{
    static constexpr int IS_LINEAR = 0;
};

struct WrongConstitutiveReturn
{
    static constexpr std::size_t N = 1;
    struct State
    {
        std::array<double, 1> value {};
    };
    float operator()(State const&, int) const;
    float value(State const&, int) const;
    float jacobian(State const&, int) const;
    float inverse(State const&, int) const;
    float inverse_value(State const&, int) const;
};

struct MissingExtrapolationBasis
{
    template <class Field, class Element>
    double operator()(Field const&, Element) const;
};

struct MissingIndexAccess : sil::tensor::TensorFullIndex<sil::tensor::ScalarIndex>
{
    static void process_access() = delete;
};

struct MissingIndexMemoryMapping : sil::tensor::TensorFullIndex<sil::tensor::ScalarIndex>
{
    static void mem_id() = delete;
};

static_assert(!sil::tensor::TensorIndexConcept<MissingInterface>);
static_assert(!sil::tensor::TensorIndexConcept<MissingIndexAccess>);
static_assert(!sil::tensor::TensorIndexConcept<MissingIndexMemoryMapping>);
static_assert(!sil::tensor::ExtrapolationRuleConcept<
              MissingExtrapolationBasis,
              sil::tensor::detail::FieldConcept<sil::tensor::ScalarIndex>,
              ddc::DiscreteElement<sil::tensor::detail::GridConcept>>);
static_assert(!similie::physics::ConstitutiveLawConcept<WrongConstitutiveReturn, int>);
static_assert(similie::physics::PhysicsConcept<NonHamiltonianPhysics>);
static_assert(!similie::physics::HamiltonianPhysicsConcept<NonHamiltonianPhysics, int>);
static_assert(!similie::physics::PhysicsConcept<WrongPhysicsMetadata>);
static_assert(!similie::physics::PhysicsConcept<MissingInterface>);
static_assert(!similie::physics::HamiltonianPhysicsConcept<MissingInterface, int>);

} // namespace concept_test
