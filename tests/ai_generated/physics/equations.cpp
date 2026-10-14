// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#include <array>
#include <span>

#include <gtest/gtest.h>
#include <similie/physics/dedonder_weyl_equations.hpp>
#include <similie/physics/elasticity/linear_elasticity.hpp>
#include <similie/physics/elasticity/linear_elasticity_constitutive_law.hpp>
#include <similie/physics/hamilton_equations.hpp>
#include <similie/physics/scalar_field/scalar_field_with_power_coupling.hpp>

namespace {

struct Time
{
};
struct X
{
};
struct Y
{
};

struct ComponentHamiltonian
{
    static constexpr std::size_t N = 1;
    static constexpr bool IS_LINEAR = true;

    template <class Index>
    KOKKOS_FUNCTION constexpr double dhamiltonian_dmoments(double moment) const
    {
        return 2.0 * moment;
    }

    KOKKOS_FUNCTION constexpr double dhamiltonian_dpotential(double potential) const
    {
        return 3.0 * potential;
    }
};

struct SpatialHamiltonian
{
    static constexpr std::size_t N = 1;
    static constexpr bool IS_LINEAR = true;

    template <class Index, class Elem>
    KOKKOS_FUNCTION constexpr double dhamiltonian_dmoments(double moment, Elem elem) const
    {
        return elem * moment;
    }

    template <class Elem>
    KOKKOS_FUNCTION constexpr double dhamiltonian_dpotential(double potential, Elem elem) const
    {
        return elem * potential;
    }
};

struct CoupledHamiltonian
{
    static constexpr std::size_t N = 2;
    static constexpr bool IS_LINEAR = true;

    template <class Index>
    KOKKOS_FUNCTION constexpr double dhamiltonian_dmoments(std::span<double const, N> moments) const
    {
        return moments[0] + 2.0 * moments[1];
    }
};

template <class Equations>
concept HasMomentSpan = requires(Equations const& equations, std::span<double const, 3> moments) {
    equations.template dpotential_dt<X>(moments);
};

template <class Equations>
concept HasElementMomentSpan
        = requires(Equations const& equations, std::span<double const, 3> moments) {
              equations.template dpotential_dt<X>(moments, 0);
          };

template <class Equations>
concept HasMomentCoefficients
        = requires(Equations const& equations) { equations.template dpotential_dt_value<X>(0); };

TEST(PhysicsAPI, SingleMomentSpanAndScalarAgree)
{
    similie::physics::HamiltonEquations equations(ComponentHamiltonian {});
    similie::physics::DeDonderWeylEquations covariant_equations(ComponentHamiltonian {});
    std::array<double, 1> const moments {4.0};
    std::array<double, 1> const potential {5.0};
    EXPECT_TRUE(equations.IS_LINEAR);
    EXPECT_TRUE(covariant_equations.IS_LINEAR);
    EXPECT_DOUBLE_EQ(equations.dpotential_dt<X>(4.0), 8.0);
    EXPECT_DOUBLE_EQ(equations.dpotential_dt<X>(std::span<double const, 1>(moments)), 8.0);
    EXPECT_DOUBLE_EQ(equations.dpotential_dt<X>(std::span<double const, 1>(moments), 7), 8.0);
    EXPECT_DOUBLE_EQ(equations.dmoments_dt(5.0), -15.0);
    EXPECT_DOUBLE_EQ(equations.dmoments_dt<X>(std::span<double const, 1>(potential)), -15.0);
    EXPECT_DOUBLE_EQ(equations.dmoments_dt<X>(5.0, 7), -15.0);
    EXPECT_DOUBLE_EQ(equations.dmoments_dt<X>(std::span<double const, 1>(potential), 7), -15.0);
    EXPECT_DOUBLE_EQ(
            covariant_equations.potential_grad<X>(std::span<double const, 1>(moments)),
            8.0);
    EXPECT_DOUBLE_EQ(covariant_equations.potential_grad<X>(4.0), 8.0);
    EXPECT_DOUBLE_EQ(covariant_equations.moments_div(5.0), -15.0);
    EXPECT_DOUBLE_EQ(
            covariant_equations.moments_div<X>(std::span<double const, 1>(potential)),
            -15.0);
    static_assert(!HasMomentCoefficients<decltype(equations)>);
}

TEST(PhysicsAPI, ElementAwareSpanUsesSpatialParameters)
{
    similie::physics::HamiltonEquations equations(SpatialHamiltonian {});
    std::array<double, 1> const state {4.0};
    EXPECT_DOUBLE_EQ(equations.dpotential_dt<X>(std::span<double const, 1>(state), 7), 28.0);
    EXPECT_DOUBLE_EQ(equations.dmoments_dt<X>(std::span<double const, 1>(state), 7), -28.0);
}

TEST(PhysicsAPI, FullSpanPreservesComponentCoupling)
{
    similie::physics::HamiltonEquations equations(CoupledHamiltonian {});
    similie::physics::DeDonderWeylEquations covariant_equations(CoupledHamiltonian {});
    std::array<double, 2> const state {4.0, 5.0};
    EXPECT_DOUBLE_EQ(equations.dpotential_dt<X>(std::span<double const, 2>(state)), 14.0);
    EXPECT_DOUBLE_EQ(equations.dpotential_dt<X>(std::span<double const, 2>(state), 7), 14.0);
    EXPECT_DOUBLE_EQ(
            covariant_equations.potential_grad<X>(std::span<double const, 2>(state)),
            14.0);
}

TEST(PhysicsAPI, GeneratedScalarFieldUsesSelectedComponent)
{
    similie::physics::scalar_field::ScalarFieldWithPowerCouplingHamiltonian<Time, X, Y>
            hamiltonian(2.0, 0.0, 4.0);
    similie::physics::HamiltonEquations equations(hamiltonian);
    EXPECT_DOUBLE_EQ(equations.dpotential_dt<Time>(3.0), -3.0);
    EXPECT_DOUBLE_EQ(equations.dpotential_dt<X>(3.0), 3.0);
    EXPECT_DOUBLE_EQ(equations.dmoments_dt(5.0), 20.0);
    static_assert(!HasMomentSpan<decltype(equations)>);
}

TEST(PhysicsAPI, GeneratedElasticityRetainsCrossComponentResponse)
{
    similie::physics::elasticity::LinearElasticityHamiltonian<X, Y> hamiltonian(12.0, 0.25);
    similie::physics::HamiltonEquations equations(hamiltonian);
    static_assert(!HasMomentSpan<decltype(equations)>);
    static_assert(!HasElementMomentSpan<decltype(equations)>);
    similie::physics::elasticity::Strain2D const strain {.xx = 1.0, .yy = 2.0, .xy = 3.0};
    EXPECT_NEAR(
            equations.dpotential_dt<similie::physics::elasticity::StrainXX>(strain, 0),
            19.2,
            1e-12);
    EXPECT_NEAR(
            equations.dpotential_dt<similie::physics::elasticity::StrainYY>(strain, 0),
            28.8,
            1e-12);
    EXPECT_NEAR(
            equations.dpotential_dt<similie::physics::elasticity::StrainXY>(strain, 0),
            57.6,
            1e-12);
}

TEST(PhysicsAPI, GeneratedConstitutiveLawDerivativeAndInverseAgree)
{
    similie::physics::elasticity::LinearElasticStrainToStress law(4.0, 2.0);
    EXPECT_DOUBLE_EQ(law(3.0, 5.0), 26.0);
    EXPECT_DOUBLE_EQ(law.value(3.0, 5.0), 4.0);
    EXPECT_DOUBLE_EQ(law.jacobian(3.0, 5.0), 4.0);
    EXPECT_DOUBLE_EQ(law.inverse(3.0, 26.0), 5.0);
    EXPECT_DOUBLE_EQ(law.inverse_value(3.0, 26.0), 0.25);
}

} // namespace
