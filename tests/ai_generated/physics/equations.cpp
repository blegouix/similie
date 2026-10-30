// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#include <array>
#include <type_traits>

#include <gtest/gtest.h>
#include <similie/physics/dedonder_weyl_equations.hpp>
#include <similie/physics/elasticity/linear_elasticity.hpp>
#include <similie/physics/elasticity/linear_elasticity_constitutive_law.hpp>
#include <similie/physics/hamilton_equations.hpp>
#include <similie/physics/magnetostatics/linear_magnetostatics_constitutive_law.hpp>
#include <similie/physics/magnetostatics/nonlinear_magnetostatics.hpp>
#include <similie/physics/scalar_field/scalar_field_with_power_coupling.hpp>

namespace physics_api_test {

struct Time
{
};
struct X
{
};
struct Y
{
};

using Component0 = std::integral_constant<std::size_t, 0>;
using Component1 = std::integral_constant<std::size_t, 1>;
using Component2 = std::integral_constant<std::size_t, 2>;

template <class Equations>
concept AcceptsScalarStateConcept = requires(Equations const& equations) {
    equations.template dpotential_dt<Component0>(4.0, 0);
};

struct SpatialHamiltonian
{
    static constexpr std::size_t N = 1;
    static constexpr bool IS_LINEAR = true;
    struct State
    {
        std::array<double, 1> potential {};
        std::array<double, N> moments {};
    };
    template <class Elem>
    KOKKOS_FUNCTION constexpr double hamiltonian(State const& state, Elem elem) const
    {
        return elem
               * (state.potential[0] * state.potential[0] + state.moments[0] * state.moments[0])
               / 2;
    }
    template <class Index, class Elem>
    KOKKOS_FUNCTION constexpr double dhamiltonian_dmoments(State const& state, Elem elem) const
    {
        return elem * state.moments[Index::value];
    }
    template <class Index, class Elem>
    KOKKOS_FUNCTION constexpr double dhamiltonian_dpotential(State const& state, Elem elem) const
    {
        return elem * state.potential[Index::value];
    }
    template <class Row, class Column, class Elem>
    KOKKOS_FUNCTION constexpr double jacobian(State const&, Elem elem) const
    {
        return Row::value == Column::value ? elem : 0;
    }
};

static_assert(similie::physics::HamiltonianPhysicsConcept<SpatialHamiltonian, int>);
template <class Equations>
concept CoefficientEquationsConcept
        = requires(Equations const& equations, typename Equations::State const& state) {
              equations.template dpotential_dt_value<Component0>(state, 0);
              equations.template dmoments_dt_value<Component0>(state, 0);
          };

static_assert(
        !CoefficientEquationsConcept<similie::physics::HamiltonEquations<SpatialHamiltonian>>);
static_assert(CoefficientEquationsConcept<similie::physics::HamiltonEquations<
                      similie::physics::detail::HamiltonianPhysicsConcept>>);

TEST(PhysicsAPI, SharedStateFeedsBothEquationWrappers)
{
    similie::physics::detail::HamiltonianPhysicsConcept::State const state {{5.0}, {4.0}};
    similie::physics::HamiltonEquations equations(
            similie::physics::detail::HamiltonianPhysicsConcept {});
    similie::physics::DeDonderWeylEquations covariant_equations(
            similie::physics::detail::HamiltonianPhysicsConcept {});
    static_assert(!AcceptsScalarStateConcept<decltype(equations)>);
    EXPECT_TRUE(equations.IS_LINEAR);
    EXPECT_DOUBLE_EQ(equations.dpotential_dt<Component0>(state, 0), 4.0);
    EXPECT_DOUBLE_EQ(equations.dmoments_dt<Component0>(state, 0), -5.0);
    EXPECT_DOUBLE_EQ(covariant_equations.potential_grad<Component0>(state, 0), 4.0);
    EXPECT_DOUBLE_EQ(covariant_equations.moments_div<Component0>(state, 0), -5.0);
    EXPECT_DOUBLE_EQ(equations.dpotential_dt_value<Component0>(state, 0), 1.0);
    EXPECT_DOUBLE_EQ(equations.dmoments_dt_value<Component0>(state, 0), -1.0);
}

TEST(PhysicsAPI, ElementArgumentReachesSpatialParameters)
{
    SpatialHamiltonian::State const state {{4.0}, {4.0}};
    similie::physics::HamiltonEquations equations(SpatialHamiltonian {});
    similie::physics::DeDonderWeylEquations covariant_equations(SpatialHamiltonian {});
    EXPECT_DOUBLE_EQ(equations.dpotential_dt<Component0>(state, 7), 28.0);
    EXPECT_DOUBLE_EQ(equations.dmoments_dt<Component0>(state, 7), -28.0);
    EXPECT_DOUBLE_EQ(covariant_equations.potential_grad<Component0>(state, 7), 28.0);
    EXPECT_DOUBLE_EQ(covariant_equations.moments_div<Component0>(state, 7), -28.0);
}

TEST(PhysicsAPI, GeneratedScalarFieldUsesSelectedComponent)
{
    similie::physics::scalar_field::ScalarFieldWithPowerCouplingHamiltonian<Time, X, Y>
            hamiltonian(2.0, 0.0, 4.0);
    similie::physics::scalar_field::ScalarFieldWithPowerCouplingHamiltonian<Time, X, Y>::State const
            state {{5.0}, {3.0, 3.0, 0.0}};
    similie::physics::HamiltonEquations equations(hamiltonian);
    EXPECT_DOUBLE_EQ(equations.dpotential_dt<Component0>(state, 0), -3.0);
    EXPECT_DOUBLE_EQ(equations.dpotential_dt<Component1>(state, 0), 3.0);
    EXPECT_DOUBLE_EQ(equations.dmoments_dt<Component0>(state, 0), 20.0);
    EXPECT_DOUBLE_EQ(hamiltonian.hamiltonian(state, 0), -50.0);
}

TEST(PhysicsAPI, GeneratedElasticityRetainsCoupledState)
{
    similie::physics::elasticity::LinearElasticityHamiltonian<X, Y> hamiltonian(12.0, 0.25);
    similie::physics::elasticity::LinearElasticityHamiltonian<X, Y>::State state {};
    state.moments = {1.0, 3.0, 2.0};
    state.f = {4.0, 5.0};
    similie::physics::HamiltonEquations equations(hamiltonian);
    EXPECT_NEAR(equations.dpotential_dt<Component0>(state, 0), 19.2, 1e-12);
    EXPECT_NEAR(equations.dpotential_dt<Component2>(state, 0), 28.8, 1e-12);
    EXPECT_NEAR(equations.dpotential_dt<Component1>(state, 0), 57.6, 1e-12);
    EXPECT_DOUBLE_EQ(equations.dmoments_dt<Component0>(state, 0), 4.0);
    EXPECT_DOUBLE_EQ(equations.dmoments_dt<Component1>(state, 0), 5.0);
    EXPECT_NEAR((equations.jacobian<Component0, Component2>(state, 0)), 3.2, 1e-12);
    EXPECT_NEAR(hamiltonian.hamiltonian(state, 0), 124.8, 1e-12);
}

TEST(PhysicsAPI, ScalarConstitutiveLawSharesStateSignature)
{
    similie::physics::elasticity::LinearElasticStrainToStress law(4.0, 2.0);
    similie::physics::elasticity::LinearElasticStrainToStress::State state {{5.0}, 3.0};
    EXPECT_DOUBLE_EQ(law(state, 0)[0], 26.0);
    EXPECT_DOUBLE_EQ(law.value(state, 0)[0], 4.0);
    EXPECT_DOUBLE_EQ(law.jacobian(state, 0)[0], 4.0);
    state.value[0] = 26.0;
    EXPECT_DOUBLE_EQ(law.inverse(state, 0)[0], 5.0);
    EXPECT_DOUBLE_EQ(law.inverse_value(state, 0)[0], 0.25);
}

TEST(PhysicsAPI, VectorConstitutiveLawRetainsInverseAndJacobian)
{
    similie::physics::magnetostatics::InterpolatedNonlinearBHCurve<4>
            curve(std::array<double, 4> {0.0, 1.0, 2.0, 3.0},
                  std::array<double, 4> {0.0, 2.0, 7.0, 20.0});
    similie::physics::magnetostatics::NonlinearMagneticInductionToMagneticField law(curve);
    similie::physics::magnetostatics::NonlinearMagneticInductionToMagneticField<
            decltype(curve)>::State state {{1.25, 0.5, 0.2}};
    std::array<double, 3> const response = law(state, 0);
    std::array<double, 9> const jacobian = law.jacobian(state, 0);
    for (std::size_t column = 0; column < 3; ++column) {
        double const step = 1e-6;
        state.value[column] += step;
        std::array<double, 3> const upper = law(state, 0);
        state.value[column] -= 2 * step;
        std::array<double, 3> const lower = law(state, 0);
        state.value[column] += step;
        for (std::size_t row = 0; row < 3; ++row) {
            EXPECT_NEAR(jacobian[3 * row + column], (upper[row] - lower[row]) / (2 * step), 1e-8);
        }
    }
    std::array<double, 3> const original = state.value;
    state.value = response;
    std::array<double, 3> const inverse = law.inverse(state, 0);
    std::array<double, 9> const inverse_jacobian = law.inverse_value(state, 0);
    for (std::size_t row = 0; row < 3; ++row) {
        EXPECT_NEAR(inverse[row], original[row], 1e-12);
        for (std::size_t column = 0; column < 3; ++column) {
            double product = 0;
            for (std::size_t k = 0; k < 3; ++k) {
                product += jacobian[3 * row + k] * inverse_jacobian[3 * k + column];
            }
            EXPECT_NEAR(product, row == column ? 1.0 : 0.0, 1e-12);
        }
    }
}

TEST(PhysicsAPI, MagneticConstitutiveLawsUsePointwiseFields)
{
    similie::physics::magnetostatics::LinearMagneticInductionToMagneticField const linear(4.0);
    similie::physics::magnetostatics::LinearMagneticInductionToMagneticField::State state {{2.0}};
    EXPECT_DOUBLE_EQ(linear(state, 0)[0], 0.5);
    EXPECT_DOUBLE_EQ(linear.jacobian(state, 0)[0], 0.25);
    state.value[0] = 0.5;
    EXPECT_DOUBLE_EQ(linear.inverse(state, 0)[0], 2.0);
    EXPECT_DOUBLE_EQ(linear.inverse_value(state, 0)[0], 4.0);

    similie::physics::magnetostatics::InterpolatedNonlinearBHCurve<3> const
            curve(std::array<double, 3> {0.0, 1.0, 3.0}, std::array<double, 3> {0.0, 2.0, 10.0});
    similie::physics::magnetostatics::NonlinearMagneticInductionToMagneticField const law(curve);
    similie::physics::magnetostatics::NonlinearMagneticState const origin {};
    std::array<double, 3> const field = law(origin, 0);
    std::array<double, 3> const inverse = law.inverse(origin, 0);
    std::array<double, 9> const jacobian = law.jacobian(origin, 0);
    for (std::size_t row = 0; row < 3; ++row) {
        EXPECT_DOUBLE_EQ(field[row], 0.0);
        EXPECT_DOUBLE_EQ(inverse[row], 0.0);
        for (std::size_t column = 0; column < 3; ++column)
            EXPECT_DOUBLE_EQ(jacobian[3 * row + column], row == column ? 2.0 : 0.0);
    }
    similie::physics::magnetostatics::NonlinearMagneticState const induction {{1.2, 0.9, 0.0}};
    std::array<double, 3> const response = law(induction, 0);
    EXPECT_NEAR(response[0], 3.2, 1e-12);
    EXPECT_NEAR(response[1], 2.4, 1e-12);
    std::array<double, 9> const tangent = law.jacobian(induction, 0);
    for (std::size_t row = 0; row < 3; ++row)
        for (std::size_t column = 0; column < 3; ++column)
            EXPECT_DOUBLE_EQ(tangent[3 * row + column], tangent[3 * column + row]);
}

TEST(PhysicsAPI, NonlinearHamiltonianRetainsCrossComponentDerivatives)
{
    similie::physics::magnetostatics::InterpolatedNonlinearBHCurve<4>
            curve(std::array<double, 4> {0.0, 1.0, 2.0, 3.0},
                  std::array<double, 4> {0.0, 2.0, 7.0, 20.0});
    similie::physics::magnetostatics::NonlinearMagnetostaticsHamiltonian hamiltonian(curve);
    similie::physics::magnetostatics::NonlinearMagnetostaticsHamiltonian<decltype(curve)>::State
            state {};
    state.moments = {1.25, 0.5, 0.2};
    double const derivative = hamiltonian.dhamiltonian_dmoments<Component0>(state, 0);
    double const cross_derivative = hamiltonian.jacobian<Component0, Component1>(state, 0);
    double const step = 1e-6;
    state.moments[0] += step;
    double const upper_energy = hamiltonian.hamiltonian(state, 0);
    state.moments[0] -= 2 * step;
    double const lower_energy = hamiltonian.hamiltonian(state, 0);
    state.moments[0] += step;
    EXPECT_NEAR(derivative, (upper_energy - lower_energy) / (2 * step), 1e-8);
    state.moments[1] += step;
    double const upper_derivative = hamiltonian.dhamiltonian_dmoments<Component0>(state, 0);
    state.moments[1] -= 2 * step;
    double const lower_derivative = hamiltonian.dhamiltonian_dmoments<Component0>(state, 0);
    EXPECT_NEAR(cross_derivative, (upper_derivative - lower_derivative) / (2 * step), 1e-8);
}

} // namespace physics_api_test
