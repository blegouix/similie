// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: MIT

#include <filesystem>

#include <gtest/gtest.h>
#include <similie/physics/dedonder_weyl_equations.hpp>
#include <similie/physics/hamilton_equations.hpp>
#include <similie/physics/magnetostatics/linear_magnetostatics.hpp>
#include <similie/physics/magnetostatics/magnetostatics_quantities.hpp>
#include <similie/physics/structured_scalar_potential_equations.hpp>
#include <similie/physics/scalar_field/scalar_field_with_power_coupling.hpp>
#include <similie/physics/stationary_equations_operator.hpp>

#include "onelab_interface.hpp"

namespace {

std::filesystem::path test_file(char const* name)
{
    return std::filesystem::path(SIMILIE_ONELAB_TEST_DIR) / name;
}

template <class ReferenceOperator, class WrappedOperator, class OutputView, class InputView>
struct CompareStationaryOperatorsFunctor
{
    ReferenceOperator m_reference_operator;
    WrappedOperator m_wrapped_operator;
    OutputView m_output_reference;
    OutputView m_output_wrapped;
    InputView m_input;

    KOKKOS_FUNCTION void operator()(std::size_t const row) const
    {
        m_reference_operator.apply_at(m_output_reference, m_input, row);
        m_wrapped_operator.apply_at(m_output_wrapped, m_input, row);
    }
};

} // namespace

TEST(OnelabInterface, ParseMagnetostaticsSilpro)
{
    similie::onelab_interface::SilproProblem const problem
            = similie::onelab_interface::OnelabInterface::parse_silpro_file(
                    test_file("magnetostatics.silpro"));

    EXPECT_EQ(problem.name, "Test magnetostatics problem");
    EXPECT_EQ(problem.physics, similie::onelab_interface::SupportedPhysics::Magnetostatics);
    EXPECT_EQ(
            problem.solver,
            similie::onelab_interface::SupportedSolver::MinimizeStrongFormulationResidual);
    EXPECT_EQ(problem.solver_settings.max_iterations, 4321U);
    EXPECT_DOUBLE_EQ(problem.solver_settings.relative_tolerance, 1e-8);
    EXPECT_EQ(problem.solver_settings.jacobi_max_block_size, 3U);
    EXPECT_FALSE(problem.solver_settings.use_matrix_free);
}

TEST(OnelabInterface, ParseScalarFieldSilpro)
{
    similie::onelab_interface::SilproProblem const problem
            = similie::onelab_interface::OnelabInterface::parse_silpro_file(
                    test_file("scalar_field.silpro"));

    EXPECT_EQ(problem.name, "Test scalar field problem");
    EXPECT_EQ(
            problem.physics,
            similie::onelab_interface::SupportedPhysics::ScalarFieldWithPowerCoupling);
    EXPECT_EQ(problem.solver_settings.max_iterations, 250U);
    EXPECT_DOUBLE_EQ(problem.solver_settings.relative_tolerance, 1e-6);
    EXPECT_TRUE(problem.solver_settings.use_matrix_free);
    EXPECT_DOUBLE_EQ(problem.scalar_field.mass, 2.5);
    EXPECT_DOUBLE_EQ(problem.scalar_field.coupling_constant, 3.25);
    EXPECT_DOUBLE_EQ(problem.scalar_field.coupling_power, 6.0);
}

TEST(OnelabInterface, HamiltonEquationsStaticPotentialDerivative)
{
    using similie::physics::HamiltonEquations;
    using similie::physics::magnetostatics::LinearMagnetostaticsHamiltonian;

    HamiltonEquations const equations(LinearMagnetostaticsHamiltonian(2.0));

    EXPECT_DOUBLE_EQ(equations.template dpotential_dt<0>(4.0), 2.0);
    EXPECT_DOUBLE_EQ(equations.template dpotential_dt<1>(6.0), 3.0);
    EXPECT_DOUBLE_EQ(equations.template dpotential_dt<2>(8.0), 4.0);
}

TEST(OnelabInterface, HamiltonEquationsStaticMomentumDerivative)
{
    using similie::physics::HamiltonEquations;
    using similie::physics::scalar_field::ScalarFieldWithPowerCouplingHamiltonian;

    HamiltonEquations const equations(ScalarFieldWithPowerCouplingHamiltonian(2.0, 0.0, 4.0));

    EXPECT_DOUBLE_EQ(equations.template dmoments_dt<0>(3.0), 12.0);
    EXPECT_DOUBLE_EQ(equations.template dmoments_dt<1>(3.0), 12.0);
}

TEST(OnelabInterface, DeDonderWeylEquationsStaticInterfaces)
{
    using similie::physics::DeDonderWeylEquations;
    using similie::physics::scalar_field::ScalarFieldWithPowerCouplingHamiltonian;

    DeDonderWeylEquations const equations(ScalarFieldWithPowerCouplingHamiltonian(2.0, 0.0, 4.0));

    EXPECT_DOUBLE_EQ(equations.template potential_grad<0>(5.0), -5.0);
    EXPECT_DOUBLE_EQ(equations.template potential_grad<1>(5.0), 5.0);
    EXPECT_DOUBLE_EQ(equations.moments_div(3.0), 12.0);
}

TEST(OnelabInterface, LinearMagneticInductionToMagneticFieldValueAndApplication)
{
    using similie::physics::magnetostatics::LinearMagneticInductionToMagneticField;

    LinearMagneticInductionToMagneticField const constitutive_law(2.0);
    EXPECT_DOUBLE_EQ(constitutive_law.forward_value(3.0, 4.0), 1.5);
    EXPECT_DOUBLE_EQ(constitutive_law.forward(3.0, 4.0), 6.0);
    EXPECT_DOUBLE_EQ(constitutive_law.inverse_value(3.0, 5.0), 2.0 / 3.0);
    EXPECT_DOUBLE_EQ(constitutive_law.inverse(3.0, 5.0), 10.0 / 3.0);
}

TEST(OnelabInterface, StationaryMagnetostaticsOperatorMatchesPrediscretizedForMuOne)
{
    using namespace similie::physics::magnetostatics;

    Kokkos::View<double*> x_coords("x", 5);
    Kokkos::View<double*> y_coords("y", 5);
    auto x_host = Kokkos::create_mirror_view(x_coords);
    auto y_host = Kokkos::create_mirror_view(y_coords);
    for (std::size_t i = 0; i < 5; ++i) {
        x_host(i) = static_cast<double>(i);
        y_host(i) = static_cast<double>(i);
    }
    Kokkos::deep_copy(x_coords, x_host);
    Kokkos::deep_copy(y_coords, y_host);

    similie::solvers::StructuredScalarPoissonStrongFormOperator2D<
            X,
            Y,
            typename Kokkos::DefaultExecutionSpace::memory_space>
            reference_operator(x_coords, y_coords);
    similie::physics::HamiltonEquations equations(LinearMagnetostaticsHamiltonian(1.0));
    auto wrapped_operator = similie::physics::make_stationary_equations_operator(
            equations,
            similie::solvers::StructuredScalarPoissonStrongFormOperator2D<
                    X,
                    Y,
                    typename Kokkos::DefaultExecutionSpace::memory_space>(x_coords, y_coords));

    Kokkos::View<double**> input("input", 25, 1);
    Kokkos::View<double**> output_reference("output_reference", 25, 1);
    Kokkos::View<double**> output_wrapped("output_wrapped", 25, 1);
    auto input_host = Kokkos::create_mirror_view(input);
    for (std::size_t j = 0; j < 5; ++j) {
        for (std::size_t i = 0; i < 5; ++i) {
            input_host(i + 5 * j, 0) = static_cast<double>(i * i + 2 * j);
        }
    }
    Kokkos::deep_copy(input, input_host);

    Kokkos::parallel_for(
            Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, 25),
            CompareStationaryOperatorsFunctor {
                    reference_operator,
                    wrapped_operator,
                    output_reference,
                    output_wrapped,
                    input,
            });
    auto output_reference_host
            = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), output_reference);
    auto output_wrapped_host
            = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), output_wrapped);

    for (std::size_t row = 0; row < 25; ++row) {
        EXPECT_NEAR(output_wrapped_host(row, 0), output_reference_host(row, 0), 1e-12)
                << "row=" << row;
    }
}
