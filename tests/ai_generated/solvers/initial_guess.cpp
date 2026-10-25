// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#include <gtest/gtest.h>
#include <similie/solvers/minimize_strong_formulation_residual.hpp>

namespace {
class DiagonalOperator
{
public:
    static constexpr bool IS_LINEAR = true, IS_SYMMETRIC = true;
    std::size_t size() const
    {
        return 8;
    }

    template <class Input, class Output>
    void apply(Kokkos::DefaultExecutionSpace exec, Input const& x, Output const& y) const
    {
        Kokkos::parallel_for(
                "diagonal_initial_guess_test",
                Kokkos::RangePolicy<>(exec, 0, 8),
                KOKKOS_LAMBDA(int i) { y(i, 0) = (i + 1) * x(i, 0); });
    }
};

similie::solvers::StrongFormulationSolverSettings settings(bool warm_start)
{
    similie::solvers::StrongFormulationSolverSettings settings;
    settings.preconditioner = similie::solvers::PreconditionerType::Identity;
    settings.use_initial_guess = warm_start;
    settings.relative_tolerance = 1e-11;
    return settings;
}
} // namespace

TEST(SolverInitialGuess, ExactPreviousSolutionRequiresNoIterations)
{
    Kokkos::View<double**> rhs("rhs", 8, 1), solution("solution", 8, 1);
    auto host_rhs = Kokkos::create_mirror_view(rhs);
    auto host_solution = Kokkos::create_mirror_view(solution);
    for (int i = 0; i < 8; ++i) {
        host_solution(i, 0) = i + 1;
        host_rhs(i, 0) = (i + 1) * (i + 1);
    }
    Kokkos::deep_copy(rhs, host_rhs);
    Kokkos::deep_copy(solution, host_solution);
    auto const warm = similie::solvers::minimize_strong_formulation_residual(
            Kokkos::DefaultExecutionSpace(),
            DiagonalOperator {},
            rhs,
            solution,
            settings(true));
    EXPECT_EQ(warm.iterations, 0U);
    EXPECT_DOUBLE_EQ(warm.initial_residual_l2, 0.0);
    EXPECT_TRUE(warm.converged);
    auto const cold = similie::solvers::minimize_strong_formulation_residual(
            Kokkos::DefaultExecutionSpace(),
            DiagonalOperator {},
            rhs,
            solution,
            settings(false));
    EXPECT_GT(cold.iterations, 0U);
    EXPECT_LE(cold.final_relative_residual, 1e-11);
}

TEST(SolverInitialGuess, WarmStartKeepsToleranceRelativeToRightHandSide)
{
    Kokkos::View<double**> rhs("rhs", 8, 1), solution("solution", 8, 1);
    auto host_rhs = Kokkos::create_mirror_view(rhs);
    auto host_solution = Kokkos::create_mirror_view(solution);
    for (int i = 0; i < 8; ++i) {
        host_solution(i, 0) = 1.0 + 1e-7 * (i + 1);
        host_rhs(i, 0) = i + 1;
    }
    Kokkos::deep_copy(rhs, host_rhs);
    Kokkos::deep_copy(solution, host_solution);
    auto const diagnostics = similie::solvers::minimize_strong_formulation_residual(
            Kokkos::DefaultExecutionSpace(),
            DiagonalOperator {},
            rhs,
            solution,
            settings(true));
    EXPECT_LT(diagnostics.initial_residual_l2, 1e-5 * diagnostics.rhs_norm_l2);
    double residual = 0;
    Kokkos::deep_copy(host_solution, solution);
    for (int i = 0; i < 8; ++i) {
        double const r = host_rhs(i, 0) - (i + 1) * host_solution(i, 0);
        residual += r * r;
    }
    double const relative = std::sqrt(residual) / diagnostics.rhs_norm_l2;
    EXPECT_LE(relative, 1e-11);
    EXPECT_NEAR(diagnostics.final_relative_residual, relative, 1e-14);
}

TEST(SolverInitialGuess, ZeroRightHandSideClearsPreviousSolution)
{
    Kokkos::View<double**> rhs("rhs", 8, 1), solution("solution", 8, 1);
    Kokkos::deep_copy(solution, 42.0);
    auto const diagnostics = similie::solvers::minimize_strong_formulation_residual(
            Kokkos::DefaultExecutionSpace(),
            DiagonalOperator {},
            rhs,
            solution,
            settings(true));
    auto const host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), solution);
    for (int i = 0; i < 8; ++i)
        EXPECT_DOUBLE_EQ(host(i, 0), 0.0);
    EXPECT_EQ(diagnostics.iterations, 0U);
    EXPECT_TRUE(diagnostics.converged);
}

TEST(SolverInitialGuess, SingleEigenmodeChecksFinalResidual)
{
    Kokkos::View<double**> rhs("rhs", 8, 1), solution("solution", 8, 1);
    auto host_rhs = Kokkos::create_mirror_view(rhs);
    host_rhs(3, 0) = 1;
    Kokkos::deep_copy(rhs, host_rhs);
    auto const diagnostics = similie::solvers::minimize_strong_formulation_residual(
            Kokkos::DefaultExecutionSpace(),
            DiagonalOperator {},
            rhs,
            solution,
            settings(true));
    auto const host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), solution);
    EXPECT_NEAR(host(3, 0), 0.25, 1e-12);
    EXPECT_LE(diagnostics.final_relative_residual, 1e-11);
}
