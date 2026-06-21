// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: MIT

#pragma once

#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <similie/solvers/minimize_strong_formulation_residual.hpp>

namespace similie::onelab_interface::minimize_strong_formulation_residual_onelab {

inline double criterion_to_control_value(solvers::Criterion criterion)
{
    switch (criterion) {
    case solvers::Criterion::PotentialTemporalDerivative:
        return 0.0;
    case solvers::Criterion::MomentsTemporalDerivative:
        return 1.0;
    case solvers::Criterion::PotentialAndMomentsTemporalDerivative:
        return 2.0;
    }
    return 1.0;
}

inline solvers::Criterion criterion_from_control_value(double value)
{
    int const rounded = static_cast<int>(value + 0.5);
    switch (rounded) {
    case 0:
        return solvers::Criterion::PotentialTemporalDerivative;
    case 1:
        return solvers::Criterion::MomentsTemporalDerivative;
    case 2:
        return solvers::Criterion::PotentialAndMomentsTemporalDerivative;
    default:
        throw std::runtime_error("invalid strong-formulation solver criterion control value");
    }
}

template <class SolverSettings, class ProblemParameterName, class PublishNumber>
void synchronize_controls(
        SolverSettings const& solver_settings,
        ProblemParameterName&& problem_parameter_name,
        PublishNumber&& publish_or_sync_number)
{
    publish_or_sync_number(
            problem_parameter_name("1Solver", "0Max iterations"),
            "Max iterations",
            "Maximum number of iterations for the stationary strong-formulation solver.",
            static_cast<double>(solver_settings.max_iterations),
            1.0,
            1.e9,
            1.0);
    publish_or_sync_number(
            problem_parameter_name("1Solver", "1Relative tolerance"),
            "Relative tolerance",
            "Relative convergence tolerance for the stationary strong-formulation solver.",
            solver_settings.relative_tolerance,
            0.0,
            1.0,
            1.e-12);
    publish_or_sync_number(
            problem_parameter_name("1Solver", "2Jacobi max block size"),
            "Jacobi max block size",
            "Maximum Jacobi block size used by the auxiliary preconditioner.",
            static_cast<double>(solver_settings.jacobi_max_block_size),
            1.0,
            1.e9,
            1.0);
    publish_or_sync_number(
            problem_parameter_name("1Solver", "3Use matrix-free"),
            "Use matrix-free",
            "When enabled, the strong-form operator is applied matrix-free and only the Jacobi "
            "preconditioner uses an auxiliary assembled matrix.",
            solver_settings.use_matrix_free ? 1.0 : 0.0,
            0.0,
            1.0,
            1.0,
            std::vector<double> {0.0, 1.0},
            std::map<double, std::string> {{0.0, "No"}, {1.0, "Yes"}});
    publish_or_sync_number(
            problem_parameter_name("1Solver", "4Criterion"),
            "Criterion",
            "Stationary strong-formulation criterion used to define the residual equation set.",
            criterion_to_control_value(solver_settings.criterion),
            0.0,
            2.0,
            1.0,
            std::vector<double> {0.0, 1.0, 2.0},
            std::map<double, std::string> {
                    {0.0, "PotentialTemporalDerivative"},
                    {1.0, "MomentsTemporalDerivative"},
                    {2.0, "PotentialAndMomentsTemporalDerivative"},
            });
}

template <class SolverSettings, class ProblemParameterName, class GetFirstNumberValue>
SolverSettings apply_control_overrides(
        SolverSettings solver_settings,
        ProblemParameterName&& problem_parameter_name,
        GetFirstNumberValue&& get_first_number_value)
{
    solver_settings.max_iterations = static_cast<unsigned int>(get_first_number_value(
            problem_parameter_name("1Solver", "0Max iterations"),
            static_cast<double>(solver_settings.max_iterations)));
    solver_settings.relative_tolerance = get_first_number_value(
            problem_parameter_name("1Solver", "1Relative tolerance"),
            solver_settings.relative_tolerance);
    solver_settings.jacobi_max_block_size = static_cast<unsigned int>(get_first_number_value(
            problem_parameter_name("1Solver", "2Jacobi max block size"),
            static_cast<double>(solver_settings.jacobi_max_block_size)));
    solver_settings.use_matrix_free
            = (get_first_number_value(
                       problem_parameter_name("1Solver", "3Use matrix-free"),
                       solver_settings.use_matrix_free ? 1.0 : 0.0)
               != 0.0);
    solver_settings.criterion = criterion_from_control_value(get_first_number_value(
            problem_parameter_name("1Solver", "4Criterion"),
            criterion_to_control_value(solver_settings.criterion)));
    return solver_settings;
}

} // namespace similie::onelab_interface::minimize_strong_formulation_residual_onelab
