// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: MIT

#pragma once

#include <map>
#include <optional>
#include <string>
#include <vector>

namespace similie::onelab_interface::minimize_strong_formulation_residual_onelab {

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
    return solver_settings;
}

} // namespace similie::onelab_interface::minimize_strong_formulation_residual_onelab
