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

inline double preconditioner_to_control_value(solvers::PreconditionerType preconditioner)
{
    switch (preconditioner) {
    case solvers::PreconditionerType::Identity:
        return 0.0;
    case solvers::PreconditionerType::Jacobi:
        return 1.0;
    case solvers::PreconditionerType::SpdIsai:
        return 2.0;
    case solvers::PreconditionerType::SymmetricGaussSeidel:
        return 3.0;
    case solvers::PreconditionerType::Ssor:
        return 4.0;
    case solvers::PreconditionerType::ChebyshevJacobi:
        return 5.0;
    case solvers::PreconditionerType::IrJacobi:
        return 6.0;
    case solvers::PreconditionerType::GeneralIsai:
        return 7.0;
    }
    return 1.0;
}

inline solvers::PreconditionerType preconditioner_from_control_value(double value)
{
    int const rounded = static_cast<int>(value + 0.5);
    switch (rounded) {
    case 0:
        return solvers::PreconditionerType::Identity;
    case 1:
        return solvers::PreconditionerType::Jacobi;
    case 2:
        return solvers::PreconditionerType::SpdIsai;
    case 3:
        return solvers::PreconditionerType::SymmetricGaussSeidel;
    case 4:
        return solvers::PreconditionerType::Ssor;
    case 5:
        return solvers::PreconditionerType::ChebyshevJacobi;
    case 6:
        return solvers::PreconditionerType::IrJacobi;
    case 7:
        return solvers::PreconditionerType::GeneralIsai;
    default:
        throw std::runtime_error("invalid strong-formulation preconditioner control value");
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
    publish_or_sync_number(
            problem_parameter_name("1Solver", "5Preconditioner"),
            "Preconditioner",
            "Ginkgo preconditioner used by the stationary strong-formulation solver.",
            preconditioner_to_control_value(solver_settings.preconditioner),
            0.0,
            7.0,
            1.0,
            std::vector<double> {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0},
            std::map<double, std::string> {
                    {0.0, "Identity"},
                    {1.0, "Jacobi"},
                    {2.0, "SpdIsai"},
                    {3.0, "SymmetricGaussSeidel"},
                    {4.0, "Ssor"},
                    {5.0, "ChebyshevJacobi"},
                    {6.0, "IrJacobi"},
                    {7.0, "GeneralIsai"},
            });
    publish_or_sync_number(
            problem_parameter_name("1Solver", "6SOR relaxation factor"),
            "SOR relaxation factor",
            "Relaxation factor used by the SSOR preconditioner.",
            solver_settings.sor_relaxation_factor,
            1.e-6,
            1.999999,
            0.1);
    publish_or_sync_number(
            problem_parameter_name("1Solver", "7Vector potential gauge penalty"),
            "Vector potential gauge penalty",
            "Gauge penalty used by the 3D vector-potential solver.",
            solver_settings.vector_potential_gauge_penalty,
            0.0,
            1.e12,
            1.0);
    publish_or_sync_number(
            problem_parameter_name("1Solver", "8Chebyshev iterations"),
            "Chebyshev iterations",
            "Maximum inner Chebyshev iterations used by the Chebyshev-Jacobi preconditioner.",
            static_cast<double>(solver_settings.chebyshev_iterations),
            1.0,
            1.e6,
            1.0);
    publish_or_sync_number(
            problem_parameter_name("1Solver", "9Chebyshev lower bound"),
            "Chebyshev lower bound",
            "Lower spectral focus used by the Chebyshev-Jacobi preconditioner.",
            solver_settings.chebyshev_lower_bound,
            1.e-12,
            1.e12,
            0.1);
    publish_or_sync_number(
            problem_parameter_name("1Solver", "10Chebyshev upper bound"),
            "Chebyshev upper bound",
            "Upper spectral focus used by the Chebyshev-Jacobi preconditioner.",
            solver_settings.chebyshev_upper_bound,
            1.e-12,
            1.e12,
            0.1);
    publish_or_sync_number(
            problem_parameter_name("1Solver", "11IR iterations"),
            "IR iterations",
            "Maximum inner iterations used by the IR-Jacobi preconditioner.",
            static_cast<double>(solver_settings.ir_iterations),
            1.0,
            1.e6,
            1.0);
    publish_or_sync_number(
            problem_parameter_name("1Solver", "12IR relaxation factor"),
            "IR relaxation factor",
            "Relaxation factor used by the IR-Jacobi preconditioner.",
            solver_settings.ir_relaxation_factor,
            1.e-12,
            2.0,
            0.1);
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
    solver_settings.preconditioner = preconditioner_from_control_value(get_first_number_value(
            problem_parameter_name("1Solver", "5Preconditioner"),
            preconditioner_to_control_value(solver_settings.preconditioner)));
    solver_settings.sor_relaxation_factor = get_first_number_value(
            problem_parameter_name("1Solver", "6SOR relaxation factor"),
            solver_settings.sor_relaxation_factor);
    solver_settings.vector_potential_gauge_penalty = get_first_number_value(
            problem_parameter_name("1Solver", "7Vector potential gauge penalty"),
            solver_settings.vector_potential_gauge_penalty);
    solver_settings.chebyshev_iterations = static_cast<unsigned int>(get_first_number_value(
            problem_parameter_name("1Solver", "8Chebyshev iterations"),
            static_cast<double>(solver_settings.chebyshev_iterations)));
    solver_settings.chebyshev_lower_bound = get_first_number_value(
            problem_parameter_name("1Solver", "9Chebyshev lower bound"),
            solver_settings.chebyshev_lower_bound);
    solver_settings.chebyshev_upper_bound = get_first_number_value(
            problem_parameter_name("1Solver", "10Chebyshev upper bound"),
            solver_settings.chebyshev_upper_bound);
    solver_settings.ir_iterations = static_cast<unsigned int>(get_first_number_value(
            problem_parameter_name("1Solver", "11IR iterations"),
            static_cast<double>(solver_settings.ir_iterations)));
    solver_settings.ir_relaxation_factor = get_first_number_value(
            problem_parameter_name("1Solver", "12IR relaxation factor"),
            solver_settings.ir_relaxation_factor);
    return solver_settings;
}

} // namespace similie::onelab_interface::minimize_strong_formulation_residual_onelab
