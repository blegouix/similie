// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: MIT

#pragma once

#include <cmath>

#include <Kokkos_Core.hpp>

#include <similie/physics/dedonder_weyl.hpp>

namespace similie::solvers {

struct StrongFormulationSolverSettings
{
    unsigned int max_iterations = 2000U;
    double relative_tolerance = 1.0e-12;
};

struct StrongFormulationSolverDiagnostics
{
    unsigned int iterations = 0U;
    double initial_residual_l2 = 0.0;
    double final_residual_l2 = 0.0;
    double final_relative_residual = 0.0;
    bool converged = true;
};

namespace detail {

template <class ExecSpace, class OperatorModel, class InputView, class OutputView>
void apply_operator(ExecSpace exec_space, OperatorModel const& operator_model, InputView input, OutputView output)
{
    Kokkos::parallel_for(
            "similie_apply_operator",
            Kokkos::RangePolicy<ExecSpace>(exec_space, 0, operator_model.size()),
            KOKKOS_LAMBDA(std::size_t row) { operator_model.apply_at(output, input, row); });
    exec_space.fence();
}

template <class ExecSpace, class ViewType1, class ViewType2>
double dot(ExecSpace exec_space, ViewType1 lhs, ViewType2 rhs)
{
    double result = 0.0;
    Kokkos::parallel_reduce(
            "similie_dot",
            Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>(
                    exec_space,
                    {0, 0},
                    {lhs.extent(0), lhs.extent(1)}),
            KOKKOS_LAMBDA(std::size_t row, std::size_t column, double& local_sum) {
                local_sum += lhs(row, column) * rhs(row, column);
            },
            result);
    exec_space.fence();
    return result;
}

template <class ExecSpace, class ViewType>
void fill(ExecSpace exec_space, ViewType view, double value)
{
    Kokkos::parallel_for(
            "similie_fill",
            Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>(
                    exec_space,
                    {0, 0},
                    {view.extent(0), view.extent(1)}),
            KOKKOS_LAMBDA(std::size_t row, std::size_t column) { view(row, column) = value; });
    exec_space.fence();
}

template <class ExecSpace, class DestinationView, class SourceView>
void copy(ExecSpace exec_space, DestinationView destination, SourceView source)
{
    Kokkos::parallel_for(
            "similie_copy",
            Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>(
                    exec_space,
                    {0, 0},
                    {destination.extent(0), destination.extent(1)}),
            KOKKOS_LAMBDA(std::size_t row, std::size_t column) {
                destination(row, column) = source(row, column);
            });
    exec_space.fence();
}

template <class ExecSpace, class ViewType1, class ViewType2, class ViewType3>
void update_axpby(ExecSpace exec_space, ViewType1 destination, double alpha, ViewType2 x, double beta, ViewType3 y)
{
    Kokkos::parallel_for(
            "similie_axpby",
            Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>(
                    exec_space,
                    {0, 0},
                    {destination.extent(0), destination.extent(1)}),
            KOKKOS_LAMBDA(std::size_t row, std::size_t column) {
                destination(row, column) = alpha * x(row, column) + beta * y(row, column);
            });
    exec_space.fence();
}

template <class ExecSpace, class ViewType1, class ViewType2>
void axpy_inplace(ExecSpace exec_space, ViewType1 destination, double alpha, ViewType2 source)
{
    Kokkos::parallel_for(
            "similie_axpy_inplace",
            Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>(
                    exec_space,
                    {0, 0},
                    {destination.extent(0), destination.extent(1)}),
            KOKKOS_LAMBDA(std::size_t row, std::size_t column) {
                destination(row, column) += alpha * source(row, column);
            });
    exec_space.fence();
}

} // namespace detail

template <class ExecSpace, class OperatorModel, class RHSViewType, class SolutionViewType>
StrongFormulationSolverDiagnostics minimize_strong_formulation_residual(
        ExecSpace exec_space,
        physics::dedonder_weyl::StationaryStrongFormulation<OperatorModel> const& formulation,
        RHSViewType rhs,
        SolutionViewType solution,
        StrongFormulationSolverSettings settings = {})
{
    using memory_space = typename SolutionViewType::memory_space;
    StrongFormulationSolverDiagnostics diagnostics;

    Kokkos::View<double**, memory_space> residual("similie_residual", rhs.extent(0), rhs.extent(1));
    Kokkos::View<double**, memory_space> search_direction("similie_search_direction", rhs.extent(0), rhs.extent(1));
    Kokkos::View<double**, memory_space> operator_search_direction(
            "similie_operator_search_direction",
            rhs.extent(0),
            rhs.extent(1));

    detail::fill(exec_space, solution, 0.0);
    detail::copy(exec_space, residual, rhs);
    detail::copy(exec_space, search_direction, residual);

    double const initial_norm_sq = detail::dot(exec_space, residual, residual);
    diagnostics.initial_residual_l2 = std::sqrt(initial_norm_sq);
    diagnostics.final_residual_l2 = diagnostics.initial_residual_l2;
    diagnostics.final_relative_residual = diagnostics.initial_residual_l2 == 0.0 ? 0.0 : 1.0;
    if (initial_norm_sq == 0.0) {
        return diagnostics;
    }

    double residual_norm_sq = initial_norm_sq;
    diagnostics.converged = false;
    for (unsigned int iteration = 0; iteration < settings.max_iterations; ++iteration) {
        diagnostics.iterations = iteration + 1;
        detail::apply_operator(
                exec_space,
                formulation.strong_form_operator,
                search_direction,
                operator_search_direction);

        double const denominator = detail::dot(exec_space, search_direction, operator_search_direction);
        if (std::abs(denominator) < 1.0e-30) {
            break;
        }

        double const alpha = residual_norm_sq / denominator;
        detail::axpy_inplace(exec_space, solution, alpha, search_direction);
        detail::axpy_inplace(exec_space, residual, -alpha, operator_search_direction);

        double const new_residual_norm_sq = detail::dot(exec_space, residual, residual);
        diagnostics.final_residual_l2 = std::sqrt(new_residual_norm_sq);
        diagnostics.final_relative_residual = diagnostics.initial_residual_l2 == 0.0
                                                     ? 0.0
                                                     : diagnostics.final_residual_l2
                                                               / diagnostics.initial_residual_l2;
        if (new_residual_norm_sq <= settings.relative_tolerance * settings.relative_tolerance * initial_norm_sq) {
            diagnostics.converged = true;
            break;
        }

        double const beta = new_residual_norm_sq / residual_norm_sq;
        detail::update_axpby(exec_space, search_direction, 1.0, residual, beta, search_direction);
        residual_norm_sq = new_residual_norm_sq;
    }

    return diagnostics;
}

} // namespace similie::solvers
