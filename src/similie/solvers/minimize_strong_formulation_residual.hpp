// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: MIT

#pragma once

#include <cmath>
#include <memory>
#include <stdexcept>

#include <ginkgo/extensions/kokkos.hpp>
#include <ginkgo/core/log/convergence.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/preconditioner/jacobi.hpp>
#include <ginkgo/core/solver/cg.hpp>
#include <ginkgo/core/stop/iteration.hpp>
#include <ginkgo/core/stop/residual_norm.hpp>

#include <Kokkos_Core.hpp>

#include <similie/physics/dedonder_weyl.hpp>
#include <similie/physics/magnetostatics/structured_linear_magnetostatics.hpp>

namespace similie::solvers {

struct StrongFormulationSolverSettings
{
    unsigned int max_iterations = 2000U;
    double relative_tolerance = 1.0e-12;
    unsigned int jacobi_max_block_size = 1U;
    bool use_matrix_free = true;
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

template <class KokkosViewType>
auto to_gko_dense(std::shared_ptr<gko::Executor const> const& gko_exec, KokkosViewType const& view)
{
    static_assert(Kokkos::is_view_v<KokkosViewType> && KokkosViewType::rank == 2);
    using value_type = typename KokkosViewType::traits::value_type;

    if (view.stride(1) != 1) {
        throw std::runtime_error("The Kokkos view passed to Ginkgo must be contiguous in the second dimension");
    }

    return gko::matrix::Dense<value_type>::create(
            gko_exec,
            gko::dim<2>(view.extent(0), view.extent(1)),
            gko::array<value_type>::view(gko_exec, view.span(), view.data()),
            view.stride(0));
}

template <class MemorySpace>
gko::matrix_data<double, gko::int32> assemble_matrix_data(
        physics::magnetostatics::StructuredScalarPoissonStrongFormOperator2D<MemorySpace> const& operator_model)
{
    auto const x_coords_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), operator_model.x_coords());
    auto const y_coords_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), operator_model.y_coords());

    std::size_t const nx = operator_model.nx();
    std::size_t const ny = operator_model.ny();

    gko::matrix_data<double, gko::int32> matrix_data(gko::dim<2>(operator_model.size(), operator_model.size()));
    matrix_data.nonzeros.reserve(5 * operator_model.size());

    auto flat_index = [nx](std::size_t i, std::size_t j) { return static_cast<gko::int32>(i + nx * j); };

    for (std::size_t j = 0; j < ny; ++j) {
        for (std::size_t i = 0; i < nx; ++i) {
            gko::int32 const row = flat_index(i, j);
            bool const boundary = (i == 0 || j == 0 || i + 1 == nx || j + 1 == ny);
            if (boundary) {
                matrix_data.nonzeros.emplace_back(row, row, 1.0);
                continue;
            }

            double const dxm = x_coords_host(i) - x_coords_host(i - 1);
            double const dxp = x_coords_host(i + 1) - x_coords_host(i);
            double const dym = y_coords_host(j) - y_coords_host(j - 1);
            double const dyp = y_coords_host(j + 1) - y_coords_host(j);

            double const coeff_im1 = -2.0 / (dxm * (dxm + dxp));
            double const coeff_ip1 = -2.0 / (dxp * (dxm + dxp));
            double const coeff_jm1 = -2.0 / (dym * (dym + dyp));
            double const coeff_jp1 = -2.0 / (dyp * (dym + dyp));
            double const coeff_center = -coeff_im1 - coeff_ip1 - coeff_jm1 - coeff_jp1;

            matrix_data.nonzeros.emplace_back(row, flat_index(i - 1, j), coeff_im1);
            matrix_data.nonzeros.emplace_back(row, flat_index(i + 1, j), coeff_ip1);
            matrix_data.nonzeros.emplace_back(row, flat_index(i, j - 1), coeff_jm1);
            matrix_data.nonzeros.emplace_back(row, flat_index(i, j + 1), coeff_jp1);
            matrix_data.nonzeros.emplace_back(row, row, coeff_center);
        }
    }

    matrix_data.sort_row_major();
    return matrix_data;
}

template <class MemorySpace>
std::shared_ptr<gko::matrix::Csr<double, gko::int32>> build_matrix(
        std::shared_ptr<gko::Executor const> const& gko_exec,
        physics::magnetostatics::StructuredScalarPoissonStrongFormOperator2D<MemorySpace> const& operator_model)
{
    using matrix_type = gko::matrix::Csr<double, gko::int32>;
    auto matrix = std::shared_ptr<matrix_type>(
            matrix_type::create(gko_exec, gko::dim<2>(operator_model.size(), operator_model.size())).release());
    auto matrix_data = assemble_matrix_data(operator_model);
    matrix->read(matrix_data);
    return matrix;
}

inline auto build_jacobi_preconditioner_factory(
        std::shared_ptr<gko::Executor const> const& gko_exec,
        StrongFormulationSolverSettings const& settings)
{
    return gko::preconditioner::Jacobi<double>::build()
            .with_max_block_size(settings.jacobi_max_block_size)
            .on(gko_exec);
}

inline std::shared_ptr<gko::LinOp> build_jacobi_preconditioner(
        std::shared_ptr<gko::Executor const> const& gko_exec,
        std::shared_ptr<gko::LinOp const> const& matrix,
        StrongFormulationSolverSettings const& settings)
{
    auto preconditioner_factory = build_jacobi_preconditioner_factory(gko_exec, settings);
    return std::shared_ptr<gko::LinOp>(preconditioner_factory->generate(matrix).release());
}

template <class ExecSpace, class ViewType>
double residual_norm_l2(ExecSpace exec_space, ViewType residual)
{
    return std::sqrt(dot(exec_space, residual, residual));
}

} // namespace detail

template <class ExecSpace, class MemorySpace, class RHSViewType, class SolutionViewType>
StrongFormulationSolverDiagnostics minimize_strong_formulation_residual(
        ExecSpace exec_space,
        physics::dedonder_weyl::StationaryStrongFormulation<
                physics::magnetostatics::StructuredScalarPoissonStrongFormOperator2D<MemorySpace>> const&
                formulation,
        RHSViewType rhs,
        SolutionViewType solution,
        StrongFormulationSolverSettings settings = {})
{
    using operator_model_type
            = physics::magnetostatics::StructuredScalarPoissonStrongFormOperator2D<MemorySpace>;
    StrongFormulationSolverDiagnostics diagnostics;

    detail::fill(exec_space, solution, 0.0);
    diagnostics.initial_residual_l2 = detail::residual_norm_l2(exec_space, rhs);
    diagnostics.final_residual_l2 = diagnostics.initial_residual_l2;
    diagnostics.final_relative_residual = diagnostics.initial_residual_l2 == 0.0 ? 0.0 : 1.0;
    if (diagnostics.initial_residual_l2 == 0.0) {
        return diagnostics;
    }

    operator_model_type const& operator_model = formulation.strong_form_operator;
    auto const gko_exec = gko::ext::kokkos::create_executor(exec_space);
    auto matrix = detail::build_matrix(gko_exec, operator_model);
    auto preconditioner = detail::build_jacobi_preconditioner(
            gko_exec,
            std::static_pointer_cast<gko::LinOp const>(matrix),
            settings);

    if (!settings.use_matrix_free) {
        using solver_type = gko::solver::Cg<double>;
        auto residual_criterion = gko::stop::ResidualNorm<double>::build()
                                          .with_reduction_factor(settings.relative_tolerance)
                                          .on(gko_exec);
        auto iterations_criterion
                = gko::stop::Iteration::build().with_max_iters(settings.max_iterations).on(gko_exec);
        auto preconditioner_factory = detail::build_jacobi_preconditioner_factory(gko_exec, settings);
        auto solver_factory = solver_type::build()
                                      .with_preconditioner(std::move(preconditioner_factory))
                                      .with_criteria(std::move(residual_criterion), std::move(iterations_criterion))
                                      .on(gko_exec);
        auto solver = solver_factory->generate(std::static_pointer_cast<gko::LinOp const>(matrix));
        auto convergence_logger = std::shared_ptr<gko::log::Convergence<double>>(
                gko::log::Convergence<double>::create().release());
        solver->add_logger(convergence_logger);

        auto rhs_gko = detail::to_gko_dense(gko_exec, rhs);
        auto solution_gko = detail::to_gko_dense(gko_exec, solution);
        solver->apply(rhs_gko, solution_gko);
        gko_exec->synchronize();
        solver->remove_logger(convergence_logger);

        diagnostics.iterations = static_cast<unsigned int>(convergence_logger->get_num_iterations());
        diagnostics.converged = convergence_logger->has_converged();
        auto residual_norm_dense
                = dynamic_cast<gko::matrix::Dense<double> const*>(convergence_logger->get_residual_norm());
        if (residual_norm_dense != nullptr) {
            auto host_dense = gko::matrix::Dense<double>::create(
                    gko_exec->get_master(),
                    residual_norm_dense->get_size());
            residual_norm_dense->convert_to(host_dense.get());
            diagnostics.final_residual_l2 = host_dense->at(0, 0);
        }
        diagnostics.final_relative_residual = diagnostics.initial_residual_l2 == 0.0
                                                      ? 0.0
                                                      : diagnostics.final_residual_l2
                                                                / diagnostics.initial_residual_l2;
        return diagnostics;
    }

    using memory_space = typename SolutionViewType::memory_space;
    Kokkos::View<double**, memory_space> residual("similie_residual", rhs.extent(0), rhs.extent(1));
    Kokkos::View<double**, memory_space> preconditioned_residual(
            "similie_preconditioned_residual",
            rhs.extent(0),
            rhs.extent(1));
    Kokkos::View<double**, memory_space> search_direction("similie_search_direction", rhs.extent(0), rhs.extent(1));
    Kokkos::View<double**, memory_space> operator_search_direction(
            "similie_operator_search_direction",
            rhs.extent(0),
            rhs.extent(1));

    detail::copy(exec_space, residual, rhs);
    {
        auto residual_gko = detail::to_gko_dense(gko_exec, residual);
        auto preconditioned_residual_gko = detail::to_gko_dense(gko_exec, preconditioned_residual);
        preconditioner->apply(residual_gko, preconditioned_residual_gko);
        gko_exec->synchronize();
    }
    detail::copy(exec_space, search_direction, preconditioned_residual);

    double residual_norm_sq = detail::dot(exec_space, residual, residual);
    double rz_dot = detail::dot(exec_space, residual, preconditioned_residual);
    diagnostics.converged = false;
    double const initial_norm_sq = residual_norm_sq;

    for (unsigned int iteration = 0; iteration < settings.max_iterations; ++iteration) {
        diagnostics.iterations = iteration + 1;
        detail::apply_operator(
                exec_space,
                operator_model,
                search_direction,
                operator_search_direction);

        double const denominator = detail::dot(exec_space, search_direction, operator_search_direction);
        if (std::abs(denominator) < 1.0e-30) {
            break;
        }

        double const alpha = rz_dot / denominator;
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

        {
            auto residual_gko = detail::to_gko_dense(gko_exec, residual);
            auto preconditioned_residual_gko = detail::to_gko_dense(gko_exec, preconditioned_residual);
            preconditioner->apply(residual_gko, preconditioned_residual_gko);
            gko_exec->synchronize();
        }
        double const new_rz_dot = detail::dot(exec_space, residual, preconditioned_residual);
        if (std::abs(rz_dot) < 1.0e-30) {
            break;
        }
        double const beta = new_rz_dot / rz_dot;
        detail::update_axpby(exec_space, search_direction, 1.0, preconditioned_residual, beta, search_direction);
        residual_norm_sq = new_residual_norm_sq;
        rz_dot = new_rz_dot;
    }

    return diagnostics;
}

} // namespace similie::solvers
