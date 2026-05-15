// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: MIT

#pragma once

#include <cmath>
#include <memory>
#include <stdexcept>
#include <type_traits>

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

    return matrix_data;
}

template <class ValueType>
double dense_scalar_to_double(
        std::shared_ptr<gko::Executor const> const& exec,
        gko::matrix::Dense<ValueType> const* dense)
{
    if (dense == nullptr) {
        return 0.0;
    }
    auto host_dense = gko::matrix::Dense<ValueType>::create(exec->get_master(), dense->get_size());
    dense->convert_to(host_dense.get());
    return static_cast<double>(host_dense->at(0, 0));
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
    using matrix_type = gko::matrix::Csr<double, gko::int32>;
    using solver_type = gko::solver::Cg<double>;

    StrongFormulationSolverDiagnostics diagnostics;
    auto const gko_exec = gko::ext::kokkos::create_executor(exec_space);

    double initial_norm_sq = 0.0;
    Kokkos::parallel_reduce(
            "similie_initial_residual_norm_sq",
            Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>(exec_space, {0, 0}, {rhs.extent(0), rhs.extent(1)}),
            KOKKOS_LAMBDA(std::size_t row, std::size_t col, double& local_sum) {
                local_sum += rhs(row, col) * rhs(row, col);
            },
            initial_norm_sq);
    exec_space.fence();
    diagnostics.initial_residual_l2 = std::sqrt(initial_norm_sq);
    diagnostics.final_residual_l2 = diagnostics.initial_residual_l2;
    diagnostics.final_relative_residual = diagnostics.initial_residual_l2 == 0.0 ? 0.0 : 1.0;
    if (initial_norm_sq == 0.0) {
        Kokkos::deep_copy(solution, 0.0);
        return diagnostics;
    }

    auto matrix = std::shared_ptr<matrix_type>(
            matrix_type::create(
                    gko_exec,
                    gko::dim<2>(
                            formulation.strong_form_operator.size(),
                            formulation.strong_form_operator.size()))
                    .release());
    auto matrix_data = detail::assemble_matrix_data(formulation.strong_form_operator);
    matrix_data.sort_row_major();
    matrix->read(matrix_data);

    auto residual_criterion = gko::stop::ResidualNorm<double>::build()
                                      .with_reduction_factor(settings.relative_tolerance)
                                      .on(gko_exec);
    auto iterations_criterion
            = gko::stop::Iteration::build().with_max_iters(settings.max_iterations).on(gko_exec);

    auto preconditioner_factory = gko::preconditioner::Jacobi<double>::build()
                                          .with_max_block_size(settings.jacobi_max_block_size)
                                          .on(gko_exec);

    auto solver_factory = solver_type::build()
                                  .with_preconditioner(std::move(preconditioner_factory))
                                  .with_criteria(std::move(residual_criterion), std::move(iterations_criterion))
                                  .on(gko_exec);
    auto solver = solver_factory->generate(std::static_pointer_cast<gko::LinOp const>(matrix));
    auto convergence_logger = std::shared_ptr<gko::log::Convergence<double>>(
            gko::log::Convergence<double>::create().release());
    solver->add_logger(convergence_logger);

    Kokkos::deep_copy(solution, 0.0);
    auto rhs_gko = detail::to_gko_dense(gko_exec, rhs);
    auto solution_gko = detail::to_gko_dense(gko_exec, solution);
    solver->apply(rhs_gko, solution_gko);
    gko_exec->synchronize();
    solver->remove_logger(convergence_logger);

    diagnostics.iterations = static_cast<unsigned int>(convergence_logger->get_num_iterations());
    diagnostics.converged = convergence_logger->has_converged();
    diagnostics.final_residual_l2 = detail::dense_scalar_to_double(
            gko_exec,
            dynamic_cast<gko::matrix::Dense<double> const*>(convergence_logger->get_residual_norm()));
    diagnostics.final_relative_residual = diagnostics.initial_residual_l2 == 0.0
                                                  ? 0.0
                                                  : diagnostics.final_residual_l2 / diagnostics.initial_residual_l2;

    return diagnostics;
}

} // namespace similie::solvers
