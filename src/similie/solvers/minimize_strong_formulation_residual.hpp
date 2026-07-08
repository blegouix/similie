// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: MIT

#pragma once

#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <type_traits>

#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/log/convergence.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/preconditioner/jacobi.hpp>
#include <ginkgo/core/solver/cg.hpp>
#include <ginkgo/core/stop/iteration.hpp>
#include <ginkgo/core/stop/residual_norm.hpp>
#include <ginkgo/extensions/kokkos.hpp>

#include <Kokkos_Core.hpp>

namespace similie::solvers {

enum class Criterion {
    PotentialTemporalDerivative,
    MomentsTemporalDerivative,
    PotentialAndMomentsTemporalDerivative,
};

struct StrongFormulationSolverSettings
{
    unsigned int max_iterations = 2000U;
    double relative_tolerance = 1.0e-12;
    unsigned int jacobi_max_block_size = 1U;
    bool use_matrix_free = true;
    Criterion criterion = Criterion::MomentsTemporalDerivative;
};

struct StrongFormulationSolverDiagnostics
{
    unsigned int iterations = 0U;
    double initial_residual_l2 = 0.0;
    double final_residual_l2 = 0.0;
    double final_relative_residual = 0.0;
    double duration = 0.0;
    bool converged = true;
};

namespace detail {

template <class ExecSpace, class ViewType>
double residual_norm_l2(ExecSpace exec_space, ViewType residual);

template <class ExecSpace, class OperatorModel, class InputView, class OutputView>
void apply_operator(
        ExecSpace exec_space,
        OperatorModel const& operator_model,
        InputView input,
        OutputView output)
{
    if constexpr (requires(OperatorModel const& model, ExecSpace ex, InputView in, OutputView out) {
                      model.apply(ex, in, out);
                  }) {
        operator_model.apply(exec_space, input, output);
    } else {
        Kokkos::parallel_for(
                "similie_apply_operator",
                Kokkos::RangePolicy<ExecSpace>(exec_space, 0, operator_model.size()),
                KOKKOS_LAMBDA(std::size_t row) { operator_model.apply_at(output, input, row); });
        exec_space.fence();
    }
}

template <class ExecSpace, class ViewType1, class ViewType2>
double dot(ExecSpace exec_space, ViewType1 lhs, ViewType2 rhs)
{
    double result = 0.0;
    Kokkos::parallel_reduce(
            "similie_dot",
            Kokkos::MDRangePolicy<
                    ExecSpace,
                    Kokkos::Rank<2>>(exec_space, {0, 0}, {lhs.extent(0), lhs.extent(1)}),
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
            Kokkos::MDRangePolicy<
                    ExecSpace,
                    Kokkos::Rank<2>>(exec_space, {0, 0}, {view.extent(0), view.extent(1)}),
            KOKKOS_LAMBDA(std::size_t row, std::size_t column) { view(row, column) = value; });
    exec_space.fence();
}

template <class ExecSpace, class DestinationView, class SourceView>
void copy(ExecSpace exec_space, DestinationView destination, SourceView source)
{
    Kokkos::parallel_for(
            "similie_copy",
            Kokkos::MDRangePolicy<
                    ExecSpace,
                    Kokkos::Rank<
                            2>>(exec_space, {0, 0}, {destination.extent(0), destination.extent(1)}),
            KOKKOS_LAMBDA(std::size_t row, std::size_t column) {
                destination(row, column) = source(row, column);
            });
    exec_space.fence();
}

template <class ExecSpace, class ViewType1, class ViewType2, class ViewType3>
void update_axpby(
        ExecSpace exec_space,
        ViewType1 destination,
        double alpha,
        ViewType2 x,
        double beta,
        ViewType3 y)
{
    Kokkos::parallel_for(
            "similie_axpby",
            Kokkos::MDRangePolicy<
                    ExecSpace,
                    Kokkos::Rank<
                            2>>(exec_space, {0, 0}, {destination.extent(0), destination.extent(1)}),
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
            Kokkos::MDRangePolicy<
                    ExecSpace,
                    Kokkos::Rank<
                            2>>(exec_space, {0, 0}, {destination.extent(0), destination.extent(1)}),
            KOKKOS_LAMBDA(std::size_t row, std::size_t column) {
                destination(row, column) += alpha * source(row, column);
            });
    exec_space.fence();
}

template <class KokkosViewType>
struct GkoDenseHandle
{
    std::shared_ptr<gko::matrix::Dense<typename KokkosViewType::non_const_value_type>> dense;
    std::optional<Kokkos::View<
            typename KokkosViewType::non_const_value_type**,
            Kokkos::LayoutRight,
            typename KokkosViewType::memory_space>>
            owned_view;
};

template <class KokkosViewType>
auto to_gko_dense(std::shared_ptr<gko::Executor const> const& gko_exec, KokkosViewType const& view)
{
    static_assert(Kokkos::is_view_v<KokkosViewType> && KokkosViewType::rank == 2);
    using value_type = typename KokkosViewType::traits::value_type;
    using non_const_value_type = typename KokkosViewType::non_const_value_type;
    using owning_view_type = Kokkos::View<
            non_const_value_type**,
            Kokkos::LayoutRight,
            typename KokkosViewType::memory_space>;

    GkoDenseHandle<KokkosViewType> handle;
    if (view.stride(1) == 1) {
        handle.dense = gko::matrix::Dense<value_type>::
                create(gko_exec,
                       gko::dim<2>(view.extent(0), view.extent(1)),
                       gko::array<value_type>::view(gko_exec, view.span(), view.data()),
                       view.stride(0));
        return handle;
    }

    handle.owned_view.emplace("similie_gko_dense_bridge", view.extent(0), view.extent(1));
    Kokkos::deep_copy(*handle.owned_view, view);
    handle.dense = gko::matrix::Dense<value_type>::
            create(gko_exec,
                   gko::dim<2>(handle.owned_view->extent(0), handle.owned_view->extent(1)),
                   gko::array<value_type>::
                           view(gko_exec, handle.owned_view->span(), handle.owned_view->data()),
                   handle.owned_view->stride(0));
    return handle;
}

template <class DestinationView, class KokkosViewType>
void copy_back_from_gko_dense_bridge(
        DestinationView destination,
        GkoDenseHandle<KokkosViewType> const& handle)
{
    if (handle.owned_view.has_value()) {
        Kokkos::deep_copy(destination, *handle.owned_view);
    }
}

template <class OperatorModel>
std::shared_ptr<gko::matrix::Csr<double, gko::int32>> build_matrix(
        std::shared_ptr<gko::Executor const> const& gko_exec,
        OperatorModel const& operator_model)
{
    using matrix_type = gko::matrix::Csr<double, gko::int32>;
    auto matrix = std::shared_ptr<matrix_type>(
            matrix_type::create(gko_exec, gko::dim<2>(operator_model.size(), operator_model.size()))
                    .release());
    auto matrix_data = assemble_matrix_data(operator_model);
    matrix->read(matrix_data);
    return matrix;
}

template <class OperatorModel, class StateView>
std::shared_ptr<gko::matrix::Csr<double, gko::int32>> build_matrix(
        std::shared_ptr<gko::Executor const> const& gko_exec,
        OperatorModel const& operator_model,
        StateView state)
{
    using matrix_type = gko::matrix::Csr<double, gko::int32>;
    auto matrix = std::shared_ptr<matrix_type>(
            matrix_type::create(gko_exec, gko::dim<2>(operator_model.size(), operator_model.size()))
                    .release());
    auto matrix_data = assemble_matrix_data(operator_model, state);
    matrix->read(matrix_data);
    return matrix;
}

template <class ExecSpace, class OperatorModel>
class MatrixFreeLinOp : public gko::EnableLinOp<MatrixFreeLinOp<ExecSpace, OperatorModel>>
{
    using value_type = double;
    using dense_type = gko::matrix::Dense<value_type>;
    using memory_space = typename ExecSpace::memory_space;
    using base_type = gko::EnableLinOp<MatrixFreeLinOp<ExecSpace, OperatorModel>>;

    ExecSpace m_exec_space;
    std::shared_ptr<OperatorModel const> m_operator_model;

public:
    explicit MatrixFreeLinOp(std::shared_ptr<gko::Executor const> exec)
        : base_type(std::move(exec))
        , m_exec_space()
        , m_operator_model()
    {
    }

    MatrixFreeLinOp(
            std::shared_ptr<gko::Executor const> exec,
            ExecSpace exec_space,
            std::shared_ptr<OperatorModel const> operator_model)
        : base_type(exec, gko::dim<2>(operator_model->size(), operator_model->size()))
        , m_exec_space(exec_space)
        , m_operator_model(std::move(operator_model))
    {
    }

public:
    void apply_impl(gko::LinOp const* b, gko::LinOp* x) const override
    {
        auto const* b_dense = dynamic_cast<dense_type const*>(b);
        auto* x_dense = dynamic_cast<dense_type*>(x);
        if (b_dense == nullptr || x_dense == nullptr) {
            throw std::invalid_argument("MatrixFreeLinOp expects dense inputs and outputs");
        }
        auto b_view = gko::ext::kokkos::map_data<memory_space>(*b_dense);
        auto x_view = gko::ext::kokkos::map_data<memory_space>(*x_dense);
        apply_operator(m_exec_space, *m_operator_model, b_view, x_view);
    }

    void apply_impl(
            gko::LinOp const* alpha,
            gko::LinOp const* b,
            gko::LinOp const* beta,
            gko::LinOp* x) const override
    {
        auto const* alpha_dense = dynamic_cast<dense_type const*>(alpha);
        auto const* b_dense = dynamic_cast<dense_type const*>(b);
        auto const* beta_dense = dynamic_cast<dense_type const*>(beta);
        auto* x_dense = dynamic_cast<dense_type*>(x);
        if (alpha_dense == nullptr || b_dense == nullptr || beta_dense == nullptr
            || x_dense == nullptr) {
            throw std::invalid_argument(
                    "MatrixFreeLinOp expects dense alpha, beta, input, and output");
        }

        auto alpha_view = gko::ext::kokkos::map_data<memory_space>(*alpha_dense);
        auto b_view = gko::ext::kokkos::map_data<memory_space>(*b_dense);
        auto beta_view = gko::ext::kokkos::map_data<memory_space>(*beta_dense);
        auto x_view = gko::ext::kokkos::map_data<memory_space>(*x_dense);
        Kokkos::View<double**, Kokkos::LayoutRight, memory_space>
                applied("similie_matrix_free_linop_apply", x_view.extent(0), x_view.extent(1));
        apply_operator(m_exec_space, *m_operator_model, b_view, applied);
        Kokkos::parallel_for(
                "similie_matrix_free_linop_advanced_apply",
                Kokkos::MDRangePolicy<
                        ExecSpace,
                        Kokkos::Rank<
                                2>>(m_exec_space, {0, 0}, {x_view.extent(0), x_view.extent(1)}),
                KOKKOS_LAMBDA(std::size_t row, std::size_t column) {
                    x_view(row, column) = alpha_view(0, 0) * applied(row, column)
                                          + beta_view(0, 0) * x_view(row, column);
                });
        m_exec_space.fence();
    }
};

template <class ExecSpace, class OperatorModel, class StateView>
class StateDependentMatrixFreeLinOp
    : public gko::EnableLinOp<StateDependentMatrixFreeLinOp<ExecSpace, OperatorModel, StateView>>
{
    using value_type = double;
    using dense_type = gko::matrix::Dense<value_type>;
    using memory_space = typename ExecSpace::memory_space;
    using base_type
            = gko::EnableLinOp<StateDependentMatrixFreeLinOp<ExecSpace, OperatorModel, StateView>>;

    ExecSpace m_exec_space;
    std::shared_ptr<OperatorModel const> m_operator_model;
    StateView m_state;

public:
    explicit StateDependentMatrixFreeLinOp(std::shared_ptr<gko::Executor const> exec)
        : base_type(std::move(exec))
        , m_exec_space()
        , m_operator_model()
        , m_state()
    {
    }

    StateDependentMatrixFreeLinOp(
            std::shared_ptr<gko::Executor const> exec,
            ExecSpace exec_space,
            std::shared_ptr<OperatorModel const> operator_model,
            StateView state)
        : base_type(exec, gko::dim<2>(operator_model->size(), operator_model->size()))
        , m_exec_space(exec_space)
        , m_operator_model(std::move(operator_model))
        , m_state(state)
    {
    }

public:
    void apply_impl(gko::LinOp const* b, gko::LinOp* x) const override
    {
        auto const* b_dense = dynamic_cast<dense_type const*>(b);
        auto* x_dense = dynamic_cast<dense_type*>(x);
        if (b_dense == nullptr || x_dense == nullptr) {
            throw std::invalid_argument(
                    "StateDependentMatrixFreeLinOp expects dense inputs and outputs");
        }
        auto b_view = gko::ext::kokkos::map_data<memory_space>(*b_dense);
        auto x_view = gko::ext::kokkos::map_data<memory_space>(*x_dense);
        apply_jacobian(m_exec_space, *m_operator_model, m_state, b_view, x_view);
    }

    void apply_impl(
            gko::LinOp const* alpha,
            gko::LinOp const* b,
            gko::LinOp const* beta,
            gko::LinOp* x) const override
    {
        auto const* alpha_dense = dynamic_cast<dense_type const*>(alpha);
        auto const* b_dense = dynamic_cast<dense_type const*>(b);
        auto const* beta_dense = dynamic_cast<dense_type const*>(beta);
        auto* x_dense = dynamic_cast<dense_type*>(x);
        if (alpha_dense == nullptr || b_dense == nullptr || beta_dense == nullptr
            || x_dense == nullptr) {
            throw std::invalid_argument(
                    "StateDependentMatrixFreeLinOp expects dense alpha, beta, input, and output");
        }
        auto alpha_view = gko::ext::kokkos::map_data<memory_space>(*alpha_dense);
        auto b_view = gko::ext::kokkos::map_data<memory_space>(*b_dense);
        auto beta_view = gko::ext::kokkos::map_data<memory_space>(*beta_dense);
        auto x_view = gko::ext::kokkos::map_data<memory_space>(*x_dense);
        Kokkos::View<double**, Kokkos::LayoutRight, memory_space>
                applied("similie_state_dependent_linop_apply", x_view.extent(0), x_view.extent(1));
        apply_jacobian(m_exec_space, *m_operator_model, m_state, b_view, applied);
        Kokkos::parallel_for(
                "similie_state_dependent_linop_advanced_apply",
                Kokkos::MDRangePolicy<
                        ExecSpace,
                        Kokkos::Rank<
                                2>>(m_exec_space, {0, 0}, {x_view.extent(0), x_view.extent(1)}),
                KOKKOS_LAMBDA(std::size_t row, std::size_t column) {
                    x_view(row, column) = alpha_view(0, 0) * applied(row, column)
                                          + beta_view(0, 0) * x_view(row, column);
                });
        m_exec_space.fence();
    }
};

inline auto build_jacobi_preconditioner_factory(
        std::shared_ptr<gko::Executor const> const& gko_exec,
        StrongFormulationSolverSettings const& settings)
{
    return gko::preconditioner::Jacobi<double>::build()
            .with_max_block_size(settings.jacobi_max_block_size)
            .on(gko_exec);
}

inline std::shared_ptr<gko::LinOp const> build_jacobi_preconditioner(
        std::shared_ptr<gko::Executor const> const& gko_exec,
        std::shared_ptr<gko::LinOp const> const& matrix,
        StrongFormulationSolverSettings const& settings)
{
    auto preconditioner_factory = build_jacobi_preconditioner_factory(gko_exec, settings);
    return std::shared_ptr<gko::LinOp const>(preconditioner_factory->generate(matrix).release());
}

template <class KokkosViewType>
void apply_preconditioner(
        std::shared_ptr<gko::Executor const> const& gko_exec,
        std::shared_ptr<gko::LinOp const> const& preconditioner,
        KokkosViewType rhs,
        KokkosViewType solution)
{
    auto rhs_gko = to_gko_dense(gko_exec, rhs);
    auto solution_gko = to_gko_dense(gko_exec, solution);
    preconditioner->apply(rhs_gko.dense, solution_gko.dense);
    gko_exec->synchronize();
    copy_back_from_gko_dense_bridge(solution, solution_gko);
}

template <class OperatorModel, class StateView>
struct LinearizedOperatorModel
{
    OperatorModel const* operator_model;
    StateView state;

    [[nodiscard]] std::size_t size() const
    {
        return operator_model->size();
    }

    template <class ExecSpace, class InputView, class OutputView>
    void apply(ExecSpace exec_space, InputView input, OutputView output) const
    {
        apply_jacobian(exec_space, *operator_model, state, input, output);
    }
};

template <class ExecSpace, class OperatorModel, class RHSViewType, class SolutionViewType>
StrongFormulationSolverDiagnostics solve_matrix_free_linearized_system(
        ExecSpace exec_space,
        std::shared_ptr<gko::Executor const> const& gko_exec,
        OperatorModel const& operator_model,
        RHSViewType rhs,
        SolutionViewType solution,
        StrongFormulationSolverSettings const& settings,
        std::shared_ptr<gko::LinOp const> const& preconditioner)
{
    using memory_space = typename SolutionViewType::memory_space;
    StrongFormulationSolverDiagnostics diagnostics;
    diagnostics.initial_residual_l2 = residual_norm_l2(exec_space, rhs);
    diagnostics.final_residual_l2 = diagnostics.initial_residual_l2;
    diagnostics.final_relative_residual = diagnostics.initial_residual_l2 == 0.0 ? 0.0 : 1.0;
    if (diagnostics.initial_residual_l2 == 0.0) {
        fill(exec_space, solution, 0.0);
        return diagnostics;
    }

    Kokkos::View<double**, memory_space>
            residual("similie_matrix_free_cg_residual", rhs.extent(0), rhs.extent(1));
    Kokkos::View<double**, memory_space>
            preconditioned("similie_matrix_free_cg_preconditioned", rhs.extent(0), rhs.extent(1));
    Kokkos::View<double**, memory_space>
            direction("similie_matrix_free_cg_direction", rhs.extent(0), rhs.extent(1));
    Kokkos::View<double**, memory_space> operator_direction(
            "similie_matrix_free_cg_operator_direction",
            rhs.extent(0),
            rhs.extent(1));

    fill(exec_space, solution, 0.0);
    copy(exec_space, residual, rhs);
    auto const optimization_start = std::chrono::steady_clock::now();
    apply_preconditioner(gko_exec, preconditioner, residual, preconditioned);
    copy(exec_space, direction, preconditioned);
    double rho = dot(exec_space, residual, preconditioned);
    if (!std::isfinite(rho) || rho <= 0.0) {
        diagnostics.converged = false;
        auto const optimization_end = std::chrono::steady_clock::now();
        diagnostics.duration
                = std::chrono::duration<double>(optimization_end - optimization_start).count();
        return diagnostics;
    }

    diagnostics.converged = false;
    for (unsigned int iteration = 0; iteration < settings.max_iterations; ++iteration) {
        apply_operator(exec_space, operator_model, direction, operator_direction);
        double const denominator = dot(exec_space, direction, operator_direction);
        if (!std::isfinite(denominator) || denominator == 0.0) {
            break;
        }

        double const alpha = rho / denominator;
        axpy_inplace(exec_space, solution, alpha, direction);
        axpy_inplace(exec_space, residual, -alpha, operator_direction);

        diagnostics.iterations = iteration + 1U;
        diagnostics.final_residual_l2 = residual_norm_l2(exec_space, residual);
        diagnostics.final_relative_residual
                = diagnostics.final_residual_l2 / diagnostics.initial_residual_l2;
        if (diagnostics.final_relative_residual <= settings.relative_tolerance) {
            diagnostics.converged = true;
            break;
        }

        apply_preconditioner(gko_exec, preconditioner, residual, preconditioned);
        double const next_rho = dot(exec_space, residual, preconditioned);
        if (!std::isfinite(next_rho) || next_rho <= 0.0) {
            break;
        }
        double const beta = next_rho / rho;
        update_axpby(exec_space, direction, 1.0, preconditioned, beta, direction);
        rho = next_rho;
    }

    auto const optimization_end = std::chrono::steady_clock::now();
    diagnostics.duration
            = std::chrono::duration<double>(optimization_end - optimization_start).count();
    return diagnostics;
}

template <class ExecSpace, class OperatorModel, class RHSViewType, class SolutionViewType>
StrongFormulationSolverDiagnostics solve_linearized_system(
        ExecSpace exec_space,
        std::shared_ptr<gko::Executor const> const& gko_exec,
        OperatorModel const& operator_model,
        RHSViewType rhs,
        SolutionViewType solution,
        StrongFormulationSolverSettings const& settings,
        std::shared_ptr<gko::LinOp const> const& assembled_matrix,
        std::shared_ptr<gko::LinOp const> const& preconditioner)
{
    using solver_type = gko::solver::Cg<double>;
    StrongFormulationSolverDiagnostics diagnostics;
    diagnostics.initial_residual_l2 = residual_norm_l2(exec_space, rhs);
    diagnostics.final_residual_l2 = diagnostics.initial_residual_l2;
    diagnostics.final_relative_residual = diagnostics.initial_residual_l2 == 0.0 ? 0.0 : 1.0;
    if (diagnostics.initial_residual_l2 == 0.0) {
        fill(exec_space, solution, 0.0);
        return diagnostics;
    }
    if (settings.use_matrix_free) {
        return solve_matrix_free_linearized_system(
                exec_space,
                gko_exec,
                operator_model,
                rhs,
                solution,
                settings,
                preconditioner);
    }

    auto residual_criterion = gko::stop::ResidualNorm<double>::build()
                                      .with_reduction_factor(settings.relative_tolerance)
                                      .on(gko_exec);
    auto iterations_criterion
            = gko::stop::Iteration::build().with_max_iters(settings.max_iterations).on(gko_exec);
    auto solver_factory
            = solver_type::build()
                      .with_generated_preconditioner(preconditioner)
                      .with_criteria(std::move(residual_criterion), std::move(iterations_criterion))
                      .on(gko_exec);
    auto solver = solver_factory->generate(assembled_matrix);
    auto convergence_logger = std::shared_ptr<gko::log::Convergence<double>>(
            gko::log::Convergence<double>::create().release());
    solver->add_logger(convergence_logger);

    fill(exec_space, solution, 0.0);
    auto rhs_gko = to_gko_dense(gko_exec, rhs);
    auto solution_gko = to_gko_dense(gko_exec, solution);
    auto const optimization_start = std::chrono::steady_clock::now();
    solver->apply(rhs_gko.dense, solution_gko.dense);
    gko_exec->synchronize();
    copy_back_from_gko_dense_bridge(solution, solution_gko);
    auto const optimization_end = std::chrono::steady_clock::now();
    solver->remove_logger(convergence_logger);
    diagnostics.duration
            = std::chrono::duration<double>(optimization_end - optimization_start).count();
    diagnostics.iterations = static_cast<unsigned int>(convergence_logger->get_num_iterations());
    diagnostics.converged = convergence_logger->has_converged();
    auto residual_norm_dense = dynamic_cast<gko::matrix::Dense<double> const*>(
            convergence_logger->get_residual_norm());
    if (residual_norm_dense != nullptr) {
        auto host_dense = gko::matrix::Dense<
                double>::create(gko_exec->get_master(), residual_norm_dense->get_size());
        residual_norm_dense->convert_to(host_dense.get());
        diagnostics.final_residual_l2 = host_dense->at(0, 0);
    }
    diagnostics.final_relative_residual
            = diagnostics.initial_residual_l2 == 0.0
                      ? 0.0
                      : diagnostics.final_residual_l2 / diagnostics.initial_residual_l2;
    return diagnostics;
}

template <class ExecSpace, class ViewType>
double residual_norm_l2(ExecSpace exec_space, ViewType residual)
{
    return std::sqrt(dot(exec_space, residual, residual));
}

} // namespace detail

template <class ExecSpace, class OperatorModel, class RHSViewType, class SolutionViewType>
StrongFormulationSolverDiagnostics minimize_strong_formulation_residual(
        ExecSpace exec_space,
        OperatorModel const& operator_model,
        RHSViewType rhs,
        SolutionViewType solution,
        StrongFormulationSolverSettings settings = {})
{
    StrongFormulationSolverDiagnostics diagnostics;

    detail::fill(exec_space, solution, 0.0);
    auto const gko_exec = gko::ext::kokkos::create_executor(exec_space);
    if constexpr (OperatorModel::IS_LINEAR) {
        diagnostics.initial_residual_l2 = detail::residual_norm_l2(exec_space, rhs);
        diagnostics.final_residual_l2 = diagnostics.initial_residual_l2;
        diagnostics.final_relative_residual = diagnostics.initial_residual_l2 == 0.0 ? 0.0 : 1.0;
        if (diagnostics.initial_residual_l2 == 0.0) {
            return diagnostics;
        }
        auto matrix = detail::build_matrix(gko_exec, operator_model);
        auto preconditioner = detail::build_jacobi_preconditioner(
                gko_exec,
                std::static_pointer_cast<gko::LinOp const>(matrix),
                settings);
        return detail::solve_linearized_system(
                exec_space,
                gko_exec,
                operator_model,
                rhs,
                solution,
                settings,
                std::static_pointer_cast<gko::LinOp const>(matrix),
                preconditioner);
    } else {
        using memory_space = typename SolutionViewType::memory_space;
        Kokkos::View<double**, memory_space>
                residual("similie_nonlinear_residual", rhs.extent(0), rhs.extent(1));
        Kokkos::View<double**, memory_space>
                operator_value("similie_nonlinear_operator_value", rhs.extent(0), rhs.extent(1));
        Kokkos::View<double**, memory_space>
                correction_rhs("similie_nonlinear_correction_rhs", rhs.extent(0), rhs.extent(1));
        Kokkos::View<double**, memory_space>
                delta("similie_nonlinear_delta", rhs.extent(0), rhs.extent(1));
        auto const optimization_start = std::chrono::steady_clock::now();
        diagnostics.converged = false;

        detail::apply_operator(exec_space, operator_model, solution, operator_value);
        detail::copy(exec_space, residual, rhs);
        detail::axpy_inplace(exec_space, residual, -1.0, operator_value);
        diagnostics.initial_residual_l2 = detail::residual_norm_l2(exec_space, residual);
        diagnostics.final_residual_l2 = diagnostics.initial_residual_l2;
        diagnostics.final_relative_residual = diagnostics.initial_residual_l2 == 0.0 ? 0.0 : 1.0;
        if (diagnostics.initial_residual_l2 == 0.0) {
            diagnostics.converged = true;
            return diagnostics;
        }

        constexpr unsigned int NONLINEAR_MAX_ITERS = 30U;
        for (unsigned int iteration = 0; iteration < NONLINEAR_MAX_ITERS; ++iteration) {
            diagnostics.final_residual_l2 = detail::residual_norm_l2(exec_space, residual);
            diagnostics.final_relative_residual
                    = diagnostics.initial_residual_l2 == 0.0
                              ? 0.0
                              : diagnostics.final_residual_l2 / diagnostics.initial_residual_l2;
            if (diagnostics.final_relative_residual <= settings.relative_tolerance) {
                diagnostics.converged = true;
                break;
            }

            auto matrix = detail::build_matrix(gko_exec, operator_model, solution);
            auto preconditioner = detail::build_jacobi_preconditioner(
                    gko_exec,
                    std::static_pointer_cast<gko::LinOp const>(matrix),
                    settings);

            detail::copy(exec_space, correction_rhs, residual);
            if (settings.use_matrix_free) {
                detail::LinearizedOperatorModel<OperatorModel, SolutionViewType>
                        linearized_operator {&operator_model, solution};
                auto inner = detail::solve_matrix_free_linearized_system(
                        exec_space,
                        gko_exec,
                        linearized_operator,
                        correction_rhs,
                        delta,
                        settings,
                        preconditioner);
                diagnostics.iterations += inner.iterations;
            } else {
                auto inner = detail::solve_linearized_system(
                        exec_space,
                        gko_exec,
                        operator_model,
                        correction_rhs,
                        delta,
                        settings,
                        std::static_pointer_cast<gko::LinOp const>(matrix),
                        preconditioner);
                diagnostics.iterations += inner.iterations;
            }
            detail::axpy_inplace(exec_space, solution, 1.0, delta);
            detail::apply_operator(exec_space, operator_model, solution, operator_value);
            detail::copy(exec_space, residual, rhs);
            detail::axpy_inplace(exec_space, residual, -1.0, operator_value);
        }
        auto const optimization_end = std::chrono::steady_clock::now();
        diagnostics.duration
                = std::chrono::duration<double>(optimization_end - optimization_start).count();
        diagnostics.final_residual_l2 = detail::residual_norm_l2(exec_space, residual);
        diagnostics.final_relative_residual
                = diagnostics.initial_residual_l2 == 0.0
                          ? 0.0
                          : diagnostics.final_residual_l2 / diagnostics.initial_residual_l2;
        return diagnostics;
    }
}

} // namespace similie::solvers
