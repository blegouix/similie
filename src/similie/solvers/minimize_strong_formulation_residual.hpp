// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/log/convergence.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/identity.hpp>
#include <ginkgo/core/preconditioner/gauss_seidel.hpp>
#include <ginkgo/core/preconditioner/isai.hpp>
#include <ginkgo/core/preconditioner/jacobi.hpp>
#include <ginkgo/core/preconditioner/sor.hpp>
#include <ginkgo/core/solver/bicgstab.hpp>
#include <ginkgo/core/solver/cg.hpp>
#include <ginkgo/core/solver/chebyshev.hpp>
#include <ginkgo/core/solver/fcg.hpp>
#include <ginkgo/core/solver/gcr.hpp>
#include <ginkgo/core/solver/gmres.hpp>
#include <ginkgo/core/solver/idr.hpp>
#include <ginkgo/core/solver/ir.hpp>
#include <ginkgo/core/solver/minres.hpp>
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

enum class PreconditionerType {
    Identity,
    Jacobi,
    SpdIsai,
    SymmetricGaussSeidel,
    Ssor,
    ChebyshevJacobi,
    IrJacobi,
    GeneralIsai,
};

inline constexpr std::string_view preconditioner_name(PreconditionerType preconditioner)
{
    switch (preconditioner) {
    case PreconditionerType::Identity:
        return "Identity";
    case PreconditionerType::Jacobi:
        return "Jacobi";
    case PreconditionerType::SpdIsai:
        return "SpdIsai";
    case PreconditionerType::SymmetricGaussSeidel:
        return "SymmetricGaussSeidel";
    case PreconditionerType::Ssor:
        return "Ssor";
    case PreconditionerType::ChebyshevJacobi:
        return "ChebyshevJacobi";
    case PreconditionerType::IrJacobi:
        return "IrJacobi";
    case PreconditionerType::GeneralIsai:
        return "GeneralIsai";
    }
    return "Jacobi";
}

inline PreconditionerType parse_preconditioner(std::string_view name)
{
    if (name == "Identity" || name == "identity") {
        return PreconditionerType::Identity;
    }
    if (name == "Jacobi" || name == "jacobi") {
        return PreconditionerType::Jacobi;
    }
    if (name == "SpdIsai" || name == "spd-isai" || name == "spd_isai") {
        return PreconditionerType::SpdIsai;
    }
    if (name == "SymmetricGaussSeidel" || name == "symmetric-gauss-seidel"
        || name == "symmetric_gauss_seidel") {
        return PreconditionerType::SymmetricGaussSeidel;
    }
    if (name == "Ssor" || name == "SSOR" || name == "ssor") {
        return PreconditionerType::Ssor;
    }
    if (name == "ChebyshevJacobi" || name == "chebyshev-jacobi" || name == "chebyshev_jacobi") {
        return PreconditionerType::ChebyshevJacobi;
    }
    if (name == "IrJacobi" || name == "ir-jacobi" || name == "ir_jacobi") {
        return PreconditionerType::IrJacobi;
    }
    if (name == "GeneralIsai" || name == "general-isai" || name == "general_isai"
        || name == "isai") {
        return PreconditionerType::GeneralIsai;
    }
    throw std::runtime_error("unknown strong-formulation preconditioner: " + std::string(name));
}

struct StrongFormulationSolverSettings
{
    unsigned int max_iterations = 2000U;
    double relative_tolerance = 1.0e-12;
    unsigned int jacobi_max_block_size = 1U;
    bool use_matrix_free = true;
    Criterion criterion = Criterion::MomentsTemporalDerivative;
    PreconditionerType preconditioner = PreconditionerType::Jacobi;
    double sor_relaxation_factor = 1.2;
    unsigned int chebyshev_iterations = 8U;
    double chebyshev_lower_bound = 0.05;
    double chebyshev_upper_bound = 1.5;
    unsigned int ir_iterations = 8U;
    double ir_relaxation_factor = 1.0;
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

inline unsigned int solver_progress_stride()
{
    char const* const value = std::getenv("SIMILIE_SOLVER_PROGRESS_STRIDE");
    if (value == nullptr || value[0] == '\0') {
        return 0U;
    }

    char* parse_end = nullptr;
    unsigned long const parsed = std::strtoul(value, &parse_end, 10);
    if (parse_end == value || parsed == 0UL) {
        return 1U;
    }
    if (parsed > std::numeric_limits<unsigned int>::max()) {
        return std::numeric_limits<unsigned int>::max();
    }
    return static_cast<unsigned int>(parsed);
}

inline bool solver_progress_enabled()
{
    return solver_progress_stride() != 0U;
}

inline bool env_flag_enabled(char const* name)
{
    char const* const value = std::getenv(name);
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

inline bool env_value_equals(char const* name, char const* expected)
{
    char const* const value = std::getenv(name);
    return value != nullptr && std::string_view(value) == expected;
}

inline double env_double_or(char const* name, double default_value)
{
    char const* const value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return default_value;
    }
    char* parse_end = nullptr;
    double const parsed = std::strtod(value, &parse_end);
    if (parse_end == value) {
        return default_value;
    }
    return parsed;
}

inline int env_int_or(char const* name, int default_value)
{
    char const* const value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return default_value;
    }
    char* parse_end = nullptr;
    long const parsed = std::strtol(value, &parse_end, 10);
    if (parse_end == value) {
        return default_value;
    }
    if (parsed > std::numeric_limits<int>::max()) {
        return std::numeric_limits<int>::max();
    }
    if (parsed < std::numeric_limits<int>::min()) {
        return std::numeric_limits<int>::min();
    }
    return static_cast<int>(parsed);
}

template <class MatrixData>
void log_matrix_diagnostics(MatrixData const& matrix_data)
{
    std::size_t const size = matrix_data.size[0];
    std::vector<double> diagonal(size, 0.0);
    std::size_t negative_diagonal_count = 0;
    std::size_t zero_diagonal_count = 0;
    double min_diagonal = std::numeric_limits<double>::infinity();
    double max_diagonal = -std::numeric_limits<double>::infinity();
    std::vector<double> off_diagonal_abs_row_sum(size, 0.0);
    for (auto const& entry : matrix_data.nonzeros) {
        if (entry.row == entry.column) {
            diagonal[static_cast<std::size_t>(entry.row)] += entry.value;
        } else {
            off_diagonal_abs_row_sum[static_cast<std::size_t>(entry.row)] += std::abs(entry.value);
        }
    }
    double jacobi_gershgorin_lower = std::numeric_limits<double>::infinity();
    double jacobi_gershgorin_upper = -std::numeric_limits<double>::infinity();
    double jacobi_max_abs_off_diagonal_row_sum = 0.0;
    for (double const value : diagonal) {
        if (value < 0.0) {
            ++negative_diagonal_count;
        }
        if (value == 0.0) {
            ++zero_diagonal_count;
        }
        min_diagonal = std::min(min_diagonal, value);
        max_diagonal = std::max(max_diagonal, value);
    }
    for (std::size_t row = 0; row < size; ++row) {
        if (diagonal[row] == 0.0) {
            continue;
        }
        double const scaled_radius = off_diagonal_abs_row_sum[row] / std::abs(diagonal[row]);
        jacobi_max_abs_off_diagonal_row_sum
                = std::max(jacobi_max_abs_off_diagonal_row_sum, scaled_radius);
        jacobi_gershgorin_lower = std::min(jacobi_gershgorin_lower, 1.0 - scaled_radius);
        jacobi_gershgorin_upper = std::max(jacobi_gershgorin_upper, 1.0 + scaled_radius);
    }

    double max_abs_asymmetry = 0.0;
    std::size_t asymmetric_entry_count = 0;
    bool const symmetry_diagnostics_skipped
            = env_flag_enabled("SIMILIE_SKIP_MATRIX_SYMMETRY_DIAGNOSTICS");
    if (!symmetry_diagnostics_skipped) {
        auto matrix_entries = std::unordered_map<std::uint64_t, double>();
        matrix_entries.reserve(matrix_data.nonzeros.size());
        auto const entry_key = [size](std::size_t row, std::size_t column) {
            return static_cast<std::uint64_t>(row) * static_cast<std::uint64_t>(size)
                   + static_cast<std::uint64_t>(column);
        };
        for (auto const& entry : matrix_data.nonzeros) {
            matrix_entries[entry_key(
                    static_cast<std::size_t>(entry.row),
                    static_cast<std::size_t>(entry.column))]
                    += entry.value;
        }
        for (auto const& [key, value] : matrix_entries) {
            std::size_t const row
                    = static_cast<std::size_t>(key / static_cast<std::uint64_t>(size));
            std::size_t const column
                    = static_cast<std::size_t>(key % static_cast<std::uint64_t>(size));
            double transposed_value = 0.0;
            if (auto const transposed = matrix_entries.find(entry_key(column, row));
                transposed != matrix_entries.end()) {
                transposed_value = transposed->second;
            }
            double const asymmetry = std::abs(value - transposed_value);
            if (asymmetry
                > 1.0e-10 * std::max({1.0, std::abs(value), std::abs(transposed_value)})) {
                ++asymmetric_entry_count;
            }
            max_abs_asymmetry = std::max(max_abs_asymmetry, asymmetry);
        }
    }

    std::cout << "SimiLie matrix diagnostics: size=" << size
              << " nonzeros=" << matrix_data.nonzeros.size() << " diagonal_min=" << min_diagonal
              << " diagonal_max=" << max_diagonal
              << " diagonal_negative_count=" << negative_diagonal_count
              << " diagonal_zero_count=" << zero_diagonal_count
              << " jacobi_gershgorin_lower=" << jacobi_gershgorin_lower
              << " jacobi_gershgorin_upper=" << jacobi_gershgorin_upper
              << " jacobi_max_abs_off_diagonal_row_sum=" << jacobi_max_abs_off_diagonal_row_sum
              << " symmetry_diagnostics_skipped=" << symmetry_diagnostics_skipped
              << " asymmetric_entry_count=" << asymmetric_entry_count
              << " max_abs_asymmetry=" << max_abs_asymmetry << '\n';
}

class SolverProgressLogger final : public gko::log::Logger
{
public:
    SolverProgressLogger(
            std::shared_ptr<gko::Executor const> master_executor,
            double initial_residual_l2,
            unsigned int stride)
        : Logger(gko::log::Logger::iteration_complete_mask)
        , m_master_executor(std::move(master_executor))
        , m_initial_residual_l2(initial_residual_l2)
        , m_stride(stride)
    {
    }

protected:
    void on_iteration_complete(
            gko::LinOp const*,
            gko::LinOp const*,
            gko::LinOp const*,
            gko::size_type const& num_iterations,
            gko::LinOp const* residual,
            gko::LinOp const* residual_norm,
            gko::LinOp const*,
            gko::array<gko::stopping_status> const*,
            bool) const override
    {
        if (m_stride == 0U || num_iterations % m_stride != 0U) {
            return;
        }
        std::optional<double> residual_l2 = residual_norm_value(residual_norm);
        if (!residual_l2.has_value()) {
            residual_l2 = residual_vector_l2(residual);
        }
        if (!residual_l2.has_value()) {
            return;
        }
        double const relative_residual
                = m_initial_residual_l2 == 0.0 ? 0.0 : *residual_l2 / m_initial_residual_l2;
        std::cout << "SimiLie solver progress: iteration=" << num_iterations
                  << " residual_l2=" << *residual_l2 << " relative_residual=" << relative_residual
                  << std::endl;
    }

private:
    std::optional<double> residual_norm_value(gko::LinOp const* residual_norm) const
    {
        auto const* residual_norm_dense
                = dynamic_cast<gko::matrix::Dense<double> const*>(residual_norm);
        if (residual_norm_dense == nullptr) {
            return std::nullopt;
        }
        auto host_dense = gko::matrix::Dense<
                double>::create(m_master_executor, residual_norm_dense->get_size());
        residual_norm_dense->convert_to(host_dense.get());
        return host_dense->at(0, 0);
    }

    std::optional<double> residual_vector_l2(gko::LinOp const* residual) const
    {
        auto const* residual_dense = dynamic_cast<gko::matrix::Dense<double> const*>(residual);
        if (residual_dense == nullptr) {
            return std::nullopt;
        }
        auto host_dense
                = gko::matrix::Dense<double>::create(m_master_executor, residual_dense->get_size());
        residual_dense->convert_to(host_dense.get());
        double squared_norm = 0.0;
        for (gko::size_type row = 0; row < host_dense->get_size()[0]; ++row) {
            for (gko::size_type column = 0; column < host_dense->get_size()[1]; ++column) {
                double const value = host_dense->at(row, column);
                squared_norm += value * value;
            }
        }
        return std::sqrt(squared_norm);
    }

    std::shared_ptr<gko::Executor const> m_master_executor;
    double m_initial_residual_l2;
    unsigned int m_stride;
};

inline void log_nonlinear_progress(
        unsigned int iteration,
        StrongFormulationSolverDiagnostics const& diagnostics)
{
    if (!solver_progress_enabled()) {
        return;
    }
    std::cout << "SimiLie nonlinear progress: iteration=" << iteration
              << " residual_l2=" << diagnostics.final_residual_l2
              << " relative_residual=" << diagnostics.final_relative_residual << std::endl;
}

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

template <class ExecSpace, class OperatorModel, class = void>
struct MatrixFreeWorkspaceTraits
{
    static constexpr bool enabled = false;
    struct type
    {
    };
};

template <class ExecSpace, class OperatorModel>
struct MatrixFreeWorkspaceTraits<
        ExecSpace,
        OperatorModel,
        std::void_t<decltype(std::declval<OperatorModel const&>().create_matrix_free_workspace(
                std::declval<ExecSpace>()))>>
{
    static constexpr bool enabled = true;
    using type = decltype(std::declval<OperatorModel const&>().create_matrix_free_workspace(
            std::declval<ExecSpace>()));
};

template <class ExecSpace, class OperatorModel, class InputView, class OutputView, class Workspace>
void apply_operator(
        ExecSpace exec_space,
        OperatorModel const& operator_model,
        InputView input,
        OutputView output,
        Workspace& workspace)
{
    if constexpr (requires(
                          OperatorModel const& model,
                          ExecSpace ex,
                          InputView in,
                          OutputView out,
                          Workspace& work) { model.apply(ex, in, out, work); }) {
        operator_model.apply(exec_space, input, output, workspace);
    } else {
        apply_operator(exec_space, operator_model, input, output);
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
    if (env_flag_enabled("SIMILIE_MATRIX_DIAGNOSTICS")) {
        log_matrix_diagnostics(matrix_data);
    }
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
    if (env_flag_enabled("SIMILIE_MATRIX_DIAGNOSTICS")) {
        log_matrix_diagnostics(matrix_data);
    }
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
    using workspace_traits = MatrixFreeWorkspaceTraits<ExecSpace, OperatorModel>;
    using workspace_type = typename workspace_traits::type;

    ExecSpace m_exec_space;
    std::shared_ptr<OperatorModel const> m_operator_model;
    mutable std::shared_ptr<workspace_type> m_workspace;
    mutable std::size_t m_apply_count = 0;
    mutable std::size_t m_advanced_apply_count = 0;
    mutable double m_apply_duration = 0.0;
    mutable double m_advanced_apply_duration = 0.0;

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
        if constexpr (workspace_traits::enabled) {
            m_workspace = std::make_shared<workspace_type>(
                    m_operator_model->create_matrix_free_workspace(m_exec_space));
        }
    }

public:
    void apply_impl(gko::LinOp const* b, gko::LinOp* x) const override
    {
        auto const apply_start = std::chrono::steady_clock::now();
        auto const* b_dense = dynamic_cast<dense_type const*>(b);
        auto* x_dense = dynamic_cast<dense_type*>(x);
        if (b_dense == nullptr || x_dense == nullptr) {
            throw std::invalid_argument("MatrixFreeLinOp expects dense inputs and outputs");
        }
        auto b_view = gko::ext::kokkos::map_data<memory_space>(*b_dense);
        auto x_view = gko::ext::kokkos::map_data<memory_space>(*x_dense);
        if constexpr (workspace_traits::enabled) {
            if (m_workspace == nullptr) {
                m_workspace = std::make_shared<workspace_type>(
                        m_operator_model->create_matrix_free_workspace(m_exec_space));
            }
            apply_operator(m_exec_space, *m_operator_model, b_view, x_view, *m_workspace);
        } else {
            apply_operator(m_exec_space, *m_operator_model, b_view, x_view);
        }
        auto const apply_end = std::chrono::steady_clock::now();
        ++m_apply_count;
        m_apply_duration += std::chrono::duration<double>(apply_end - apply_start).count();
    }

    void apply_impl(
            gko::LinOp const* alpha,
            gko::LinOp const* b,
            gko::LinOp const* beta,
            gko::LinOp* x) const override
    {
        auto const apply_start = std::chrono::steady_clock::now();
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
        if constexpr (workspace_traits::enabled) {
            if (m_workspace == nullptr) {
                m_workspace = std::make_shared<workspace_type>(
                        m_operator_model->create_matrix_free_workspace(m_exec_space));
            }
            apply_operator(m_exec_space, *m_operator_model, b_view, applied, *m_workspace);
        } else {
            apply_operator(m_exec_space, *m_operator_model, b_view, applied);
        }
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
        auto const apply_end = std::chrono::steady_clock::now();
        ++m_advanced_apply_count;
        m_advanced_apply_duration += std::chrono::duration<double>(apply_end - apply_start).count();
    }

    void log_timing() const
    {
        if (!env_flag_enabled("SIMILIE_SOLVER_TIMING")) {
            return;
        }
        std::cout << "SimiLie matrix-free operator timing: apply_count=" << m_apply_count
                  << " apply_duration=" << m_apply_duration
                  << " advanced_apply_count=" << m_advanced_apply_count
                  << " advanced_apply_duration=" << m_advanced_apply_duration << '\n';
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

inline PreconditionerType selected_preconditioner(StrongFormulationSolverSettings const& settings)
{
    char const* const value = std::getenv("SIMILIE_PRECONDITIONER");
    if (value == nullptr || value[0] == '\0') {
        if (env_flag_enabled("SIMILIE_DISABLE_JACOBI_PRECONDITIONER")) {
            return PreconditionerType::Identity;
        }
        return settings.preconditioner;
    }
    return parse_preconditioner(value);
}

inline std::shared_ptr<gko::LinOp const> build_identity_preconditioner(
        std::shared_ptr<gko::Executor const> const& gko_exec,
        gko::size_type size)
{
    return gko::matrix::Identity<double>::create(gko_exec, size);
}

inline std::shared_ptr<gko::LinOp const> build_preconditioner(
        std::shared_ptr<gko::Executor const> const& gko_exec,
        std::shared_ptr<gko::LinOp const> const& matrix,
        StrongFormulationSolverSettings const& settings)
{
    PreconditionerType const preconditioner = selected_preconditioner(settings);
    std::cout << "SimiLie Ginkgo preconditioner: " << preconditioner_name(preconditioner) << '\n';
    switch (preconditioner) {
    case PreconditionerType::Identity:
        return build_identity_preconditioner(gko_exec, matrix->get_size()[0]);
    case PreconditionerType::Jacobi: {
        auto preconditioner_factory = build_jacobi_preconditioner_factory(gko_exec, settings);
        return std::shared_ptr<gko::LinOp const>(
                preconditioner_factory->generate(matrix).release());
    }
    case PreconditionerType::SpdIsai: {
        auto preconditioner_factory
                = gko::preconditioner::SpdIsai<double, gko::int32>::build().on(gko_exec);
        return std::shared_ptr<gko::LinOp const>(
                preconditioner_factory->generate(matrix).release());
    }
    case PreconditionerType::GeneralIsai: {
        auto preconditioner_factory
                = gko::preconditioner::GeneralIsai<double, gko::int32>::build().on(gko_exec);
        return std::shared_ptr<gko::LinOp const>(
                preconditioner_factory->generate(matrix).release());
    }
    case PreconditionerType::SymmetricGaussSeidel: {
        auto preconditioner_factory = gko::preconditioner::GaussSeidel<double, gko::int32>::build()
                                              .with_symmetric(true)
                                              .on(gko_exec);
        return std::shared_ptr<gko::LinOp const>(
                preconditioner_factory->generate(matrix).release());
    }
    case PreconditionerType::Ssor: {
        auto preconditioner_factory = gko::preconditioner::Sor<double, gko::int32>::build()
                                              .with_symmetric(true)
                                              .with_relaxation_factor(env_double_or(
                                                      "SIMILIE_SOR_RELAXATION_FACTOR",
                                                      settings.sor_relaxation_factor))
                                              .on(gko_exec);
        return std::shared_ptr<gko::LinOp const>(
                preconditioner_factory->generate(matrix).release());
    }
    case PreconditionerType::ChebyshevJacobi: {
        auto jacobi_factory = build_jacobi_preconditioner_factory(gko_exec, settings);
        auto iterations_criterion = gko::stop::Iteration::build()
                                            .with_max_iters(settings.chebyshev_iterations)
                                            .on(gko_exec);
        auto preconditioner_factory
                = gko::solver::Chebyshev<double>::build()
                          .with_criteria(std::move(iterations_criterion))
                          .with_preconditioner(std::move(jacobi_factory))
                          .with_foci(settings.chebyshev_lower_bound, settings.chebyshev_upper_bound)
                          .with_default_initial_guess(gko::solver::initial_guess_mode::zero)
                          .on(gko_exec);
        return std::shared_ptr<gko::LinOp const>(
                preconditioner_factory->generate(matrix).release());
    }
    case PreconditionerType::IrJacobi: {
        auto jacobi_factory = build_jacobi_preconditioner_factory(gko_exec, settings);
        auto iterations_criterion
                = gko::stop::Iteration::build().with_max_iters(settings.ir_iterations).on(gko_exec);
        auto preconditioner_factory
                = gko::solver::Ir<double>::build()
                          .with_criteria(std::move(iterations_criterion))
                          .with_solver(std::move(jacobi_factory))
                          .with_relaxation_factor(settings.ir_relaxation_factor)
                          .with_default_initial_guess(gko::solver::initial_guess_mode::zero)
                          .on(gko_exec);
        return std::shared_ptr<gko::LinOp const>(
                preconditioner_factory->generate(matrix).release());
    }
    }
    throw std::runtime_error("unsupported strong-formulation preconditioner");
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
    StrongFormulationSolverDiagnostics diagnostics;
    diagnostics.initial_residual_l2 = residual_norm_l2(exec_space, rhs);
    diagnostics.final_residual_l2 = diagnostics.initial_residual_l2;
    diagnostics.final_relative_residual = diagnostics.initial_residual_l2 == 0.0 ? 0.0 : 1.0;
    if (diagnostics.initial_residual_l2 == 0.0) {
        fill(exec_space, solution, 0.0);
        return diagnostics;
    }

    auto residual_criterion = gko::stop::ResidualNorm<double>::build()
                                      .with_reduction_factor(settings.relative_tolerance)
                                      .on(gko_exec);
    auto iterations_criterion
            = gko::stop::Iteration::build().with_max_iters(settings.max_iterations).on(gko_exec);
    std::unique_ptr<gko::LinOpFactory> solver_factory;
    if (detail::env_value_equals("SIMILIE_SOLVER", "minres")) {
        solver_factory = gko::solver::Minres<double>::build()
                                 .with_generated_preconditioner(preconditioner)
                                 .with_criteria(
                                         std::move(residual_criterion),
                                         std::move(iterations_criterion))
                                 .on(gko_exec);
    } else if (detail::env_value_equals("SIMILIE_SOLVER", "fcg")) {
        solver_factory = gko::solver::Fcg<double>::build()
                                 .with_generated_preconditioner(preconditioner)
                                 .with_criteria(
                                         std::move(residual_criterion),
                                         std::move(iterations_criterion))
                                 .on(gko_exec);
    } else if (detail::env_value_equals("SIMILIE_SOLVER", "gmres")) {
        solver_factory = gko::solver::Gmres<double>::build()
                                 .with_generated_preconditioner(preconditioner)
                                 .with_criteria(
                                         std::move(residual_criterion),
                                         std::move(iterations_criterion))
                                 .with_krylov_dim(
                                         static_cast<gko::size_type>(
                                                 env_int_or("SIMILIE_GMRES_KRYLOV_DIM", 100)))
                                 .on(gko_exec);
    } else if (detail::env_value_equals("SIMILIE_SOLVER", "bicgstab")) {
        solver_factory = gko::solver::Bicgstab<double>::build()
                                 .with_generated_preconditioner(preconditioner)
                                 .with_criteria(
                                         std::move(residual_criterion),
                                         std::move(iterations_criterion))
                                 .on(gko_exec);
    } else if (detail::env_value_equals("SIMILIE_SOLVER", "gcr")) {
        solver_factory
                = gko::solver::Gcr<double>::build()
                          .with_generated_preconditioner(preconditioner)
                          .with_criteria(
                                  std::move(residual_criterion),
                                  std::move(iterations_criterion))
                          .with_krylov_dim(
                                  static_cast<gko::size_type>(
                                          std::max(1, env_int_or("SIMILIE_GCR_KRYLOV_DIM", 100))))
                          .on(gko_exec);
    } else if (detail::env_value_equals("SIMILIE_SOLVER", "idr")) {
        solver_factory
                = gko::solver::Idr<double>::build()
                          .with_generated_preconditioner(preconditioner)
                          .with_criteria(
                                  std::move(residual_criterion),
                                  std::move(iterations_criterion))
                          .with_subspace_dim(
                                  static_cast<gko::size_type>(
                                          std::max(1, env_int_or("SIMILIE_IDR_SUBSPACE_DIM", 8))))
                          .with_deterministic(true)
                          .on(gko_exec);
    } else {
        solver_factory = gko::solver::Cg<double>::build()
                                 .with_generated_preconditioner(preconditioner)
                                 .with_criteria(
                                         std::move(residual_criterion),
                                         std::move(iterations_criterion))
                                 .on(gko_exec);
    }
    std::shared_ptr<MatrixFreeLinOp<ExecSpace, OperatorModel> const> matrix_free_system_matrix;
    std::shared_ptr<gko::LinOp const> system_matrix;
    if (settings.use_matrix_free) {
        auto operator_model_ptr
                = std::shared_ptr<OperatorModel const>(&operator_model, [](OperatorModel const*) {
                  });
        matrix_free_system_matrix = std::make_shared<MatrixFreeLinOp<
                ExecSpace,
                OperatorModel>>(gko_exec, exec_space, std::move(operator_model_ptr));
        system_matrix = matrix_free_system_matrix;
    } else {
        system_matrix = assembled_matrix;
    }
    if (settings.use_matrix_free && matrix_free_system_matrix != nullptr
        && assembled_matrix != nullptr && env_flag_enabled("SIMILIE_COMPARE_MATRIX_FREE_APPLY")) {
        using memory_space = typename RHSViewType::memory_space;
        Kokkos::View<double**, memory_space> matrix_free_applied(
                "similie_compare_matrix_free_applied",
                rhs.extent(0),
                rhs.extent(1));
        Kokkos::View<double**, memory_space> assembled_applied(
                "similie_compare_assembled_applied",
                rhs.extent(0),
                rhs.extent(1));
        Kokkos::View<double**, memory_space>
                difference("similie_compare_apply_difference", rhs.extent(0), rhs.extent(1));
        auto probe_gko = to_gko_dense(gko_exec, rhs);
        auto matrix_free_applied_gko = to_gko_dense(gko_exec, matrix_free_applied);
        auto assembled_applied_gko = to_gko_dense(gko_exec, assembled_applied);
        matrix_free_system_matrix->apply(probe_gko.dense, matrix_free_applied_gko.dense);
        assembled_matrix->apply(probe_gko.dense, assembled_applied_gko.dense);
        gko_exec->synchronize();
        copy_back_from_gko_dense_bridge(matrix_free_applied, matrix_free_applied_gko);
        copy_back_from_gko_dense_bridge(assembled_applied, assembled_applied_gko);
        copy(exec_space, difference, matrix_free_applied);
        axpy_inplace(exec_space, difference, -1.0, assembled_applied);
        double const difference_norm = residual_norm_l2(exec_space, difference);
        double const assembled_norm = residual_norm_l2(exec_space, assembled_applied);
        std::cout << "SimiLie matrix-free apply comparison: difference_l2=" << difference_norm
                  << " assembled_l2=" << assembled_norm << " relative_difference="
                  << (assembled_norm == 0.0 ? 0.0 : difference_norm / assembled_norm) << '\n';
        if (rhs.extent(1) == 1 && rhs.extent(0) % 3 == 0) {
            auto const difference_host
                    = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), difference);
            auto const assembled_host
                    = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), assembled_applied);
            std::array<double, 3> component_difference_norms {};
            std::array<double, 3> component_assembled_norms {};
            for (std::size_t row = 0; row < rhs.extent(0); ++row) {
                std::size_t const component = row % 3;
                component_difference_norms[component]
                        += difference_host(row, 0) * difference_host(row, 0);
                component_assembled_norms[component]
                        += assembled_host(row, 0) * assembled_host(row, 0);
            }
            std::cout << "SimiLie matrix-free apply component comparison:";
            for (std::size_t component = 0; component < 3; ++component) {
                double const component_difference
                        = std::sqrt(component_difference_norms[component]);
                double const component_assembled = std::sqrt(component_assembled_norms[component]);
                std::cout << " component" << component << "_difference_l2=" << component_difference
                          << " component" << component << "_relative_difference="
                          << (component_assembled == 0.0
                                      ? 0.0
                                      : component_difference / component_assembled);
            }
            std::cout << '\n';
        }
    }
    auto solver = solver_factory->generate(system_matrix);
    auto convergence_logger = std::shared_ptr<gko::log::Convergence<double>>(
            gko::log::Convergence<double>::create().release());
    solver->add_logger(convergence_logger);
    std::shared_ptr<SolverProgressLogger> progress_logger;
    if (unsigned int const progress_stride = solver_progress_stride(); progress_stride != 0U) {
        progress_logger = std::make_shared<SolverProgressLogger>(
                gko_exec->get_master(),
                diagnostics.initial_residual_l2,
                progress_stride);
        solver->add_logger(progress_logger);
    }

    fill(exec_space, solution, 0.0);
    auto rhs_gko = to_gko_dense(gko_exec, rhs);
    auto solution_gko = to_gko_dense(gko_exec, solution);
    auto const optimization_start = std::chrono::steady_clock::now();
    solver->apply(rhs_gko.dense, solution_gko.dense);
    gko_exec->synchronize();
    copy_back_from_gko_dense_bridge(solution, solution_gko);
    auto const optimization_end = std::chrono::steady_clock::now();
    diagnostics.duration
            = std::chrono::duration<double>(optimization_end - optimization_start).count();
    if (matrix_free_system_matrix != nullptr) {
        matrix_free_system_matrix->log_timing();
    }
    if (progress_logger != nullptr) {
        solver->remove_logger(progress_logger);
    }
    solver->remove_logger(convergence_logger);
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
    if (env_flag_enabled("SIMILIE_TRUE_RESIDUAL_DIAGNOSTICS")) {
        using memory_space = typename SolutionViewType::memory_space;
        Kokkos::View<double**, memory_space>
                applied("similie_true_residual_applied", rhs.extent(0), rhs.extent(1));
        Kokkos::View<double**, memory_space>
                true_residual("similie_true_residual", rhs.extent(0), rhs.extent(1));
        if (settings.use_matrix_free && matrix_free_system_matrix != nullptr) {
            auto solution_gko_for_residual = to_gko_dense(gko_exec, solution);
            auto applied_gko_for_residual = to_gko_dense(gko_exec, applied);
            matrix_free_system_matrix
                    ->apply(solution_gko_for_residual.dense, applied_gko_for_residual.dense);
            gko_exec->synchronize();
            copy_back_from_gko_dense_bridge(applied, applied_gko_for_residual);
        } else {
            apply_operator(exec_space, operator_model, solution, applied);
        }
        copy(exec_space, true_residual, rhs);
        axpy_inplace(exec_space, true_residual, -1.0, applied);
        double const true_residual_l2 = residual_norm_l2(exec_space, true_residual);
        double const true_relative_residual
                = diagnostics.initial_residual_l2 == 0.0
                          ? 0.0
                          : true_residual_l2 / diagnostics.initial_residual_l2;
        std::cout << "SimiLie true residual diagnostics: residual_l2=" << true_residual_l2
                  << " relative_residual=" << true_relative_residual << '\n';
    }
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
        auto const matrix_build_start = std::chrono::steady_clock::now();
        std::shared_ptr<gko::matrix::Csr<double, gko::int32>> matrix;
        auto const matrix_build_end = std::chrono::steady_clock::now();
        std::shared_ptr<gko::LinOp const> preconditioner;
        PreconditionerType const preconditioner_type = detail::selected_preconditioner(settings);
        if (settings.use_matrix_free && preconditioner_type == PreconditionerType::Identity) {
            std::cout << "SimiLie Ginkgo preconditioner: "
                      << preconditioner_name(preconditioner_type) << '\n';
            preconditioner = detail::build_identity_preconditioner(gko_exec, operator_model.size());
        } else {
            matrix = detail::build_matrix(gko_exec, operator_model);
            auto const actual_matrix_build_end = std::chrono::steady_clock::now();
            preconditioner = detail::build_preconditioner(
                    gko_exec,
                    std::static_pointer_cast<gko::LinOp const>(matrix),
                    settings);
            if (detail::env_flag_enabled("SIMILIE_SOLVER_TIMING")) {
                std::cout << "SimiLie linear setup timing: matrix_build_duration="
                          << std::chrono::duration<double>(
                                     actual_matrix_build_end - matrix_build_start)
                                     .count()
                          << " preconditioner_build_duration="
                          << std::chrono::duration<double>(
                                     std::chrono::steady_clock::now() - actual_matrix_build_end)
                                     .count()
                          << '\n';
            }
            return detail::solve_linearized_system(
                    exec_space,
                    gko_exec,
                    operator_model,
                    rhs,
                    solution,
                    settings,
                    std::static_pointer_cast<gko::LinOp const>(matrix),
                    preconditioner);
        }
        auto const preconditioner_build_end = std::chrono::steady_clock::now();
        if (detail::env_flag_enabled("SIMILIE_SOLVER_TIMING")) {
            std::cout << "SimiLie linear setup timing: matrix_build_duration="
                      << std::chrono::duration<double>(matrix_build_end - matrix_build_start)
                                 .count()
                      << " preconditioner_build_duration="
                      << std::chrono::duration<double>(preconditioner_build_end - matrix_build_end)
                                 .count()
                      << '\n';
        }
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
            detail::log_nonlinear_progress(iteration, diagnostics);
            if (diagnostics.final_relative_residual <= settings.relative_tolerance) {
                diagnostics.converged = true;
                break;
            }

            auto matrix = detail::build_matrix(gko_exec, operator_model, solution);
            auto preconditioner = detail::build_preconditioner(
                    gko_exec,
                    std::static_pointer_cast<gko::LinOp const>(matrix),
                    settings);

            detail::copy(exec_space, correction_rhs, residual);
            if (settings.use_matrix_free) {
                using solver_type = gko::solver::Cg<double>;
                auto residual_criterion
                        = gko::stop::ResidualNorm<double>::build()
                                  .with_reduction_factor(settings.relative_tolerance)
                                  .on(gko_exec);
                auto iterations_criterion = gko::stop::Iteration::build()
                                                    .with_max_iters(settings.max_iterations)
                                                    .on(gko_exec);
                auto solver_factory = solver_type::build()
                                              .with_generated_preconditioner(preconditioner)
                                              .with_criteria(
                                                      std::move(residual_criterion),
                                                      std::move(iterations_criterion))
                                              .on(gko_exec);
                auto operator_model_ptr = std::shared_ptr<
                        OperatorModel const>(&operator_model, [](OperatorModel const*) {});
                auto system_matrix = std::shared_ptr<gko::LinOp const>(
                        std::make_shared<detail::StateDependentMatrixFreeLinOp<
                                ExecSpace,
                                OperatorModel,
                                SolutionViewType>>(
                                gko_exec,
                                exec_space,
                                operator_model_ptr,
                                solution));
                auto solver = solver_factory->generate(system_matrix);
                auto convergence_logger = std::shared_ptr<gko::log::Convergence<double>>(
                        gko::log::Convergence<double>::create().release());
                solver->add_logger(convergence_logger);
                std::shared_ptr<detail::SolverProgressLogger> progress_logger;
                if (unsigned int const progress_stride = detail::solver_progress_stride();
                    progress_stride != 0U) {
                    progress_logger = std::make_shared<detail::SolverProgressLogger>(
                            gko_exec->get_master(),
                            detail::residual_norm_l2(exec_space, correction_rhs),
                            progress_stride);
                    solver->add_logger(progress_logger);
                }
                detail::fill(exec_space, delta, 0.0);
                auto rhs_gko = detail::to_gko_dense(gko_exec, correction_rhs);
                auto delta_gko = detail::to_gko_dense(gko_exec, delta);
                solver->apply(rhs_gko.dense, delta_gko.dense);
                gko_exec->synchronize();
                detail::copy_back_from_gko_dense_bridge(delta, delta_gko);
                if (progress_logger != nullptr) {
                    solver->remove_logger(progress_logger);
                }
                solver->remove_logger(convergence_logger);
                diagnostics.iterations
                        += static_cast<unsigned int>(convergence_logger->get_num_iterations());
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
