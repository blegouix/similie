// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#pragma once

#include <chrono>
#include <iostream>
#include <memory>

#include <similie/solvers/minimize_strong_formulation_residual.hpp>

#include <Kokkos_Core.hpp>

namespace taylor_green_vortex {
/** Reusable Ginkgo preconditioning for the periodic scalar and vector solves.
 * The seven-point Laplacian plus a unit shift is an SPD approximation used only
 * for preconditioning. The outer Krylov method still applies the native DEC
 * operator and checks its residual. The shift makes the coarse systems invertible
 * without changing the physical equation or its constant-potential gauge.
 * \important This operator and documentation is fully AI-generated.
 */
class PoissonSolver
{
    int const m_n, m_components;
    double const m_h;
    std::shared_ptr<gko::LinOp const> m_preconditioner;
    similie::solvers::PreconditionerType m_type = similie::solvers::PreconditionerType::Identity;

public:
    PoissonSolver(int n, int components, double h) : m_n(n), m_components(components), m_h(h) {}

    similie::solvers::StrongFormulationSolverSettings settings()
    {
        similie::solvers::StrongFormulationSolverSettings settings;
        // The CI meshes benefit from sparse polynomial preconditioning;
        // tiny validation meshes converge faster without its extra kernels.
        settings.preconditioner = m_n >= 64 ? similie::solvers::PreconditionerType::ChebyshevJacobi
                                            : similie::solvers::PreconditionerType::Identity;
        settings.use_initial_guess
                = !similie::solvers::detail::env_value_equals("SIMILIE_TGV_WARM_START", "0");
        settings.relative_tolerance = 1e-11;
        settings.max_iterations = 8 * m_n;
        // Exact spectral bounds for the Jacobi-scaled, shifted surrogate.
        double const diagonal = 6 / (m_h * m_h) + 1;
        settings.chebyshev_lower_bound = 1 / diagonal;
        settings.chebyshev_upper_bound = (12 / (m_h * m_h) + 1) / diagonal;
        settings.chebyshev_iterations = static_cast<unsigned int>(
                std::
                        max(1,
                            similie::solvers::detail::
                                    env_int_or("SIMILIE_CHEBYSHEV_ITERATIONS", 8)));
        similie::solvers::PreconditionerType const type
                = similie::solvers::detail::selected_preconditioner(settings);
        if (!m_preconditioner || type != m_type) {
            std::chrono::steady_clock::time_point const start = std::chrono::steady_clock::now();
            Kokkos::DefaultExecutionSpace exec;
            std::shared_ptr<gko::Executor> const executor = gko::ext::kokkos::create_executor(exec);
            int const n = m_n, cells = n * n * n, rows = cells * m_components;
            if (type == similie::solvers::PreconditionerType::Identity) {
                m_preconditioner
                        = similie::solvers::detail::build_identity_preconditioner(executor, rows);
            } else {
                std::shared_ptr<gko::matrix::Csr<double, gko::int32>> matrix
                        = gko::matrix::Csr<double, gko::int32>::
                                create(executor, gko::dim<2>(rows, rows), 7 * std::size_t(rows));
                double* const values = matrix->get_values();
                gko::int32* const columns = matrix->get_col_idxs();
                gko::int32* const pointers = matrix->get_row_ptrs();
                double const scale = 1 / (m_h * m_h);
                Kokkos::parallel_for(
                        "periodic_poisson_preconditioner",
                        Kokkos::RangePolicy<>(exec, 0, rows + 1),
                        KOKKOS_LAMBDA(int row) {
                            pointers[row] = 7 * row;
                            if (row == rows)
                                return;
                            int const i = row % cells, component = row / cells;
                            values[7 * row] = 6 * scale + 1;
                            columns[7 * row] = row;
                            for (int axis = 0; axis < 3; ++axis) {
                                int const stride = axis == 0 ? n * n : (axis == 1 ? n : 1);
                                int const coordinate = (i / stride) % n;
                                for (int side = 0; side < 2; ++side) {
                                    int const offset = 1 + 2 * axis + side;
                                    int const neighbor = i
                                                         + (((coordinate + (side ? 1 : n - 1)) % n)
                                                            - coordinate)
                                                                   * stride;
                                    values[7 * row + offset] = -scale;
                                    columns[7 * row + offset] = component * cells + neighbor;
                                }
                            }
                        });
                exec.fence();
                matrix->sort_by_column_index();
                matrix->set_strategy(matrix->get_strategy());
                m_preconditioner = similie::solvers::detail::
                        build_preconditioner(executor, matrix, settings);
            }
            executor->synchronize();
            m_type = type;
            char const* const algorithm = std::getenv("SIMILIE_SOLVER");
            std::cout << "TGV Poisson solver: N=" << n << " components=" << m_components
                      << " algorithm=" << (algorithm ? algorithm : "cg")
                      << " preconditioner=" << similie::solvers::preconditioner_name(type)
                      << " warm_start=" << settings.use_initial_guess
                      << " relative_tolerance=" << settings.relative_tolerance
                      << " max_iterations=" << settings.max_iterations
                      << " chebyshev_iterations=" << settings.chebyshev_iterations
                      << " preconditioner_setup_seconds="
                      << std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
                                 .count()
                      << std::endl;
        }
        settings.generated_preconditioner = m_preconditioner;
        return settings;
    }
};
} // namespace taylor_green_vortex
