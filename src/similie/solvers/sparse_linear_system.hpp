// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#pragma once

#include <cmath>
#include <stdexcept>

#include "minimize_strong_formulation_residual.hpp"

namespace similie::solvers {
/** Sparse scalar operator shared by assembled and matrix-free Ginkgo solves.
 * \important This operator and documentation is fully AI-generated.
 * Stores the device CSR arrays assembled from DEC stencils.
 * apply() and create_matrix() use the same coefficients;
 * assemble_matrix_data() exports them to the host only for diagnostics.
 * Nodal row constraints may break symmetry, so no symmetry assumption is made.
 */
class SparseLinearOperator
{
    Kokkos::View<int*> m_offsets;
    Kokkos::View<int*> m_columns;
    Kokkos::View<double*> m_values;
    std::size_t m_size;

public:
    static constexpr bool IS_LINEAR = true;
    static constexpr bool IS_SYMMETRIC = false;

    /** Adopt CSR arrays assembled in the default memory space, without host packing. */
    SparseLinearOperator(
            Kokkos::View<int*> offsets,
            Kokkos::View<int*> columns,
            Kokkos::View<double*> values)
        : m_offsets(offsets)
        , m_columns(columns)
        , m_values(values)
        , m_size(offsets.extent(0) - 1)
    {
        if (offsets.extent(0) == 0 || columns.extent(0) != values.extent(0))
            throw std::invalid_argument("inconsistent CSR array extents");
    }

    /** Transfer CSR directly to the solver executor; no host matrix_data is built. */
    std::shared_ptr<gko::matrix::Csr<double, gko::int32>> create_matrix(
            std::shared_ptr<gko::Executor const> const& executor) const
    {
        auto const source = gko::ext::kokkos::create_executor(Kokkos::DefaultExecutionSpace {});
        gko::array<double> values(executor, m_values.extent(0));
        gko::array<gko::int32> columns(executor, m_columns.extent(0)),
                offsets(executor, m_offsets.extent(0));
        executor->copy_from(source.get(), m_values.extent(0), m_values.data(), values.get_data());
        executor->copy_from(
                source.get(),
                m_columns.extent(0),
                m_columns.data(),
                columns.get_data());
        executor->copy_from(
                source.get(),
                m_offsets.extent(0),
                m_offsets.data(),
                offsets.get_data());
        std::shared_ptr<gko::matrix::Csr<double, gko::int32>> matrix(
                gko::matrix::Csr<double, gko::int32>::
                        create(executor,
                               gko::dim<2>(m_size, m_size),
                               std::move(values),
                               std::move(columns),
                               std::move(offsets))
                                .release());
        matrix->sort_by_column_index();
        return matrix;
    }

    [[nodiscard]] std::size_t size() const
    {
        return m_size;
    }

    template <class ExecSpace, class InputView, class OutputView>
    void apply(ExecSpace exec_space, InputView input, OutputView output) const
    {
        auto const offsets = m_offsets;
        auto const columns = m_columns;
        auto const values = m_values;
        Kokkos::parallel_for(
                "similie_sparse_operator",
                Kokkos::RangePolicy<ExecSpace>(exec_space, 0, m_size),
                KOKKOS_LAMBDA(std::size_t row) {
                    double sum = 0.0;
                    for (int slot = offsets(row); slot < offsets(row + 1); ++slot)
                        sum += values(slot) * input(columns(slot), 0);
                    output(row, 0) = sum;
                });
        exec_space.fence();
    }

    friend gko::matrix_data<double, gko::int32> assemble_matrix_data(
            SparseLinearOperator const& matrix)
    {
        gko::matrix_data<double, gko::int32> data(gko::dim<2>(matrix.m_size, matrix.m_size));
        auto const offsets
                = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), matrix.m_offsets);
        auto const columns
                = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), matrix.m_columns);
        auto const values
                = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), matrix.m_values);
        for (std::size_t row = 0; row < matrix.m_size; ++row)
            for (int slot = offsets(row); slot < offsets(row + 1); ++slot)
                data.nonzeros
                        .emplace_back(static_cast<gko::int32>(row), columns(slot), values(slot));
        return data;
    }
};

template <class RhsView>
    requires Kokkos::is_view<RhsView>::value
Kokkos::View<double**> solve_linear_system(
        SparseLinearOperator const& matrix,
        RhsView rhs,
        StrongFormulationSolverSettings const& settings,
        StrongFormulationSolverDiagnostics& diagnostics)
{
    if (rhs.extent(0) != matrix.size() || rhs.extent(1) != 1)
        throw std::invalid_argument("linear system RHS shape does not match operator");
    Kokkos::View<double**> solution("similie_sparse_solution", matrix.size(), 1);
    diagnostics = minimize_strong_formulation_residual(
            Kokkos::DefaultExecutionSpace {},
            matrix,
            rhs,
            solution,
            settings);
    if (!std::isfinite(diagnostics.final_relative_residual)
        || diagnostics.final_relative_residual > settings.relative_tolerance * 10.0)
        throw std::runtime_error("scalar linear solve did not converge");
    return solution;
}

struct ScalarLinearSystem
{
    SparseLinearOperator matrix;
    Kokkos::View<double**> rhs;
};
} // namespace similie::solvers
