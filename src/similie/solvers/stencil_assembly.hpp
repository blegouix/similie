// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#pragma once

#include <limits>

#include <similie/tensor/owning_tensor.hpp>

#include "minimize_strong_formulation_residual.hpp"

namespace similie::solvers {
struct AssembledLinearSystem
{
    std::shared_ptr<gko::matrix::Csr<double, gko::int32>> matrix;
    Kokkos::View<double**> rhs;
    Kokkos::View<double*> row_scales;
};
} // namespace similie::solvers

namespace similie::solvers::detail {
struct StencilColumn
{
};

template <std::size_t Capacity>
KOKKOS_FUNCTION auto make_row_stencil()
{
    std::array<double, Capacity> storage {};
    ddc::DiscreteDomain<StencilColumn> const
            domain(ddc::DiscreteElement<StencilColumn>(0),
                   ddc::DiscreteVector<StencilColumn>(Capacity));
    sil::tensor::Tensor<
            double,
            ddc::DiscreteDomain<StencilColumn>,
            Kokkos::layout_right,
            Kokkos::DefaultExecutionSpace::memory_space>
            tensor(ddc::ChunkSpan<
                    double,
                    ddc::DiscreteDomain<StencilColumn>,
                    Kokkos::layout_right,
                    Kokkos::DefaultExecutionSpace::memory_space>(storage.data(), domain));
    return sil::tensor::OwningTensor(tensor, std::move(storage));
}

template <class Weights, class Indices>
KOKKOS_FUNCTION bool add_stencil_coefficient(
        Weights const& weights,
        Indices indices,
        std::size_t& count,
        std::size_t column,
        double weight)
{
    if (weight == 0.0)
        return true;
    std::size_t i = 0;
    for (; i < count; ++i)
        if (indices[i] == column)
            break;
    if (i == count) {
        if (count == weights.domain().size())
            return false;
        indices[count++] = column;
    }
    weights.mem(ddc::DiscreteElement<StencilColumn>(i)) += weight;
    return true;
}

// Native tensor coefficients and their global column mapping are kept apart.
// RHS-only assembly allocates neither coefficient rows nor sparse matrix arrays.
template <std::size_t Capacity>
struct RowAssembly
{
    static_assert(Capacity > 0);
    std::size_t column_count;
    Kokkos::View<std::size_t**, Kokkos::LayoutRight> columns;
    Kokkos::View<double**, Kokkos::LayoutRight> weights;
    Kokkos::View<std::size_t*> counts;
    Kokkos::View<double*> constants, lower, upper;
    Kokkos::View<int> errors;

    explicit RowAssembly(std::size_t size, bool matrix = true, std::size_t input_size = 0)
        : column_count(input_size == 0 ? size : input_size)
        , columns("dec_columns", matrix ? size : 0, Capacity)
        , weights("dec_weights", matrix ? size : 0, Capacity)
        , counts("dec_counts", matrix ? size : 0)
        , constants("dec_constants", size)
        , lower("dirichlet_lower", size)
        , upper("dirichlet_upper", size)
        , errors("dec_errors")
    {
        if (size == 0 || size > std::size_t(std::numeric_limits<int>::max()) / Capacity
            || column_count > std::size_t(std::numeric_limits<int>::max()))
            throw std::invalid_argument("differential matrix exceeds 32-bit CSR capacity");
        Kokkos::deep_copy(lower, std::numeric_limits<double>::infinity());
        Kokkos::deep_copy(upper, -std::numeric_limits<double>::infinity());
    }

    KOKKOS_FUNCTION void fail(int code) const
    {
        Kokkos::atomic_fetch_or(&errors(), code);
    }

    KOKKOS_FUNCTION auto coefficients(std::size_t row) const
    {
        ddc::DiscreteDomain<StencilColumn> const
                domain(ddc::DiscreteElement<StencilColumn>(0),
                       ddc::DiscreteVector<StencilColumn>(Capacity));
        return sil::tensor::Tensor<
                double,
                ddc::DiscreteDomain<StencilColumn>,
                Kokkos::layout_right,
                Kokkos::DefaultExecutionSpace::memory_space>(
                ddc::ChunkSpan<
                        double,
                        ddc::DiscreteDomain<StencilColumn>,
                        Kokkos::layout_right,
                        Kokkos::DefaultExecutionSpace::memory_space>(&weights(row, 0), domain));
    }

    KOKKOS_FUNCTION void add(std::size_t row, std::size_t column, double weight) const
    {
        if (column >= column_count) {
            fail(4);
            return;
        }
        if (!add_stencil_coefficient(
                    coefficients(row),
                    &columns(row, 0),
                    counts(row),
                    column,
                    weight))
            fail(2);
    }

    KOKKOS_FUNCTION void prescribe(std::size_t index, double value) const
    {
        if (index >= constants.extent(0) || index >= column_count || !Kokkos::isfinite(value)) {
            fail(8);
            return;
        }
        Kokkos::atomic_min(&lower(index), value);
        Kokkos::atomic_max(&upper(index), value);
    }

    void check() const
    {
        int error = 0;
        Kokkos::deep_copy(error, errors);
        if (error & 2)
            throw std::runtime_error("differential stencil exceeds its column capacity");
        if (error & 8)
            throw std::runtime_error("incompatible or invalid Dirichlet constraints");
        if (error)
            throw std::runtime_error("invalid or empty differential row");
    }

    Kokkos::View<double**> finalize_rhs(
            Kokkos::DefaultExecutionSpace const& exec,
            Kokkos::View<double*> scales) const
    {
        check();
        RowAssembly const data = *this;
        Kokkos::View<double**> rhs("dec_rhs", constants.extent(0), 1);
        Kokkos::parallel_for(
                "finish_dec_rhs",
                Kokkos::RangePolicy(exec, 0, constants.extent(0)),
                KOKKOS_LAMBDA(std::size_t row) {
                    if (Kokkos::isfinite(data.lower(row))) {
                        if (Kokkos::abs(data.upper(row) - data.lower(row))
                            > 1.e-12 * (1.0 + Kokkos::abs(data.lower(row))))
                            data.fail(8);
                        data.constants(row) = -data.lower(row);
                    }
                    if (!Kokkos::isfinite(data.constants(row)))
                        data.fail(4);
                    rhs(row, 0) = -data.constants(row) / scales(row);
                });
        exec.fence("scalar DEC rhs ready");
        check();
        return rhs;
    }

    AssembledLinearSystem finalize(
            Kokkos::DefaultExecutionSpace const& exec,
            bool normalize,
            bool allow_empty_rows = false) const
    {
        check();
        RowAssembly const data = *this;
        Kokkos::View<int*> offsets("dec_offsets", constants.extent(0) + 1);
        Kokkos::View<double*> scales("dec_row_scales", constants.extent(0));
        Kokkos::parallel_for(
                "finish_dec_rows",
                Kokkos::RangePolicy(exec, 0, constants.extent(0)),
                KOKKOS_LAMBDA(std::size_t row) {
                    if (Kokkos::isfinite(data.lower(row))) {
                        data.counts(row) = 1;
                        data.columns(row, 0) = row;
                        data.weights(row, 0) = 1.0;
                    }
                    double scale = 0.0;
                    std::size_t count = 0;
                    for (std::size_t i = 0; i < data.counts(row); ++i) {
                        double const weight = data.weights(row, i);
                        if (!Kokkos::isfinite(weight))
                            data.fail(4);
                        if (weight != 0.0) {
                            data.columns(row, count) = data.columns(row, i);
                            data.weights(row, count++) = weight;
                            scale = Kokkos::max(scale, Kokkos::abs(weight));
                        }
                    }
                    data.counts(row) = count;
                    if (!(scale > 0.0)) {
                        if (allow_empty_rows)
                            scale = 1.0;
                        else
                            data.fail(4);
                    }
                    scales(row) = normalize ? scale : 1.0;
                    for (std::size_t i = 0; i < count; ++i)
                        data.weights(row, i) /= scales(row);
                });
        int total = 0;
        Kokkos::parallel_scan(
                "scan_dec_rows",
                Kokkos::RangePolicy(exec, 0, constants.extent(0) + 1),
                KOKKOS_LAMBDA(std::size_t row, int& sum, bool final) {
                    if (final)
                        offsets(row) = sum;
                    if (row < data.constants.extent(0))
                        sum += static_cast<int>(data.counts(row));
                },
                total);
        check();
        Kokkos::View<int*> csr_columns("dec_csr_columns", total);
        Kokkos::View<double*> values("dec_values", total);
        Kokkos::parallel_for(
                "pack_dec_rows",
                Kokkos::RangePolicy(exec, 0, constants.extent(0)),
                KOKKOS_LAMBDA(std::size_t row) {
                    for (std::size_t i = 0; i < data.counts(row); ++i) {
                        csr_columns(offsets(row) + i) = static_cast<int>(data.columns(row, i));
                        values(offsets(row) + i) = data.weights(row, i);
                    }
                });
        Kokkos::View<double**> const rhs = finalize_rhs(exec, scales);
        auto const executor = gko::ext::kokkos::create_executor(exec);
        gko::array<double> matrix_values(executor, values.extent(0));
        gko::array<gko::int32> matrix_columns(executor, csr_columns.extent(0)),
                matrix_offsets(executor, offsets.extent(0));
        executor->copy_from(
                executor.get(),
                values.extent(0),
                values.data(),
                matrix_values.get_data());
        executor->copy_from(
                executor.get(),
                csr_columns.extent(0),
                csr_columns.data(),
                matrix_columns.get_data());
        executor->copy_from(
                executor.get(),
                offsets.extent(0),
                offsets.data(),
                matrix_offsets.get_data());
        std::shared_ptr<gko::matrix::Csr<double, gko::int32>> matrix
                = gko::matrix::Csr<double, gko::int32>::
                        create(executor,
                               gko::dim<2>(constants.extent(0), column_count),
                               std::move(matrix_values),
                               std::move(matrix_columns),
                               std::move(matrix_offsets));
        matrix->sort_by_column_index();
        return {matrix, rhs, scales};
    }
};
} // namespace similie::solvers::detail
