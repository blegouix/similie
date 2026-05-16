// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: MIT

#pragma once

#include <cassert>
#include <utility>

#include <ginkgo/core/base/matrix_data.hpp>
#include <ginkgo/core/base/types.hpp>

#include <Kokkos_Core.hpp>

namespace similie::physics {

namespace detail {

struct CanonicalBasisInputView
{
    std::size_t active_row;

    KOKKOS_INLINE_FUNCTION double operator()(std::size_t row, [[maybe_unused]] std::size_t column) const
    {
        return row == active_row ? 1.0 : 0.0;
    }
};

struct SingleRowOutputView
{
    std::size_t active_row;
    double value = 0.0;

    KOKKOS_INLINE_FUNCTION double& operator()(std::size_t row, [[maybe_unused]] std::size_t column)
    {
        assert(row == active_row);
        return value;
    }
};

template <class Operator>
gko::matrix_data<double, gko::int32> assemble_matrix_data_from_operator_action(
        Operator const& operator_model)
{
    gko::matrix_data<double, gko::int32> matrix_data(
            gko::dim<2>(operator_model.size(), operator_model.size()));

    for (std::size_t row = 0; row < operator_model.size(); ++row) {
        operator_model.operator_model().for_each_nonzero_column(row, [&](std::size_t column) {
            SingleRowOutputView output {.active_row = row};
            CanonicalBasisInputView input {.active_row = column};
            operator_model.apply_at(output, input, row);
            if (output.value != 0.0) {
                matrix_data.nonzeros.emplace_back(
                        static_cast<gko::int32>(row),
                        static_cast<gko::int32>(column),
                        output.value);
            }
        });
    }

    matrix_data.sort_row_major();
    return matrix_data;
}

} // namespace detail

template <class Equations, class OperatorModel>
class StationaryEquationsOperator
{
    Equations m_equations;
    OperatorModel m_operator_model;

public:
    constexpr StationaryEquationsOperator(Equations equations, OperatorModel operator_model)
        : m_equations(std::move(equations))
        , m_operator_model(std::move(operator_model))
    {
    }

    [[nodiscard]] constexpr Equations const& equations() const
    {
        return m_equations;
    }

    [[nodiscard]] constexpr OperatorModel const& operator_model() const
    {
        return m_operator_model;
    }

    [[nodiscard]] constexpr std::size_t size() const
    {
        return m_operator_model.size();
    }

    template <class ColumnIndex>
    [[nodiscard]] KOKKOS_INLINE_FUNCTION double value(std::size_t row, ColumnIndex column) const
    {
        detail::SingleRowOutputView output {.active_row = row};
        detail::CanonicalBasisInputView input {.active_row = static_cast<std::size_t>(column)};
        apply_at(output, input, row);
        return output.value;
    }

    template <class InputView, class OutputView>
    KOKKOS_INLINE_FUNCTION void apply_at(OutputView output, InputView input, std::size_t row) const
    {
        // Matrix-free applications may override the wrapped operator by exposing
        // an equations-aware free function through ADL. This lets the iterative
        // solve follow the physics equations directly, while the wrapped operator
        // still provides the assembled/preconditioning path when needed.
        if constexpr (requires {
                          apply_stationary_equations_at(
                                  output,
                                  input,
                                  row,
                                  m_equations,
                                  m_operator_model);
                      }) {
            apply_stationary_equations_at(output, input, row, m_equations, m_operator_model);
        } else {
            m_operator_model.apply_at(output, input, row);
        }
    }
};

template <class Equations, class OperatorModel>
auto assemble_matrix_data(
        StationaryEquationsOperator<Equations, OperatorModel> const& operator_model)
{
    if constexpr (requires {
                      operator_model.operator_model().for_each_nonzero_column(
                              std::size_t {},
                              [](std::size_t) {});
                  }) {
        return detail::assemble_matrix_data_from_operator_action(operator_model);
    } else {
        return assemble_matrix_data(operator_model.operator_model());
    }
}

template <class Equations, class OperatorModel>
constexpr auto make_stationary_equations_operator(Equations equations, OperatorModel operator_model)
{
    return StationaryEquationsOperator<
            Equations,
            OperatorModel>(std::move(equations), std::move(operator_model));
}

} // namespace similie::physics
