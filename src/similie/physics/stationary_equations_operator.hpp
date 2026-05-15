// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: MIT

#pragma once

#include <utility>

namespace similie::physics {

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
    // For assembled solves and preconditioner setup, we intentionally delegate to
    // the wrapped prediscretized operator model.
    return assemble_matrix_data(operator_model.operator_model());
}

template <class Equations, class OperatorModel>
constexpr auto make_stationary_equations_operator(Equations equations, OperatorModel operator_model)
{
    return StationaryEquationsOperator<
            Equations,
            OperatorModel>(std::move(equations), std::move(operator_model));
}

} // namespace similie::physics
