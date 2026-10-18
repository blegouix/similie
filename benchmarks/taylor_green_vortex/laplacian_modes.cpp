// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#include "laplacian_modes.hpp"

namespace taylor_green_vortex {
namespace {
template <int Mode = 0, class Tensor>
std::unique_ptr<LaplacianOperator<Tensor>> dispatch(
        int selected,
        Tensor const& output,
        Tensor const& input,
        MetricTensor const& metric,
        PositionTensor const& position)
{
    if (selected == Mode)
        return detail::make_laplacian_mode<Mode>(output, input, metric, position);
    if constexpr (Mode < 10)
        return dispatch<Mode + 1>(selected, output, input, metric, position);
    throw std::invalid_argument("Unknown Taylor-Green Laplacian staging mode");
}
} // namespace

std::unique_ptr<LaplacianOperator<ScalarTensor>> make_scalar_laplacian(
        Configuration const& selected,
        ScalarTensor const& output,
        ScalarTensor const& input,
        MetricTensor const& metric,
        PositionTensor const& position)
{
    return dispatch(selected.laplacian_mode(), output, input, metric, position);
}
std::unique_ptr<LaplacianOperator<TwoFormTensor>> make_two_form_laplacian(
        Configuration const& selected,
        TwoFormTensor const& output,
        TwoFormTensor const& input,
        MetricTensor const& metric,
        PositionTensor const& position)
{
    return dispatch(selected.laplacian_mode(), output, input, metric, position);
}
} // namespace taylor_green_vortex
