// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#include <stdexcept>

#include "laplacian_mode.hpp"

namespace taylor_green_vortex {

std::unique_ptr<LaplacianOperator<ScalarTensor>> make_scalar_laplacian(
        Configuration const& selected,
        ScalarTensor const& output,
        ScalarTensor const& input,
        MetricTensor const& metric,
        PositionTensor const& position)
{
    int const mode = selected.laplacian_mode();
    if (mode >= 0 && mode <= 7)
        return detail::make_scalar_laplacian_modes_0_7(mode, output, input, metric, position);
    if (mode <= 15 && mode >= 8)
        return detail::make_scalar_laplacian_modes_8_15(mode, output, input, metric, position);
    if (mode <= 23 && mode >= 16)
        return detail::make_scalar_laplacian_modes_16_23(mode, output, input, metric, position);
    if (mode <= 31 && mode >= 24)
        return detail::make_scalar_laplacian_modes_24_31(mode, output, input, metric, position);
    if (mode <= 38 && mode >= 32)
        return detail::make_scalar_laplacian_modes_32_38(mode, output, input, metric, position);
    throw std::invalid_argument("Unknown Taylor-Green Laplacian staging mode");
}

std::unique_ptr<LaplacianOperator<TwoFormTensor>> make_two_form_laplacian(
        Configuration const& selected,
        TwoFormTensor const& output,
        TwoFormTensor const& input,
        MetricTensor const& metric,
        PositionTensor const& position)
{
    int const mode = selected.laplacian_mode();
    if (mode >= 0 && mode <= 7)
        return detail::make_two_form_laplacian_modes_0_7(mode, output, input, metric, position);
    if (mode <= 15 && mode >= 8)
        return detail::make_two_form_laplacian_modes_8_15(mode, output, input, metric, position);
    if (mode <= 23 && mode >= 16)
        return detail::make_two_form_laplacian_modes_16_23(mode, output, input, metric, position);
    if (mode <= 31 && mode >= 24)
        return detail::make_two_form_laplacian_modes_24_31(mode, output, input, metric, position);
    if (mode <= 38 && mode >= 32)
        return detail::make_two_form_laplacian_modes_32_38(mode, output, input, metric, position);
    throw std::invalid_argument("Unknown Taylor-Green Laplacian staging mode");
}

} // namespace taylor_green_vortex
