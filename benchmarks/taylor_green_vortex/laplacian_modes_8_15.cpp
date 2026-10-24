// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#include "laplacian_mode_impl.hpp"

namespace taylor_green_vortex::detail {

std::unique_ptr<LaplacianOperator<ScalarTensor>> make_scalar_laplacian_modes_8_15(
        int selected,
        ScalarTensor const& output,
        ScalarTensor const& input,
        MetricTensor const& metric,
        PositionTensor const& position)
{
    return make_laplacian_mode_in_range<8, 15>(selected, output, input, metric, position);
}

std::unique_ptr<LaplacianOperator<TwoFormTensor>> make_two_form_laplacian_modes_8_15(
        int selected,
        TwoFormTensor const& output,
        TwoFormTensor const& input,
        MetricTensor const& metric,
        PositionTensor const& position)
{
    return make_laplacian_mode_in_range<8, 15>(selected, output, input, metric, position);
}

} // namespace taylor_green_vortex::detail
