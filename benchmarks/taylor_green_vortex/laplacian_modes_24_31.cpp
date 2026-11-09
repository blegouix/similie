// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#include "laplacian_mode_impl.hpp"

namespace taylor_green_vortex::detail {

std::unique_ptr<LaplacianOperator<sil::exterior::Cochain<ScalarTensor>>>
make_scalar_laplacian_modes_24_31(
        int selected,
        sil::exterior::Cochain<ScalarTensor> const& output,
        sil::exterior::Cochain<ScalarTensor> const& input,
        MetricTensor const& metric,
        PositionTensor const& position)
{
    return make_laplacian_mode_in_range<24, 31>(selected, output, input, metric, position);
}

std::unique_ptr<LaplacianOperator<sil::exterior::Cochain<TwoFormTensor>>>
make_two_form_laplacian_modes_24_31(
        int selected,
        sil::exterior::Cochain<TwoFormTensor> const& output,
        sil::exterior::Cochain<TwoFormTensor> const& input,
        MetricTensor const& metric,
        PositionTensor const& position)
{
    return make_laplacian_mode_in_range<24, 31>(selected, output, input, metric, position);
}

} // namespace taylor_green_vortex::detail
