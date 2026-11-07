// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#pragma once

#include <memory>

#include <similie/exterior/cochain.hpp>

#include "configuration.hpp"
#include "grid_types.hpp"

namespace taylor_green_vortex {
// Host dispatch selects a compiled operator type once, before solver iterations.
template <class Tensor>
class LaplacianOperator
{
public:
    virtual ~LaplacianOperator() = default;
    virtual void apply(Tensor const& output, Tensor const& input) = 0;
};

namespace detail {
std::unique_ptr<LaplacianOperator<sil::exterior::Cochain<ScalarTensor>>>
make_scalar_laplacian_modes_0_7(
        int,
        sil::exterior::Cochain<ScalarTensor> const&,
        sil::exterior::Cochain<ScalarTensor> const&,
        MetricTensor const&,
        PositionTensor const&);
std::unique_ptr<LaplacianOperator<sil::exterior::Cochain<ScalarTensor>>>
make_scalar_laplacian_modes_8_15(
        int,
        sil::exterior::Cochain<ScalarTensor> const&,
        sil::exterior::Cochain<ScalarTensor> const&,
        MetricTensor const&,
        PositionTensor const&);
std::unique_ptr<LaplacianOperator<sil::exterior::Cochain<ScalarTensor>>>
make_scalar_laplacian_modes_16_23(
        int,
        sil::exterior::Cochain<ScalarTensor> const&,
        sil::exterior::Cochain<ScalarTensor> const&,
        MetricTensor const&,
        PositionTensor const&);
std::unique_ptr<LaplacianOperator<sil::exterior::Cochain<ScalarTensor>>>
make_scalar_laplacian_modes_24_31(
        int,
        sil::exterior::Cochain<ScalarTensor> const&,
        sil::exterior::Cochain<ScalarTensor> const&,
        MetricTensor const&,
        PositionTensor const&);
std::unique_ptr<LaplacianOperator<sil::exterior::Cochain<ScalarTensor>>>
make_scalar_laplacian_modes_32_38(
        int,
        sil::exterior::Cochain<ScalarTensor> const&,
        sil::exterior::Cochain<ScalarTensor> const&,
        MetricTensor const&,
        PositionTensor const&);

std::unique_ptr<LaplacianOperator<sil::exterior::Cochain<TwoFormTensor>>>
make_two_form_laplacian_modes_0_7(
        int,
        sil::exterior::Cochain<TwoFormTensor> const&,
        sil::exterior::Cochain<TwoFormTensor> const&,
        MetricTensor const&,
        PositionTensor const&);
std::unique_ptr<LaplacianOperator<sil::exterior::Cochain<TwoFormTensor>>>
make_two_form_laplacian_modes_8_15(
        int,
        sil::exterior::Cochain<TwoFormTensor> const&,
        sil::exterior::Cochain<TwoFormTensor> const&,
        MetricTensor const&,
        PositionTensor const&);
std::unique_ptr<LaplacianOperator<sil::exterior::Cochain<TwoFormTensor>>>
make_two_form_laplacian_modes_16_23(
        int,
        sil::exterior::Cochain<TwoFormTensor> const&,
        sil::exterior::Cochain<TwoFormTensor> const&,
        MetricTensor const&,
        PositionTensor const&);
std::unique_ptr<LaplacianOperator<sil::exterior::Cochain<TwoFormTensor>>>
make_two_form_laplacian_modes_24_31(
        int,
        sil::exterior::Cochain<TwoFormTensor> const&,
        sil::exterior::Cochain<TwoFormTensor> const&,
        MetricTensor const&,
        PositionTensor const&);
std::unique_ptr<LaplacianOperator<sil::exterior::Cochain<TwoFormTensor>>>
make_two_form_laplacian_modes_32_38(
        int,
        sil::exterior::Cochain<TwoFormTensor> const&,
        sil::exterior::Cochain<TwoFormTensor> const&,
        MetricTensor const&,
        PositionTensor const&);
} // namespace detail

std::unique_ptr<LaplacianOperator<sil::exterior::Cochain<ScalarTensor>>> make_scalar_laplacian(
        Configuration const& selected,
        sil::exterior::Cochain<ScalarTensor> const& output,
        sil::exterior::Cochain<ScalarTensor> const& input,
        MetricTensor const& metric,
        PositionTensor const& position);
std::unique_ptr<LaplacianOperator<sil::exterior::Cochain<TwoFormTensor>>> make_two_form_laplacian(
        Configuration const& selected,
        sil::exterior::Cochain<TwoFormTensor> const& output,
        sil::exterior::Cochain<TwoFormTensor> const& input,
        MetricTensor const& metric,
        PositionTensor const& position);
} // namespace taylor_green_vortex
