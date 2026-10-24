// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#pragma once

#include <memory>

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
std::unique_ptr<LaplacianOperator<ScalarTensor>> make_scalar_laplacian_modes_0_7(
        int,
        ScalarTensor const&,
        ScalarTensor const&,
        MetricTensor const&,
        PositionTensor const&);
std::unique_ptr<LaplacianOperator<ScalarTensor>> make_scalar_laplacian_modes_8_15(
        int,
        ScalarTensor const&,
        ScalarTensor const&,
        MetricTensor const&,
        PositionTensor const&);
std::unique_ptr<LaplacianOperator<ScalarTensor>> make_scalar_laplacian_modes_16_23(
        int,
        ScalarTensor const&,
        ScalarTensor const&,
        MetricTensor const&,
        PositionTensor const&);
std::unique_ptr<LaplacianOperator<ScalarTensor>> make_scalar_laplacian_modes_24_31(
        int,
        ScalarTensor const&,
        ScalarTensor const&,
        MetricTensor const&,
        PositionTensor const&);
std::unique_ptr<LaplacianOperator<ScalarTensor>> make_scalar_laplacian_modes_32_38(
        int,
        ScalarTensor const&,
        ScalarTensor const&,
        MetricTensor const&,
        PositionTensor const&);

std::unique_ptr<LaplacianOperator<TwoFormTensor>> make_two_form_laplacian_modes_0_7(
        int,
        TwoFormTensor const&,
        TwoFormTensor const&,
        MetricTensor const&,
        PositionTensor const&);
std::unique_ptr<LaplacianOperator<TwoFormTensor>> make_two_form_laplacian_modes_8_15(
        int,
        TwoFormTensor const&,
        TwoFormTensor const&,
        MetricTensor const&,
        PositionTensor const&);
std::unique_ptr<LaplacianOperator<TwoFormTensor>> make_two_form_laplacian_modes_16_23(
        int,
        TwoFormTensor const&,
        TwoFormTensor const&,
        MetricTensor const&,
        PositionTensor const&);
std::unique_ptr<LaplacianOperator<TwoFormTensor>> make_two_form_laplacian_modes_24_31(
        int,
        TwoFormTensor const&,
        TwoFormTensor const&,
        MetricTensor const&,
        PositionTensor const&);
std::unique_ptr<LaplacianOperator<TwoFormTensor>> make_two_form_laplacian_modes_32_38(
        int,
        TwoFormTensor const&,
        TwoFormTensor const&,
        MetricTensor const&,
        PositionTensor const&);
} // namespace detail

std::unique_ptr<LaplacianOperator<ScalarTensor>> make_scalar_laplacian(
        Configuration const& selected,
        ScalarTensor const& output,
        ScalarTensor const& input,
        MetricTensor const& metric,
        PositionTensor const& position);
std::unique_ptr<LaplacianOperator<TwoFormTensor>> make_two_form_laplacian(
        Configuration const& selected,
        TwoFormTensor const& output,
        TwoFormTensor const& input,
        MetricTensor const& metric,
        PositionTensor const& position);
} // namespace taylor_green_vortex
