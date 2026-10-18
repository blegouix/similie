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

std::unique_ptr<LaplacianOperator<ScalarTensor>> make_scalar_laplacian(
        Configuration const&,
        ScalarTensor const& output,
        ScalarTensor const& input,
        MetricTensor const&,
        PositionTensor const&);
std::unique_ptr<LaplacianOperator<TwoFormTensor>> make_two_form_laplacian(
        Configuration const&,
        TwoFormTensor const& output,
        TwoFormTensor const& input,
        MetricTensor const&,
        PositionTensor const&);

namespace detail {
template <int Mode, class Tensor>
std::unique_ptr<LaplacianOperator<Tensor>> make_laplacian_mode(
        Tensor const& output,
        Tensor const& input,
        MetricTensor const& metric,
        PositionTensor const& position);
} // namespace detail
} // namespace taylor_green_vortex
