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
    virtual void apply(Tensor output, Tensor input) = 0;
};

std::unique_ptr<LaplacianOperator<ScalarTensor>> make_scalar_laplacian(
        Configuration const&,
        ScalarTensor output,
        ScalarTensor input,
        MetricTensor,
        PositionTensor);
std::unique_ptr<LaplacianOperator<TwoFormTensor>> make_two_form_laplacian(
        Configuration const&,
        TwoFormTensor output,
        TwoFormTensor input,
        MetricTensor,
        PositionTensor);

namespace detail {
template <int Mode, class Tensor>
std::unique_ptr<LaplacianOperator<Tensor>> make_laplacian_mode(
        Tensor output,
        Tensor input,
        MetricTensor metric,
        PositionTensor position);
} // namespace detail
} // namespace taylor_green_vortex
