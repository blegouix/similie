// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: MIT

#pragma once

#include <similie/misc/clamp_to_domain.hpp>
#include <similie/misc/domain_contains.hpp>

#include <Kokkos_Core.hpp>

namespace sil {

namespace exterior {

namespace detail {

struct IdentityStencilEvaluator
{
    KOKKOS_FUNCTION double operator()(auto, auto) const
    {
        return 0.0;
    }

    KOKKOS_FUNCTION double value(auto sampler, auto sampled_elem, auto cochain_elem) const
    {
        return sampler(sampled_elem, cochain_elem);
    }
};

template <class TensorType>
struct ClampedTensorEvaluator
{
    TensorType tensor;

    KOKKOS_FUNCTION double operator()(auto sampled_elem, auto cochain_elem) const
    {
        auto const clamped_elem = misc::clamp_to_domain(tensor.non_indices_domain(), sampled_elem);
        return tensor.mem(clamped_elem, cochain_elem);
    }

    KOKKOS_FUNCTION double value(auto sampler, auto sampled_elem, auto cochain_elem) const
    {
        auto const clamped_elem = misc::clamp_to_domain(tensor.non_indices_domain(), sampled_elem);
        return sampler(clamped_elem, cochain_elem);
    }
};

template <class TensorType>
struct ZeroOutsideTensorEvaluator
{
    TensorType tensor;

    KOKKOS_FUNCTION double operator()(auto sampled_elem, auto cochain_elem) const
    {
        if (!misc::domain_contains(tensor.non_indices_domain(), sampled_elem)) {
            return 0.0;
        }
        return tensor.mem(sampled_elem, cochain_elem);
    }

    KOKKOS_FUNCTION double value(auto sampler, auto sampled_elem, auto cochain_elem) const
    {
        if (!misc::domain_contains(tensor.non_indices_domain(), sampled_elem)) {
            return 0.0;
        }
        return sampler(sampled_elem, cochain_elem);
    }
};

} // namespace detail

} // namespace exterior

} // namespace sil
