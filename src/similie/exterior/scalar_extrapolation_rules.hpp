// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#pragma once

#include <cstddef>
#include <variant>

#include <ddc/ddc.hpp>

#include <similie/misc/clamp_to_domain.hpp>
#include <similie/misc/domain_contains.hpp>

namespace sil::exterior {

/**
 * Use the nearest value of a cochain at an out-of-domain sample.
 * \important This operator and documentation are fully AI-generated.
 *
 * This is the historical closure of the primal coboundary in SimiLie. It is
 * explicit so callers can select another rule for a connected domain.
 */
struct ClampCochainExtrapolationRule
{
    template <class TensorType, class Element, class Component>
    KOKKOS_FUNCTION double operator()(
            TensorType tensor,
            Element sampled_element,
            Component component) const
    {
        return tensor
                .mem(sil::misc::clamp_to_domain(tensor.non_indices_domain(), sampled_element),
                     component);
    }
};

/**
 * Return zero for samples outside a cochain domain.
 * \important This operator and documentation are fully AI-generated.
 *
 * This is the historical closure used for the dual cochain in the scalar
 * codifferential. Interior samples retain their stored values.
 */
struct ZeroCochainExtrapolationRule
{
    template <class TensorType, class Element, class Component>
    KOKKOS_FUNCTION double operator()(
            TensorType tensor,
            Element sampled_element,
            Component component) const
    {
        if (!sil::misc::domain_contains(tensor.non_indices_domain(), sampled_element))
            return 0.0;
        return tensor.mem(sampled_element, component);
    }
};

/** Prescribe the scalar potential on one boundary node. */
struct PrescribedScalarExtrapolationRule
{
    double value;
};

/** Prescribe the normal scalar flux on one boundary node. */
struct NormalScalarFluxExtrapolationRule
{
    double value = 0.0;
};

/**
 * Use the one-sided scalar DEC Laplacian row at a free boundary.
 * \important This operator and documentation are fully AI-generated.
 *
 * This is the natural closure of the discrete bulk equation when no boundary
 * flux is prescribed. It is useful on smooth boundaries where the variational
 * zero-flux condition is represented by the exterior Laplacian itself.
 */
struct NaturalScalarExtrapolationRule
{
};

using ScalarBoundaryExtrapolationRule = std::variant<
        PrescribedScalarExtrapolationRule,
        NormalScalarFluxExtrapolationRule,
        NaturalScalarExtrapolationRule>;

/** Select an x-normal trace of a structured tensor domain. */
enum class ScalarTraceSide { LowerX, UpperX };

/**
 * Connect two scalar traces with an affine potential jump.
 * \important This operator and documentation are fully AI-generated.
 *
 * For each pair of matching trace nodes, the first potential equals the
 * second potential plus jump_coefficient times the global jump parameter.
 * The coupled Laplacian also matches the physical normal flux on the traces.
 */
struct ConnectedScalarExtrapolationRule
{
    std::size_t first_domain;
    ScalarTraceSide first_side;
    std::size_t second_domain;
    ScalarTraceSide second_side;
    double jump_coefficient = 0.0;
};

} // namespace sil::exterior
