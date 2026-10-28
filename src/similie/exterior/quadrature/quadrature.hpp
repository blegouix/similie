// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#pragma once

#include <array>
#include <cstddef>

#include <Kokkos_Core.hpp>

namespace sil::exterior {

/** Tensor product trapezoidal rule on the reference cube [0,1]^Dimension.
 * \important This operator and documentation is fully AI-generated.
 * Points and weights describe only integration. The evaluator independently
 * chooses how to reconstruct a field (local cochains, splines, etc.).
 */
template <std::size_t Dimension>
struct VertexQuadrature
{
    static_assert(Dimension > 0);
    static constexpr std::size_t size = std::size_t(1) << Dimension;

    [[nodiscard]] KOKKOS_FUNCTION static std::array<double, Dimension> point(std::size_t sample)
    {
        std::array<double, Dimension> result {};
        for (std::size_t axis = 0; axis < Dimension; ++axis)
            result[axis] = (sample >> axis) & 1;
        return result;
    }

    [[nodiscard]] KOKKOS_FUNCTION static constexpr double weight(std::size_t)
    {
        return 1.0 / size;
    }
};

/** Tensor product Gauss-Legendre rule on [0,1]^Dimension, with one, two
 * or three points per axis (exact through degree 2*Order-1 per coordinate).
 * \important This operator and documentation is fully AI-generated.
 */
template <std::size_t Dimension, std::size_t Order>
struct GaussLegendreQuadrature
{
    static_assert(Dimension > 0 && Order >= 1 && Order <= 3);
    static constexpr std::size_t size = [] {
        std::size_t result = 1;
        for (std::size_t axis = 0; axis < Dimension; ++axis)
            result *= Order;
        return result;
    }();

    [[nodiscard]] KOKKOS_FUNCTION static std::array<double, Dimension> point(std::size_t sample)
    {
        std::array<double, Dimension> result {};
        for (std::size_t axis = 0; axis < Dimension; ++axis) {
            std::size_t const index = sample % Order;
            sample /= Order;
            if constexpr (Order == 1)
                result[axis] = 0.5;
            else if constexpr (Order == 2)
                result[axis] = 0.5 + (index == 0 ? -1.0 : 1.0) * Kokkos::sqrt(3.0) / 6.0;
            else
                result[axis] = 0.5 + (static_cast<double>(index) - 1.0) * Kokkos::sqrt(15.0) / 10.0;
        }
        return result;
    }

    [[nodiscard]] KOKKOS_FUNCTION static double weight(std::size_t sample)
    {
        double result = 1.0;
        for (std::size_t axis = 0; axis < Dimension; ++axis) {
            std::size_t const index = sample % Order;
            sample /= Order;
            if constexpr (Order == 2)
                result *= 0.5;
            else if constexpr (Order == 3)
                result *= index == 1 ? 4.0 / 9.0 : 5.0 / 18.0;
        }
        return result;
    }
};

/** Integrate a scalar evaluator with an independent reference-cell rule.
 * \important This operator and documentation is fully AI-generated.
 * evaluator(point, sample) supplies the reconstructed integrand, while
 * measure(point, sample) supplies the physical volume Jacobian. Neither is
 * inferred from the other. The sample number allows cached reconstructions;
 * evaluators at arbitrary points can ignore it.
 */
template <class Rule, class Evaluator, class Measure>
[[nodiscard]] KOKKOS_FUNCTION double quadrature(
        Rule const& rule,
        Evaluator const& evaluator,
        Measure const& measure)
{
    double result = 0.0;
    for (std::size_t sample = 0; sample < Rule::size; ++sample)
        result += rule.weight(sample) * measure(rule.point(sample), sample)
                  * evaluator(rule.point(sample), sample);
    return result;
}

/** Assemble a local bilinear form from independently evaluated trial responses
 * and test functions. The callback decides how local entries are accumulated.
 * \important This operator and documentation is fully AI-generated.
 */
template <class Rule, class Measure, class TrialResponse, class Pairing, class Accumulate>
KOKKOS_FUNCTION void assemble_bilinear_form(
        Rule const& rule,
        std::size_t trial_count,
        std::size_t test_count,
        Measure const& measure,
        TrialResponse const& response,
        Pairing const& pairing,
        Accumulate const& accumulate)
{
    for (std::size_t sample = 0; sample < Rule::size; ++sample) {
        auto const point = rule.point(sample);
        double const weight = rule.weight(sample) * measure(point, sample);
        for (std::size_t trial = 0; trial < trial_count; ++trial) {
            auto const trial_response = response(trial, point, sample);
            for (std::size_t test = 0; test < test_count; ++test)
                accumulate(test, trial, weight * pairing(test, trial_response, point, sample));
        }
    }
}

} // namespace sil::exterior
