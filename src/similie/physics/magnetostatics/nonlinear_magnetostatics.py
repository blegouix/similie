#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Baptiste Legouix
# SPDX-License-Identifier: MIT

from __future__ import annotations

from pathlib import Path


def write_cpp_nonlinear_magnetostatics_header(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(
        """\
// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <span>
#include <stdexcept>

#include <Kokkos_Core.hpp>

#include <similie/physics/magnetostatics/magnetostatics_quantities.hpp>

namespace similie::physics::magnetostatics {

template <std::size_t MaxSamples>
struct InterpolatedNonlinearBHCurve
{
    static constexpr std::size_t MAX_SAMPLES = MaxSamples;

    std::size_t m_num_samples = 0;
    std::array<double, MAX_SAMPLES> m_b {};
    std::array<double, MAX_SAMPLES> m_h {};
    std::array<double, MAX_SAMPLES> m_q {};
    std::array<double, MAX_SAMPLES> m_nu {};
    std::array<double, MAX_SAMPLES> m_energy {};
    std::array<double, MAX_SAMPLES - 1> m_dnu_dq {};
    std::array<double, MAX_SAMPLES - 1> m_dh_db {};

    template <std::size_t NumSamples>
    explicit InterpolatedNonlinearBHCurve(
            std::array<double, NumSamples> const& magnetic_induction_samples,
            std::array<double, NumSamples> const& magnetic_field_samples,
            std::size_t num_samples = NumSamples)
        : m_num_samples(num_samples)
    {
        static_assert(NumSamples <= MAX_SAMPLES);
        if (m_num_samples < 2 || m_num_samples > NumSamples) {
            throw std::runtime_error("invalid nonlinear B-H curve sample count");
        }

        for (std::size_t i = 0; i < m_num_samples; ++i) {
            m_b[i] = magnetic_induction_samples[i];
            m_h[i] = magnetic_field_samples[i];
            m_q[i] = m_b[i] * m_b[i];
        }

        m_nu[0] = m_h[1] / m_b[1];
        for (std::size_t i = 1; i < m_num_samples; ++i) {
            m_nu[i] = m_h[i] / m_b[i];
        }
        for (std::size_t i = 0; i + 1 < m_num_samples; ++i) {
            m_dnu_dq[i] = (m_nu[i + 1] - m_nu[i]) / (m_q[i + 1] - m_q[i]);
            m_dh_db[i] = (m_h[i + 1] - m_h[i]) / (m_b[i + 1] - m_b[i]);
        }

        m_energy[0] = 0.0;
        for (std::size_t i = 0; i + 1 < m_num_samples; ++i) {
            double const dq = m_q[i + 1] - m_q[i];
            m_energy[i + 1] = m_energy[i] + 0.5 * m_nu[i] * dq + 0.25 * m_dnu_dq[i] * dq * dq;
        }
    }

    [[nodiscard]] KOKKOS_FUNCTION std::size_t bracket_q(double q_value) const
    {
        if (q_value <= m_q[0]) {
            return 0;
        }
        for (std::size_t i = 0; i + 1 < m_num_samples; ++i) {
            if (q_value <= m_q[i + 1]) {
                return i;
            }
        }
        return m_num_samples - 2;
    }

    [[nodiscard]] KOKKOS_FUNCTION std::size_t bracket_h(double h_value) const
    {
        if (h_value <= m_h[0]) {
            return 0;
        }
        for (std::size_t i = 0; i + 1 < m_num_samples; ++i) {
            if (h_value <= m_h[i + 1]) {
                return i;
            }
        }
        return m_num_samples - 2;
    }

    [[nodiscard]] KOKKOS_FUNCTION double nu_from_q(double q_value) const
    {
        std::size_t const interval = bracket_q(q_value);
        return m_nu[interval] + m_dnu_dq[interval] * (q_value - m_q[interval]);
    }

    [[nodiscard]] KOKKOS_FUNCTION double dnu_dq(double q_value) const
    {
        return m_dnu_dq[bracket_q(q_value)];
    }

    [[nodiscard]] KOKKOS_FUNCTION double h_from_b(double b_value) const
    {
        if (b_value <= m_b[0]) {
            return m_h[0];
        }
        if (b_value >= m_b[m_num_samples - 1]) {
            std::size_t const interval = m_num_samples - 2;
            return m_h[interval] + m_dh_db[interval] * (b_value - m_b[interval]);
        }
        std::size_t const interval = bracket_q(b_value * b_value);
        return m_h[interval] + m_dh_db[interval] * (b_value - m_b[interval]);
    }

    [[nodiscard]] KOKKOS_FUNCTION double b_from_h(double h_value) const
    {
        if (h_value <= m_h[0]) {
            return m_b[0];
        }
        std::size_t const interval = bracket_h(h_value);
        return m_b[interval] + (h_value - m_h[interval]) / m_dh_db[interval];
    }

    [[nodiscard]] KOKKOS_FUNCTION double db_dh(double h_value) const
    {
        return 1.0 / m_dh_db[bracket_h(h_value)];
    }

    [[nodiscard]] KOKKOS_FUNCTION double energy_from_q(double q_value) const
    {
        std::size_t const interval = bracket_q(q_value);
        double const dq = q_value - m_q[interval];
        return m_energy[interval] + 0.5 * m_nu[interval] * dq + 0.25 * m_dnu_dq[interval] * dq * dq;
    }
};

template <class BHCurve>
class NonlinearMagneticInductionToMagneticField
{
    BHCurve m_bh_curve;

    [[nodiscard]] KOKKOS_FUNCTION std::array<double, 9> jacobian_from_magnetic_induction(
            std::span<double const, 3> hodge_star,
            std::span<double const, 3> magnetic_induction) const
    {
        double const q = magnetic_induction[0] * magnetic_induction[0]
                         + magnetic_induction[1] * magnetic_induction[1]
                         + magnetic_induction[2] * magnetic_induction[2];
        double const nu = m_bh_curve.nu_from_q(q);
        double const dnu = m_bh_curve.dnu_dq(q);
        std::array<double, 9> jacobian {};
        for (std::size_t i = 0; i < 3; ++i) {
            for (std::size_t j = 0; j < 3; ++j) {
                jacobian[3 * i + j] = hodge_star[i]
                                      * ((i == j ? nu : 0.0)
                                         + 2.0 * dnu * magnetic_induction[i] * magnetic_induction[j]);
            }
        }
        return jacobian;
    }

    [[nodiscard]] KOKKOS_FUNCTION static std::array<double, 9> inverse_matrix3x3(
            std::array<double, 9> const& matrix)
    {
        double const a00 = matrix[0];
        double const a01 = matrix[1];
        double const a02 = matrix[2];
        double const a10 = matrix[3];
        double const a11 = matrix[4];
        double const a12 = matrix[5];
        double const a20 = matrix[6];
        double const a21 = matrix[7];
        double const a22 = matrix[8];

        double const c00 = a11 * a22 - a12 * a21;
        double const c01 = a02 * a21 - a01 * a22;
        double const c02 = a01 * a12 - a02 * a11;
        double const c10 = a12 * a20 - a10 * a22;
        double const c11 = a00 * a22 - a02 * a20;
        double const c12 = a02 * a10 - a00 * a12;
        double const c20 = a10 * a21 - a11 * a20;
        double const c21 = a01 * a20 - a00 * a21;
        double const c22 = a00 * a11 - a01 * a10;

        double const determinant = a00 * c00 + a01 * c10 + a02 * c20;
        return {
                c00 / determinant,
                c01 / determinant,
                c02 / determinant,
                c10 / determinant,
                c11 / determinant,
                c12 / determinant,
                c20 / determinant,
                c21 / determinant,
                c22 / determinant,
        };
    }

public:
    explicit NonlinearMagneticInductionToMagneticField(BHCurve bh_curve) : m_bh_curve(bh_curve) {}

    [[nodiscard]] KOKKOS_FUNCTION std::array<double, 3> operator()(
            std::span<double const, 3> hodge_star,
            std::span<double const, 3> magnetic_induction) const
    {
        double const q = magnetic_induction[0] * magnetic_induction[0]
                         + magnetic_induction[1] * magnetic_induction[1]
                         + magnetic_induction[2] * magnetic_induction[2];
        double const nu = m_bh_curve.nu_from_q(q);
        return {
                hodge_star[0] * nu * magnetic_induction[0],
                hodge_star[1] * nu * magnetic_induction[1],
                hodge_star[2] * nu * magnetic_induction[2],
        };
    }

    [[nodiscard]] KOKKOS_FUNCTION double value(
            std::span<double const, 3> hodge_star,
            std::span<double const, 3> magnetic_induction,
            std::size_t row,
            std::size_t column) const
    {
        return jacobian(hodge_star, magnetic_induction)[3 * row + column];
    }

    [[nodiscard]] KOKKOS_FUNCTION std::array<double, 3> inverse(
            std::span<double const, 3> hodge_star,
            std::span<double const, 3> magnetic_field) const
    {
        double const unhodge_h0 = magnetic_field[0] / hodge_star[0];
        double const unhodge_h1 = magnetic_field[1] / hodge_star[1];
        double const unhodge_h2 = magnetic_field[2] / hodge_star[2];
        double const h_norm = std::sqrt(
                unhodge_h0 * unhodge_h0 + unhodge_h1 * unhodge_h1 + unhodge_h2 * unhodge_h2);
        if (h_norm == 0.0) {
            return {0.0, 0.0, 0.0};
        }
        double const b_norm = m_bh_curve.b_from_h(h_norm);
        double const scale = b_norm / h_norm;
        return {scale * unhodge_h0, scale * unhodge_h1, scale * unhodge_h2};
    }

    [[nodiscard]] KOKKOS_FUNCTION double inverse_value(
            std::span<double const, 3> hodge_star,
            std::span<double const, 3> magnetic_field,
            std::size_t row,
            std::size_t column) const
    {
        std::array<double, 3> const magnetic_induction = inverse(hodge_star, magnetic_field);
        return inverse_matrix3x3(jacobian(hodge_star, magnetic_induction))[3 * row + column];
    }

    [[nodiscard]] KOKKOS_FUNCTION std::array<double, 9> jacobian(
            std::span<double const, 3> hodge_star,
            std::span<double const, 3> magnetic_induction) const
    {
        return jacobian_from_magnetic_induction(hodge_star, magnetic_induction);
    }
};

} // namespace similie::physics::magnetostatics
"""
    )


if __name__ == "__main__":
    write_cpp_nonlinear_magnetostatics_header(Path("nonlinear_magnetostatics.hpp"))
