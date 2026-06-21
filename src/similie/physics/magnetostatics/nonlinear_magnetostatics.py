#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Baptiste Legouix
# SPDX-License-Identifier: MIT

from __future__ import annotations

from similie_generate_cpp_hamiltonian import (
    HamiltonianDefinition,
    SymbolicFunctionDefinition,
)
from sympy import Function, simplify, sqrt, symbols


INTERPOLATED_BH_CURVE_HEADER = """\
template <std::size_t MaxSamples>
struct InterpolatedNonlinearBHCurve
{
    static constexpr std::size_t MAX_SAMPLES = MaxSamples;

    std::size_t m_num_samples = 0;
    std::array<double, MAX_SAMPLES> m_b {};
    std::array<double, MAX_SAMPLES> m_h {};
    std::array<double, MAX_SAMPLES> m_q {};
    std::array<double, MAX_SAMPLES> m_nu {};
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

    [[nodiscard]] KOKKOS_FUNCTION double dh_db_from_b(double b_value) const
    {
        if (b_value <= m_b[0]) {
            return m_dh_db[0];
        }
        if (b_value >= m_b[m_num_samples - 1]) {
            return m_dh_db[m_num_samples - 2];
        }
        return m_dh_db[bracket_q(b_value * b_value)];
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

};
"""


CONSTITUTIVE_LAW_HEADER = """\
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
        double const b_norm = std::sqrt(q);
        if (b_norm == 0.0) {
            double const dh_db = m_bh_curve.dh_db_from_b(0.0);
            return {
                    hodge_star[0] * dh_db,
                    0.0,
                    0.0,
                    0.0,
                    hodge_star[1] * dh_db,
                    0.0,
                    0.0,
                    0.0,
                    hodge_star[2] * dh_db,
            };
        }
        double const h_norm = m_bh_curve.h_from_b(b_norm);
        double const dh_db = m_bh_curve.dh_db_from_b(b_norm);
        double const scale = h_norm / b_norm;
        double const radial_scale = (dh_db - scale) / q;
        std::array<double, 9> jacobian {};
        for (std::size_t row = 0; row < 3; ++row) {
            for (std::size_t column = 0; column < 3; ++column) {
                jacobian[3 * row + column]
                        = hodge_star[row]
                          * ((row == column ? scale : 0.0)
                             + radial_scale * magnetic_induction[row] * magnetic_induction[column]);
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
        double const b_norm = std::sqrt(q);
        if (b_norm == 0.0) {
            return {0.0, 0.0, 0.0};
        }
        double const scale = m_bh_curve.h_from_b(b_norm) / b_norm;
        return {
                hodge_star[0] * scale * magnetic_induction[0],
                hodge_star[1] * scale * magnetic_induction[1],
                hodge_star[2] * scale * magnetic_induction[2],
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
"""


class NonlinearMagnetostaticsHamiltonian:
    @staticmethod
    def __call__() -> HamiltonianDefinition:
        a = symbols("A0:3")
        b = symbols("B0:3")
        j = symbols("j0:3")
        magnetic_field = Function("magnetic_field")
        q = sum(component**2 for component in b)
        b_norm = sqrt(q)
        h = [b[i] * magnetic_field(b_norm) / b_norm for i in range(3)]
        hamiltonian = simplify(sum(b[i] * h[i] / 2 for i in range(3))) - sum(
            a[i] * j[i] for i in range(3)
        )

        return HamiltonianDefinition(
            namespace="similie::physics::magnetostatics",
            struct_name="NonlinearMagnetostaticsHamiltonian",
            parameters=["bh_curve"],
            hamiltonian=hamiltonian,
            variables=[a, b, j],
            includes=["<similie/physics/magnetostatics/magnetostatics_quantities.hpp>"],
            template_parameters=["class BHCurve"],
            parameter_types={"bh_curve": "BHCurve"},
            symbolic_functions={
                "magnetic_field": SymbolicFunctionDefinition(
                    value_expression="m_bh_curve.h_from_b({argument})",
                    derivative_expressions={
                        1: "m_bh_curve.dh_db_from_b({argument})",
                        2: "0.0",
                    },
                )
            },
            moments_object_component_expression="{moments}.template get<{index}>()",
            moments_object_norm2_expression="{moments}.norm2()",
            generate_moments_jacobian=True,
            namespace_preamble=INTERPOLATED_BH_CURVE_HEADER,
            namespace_epilogue=CONSTITUTIVE_LAW_HEADER,
        )
