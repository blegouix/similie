#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Baptiste Legouix
# SPDX-License-Identifier: MIT

from __future__ import annotations

import sys
from pathlib import Path

from sympy import symbols
from sympy.printing.codeprinter import cxxcode


output_dir = Path(sys.argv[1])
output_dir.mkdir(parents=True, exist_ok=True)

b0, b1, b2, mu = symbols("b0 b1 b2 mu")
hamiltonian = (b0**2 + b1**2 + b2**2) / (2 * mu)

(output_dir / "linear_magnetostatics.hpp").write_text(
    f"""\
// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <span>

namespace similie::physics::magnetostatics {{

struct LinearMagnetostaticsHamiltonian {{
    static constexpr std::size_t N = 3;

    double mu;

    constexpr explicit LinearMagnetostaticsHamiltonian(double mu_) : mu(mu_) {{}}

    constexpr double H(std::span<double const, N> magnetic_induction) const
    {{
        return {cxxcode(hamiltonian).replace("b0", "magnetic_induction[0]").replace("b1", "magnetic_induction[1]").replace("b2", "magnetic_induction[2]")};
    }}

    constexpr double dH_dB0(double b0) const
    {{
        return {cxxcode(hamiltonian.diff(b0))};
    }}

    constexpr double dH_dB1(double b1) const
    {{
        return {cxxcode(hamiltonian.diff(b1))};
    }}

    constexpr double dH_dB2(double b2) const
    {{
        return {cxxcode(hamiltonian.diff(b2))};
    }}
}};

}} // namespace similie::physics::magnetostatics
"""
)

(output_dir / "linear_magnetic_induction_to_magnetic_field.hpp").write_text(
    """\
// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: MIT

#pragma once

#include <array>

#include <similie/misc/specialization.hpp>
#include <similie/physics/magnetostatics/magnetostatics_indices.hpp>
#include <similie/tensor/tensor.hpp>

namespace similie::physics::magnetostatics {

class LinearMagneticInductionToMagneticField
{
    double m_mu;
    std::array<double, 3> m_hodge_star;

public:
    constexpr explicit LinearMagneticInductionToMagneticField(
            double mu,
            std::array<double, 3> hodge_star = {1.0, 1.0, 1.0})
        : m_mu(mu)
        , m_hodge_star(hodge_star)
    {
    }

    [[nodiscard]] constexpr double mu() const
    {
        return m_mu;
    }

    [[nodiscard]] constexpr std::array<double, 3> hodge_star() const
    {
        return m_hodge_star;
    }

    template <
            sil::misc::Specialization<sil::tensor::Tensor> MagneticInductionTensorType,
            sil::misc::Specialization<sil::tensor::Tensor> MagneticFieldTensorType>
    KOKKOS_FUNCTION void inverse(
            MagneticInductionTensorType magnetic_induction,
            MagneticFieldTensorType magnetic_field) const
    {
        magnetic_induction(magnetic_induction.template access_element<Y, Z>())
                = m_mu * m_hodge_star[0]
                  * magnetic_field(magnetic_field.template access_element<X>());
        magnetic_induction(magnetic_induction.template access_element<X, Z>())
                = -m_mu * m_hodge_star[1]
                  * magnetic_field(magnetic_field.template access_element<Y>());
        magnetic_induction(magnetic_induction.template access_element<X, Y>())
                = m_mu * m_hodge_star[2]
                  * magnetic_field(magnetic_field.template access_element<Z>());
    }

    template <
            sil::misc::Specialization<sil::tensor::Tensor> MagneticFieldTensorType,
            sil::misc::Specialization<sil::tensor::Tensor> MagneticInductionTensorType>
    KOKKOS_FUNCTION void forward(
            MagneticFieldTensorType magnetic_field,
            MagneticInductionTensorType magnetic_induction) const
    {
        magnetic_field(magnetic_field.template access_element<X>())
                = magnetic_induction(magnetic_induction.template access_element<Y, Z>())
                  / (m_mu * m_hodge_star[0]);
        magnetic_field(magnetic_field.template access_element<Y>())
                = -magnetic_induction(magnetic_induction.template access_element<X, Z>())
                  / (m_mu * m_hodge_star[1]);
        magnetic_field(magnetic_field.template access_element<Z>())
                = magnetic_induction(magnetic_induction.template access_element<X, Y>())
                  / (m_mu * m_hodge_star[2]);
    }
};

} // namespace similie::physics::magnetostatics
"""
)
