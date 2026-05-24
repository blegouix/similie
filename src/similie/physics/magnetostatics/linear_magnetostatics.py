#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Baptiste Legouix
# SPDX-License-Identifier: MIT

from __future__ import annotations

from pathlib import Path

from similie_generate_cpp_hamiltonian import HamiltonianDefinition
from sympy import symbols


class LinearMagnetostaticsHamiltonian:
    @staticmethod
    def __call__() -> HamiltonianDefinition:
        A = symbols("A0:3")
        B = symbols("B0:3")
        j = symbols("j0:3")
        mu = symbols("mu")
        hamiltonian = sum(B[i] ** 2 / (2 * mu) - j[i] * A[i] for i in range(3))

        return HamiltonianDefinition(
            namespace="similie::physics::magnetostatics",
            struct_name="LinearMagnetostaticsHamiltonian",
            parameters=["mu"],
            hamiltonian=hamiltonian,
            variables=[A, B, j],
            includes=["<similie/physics/magnetostatics/magnetostatics_quantities.hpp>"],
            moments_computer="MagneticVectorPotentialToMagneticInduction",
            template_parameters=["class MuTensor"],
            parameter_types={"mu": "MuTensor"},
            parameter_value_expressions={
                "mu": "m_mu(elem, ddc::DiscreteElement<sil::tensor::Covariant<sil::tensor::ScalarIndex>>(0))"
            },
            is_linear=True,
        )


LINEAR_CONSTITUTIVE_CLASS = """\
class LinearMagneticInductionToMagneticField
{
    const double m_mu;

public:
    constexpr explicit LinearMagneticInductionToMagneticField(
            double mu_)
        : m_mu(mu_)
    {
    }

    [[nodiscard]] KOKKOS_FUNCTION constexpr double value(double hodge_star, double b) const
    {
        return hodge_star / m_mu;
    }

    [[nodiscard]] KOKKOS_FUNCTION constexpr double jacobian(double hodge_star, double b) const
    {
        return value(hodge_star, b);
    }

    [[nodiscard]] KOKKOS_FUNCTION constexpr double operator()(double hodge_star, double b) const
    {
        return b * value(hodge_star, b);
    }

    [[nodiscard]] KOKKOS_FUNCTION constexpr double inverse_value(double hodge_star, double h) const
    {
        return m_mu / hodge_star;
    }

    [[nodiscard]] KOKKOS_FUNCTION constexpr double inverse(double hodge_star, double h) const
    {
        return h * inverse_value(hodge_star, h);
    }
};
"""


def write_cpp_linear_magnetostatics_header(output_path: Path) -> None:
    from similie_generate_cpp_hamiltonian import generate_cpp_hamiltonian

    generate_cpp_hamiltonian(LinearMagnetostaticsHamiltonian, output_path)
    content = output_path.read_text()
    namespace_footer = "} // namespace similie::physics::magnetostatics\n"
    if not content.endswith(namespace_footer):
        raise RuntimeError("unexpected generated linear magnetostatics header layout")
    output_path.write_text(
        content.removesuffix(namespace_footer)
        + "\n"
        + LINEAR_CONSTITUTIVE_CLASS
        + "\n"
        + namespace_footer
    )


if __name__ == "__main__":
    write_cpp_linear_magnetostatics_header(Path("linear_magnetostatics.hpp"))
