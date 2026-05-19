// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: MIT

#pragma once

#include <stdexcept>
#include <string>

#include <similie/physics/scalar_field/scalar_field_with_power_coupling.hpp>

namespace similie::onelab_interface::scalar_field_with_power_coupling_onelab {

template <class Problem, class ProblemParameterName, class PublishNumber>
void synchronize_controls(
        Problem const& problem,
        ProblemParameterName&& problem_parameter_name,
        PublishNumber&& publish_or_sync_number)
{
    publish_or_sync_number(
            problem_parameter_name("2ScalarFieldWithPowerCoupling", "0Mass"),
            "Mass",
            "Scalar-field mass parameter declared in the .silpro file.",
            problem.scalar_field.mass,
            0.0,
            1.e12,
            1.e-3);
    publish_or_sync_number(
            problem_parameter_name("2ScalarFieldWithPowerCoupling", "1Coupling constant"),
            "Coupling constant",
            "Scalar-field coupling constant declared in the .silpro file.",
            problem.scalar_field.coupling_constant,
            -1.e12,
            1.e12,
            1.e-3);
    publish_or_sync_number(
            problem_parameter_name("2ScalarFieldWithPowerCoupling", "2Coupling power"),
            "Coupling power",
            "Scalar-field coupling power declared in the .silpro file.",
            problem.scalar_field.coupling_power,
            0.0,
            1.e12,
            1.0);
}

template <class Problem, class ProblemParameterName, class GetFirstNumberValue>
Problem apply_control_overrides(
        Problem problem,
        ProblemParameterName&& problem_parameter_name,
        GetFirstNumberValue&& get_first_number_value)
{
    problem.scalar_field.mass = get_first_number_value(
            problem_parameter_name("2ScalarFieldWithPowerCoupling", "0Mass"),
            problem.scalar_field.mass);
    problem.scalar_field.coupling_constant = get_first_number_value(
            problem_parameter_name("2ScalarFieldWithPowerCoupling", "1Coupling constant"),
            problem.scalar_field.coupling_constant);
    problem.scalar_field.coupling_power = get_first_number_value(
            problem_parameter_name("2ScalarFieldWithPowerCoupling", "2Coupling power"),
            problem.scalar_field.coupling_power);
    return problem;
}

template <class Problem>
auto assemble_hamiltonian(Problem const& problem)
{
    return physics::scalar_field::ScalarFieldWithPowerCouplingHamiltonian(
            problem.scalar_field.mass,
            problem.scalar_field.coupling_constant,
            problem.scalar_field.coupling_power);
}

inline void run()
{
    throw std::runtime_error(
            "ScalarFieldWithPowerCoupling .silpro files are parsed successfully, but ONELAB "
            "execution is not implemented yet in this interface");
}

} // namespace similie::onelab_interface::scalar_field_with_power_coupling_onelab
