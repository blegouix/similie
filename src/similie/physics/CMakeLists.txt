# SPDX-FileCopyrightText: 2026 Baptiste Legouix
# SPDX-License-Identifier: MIT

set(Python3_EXECUTABLE "/usr/bin/python3" CACHE FILEPATH "Python interpreter used for symbolic calculus" FORCE)
find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(SIMILIE_GENERATED_PHYSICS_DIR "${PROJECT_BINARY_DIR}/src/similie/physics")

set(SCALAR_FIELD_GENERATOR
    "${CMAKE_CURRENT_SOURCE_DIR}/scalar_field/scalar_field_with_power_coupling.py")
set(GENERIC_HAMILTONIAN_GENERATOR
    "${CMAKE_CURRENT_SOURCE_DIR}/generate_cpp_hamiltonian.py")
set(SCALAR_FIELD_HEADER
    "${SIMILIE_GENERATED_PHYSICS_DIR}/scalar_field/scalar_field_with_power_coupling.hpp")

add_custom_command(
    OUTPUT "${SCALAR_FIELD_HEADER}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory
            "${SIMILIE_GENERATED_PHYSICS_DIR}/scalar_field"
    COMMAND "${Python3_EXECUTABLE}" "${SCALAR_FIELD_GENERATOR}" 3 "${SCALAR_FIELD_HEADER}"
    DEPENDS "${SCALAR_FIELD_GENERATOR}" "${GENERIC_HAMILTONIAN_GENERATOR}"
    COMMENT "Generating scalar field functor with power coupling"
    VERBATIM
)

set(LINEAR_MAGNETOSTATICS_GENERATOR
    "${CMAKE_CURRENT_SOURCE_DIR}/magnetostatics/linear_magnetostatics.py")
set(LINEAR_MAGNETOSTATICS_HEADER
    "${SIMILIE_GENERATED_PHYSICS_DIR}/magnetostatics/linear_magnetostatics.hpp")
set(LINEAR_MAGNETIC_INDUCTION_TO_MAGNETIC_FIELD_HEADER
    "${SIMILIE_GENERATED_PHYSICS_DIR}/magnetostatics/linear_magnetic_induction_to_magnetic_field.hpp")

add_custom_command(
    OUTPUT
        "${LINEAR_MAGNETOSTATICS_HEADER}"
        "${LINEAR_MAGNETIC_INDUCTION_TO_MAGNETIC_FIELD_HEADER}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory
            "${SIMILIE_GENERATED_PHYSICS_DIR}/magnetostatics"
    COMMAND "${Python3_EXECUTABLE}" "${LINEAR_MAGNETOSTATICS_GENERATOR}"
            "${SIMILIE_GENERATED_PHYSICS_DIR}/magnetostatics"
    DEPENDS "${LINEAR_MAGNETOSTATICS_GENERATOR}" "${GENERIC_HAMILTONIAN_GENERATOR}"
    COMMENT "Generating linear magnetostatics functors"
    VERBATIM
)

add_custom_target(similie_physics_generated
    DEPENDS
        "${SCALAR_FIELD_HEADER}"
        "${LINEAR_MAGNETOSTATICS_HEADER}"
        "${LINEAR_MAGNETIC_INDUCTION_TO_MAGNETIC_FIELD_HEADER}"
)

add_library(physics INTERFACE)
add_dependencies(physics similie_physics_generated)
target_include_directories(physics
    INTERFACE
        "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/src>"
)
