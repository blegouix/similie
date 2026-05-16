# SPDX-FileCopyrightText: 2026 Baptiste Legouix
# SPDX-License-Identifier: MIT

set(Python3_EXECUTABLE "/usr/bin/python3" CACHE FILEPATH "Python interpreter used for symbolic calculus" FORCE)
find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(SIMILIE_GENERATED_PHYSICS_DIR "${PROJECT_BINARY_DIR}/src/similie/physics")

set(SCALAR_FIELD_GENERATOR
    "${CMAKE_CURRENT_SOURCE_DIR}/scalar_field/scalar_field.py")
set(GENERIC_HAMILTONIAN_GENERATOR
    "${CMAKE_CURRENT_SOURCE_DIR}/generate_cpp_hamiltonian.py")
set(GENERIC_CONSTITUTIVE_LAW_GENERATOR
    "${CMAKE_CURRENT_SOURCE_DIR}/generate_cpp_constitutive_law.py")
set(SCALAR_FIELD_HEADER
    "${SIMILIE_GENERATED_PHYSICS_DIR}/scalar_field/scalar_field_with_power_coupling.hpp")
string(CONCAT SCALAR_FIELD_GENERATION_CODE
    "import importlib.util; import sys; "
    "from pathlib import Path; "
    "root = Path(r\"${CMAKE_CURRENT_SOURCE_DIR}\"); "
    "scalar_field_path = root / \"scalar_field\" / \"scalar_field.py\"; "
    "generator_path = root / \"generate_cpp_hamiltonian.py\"; "
    "generator_spec = importlib.util.spec_from_file_location(\"similie_generate_cpp_hamiltonian\", generator_path); "
    "generator_module = importlib.util.module_from_spec(generator_spec); "
    "sys.modules[generator_spec.name] = generator_module; "
    "generator_spec.loader.exec_module(generator_module); "
    "scalar_field_spec = importlib.util.spec_from_file_location(\"similie_scalar_field\", scalar_field_path); "
    "scalar_field_module = importlib.util.module_from_spec(scalar_field_spec); "
    "sys.modules[scalar_field_spec.name] = scalar_field_module; "
    "scalar_field_spec.loader.exec_module(scalar_field_module); "
    "generator_module.generate_cpp_hamiltonian(scalar_field_module.ScalarFieldHamiltonian, Path(r\"${SCALAR_FIELD_HEADER}\"), 3)")

add_custom_command(
    OUTPUT "${SCALAR_FIELD_HEADER}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory
            "${SIMILIE_GENERATED_PHYSICS_DIR}/scalar_field"
    COMMAND "${Python3_EXECUTABLE}" -c "${SCALAR_FIELD_GENERATION_CODE}"
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
string(CONCAT LINEAR_MAGNETOSTATICS_GENERATION_CODE
    "import importlib.util; import sys; "
    "from pathlib import Path; "
    "root = Path(r\"${CMAKE_CURRENT_SOURCE_DIR}\"); "
    "magnetostatics_path = root / \"magnetostatics\" / \"linear_magnetostatics.py\"; "
    "hamiltonian_generator_path = root / \"generate_cpp_hamiltonian.py\"; "
    "constitutive_law_generator_path = root / \"generate_cpp_constitutive_law.py\"; "
    "hamiltonian_generator_spec = importlib.util.spec_from_file_location(\"similie_generate_cpp_hamiltonian\", hamiltonian_generator_path); "
    "hamiltonian_generator_module = importlib.util.module_from_spec(hamiltonian_generator_spec); "
    "sys.modules[hamiltonian_generator_spec.name] = hamiltonian_generator_module; "
    "hamiltonian_generator_spec.loader.exec_module(hamiltonian_generator_module); "
    "constitutive_law_generator_spec = importlib.util.spec_from_file_location(\"similie_generate_cpp_constitutive_law\", constitutive_law_generator_path); "
    "constitutive_law_generator_module = importlib.util.module_from_spec(constitutive_law_generator_spec); "
    "sys.modules[constitutive_law_generator_spec.name] = constitutive_law_generator_module; "
    "constitutive_law_generator_spec.loader.exec_module(constitutive_law_generator_module); "
    "magnetostatics_spec = importlib.util.spec_from_file_location(\"similie_linear_magnetostatics\", magnetostatics_path); "
    "magnetostatics_module = importlib.util.module_from_spec(magnetostatics_spec); "
    "sys.modules[magnetostatics_spec.name] = magnetostatics_module; "
    "magnetostatics_spec.loader.exec_module(magnetostatics_module); "
    "output_dir = Path(r\"${SIMILIE_GENERATED_PHYSICS_DIR}/magnetostatics\"); "
    "hamiltonian_generator_module.generate_cpp_hamiltonian(magnetostatics_module.LinearMagnetostaticsHamiltonian, output_dir / \"linear_magnetostatics.hpp\"); "
    "constitutive_law_generator_module.generate_cpp_constitutive_law(magnetostatics_module.LinearMagneticInductionToMagneticFieldConstitutiveLaw, output_dir / \"linear_magnetic_induction_to_magnetic_field.hpp\")")

add_custom_command(
    OUTPUT
        "${LINEAR_MAGNETOSTATICS_HEADER}"
        "${LINEAR_MAGNETIC_INDUCTION_TO_MAGNETIC_FIELD_HEADER}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory
            "${SIMILIE_GENERATED_PHYSICS_DIR}/magnetostatics"
    COMMAND "${Python3_EXECUTABLE}" -c "${LINEAR_MAGNETOSTATICS_GENERATION_CODE}"
    DEPENDS
        "${LINEAR_MAGNETOSTATICS_GENERATOR}"
        "${GENERIC_HAMILTONIAN_GENERATOR}"
        "${GENERIC_CONSTITUTIVE_LAW_GENERATOR}"
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
