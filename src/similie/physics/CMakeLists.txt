# SPDX-FileCopyrightText: 2026 Baptiste Legouix
# SPDX-License-Identifier: AGPL-3.0-or-later

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
set(LINEAR_MAGNETOSTATICS_CONSTITUTIVE_LAW_HEADER
    "${SIMILIE_GENERATED_PHYSICS_DIR}/magnetostatics/linear_magnetostatics_constitutive_law.hpp")
set(NONLINEAR_MAGNETOSTATICS_GENERATOR
    "${CMAKE_CURRENT_SOURCE_DIR}/magnetostatics/nonlinear_magnetostatics.py")
set(NONLINEAR_MAGNETOSTATICS_HEADER
    "${SIMILIE_GENERATED_PHYSICS_DIR}/magnetostatics/nonlinear_magnetostatics.hpp")
set(LINEAR_ELASTICITY_GENERATOR
    "${CMAKE_CURRENT_SOURCE_DIR}/elasticity/linear_elasticity.py")
set(LINEAR_ELASTICITY_HEADER
    "${SIMILIE_GENERATED_PHYSICS_DIR}/elasticity/linear_elasticity.hpp")
set(LINEAR_ELASTICITY_CONSTITUTIVE_LAW_HEADER
    "${SIMILIE_GENERATED_PHYSICS_DIR}/elasticity/linear_elasticity_constitutive_law.hpp")
string(CONCAT LINEAR_MAGNETOSTATICS_GENERATION_CODE
    "import importlib.util; import sys; "
    "from pathlib import Path; "
    "root = Path(r\"${CMAKE_CURRENT_SOURCE_DIR}\"); "
    "magnetostatics_path = root / \"magnetostatics\" / \"linear_magnetostatics.py\"; "
    "hamiltonian_generator_path = root / \"generate_cpp_hamiltonian.py\"; "
    "hamiltonian_generator_spec = importlib.util.spec_from_file_location(\"similie_generate_cpp_hamiltonian\", hamiltonian_generator_path); "
    "hamiltonian_generator_module = importlib.util.module_from_spec(hamiltonian_generator_spec); "
    "sys.modules[hamiltonian_generator_spec.name] = hamiltonian_generator_module; "
    "hamiltonian_generator_spec.loader.exec_module(hamiltonian_generator_module); "
    "constitutive_generator_path = root / \"generate_cpp_constitutive_law.py\"; "
    "constitutive_generator_spec = importlib.util.spec_from_file_location(\"similie_generate_cpp_constitutive_law\", constitutive_generator_path); "
    "constitutive_generator_module = importlib.util.module_from_spec(constitutive_generator_spec); "
    "sys.modules[constitutive_generator_spec.name] = constitutive_generator_module; "
    "constitutive_generator_spec.loader.exec_module(constitutive_generator_module); "
    "magnetostatics_spec = importlib.util.spec_from_file_location(\"similie_linear_magnetostatics\", magnetostatics_path); "
    "magnetostatics_module = importlib.util.module_from_spec(magnetostatics_spec); "
    "sys.modules[magnetostatics_spec.name] = magnetostatics_module; "
    "magnetostatics_spec.loader.exec_module(magnetostatics_module); "
    "hamiltonian_generator_module.generate_cpp_hamiltonian(magnetostatics_module.LinearMagnetostaticsHamiltonian, Path(r\"${LINEAR_MAGNETOSTATICS_HEADER}\")); "
    "constitutive_generator_module.generate_cpp_constitutive_law(magnetostatics_module.LinearMagneticInductionToMagneticField, Path(r\"${LINEAR_MAGNETOSTATICS_CONSTITUTIVE_LAW_HEADER}\"))")
string(CONCAT NONLINEAR_MAGNETOSTATICS_GENERATION_CODE
    "import importlib.util; import sys; "
    "from pathlib import Path; "
    "root = Path(r\"${CMAKE_CURRENT_SOURCE_DIR}\"); "
    "magnetostatics_path = root / \"magnetostatics\" / \"nonlinear_magnetostatics.py\"; "
    "hamiltonian_generator_path = root / \"generate_cpp_hamiltonian.py\"; "
    "hamiltonian_generator_spec = importlib.util.spec_from_file_location(\"similie_generate_cpp_hamiltonian\", hamiltonian_generator_path); "
    "hamiltonian_generator_module = importlib.util.module_from_spec(hamiltonian_generator_spec); "
    "sys.modules[hamiltonian_generator_spec.name] = hamiltonian_generator_module; "
    "hamiltonian_generator_spec.loader.exec_module(hamiltonian_generator_module); "
    "magnetostatics_spec = importlib.util.spec_from_file_location(\"similie_nonlinear_magnetostatics\", magnetostatics_path); "
    "magnetostatics_module = importlib.util.module_from_spec(magnetostatics_spec); "
    "sys.modules[magnetostatics_spec.name] = magnetostatics_module; "
    "magnetostatics_spec.loader.exec_module(magnetostatics_module); "
    "hamiltonian_generator_module.generate_cpp_hamiltonian(magnetostatics_module.NonlinearMagnetostaticsHamiltonian, Path(r\"${NONLINEAR_MAGNETOSTATICS_HEADER}\"))")
string(CONCAT LINEAR_ELASTICITY_GENERATION_CODE
    "import importlib.util; import sys; "
    "from pathlib import Path; "
    "root = Path(r\"${CMAKE_CURRENT_SOURCE_DIR}\"); "
    "elasticity_path = root / \"elasticity\" / \"linear_elasticity.py\"; "
    "hamiltonian_generator_path = root / \"generate_cpp_hamiltonian.py\"; "
    "hamiltonian_generator_spec = importlib.util.spec_from_file_location(\"similie_generate_cpp_hamiltonian\", hamiltonian_generator_path); "
    "hamiltonian_generator_module = importlib.util.module_from_spec(hamiltonian_generator_spec); "
    "sys.modules[hamiltonian_generator_spec.name] = hamiltonian_generator_module; "
    "hamiltonian_generator_spec.loader.exec_module(hamiltonian_generator_module); "
    "constitutive_generator_path = root / \"generate_cpp_constitutive_law.py\"; "
    "constitutive_generator_spec = importlib.util.spec_from_file_location(\"similie_generate_cpp_constitutive_law\", constitutive_generator_path); "
    "constitutive_generator_module = importlib.util.module_from_spec(constitutive_generator_spec); "
    "sys.modules[constitutive_generator_spec.name] = constitutive_generator_module; "
    "constitutive_generator_spec.loader.exec_module(constitutive_generator_module); "
    "elasticity_spec = importlib.util.spec_from_file_location(\"similie_linear_elasticity\", elasticity_path); "
    "elasticity_module = importlib.util.module_from_spec(elasticity_spec); "
    "sys.modules[elasticity_spec.name] = elasticity_module; "
    "elasticity_spec.loader.exec_module(elasticity_module); "
    "hamiltonian_generator_module.generate_cpp_hamiltonian(elasticity_module.LinearElasticityHamiltonian, Path(r\"${LINEAR_ELASTICITY_HEADER}\"), 2); "
    "constitutive_generator_module.generate_cpp_constitutive_law(elasticity_module.LinearElasticStrainToStress, Path(r\"${LINEAR_ELASTICITY_CONSTITUTIVE_LAW_HEADER}\"))")

add_custom_command(
    OUTPUT
        "${LINEAR_MAGNETOSTATICS_HEADER}"
        "${LINEAR_MAGNETOSTATICS_CONSTITUTIVE_LAW_HEADER}"
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

add_custom_command(
    OUTPUT "${NONLINEAR_MAGNETOSTATICS_HEADER}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory
            "${SIMILIE_GENERATED_PHYSICS_DIR}/magnetostatics"
    COMMAND "${Python3_EXECUTABLE}" -c "${NONLINEAR_MAGNETOSTATICS_GENERATION_CODE}"
    DEPENDS "${NONLINEAR_MAGNETOSTATICS_GENERATOR}" "${GENERIC_HAMILTONIAN_GENERATOR}"
    COMMENT "Generating nonlinear magnetostatics functors"
    VERBATIM
)

add_custom_command(
    OUTPUT
        "${LINEAR_ELASTICITY_HEADER}"
        "${LINEAR_ELASTICITY_CONSTITUTIVE_LAW_HEADER}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory
            "${SIMILIE_GENERATED_PHYSICS_DIR}/elasticity"
    COMMAND "${Python3_EXECUTABLE}" -c "${LINEAR_ELASTICITY_GENERATION_CODE}"
    DEPENDS
        "${LINEAR_ELASTICITY_GENERATOR}"
        "${GENERIC_HAMILTONIAN_GENERATOR}"
        "${GENERIC_CONSTITUTIVE_LAW_GENERATOR}"
    COMMENT "Generating linear elasticity functors"
    VERBATIM
)

set(NAVIER_STOKES_FOURIER_GENERATOR
    "${CMAKE_CURRENT_SOURCE_DIR}/fluid/navier_stokes_fourier.py")
set(NAVIER_STOKES_FOURIER_HEADER
    "${SIMILIE_GENERATED_PHYSICS_DIR}/fluid/navier_stokes_fourier_generated.hpp")
add_custom_command(
    OUTPUT "${NAVIER_STOKES_FOURIER_HEADER}"
    COMMAND "${Python3_EXECUTABLE}" -B "${NAVIER_STOKES_FOURIER_GENERATOR}"
            --output "${NAVIER_STOKES_FOURIER_HEADER}"
    DEPENDS "${NAVIER_STOKES_FOURIER_GENERATOR}"
    COMMENT "Generating explicit Navier-Stokes-Fourier physics from SymPy"
    VERBATIM
)

set(MAGNETODYNAMICS_GENERATOR
    "${CMAKE_CURRENT_SOURCE_DIR}/magnetodynamics/magnetodynamics.py")
set(MAGNETODYNAMICS_HEADER
    "${SIMILIE_GENERATED_PHYSICS_DIR}/magnetodynamics/magnetodynamics_hamiltonian.hpp")
set(MAGNETODYNAMICS_MODEL_HEADER
    "${SIMILIE_GENERATED_PHYSICS_DIR}/magnetodynamics/magnetodynamics.hpp")
add_custom_command(
    OUTPUT "${MAGNETODYNAMICS_HEADER}" "${MAGNETODYNAMICS_MODEL_HEADER}"
    COMMAND "${Python3_EXECUTABLE}" -B "${MAGNETODYNAMICS_GENERATOR}"
            --output "${MAGNETODYNAMICS_HEADER}"
            --model-output "${MAGNETODYNAMICS_MODEL_HEADER}"
    DEPENDS "${MAGNETODYNAMICS_GENERATOR}" "${GENERIC_HAMILTONIAN_GENERATOR}"
            "${CMAKE_CURRENT_SOURCE_DIR}/generate_cpp_field_model.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/magnetodynamics/magnetodynamics.hpp.in"
    COMMENT "Generating magnetodynamics Hamiltonian from SymPy"
    VERBATIM
)

set(ELECTRODYNAMICS_GENERATOR
    "${CMAKE_CURRENT_SOURCE_DIR}/electrodynamics/electrodynamics.py")
set(ELECTRODYNAMICS_HEADER
    "${SIMILIE_GENERATED_PHYSICS_DIR}/electrodynamics/electrodynamics_hamiltonian.hpp")
set(ELECTRODYNAMICS_MODEL_HEADER
    "${SIMILIE_GENERATED_PHYSICS_DIR}/electrodynamics/electrodynamics.hpp")
add_custom_command(
    OUTPUT "${ELECTRODYNAMICS_HEADER}" "${ELECTRODYNAMICS_MODEL_HEADER}"
    COMMAND "${Python3_EXECUTABLE}" -B "${ELECTRODYNAMICS_GENERATOR}"
            --output "${ELECTRODYNAMICS_HEADER}"
            --model-output "${ELECTRODYNAMICS_MODEL_HEADER}"
    DEPENDS "${ELECTRODYNAMICS_GENERATOR}" "${GENERIC_HAMILTONIAN_GENERATOR}"
            "${CMAKE_CURRENT_SOURCE_DIR}/generate_cpp_field_model.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/electrodynamics/electrodynamics.hpp.in"
    COMMENT "Generating electric-only Hamiltonian from SymPy"
    VERBATIM
)

set(ELECTROMAGNETISM_HEADER
    "${SIMILIE_GENERATED_PHYSICS_DIR}/electromagnetism/electromagnetism.hpp")
add_custom_command(
    OUTPUT "${ELECTROMAGNETISM_HEADER}"
    COMMAND "${Python3_EXECUTABLE}" -B
            "${CMAKE_CURRENT_SOURCE_DIR}/electromagnetism/electromagnetism.py"
            --model-output "${ELECTROMAGNETISM_HEADER}"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/electromagnetism/electromagnetism.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/electromagnetism/electromagnetism.hpp.in"
            "${CMAKE_CURRENT_SOURCE_DIR}/electrodynamics/electrodynamics.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/magnetodynamics/magnetodynamics.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/generate_cpp_field_model.py"
    COMMENT "Generating full Maxwell field model from its SymPy Hamiltonian"
    VERBATIM
)

add_custom_target(similie_physics_generated
    DEPENDS
        "${SCALAR_FIELD_HEADER}"
        "${LINEAR_MAGNETOSTATICS_HEADER}"
        "${LINEAR_MAGNETOSTATICS_CONSTITUTIVE_LAW_HEADER}"
        "${NONLINEAR_MAGNETOSTATICS_HEADER}"
        "${LINEAR_ELASTICITY_HEADER}"
        "${LINEAR_ELASTICITY_CONSTITUTIVE_LAW_HEADER}"
        "${NAVIER_STOKES_FOURIER_HEADER}"
        "${MAGNETODYNAMICS_HEADER}"
        "${MAGNETODYNAMICS_MODEL_HEADER}"
        "${ELECTRODYNAMICS_HEADER}"
        "${ELECTRODYNAMICS_MODEL_HEADER}"
        "${ELECTROMAGNETISM_HEADER}"
)

add_library(physics INTERFACE)
add_dependencies(physics similie_physics_generated)
target_include_directories(physics
    INTERFACE
        "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/src>"
)
