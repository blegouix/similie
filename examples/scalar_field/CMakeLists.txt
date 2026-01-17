# SPDX-FileCopyrightText: 2026 Baptiste Legouix
# SPDX-License-Identifier: MIT

find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(SCALAR_FIELD_HAMILTONIAN_FUNCTOR_GENERATOR
    "${CMAKE_CURRENT_SOURCE_DIR}/scalar_field_hamiltonian_functor_generator.py")
set(SCALAR_FIELD_HAMILTONIAN
    "${CMAKE_CURRENT_BINARY_DIR}/generated/scalar_field_hamiltonian.hpp")

add_custom_command(
    OUTPUT "${SCALAR_FIELD_HAMILTONIAN}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory
            "${CMAKE_CURRENT_BINARY_DIR}/generated"
    COMMAND "${Python3_EXECUTABLE}" "${SCALAR_FIELD_HAMILTONIAN_FUNCTOR_GENERATOR}" 2 "${SCALAR_FIELD_HAMILTONIAN}"
    DEPENDS "${SCALAR_FIELD_HAMILTONIAN_FUNCTOR_GENERATOR}"
    COMMENT "Generating scalar field Hamiltonian functor"
    VERBATIM
)

add_custom_target(scalar_field_hamiltonian
    DEPENDS "${SCALAR_FIELD_HAMILTONIAN}"
)

add_executable(2d_scalar_field 2d_scalar_field.cpp)
add_dependencies(2d_scalar_field scalar_field_hamiltonian)
target_sources(2d_scalar_field PRIVATE "${SCALAR_FIELD_HAMILTONIAN}")

target_include_directories(2d_scalar_field PRIVATE
    "${CMAKE_CURRENT_BINARY_DIR}/generated"
)

target_link_libraries(2d_scalar_field
    PUBLIC
        DDC::DDC
        sil::sil
)
