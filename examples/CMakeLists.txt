# SPDX-FileCopyrightText: 2024 Baptiste Legouix
# SPDX-License-Identifier: MIT

add_executable(2d_laplacian 2d_laplacian.cpp)

target_link_libraries(2d_laplacian
    PUBLIC
        DDC::DDC
        sil::sil
)

add_executable(2d_vector_laplacian 2d_vector_laplacian.cpp)

target_link_libraries(2d_vector_laplacian
    PUBLIC
        DDC::DDC
        sil::sil
)

find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(SCALAR_FIELD_HAMILTONIAN_SCRIPT
    "${CMAKE_CURRENT_SOURCE_DIR}/scalar_field/generate_scalar_field_hamiltonian.py")
set(SCALAR_FIELD_HAMILTONIAN_HEADER
    "${CMAKE_CURRENT_BINARY_DIR}/generated/sympy_scalar_field_hamiltonian.hpp")

add_custom_command(
    OUTPUT "${SCALAR_FIELD_HAMILTONIAN_HEADER}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory
            "${CMAKE_CURRENT_BINARY_DIR}/generated"
    COMMAND "${Python3_EXECUTABLE}" "${SCALAR_FIELD_HAMILTONIAN_SCRIPT}"
            --output "${SCALAR_FIELD_HAMILTONIAN_HEADER}"
    DEPENDS "${SCALAR_FIELD_HAMILTONIAN_SCRIPT}"
    COMMENT "Generating scalar field Hamiltonian derivative functor"
)

add_custom_target(generate_scalar_field_hamiltonian
    DEPENDS "${SCALAR_FIELD_HAMILTONIAN_HEADER}"
)

add_executable(2d_scalar_field
    scalar_field/2d_scalar_field.cpp
    "${SCALAR_FIELD_HAMILTONIAN_HEADER}"
)

add_dependencies(2d_scalar_field generate_scalar_field_hamiltonian)

target_include_directories(2d_scalar_field
    PRIVATE
        "${CMAKE_CURRENT_BINARY_DIR}/generated"
)

target_link_libraries(2d_scalar_field
    PUBLIC
        DDC::DDC
        sil::sil
)

if("${SIMILIE_BUILD_YOUNG_TABLEAU}")
add_executable(kretschmann_scalar kretschmann_scalar.cpp)

target_link_libraries(kretschmann_scalar
    PUBLIC
        DDC::DDC
        sil::tensor
)
endif()
