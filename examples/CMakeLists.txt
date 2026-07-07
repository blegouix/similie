# SPDX-FileCopyrightText: 2024 Baptiste Legouix
# SPDX-License-Identifier: MIT

option(SIMILIE_ASSERT_EXAMPLE_RESULTS_CORRECTNESS "Assert example result correctness at runtime" OFF)

function(similie_configure_example_correctness target)
    if("${SIMILIE_ASSERT_EXAMPLE_RESULTS_CORRECTNESS}")
        target_compile_definitions("${target}" PRIVATE SIMILIE_ASSERT_EXAMPLE_RESULTS_CORRECTNESS)
    endif()
endfunction()

add_executable(2d_laplacian 2d_laplacian.cpp)
similie_configure_example_correctness(2d_laplacian)

target_link_libraries(2d_laplacian
    PUBLIC
        DDC::core
        DDC::pdi
        DDC::splines
        sil::sil
)

add_executable(2d_vector_laplacian 2d_vector_laplacian.cpp)
similie_configure_example_correctness(2d_vector_laplacian)

target_link_libraries(2d_vector_laplacian
    PUBLIC
        DDC::core
        DDC::pdi
        DDC::splines
        sil::sil
)

add_executable(2d_scalar_field 2d_scalar_field.cpp)
add_dependencies(2d_scalar_field similie_physics_generated)

target_link_libraries(2d_scalar_field
    PUBLIC
        DDC::core
        DDC::pdi
        DDC::splines
        sil::sil
)

if("${SIMILIE_BUILD_YOUNG_TABLEAU}")
add_executable(kretschmann_scalar kretschmann_scalar.cpp)
similie_configure_example_correctness(kretschmann_scalar)

target_link_libraries(kretschmann_scalar
    PUBLIC
        DDC::core
        DDC::pdi
        DDC::splines
        sil::tensor
)
endif()
