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

add_subdirectory(free_scalar_field)

if("${SIMILIE_BUILD_YOUNG_TABLEAU}")
add_executable(kretschmann_scalar kretschmann_scalar.cpp)

target_link_libraries(kretschmann_scalar
    PUBLIC
        DDC::DDC
        sil::tensor
)
endif()
