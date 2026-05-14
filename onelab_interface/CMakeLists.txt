# SPDX-FileCopyrightText: 2026 Baptiste Legouix
# SPDX-License-Identifier: MIT

add_library(similie_onelab_support fake_similie_run.cpp)

target_link_libraries(similie_onelab_support
    PUBLIC
        DDC::core
        sil::sil
)

target_include_directories(similie_onelab_support
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}"
)

target_compile_definitions(similie_onelab_support
    PUBLIC
        SIMILIE_ONELAB_DEFAULT_OUTPUT_DIR="${CMAKE_CURRENT_BINARY_DIR}/generated"
)

add_executable(similie_onelab main.cpp)

target_include_directories(similie_onelab
    PRIVATE
        "${SIMILIE_ONELAB_SOURCE_DIR}"
)

target_link_libraries(similie_onelab
    PRIVATE
        similie_onelab_support
)
