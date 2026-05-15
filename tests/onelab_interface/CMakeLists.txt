# SPDX-FileCopyrightText: 2026 Baptiste Legouix
# SPDX-License-Identifier: MIT

include(GoogleTest)

add_executable(unit_tests_onelab_interface silpro.cpp ../main.cpp)

target_link_libraries(unit_tests_onelab_interface
    PUBLIC
        GTest::gtest
        DDC::core
        sil::sil
)

target_include_directories(unit_tests_onelab_interface
    PRIVATE
        "${PROJECT_SOURCE_DIR}/onelab_interface"
)

target_compile_definitions(unit_tests_onelab_interface
    PRIVATE
        SIMILIE_ONELAB_TEST_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
)

gtest_discover_tests(unit_tests_onelab_interface DISCOVERY_MODE PRE_TEST)
