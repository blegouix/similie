# SPDX-FileCopyrightText: 2026 Baptiste Legouix
# SPDX-License-Identifier: MIT

include(GoogleTest)

add_executable(unit_tests_onelab_interface fake_similie_run.cpp ../main.cpp)

target_link_libraries(unit_tests_onelab_interface
    PUBLIC
        GTest::gtest
        similie_onelab_support
)

gtest_discover_tests(unit_tests_onelab_interface DISCOVERY_MODE PRE_TEST)
