# SPDX-FileCopyrightText: 2024 Baptiste Legouix
# SPDX-License-Identifier: MIT

include(GoogleTest)

add_executable(unit_tests_misc discrete_element_evaluator.cpp ../main.cpp)

target_link_libraries(
    unit_tests_misc
    PUBLIC
        GTest::gtest
        DDC::DDC
        sil::misc
)

gtest_discover_tests(unit_tests_misc DISCOVERY_MODE PRE_TEST)
