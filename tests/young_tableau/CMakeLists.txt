# SPDX-FileCopyrightText: 2024 Baptiste Legouix
# SPDX-License-Identifier: MIT

include(GoogleTest)

add_executable(unit_tests_young_tableau young_tableau.cpp ../main.cpp)

target_link_libraries(unit_tests_young_tableau 
    PUBLIC
        GTest::gtest
        DDC::DDC
        sil::young_tableau
) 

gtest_discover_tests(unit_tests_young_tableau DISCOVERY_MODE PRE_TEST)
