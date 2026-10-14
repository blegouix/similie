# SPDX-FileCopyrightText: 2026 Baptiste Legouix
# SPDX-License-Identifier: AGPL-3.0-or-later
# AI-GENERATED

if(SIMILIE_BUILD_ONELAB_INTERFACE)
    include(GoogleTest)
    add_executable(unit_tests_potential_flow potential_flow.cpp ../../main.cpp)
    target_link_libraries(unit_tests_potential_flow PRIVATE GTest::gtest similie_onelab_interface)
    gtest_discover_tests(unit_tests_potential_flow DISCOVERY_MODE PRE_TEST)
    add_executable(unit_tests_elasticity elasticity.cpp ../../main.cpp)
    target_link_libraries(unit_tests_elasticity PRIVATE GTest::gtest similie_onelab_interface)
    gtest_discover_tests(unit_tests_elasticity DISCOVERY_MODE PRE_TEST)
endif()
