# SPDX-FileCopyrightText: 2026 Baptiste Legouix
# SPDX-License-Identifier: AGPL-3.0-or-later
# AI-GENERATED

if(BUILD_TESTING)
    include(GoogleTest)
    add_executable(unit_tests_electromagnetic_coupling electromagnetic_coupling.cpp ../../main.cpp)
    target_link_libraries(unit_tests_electromagnetic_coupling PRIVATE GTest::gtest sil::sil physics)
    gtest_discover_tests(unit_tests_electromagnetic_coupling DISCOVERY_MODE PRE_TEST)
    add_test(NAME ElectromagneticCoupling.MinimalHamiltonianDerivation
        COMMAND "${Python3_EXECUTABLE}" -B "${CMAKE_CURRENT_SOURCE_DIR}/electromagnetic_coupling.py")
    add_executable(unit_tests_fmhd fmhd.cpp ../../main.cpp)
    target_link_libraries(unit_tests_fmhd PRIVATE GTest::gtest sil::sil)
    gtest_discover_tests(unit_tests_fmhd DISCOVERY_MODE PRE_TEST)
    add_executable(unit_tests_ehd ehd.cpp ../../main.cpp)
    target_link_libraries(unit_tests_ehd PRIVATE GTest::gtest sil::sil)
    gtest_discover_tests(unit_tests_ehd DISCOVERY_MODE PRE_TEST)
    add_executable(unit_tests_mhd mhd.cpp ../../main.cpp)
    target_link_libraries(unit_tests_mhd PRIVATE GTest::gtest sil::sil)
    gtest_discover_tests(unit_tests_mhd DISCOVERY_MODE PRE_TEST)
endif()

if(SIMILIE_BUILD_ONELAB_INTERFACE)
    include(GoogleTest)
    add_executable(unit_tests_onelab_integration onelab_integration.cpp ../../main.cpp)
    target_link_libraries(unit_tests_onelab_integration PRIVATE GTest::gtest similie_onelab_interface)
    gtest_discover_tests(unit_tests_onelab_integration DISCOVERY_MODE PRE_TEST)
    add_executable(unit_tests_potential_flow potential_flow.cpp ../../main.cpp)
    target_link_libraries(unit_tests_potential_flow PRIVATE GTest::gtest similie_onelab_interface)
    gtest_discover_tests(unit_tests_potential_flow DISCOVERY_MODE PRE_TEST)
    add_executable(unit_tests_elasticity elasticity.cpp ../../main.cpp)
    target_link_libraries(unit_tests_elasticity PRIVATE GTest::gtest similie_onelab_interface)
    gtest_discover_tests(unit_tests_elasticity DISCOVERY_MODE PRE_TEST)
endif()

if(BUILD_TESTING)
    include(GoogleTest)
    add_executable(unit_tests_physics_api equations.cpp ../../main.cpp)
    target_link_libraries(unit_tests_physics_api PRIVATE GTest::gtest sil::sil physics)
    gtest_discover_tests(unit_tests_physics_api DISCOVERY_MODE PRE_TEST)
    add_test(NAME PhysicsAPI.SpatialHamiltonianGenerator
        COMMAND "${Python3_EXECUTABLE}" -B "${CMAKE_CURRENT_SOURCE_DIR}/generator.py"
                "${CMAKE_CXX_COMPILER}")
endif()

if(BUILD_TESTING)
    add_executable(unit_tests_magnetic_forms magnetic_forms.cpp ../../main.cpp)
    target_link_libraries(unit_tests_magnetic_forms PRIVATE GTest::gtest sil::sil physics)
    gtest_discover_tests(unit_tests_magnetic_forms DISCOVERY_MODE PRE_TEST)
endif()
