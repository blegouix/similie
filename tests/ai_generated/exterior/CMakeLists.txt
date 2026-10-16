# SPDX-FileCopyrightText: 2026 Baptiste Legouix
# SPDX-License-Identifier: AGPL-3.0-or-later
# AI-GENERATED

include(GoogleTest)

add_executable(unit_tests_staged_laplacian staged_laplacian.cpp ../../main.cpp)
target_link_libraries(unit_tests_staged_laplacian PUBLIC GTest::gtest DDC::core sil::exterior)
gtest_discover_tests(unit_tests_staged_laplacian DISCOVERY_MODE PRE_TEST)

add_executable(unit_tests_reconstruction_application reconstruction_application.cpp ../../main.cpp)
target_link_libraries(unit_tests_reconstruction_application PUBLIC GTest::gtest DDC::core sil::exterior)
gtest_discover_tests(unit_tests_reconstruction_application DISCOVERY_MODE PRE_TEST)

add_executable(unit_tests_incidence_application incidence_application.cpp ../../main.cpp)
target_link_libraries(unit_tests_incidence_application PUBLIC GTest::gtest DDC::core sil::exterior)
gtest_discover_tests(unit_tests_incidence_application DISCOVERY_MODE PRE_TEST)

add_executable(unit_tests_covariant_derivative covariant_derivative.cpp ../../main.cpp)
target_link_libraries(unit_tests_covariant_derivative PUBLIC GTest::gtest DDC::core sil::exterior)
gtest_discover_tests(unit_tests_covariant_derivative DISCOVERY_MODE PRE_TEST)

add_executable(unit_tests_extrapolation extrapolation.cpp ../../main.cpp)
target_link_libraries(unit_tests_extrapolation PUBLIC GTest::gtest DDC::core sil::exterior)
gtest_discover_tests(unit_tests_extrapolation DISCOVERY_MODE PRE_TEST)

add_executable(unit_tests_hodge_star hodge_star.cpp ../../main.cpp)

target_link_libraries(unit_tests_hodge_star
    PUBLIC
        GTest::gtest
        DDC::core
        sil::exterior
)

gtest_discover_tests(unit_tests_hodge_star DISCOVERY_MODE PRE_TEST)

add_executable(unit_tests_reduction_and_reconstruction reduction_and_reconstruction.cpp ../../main.cpp)

target_link_libraries(unit_tests_reduction_and_reconstruction
    PUBLIC
        GTest::gtest
        DDC::core
        sil::exterior
)

gtest_discover_tests(unit_tests_reduction_and_reconstruction DISCOVERY_MODE PRE_TEST)

add_executable(unit_tests_exterior_ai_generated exterior.cpp ../../main.cpp)
target_link_libraries(unit_tests_exterior_ai_generated PUBLIC GTest::gtest DDC::core sil::exterior)
gtest_discover_tests(unit_tests_exterior_ai_generated DISCOVERY_MODE PRE_TEST)

add_executable(unit_tests_stencil_indices stencil_indices.cpp ../../main.cpp)
target_link_libraries(unit_tests_stencil_indices PUBLIC GTest::gtest DDC::core sil::exterior)
gtest_discover_tests(unit_tests_stencil_indices DISCOVERY_MODE PRE_TEST)

add_executable(unit_tests_staged_local_operators staged_local_operators.cpp ../../main.cpp)
target_link_libraries(unit_tests_staged_local_operators PUBLIC GTest::gtest DDC::core sil::exterior)
gtest_discover_tests(unit_tests_staged_local_operators DISCOVERY_MODE PRE_TEST)

add_executable(diagonal_assertion_debug diagonal_assertion_probe.cpp)
add_executable(diagonal_assertion_release diagonal_assertion_probe.cpp)
target_link_libraries(diagonal_assertion_debug PRIVATE DDC::core sil::exterior)
target_link_libraries(diagonal_assertion_release PRIVATE DDC::core sil::exterior)
target_compile_definitions(diagonal_assertion_debug PRIVATE SIMILIE_TEST_DEBUG_ASSERTIONS)
target_compile_definitions(diagonal_assertion_release PRIVATE NDEBUG)
add_test(NAME StagedLocalOperators.DebugOnlyDiagonalAssertions
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/diagonal_assertions.py"
    $<TARGET_FILE:diagonal_assertion_debug> $<TARGET_FILE:diagonal_assertion_release>)
