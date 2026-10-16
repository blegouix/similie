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
