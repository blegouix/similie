# SPDX-FileCopyrightText: 2024 Baptiste Legouix
# SPDX-License-Identifier: MIT

include(GoogleTest)

add_executable(unit_tests_exterior exterior.cpp ../main.cpp)

target_link_libraries(unit_tests_exterior
    PUBLIC
        GTest::gtest
        DDC::core
        sil::exterior
)

gtest_discover_tests(unit_tests_exterior DISCOVERY_MODE PRE_TEST)

add_executable(unit_tests_derivative derivative.cpp ../main.cpp)

target_link_libraries(unit_tests_derivative
    PUBLIC
        GTest::gtest
        DDC::core
        sil::exterior
)

gtest_discover_tests(unit_tests_derivative DISCOVERY_MODE PRE_TEST)

add_executable(unit_tests_hodge_star hodge_star.cpp ../main.cpp)

target_link_libraries(unit_tests_hodge_star
    PUBLIC
        GTest::gtest
        DDC::core
        sil::exterior
)

gtest_discover_tests(unit_tests_hodge_star DISCOVERY_MODE PRE_TEST)

add_executable(unit_tests_reduction_and_reconstruction reduction_and_reconstruction.cpp ../main.cpp)

target_link_libraries(unit_tests_reduction_and_reconstruction
    PUBLIC
        GTest::gtest
        DDC::core
        sil::exterior
)

gtest_discover_tests(unit_tests_reduction_and_reconstruction DISCOVERY_MODE PRE_TEST)

add_executable(unit_tests_codifferential codifferential.cpp ../main.cpp)

target_link_libraries(unit_tests_codifferential
    PUBLIC
        GTest::gtest
        DDC::core
        sil::exterior
)

gtest_discover_tests(unit_tests_codifferential DISCOVERY_MODE PRE_TEST)

add_executable(unit_tests_laplacian laplacian.cpp ../main.cpp)

target_link_libraries(unit_tests_laplacian
    PUBLIC
        GTest::gtest
        DDC::core
        sil::exterior
)

gtest_discover_tests(unit_tests_laplacian DISCOVERY_MODE PRE_TEST)
