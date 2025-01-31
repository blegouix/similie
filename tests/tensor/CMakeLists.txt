# SPDX-FileCopyrightText: 2024 Baptiste Legouix
# SPDX-License-Identifier: MIT

include(GoogleTest)

add_executable(unit_tests_tensor tensor.cpp ../main.cpp)

target_link_libraries(unit_tests_tensor
    PUBLIC
        GTest::gtest
        DDC::DDC
        sil::tensor
)

gtest_discover_tests(unit_tests_tensor DISCOVERY_MODE PRE_TEST)

add_executable(unit_tests_tensor_prod tensor_prod.cpp ../main.cpp)

target_link_libraries(unit_tests_tensor_prod
    PUBLIC
        GTest::gtest
        DDC::DDC
        sil::tensor
)

gtest_discover_tests(unit_tests_tensor_prod DISCOVERY_MODE PRE_TEST)

add_executable(unit_tests_metric metric.cpp ../main.cpp)

target_link_libraries(unit_tests_metric
    PUBLIC
        GTest::gtest
        DDC::DDC
        sil::tensor
)

gtest_discover_tests(unit_tests_metric DISCOVERY_MODE PRE_TEST)
