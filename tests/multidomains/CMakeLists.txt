# SPDX-FileCopyrightText: 2026 Baptiste Legouix
# SPDX-License-Identifier: AGPL-3.0-or-later
# AI-GENERATED

include(GoogleTest)
add_executable(unit_tests_multidomains topology.cpp ../main.cpp)
target_link_libraries(unit_tests_multidomains PRIVATE GTest::gtest sil::multidomains)
gtest_discover_tests(unit_tests_multidomains DISCOVERY_MODE PRE_TEST)

add_executable(unit_tests_multidomains_device device_laplacian.cpp ../main.cpp)
target_link_libraries(unit_tests_multidomains_device PRIVATE GTest::gtest sil::sil)
gtest_discover_tests(unit_tests_multidomains_device DISCOVERY_MODE PRE_TEST)
