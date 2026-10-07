# SPDX-FileCopyrightText: 2026 Baptiste Legouix
# SPDX-License-Identifier: AGPL-3.0-or-later
# AI-GENERATED

include(GoogleTest)
add_executable(unit_tests_multidomains topology.cpp ../main.cpp)
target_link_libraries(unit_tests_multidomains PRIVATE GTest::gtest sil::multidomains)
gtest_discover_tests(unit_tests_multidomains DISCOVERY_MODE PRE_TEST)
