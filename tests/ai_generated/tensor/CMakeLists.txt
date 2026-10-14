# SPDX-FileCopyrightText: 2026 Baptiste Legouix
# SPDX-License-Identifier: AGPL-3.0-or-later
# AI-GENERATED

include(GoogleTest)

add_executable(unit_tests_tensor_ai_generated tensor.cpp ../../main.cpp)
target_link_libraries(unit_tests_tensor_ai_generated PUBLIC GTest::gtest DDC::core sil::tensor)
gtest_discover_tests(unit_tests_tensor_ai_generated DISCOVERY_MODE PRE_TEST)
