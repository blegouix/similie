# SPDX-License-Identifier: GPL-3.0

include(GoogleTest)

add_executable(unit_tests_mesher mesher.cpp ../main.cpp)

target_link_libraries(unit_tests_mesher 
    PUBLIC
        GTest::gtest
        DDC::DDC
        sil::mesher
) 

gtest_discover_tests(unit_tests_mesher DISCOVERY_MODE PRE_TEST)
