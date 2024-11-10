# SPDX-FileCopyrightText: 2024 Baptiste Legouix
# SPDX-License-Identifier: GPL-3.0-or-later

include(GoogleTest)

add_executable(unit_tests_csr csr.cpp ../main.cpp)

target_link_libraries(unit_tests_csr 
    PUBLIC
        GTest::gtest
        DDC::DDC
        sil::csr
) 

gtest_discover_tests(unit_tests_csr DISCOVERY_MODE PRE_TEST)
